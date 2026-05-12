
import os
os.environ['PROJ_DATA'] = "C:/Users/franzisf/PycharmProjects/.venv/Lib/site-packages/pyproj/proj_dir/share/proj"

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import rasterio
from rasterio.transform import from_bounds
import xarray as xr


REPO_ROOT   = Path(__file__).resolve().parents[1]
CONFIG_PATH = REPO_ROOT / "config.json"


def load_config() -> dict:
    if not CONFIG_PATH.exists():
        print(f"WARNING: config.json not found at {CONFIG_PATH}. Using empty config.")
        return {}
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def fix_longitude(ds: xr.Dataset) -> xr.Dataset:
    lon_name = None
    for candidate in ("longitude", "lon", "x"):
        if candidate in ds.coords:
            lon_name = candidate
            break
    if lon_name is None:
        raise ValueError("No longitude coordinate found (tried: longitude, lon, x)")

    lon = ds[lon_name].values.copy()

    lon_new = lon + 180.0
    lon_new[lon_new > 180.0] -= 360.0

    sort_idx = np.argsort(lon_new)
    lon_sorted = lon_new[sort_idx]

    ds_out = ds.copy()
    ds_out = ds_out.assign_coords(
        {lon_name: (lon_name, lon_sorted, ds[lon_name].attrs)}
    )

    for var in ds.data_vars:
        da = ds[var]
        if lon_name in da.dims:
            axis = da.dims.index(lon_name)
            reordered = np.take(da.values, sort_idx, axis=axis)
            ds_out[var] = xr.DataArray(reordered, dims=da.dims, attrs=da.attrs)

    return ds_out


def parse_age(path: Path) -> str:
    m = re.search(r"(\d+ma)", path.stem, re.IGNORECASE)
    return m.group(1) if m else path.stem.split("_")[-1]


def format_level(level_val: float) -> str:
    as_int = round(level_val * 100000)
    return f"{as_int:05d}"


def write_geotiff(data: np.ndarray, lats: np.ndarray, lons: np.ndarray,
                  out_path: Path) -> None:
    n_lat = len(lats)
    n_lon = len(lons)

    west  = -180.0
    east  =  180.0
    south =  -90.0
    north =   90.0

    transform = from_bounds(west, south, east, north, n_lon, n_lat)

    nodata   = -9999.0
    data_f32 = data.astype(np.float32)
    data_f32 = np.where(np.isnan(data_f32), nodata, data_f32)

    with rasterio.open(
        out_path,
        "w",
        driver    = "GTiff",
        height    = n_lat,
        width     = n_lon,
        count     = 1,
        dtype     = np.float32,
        crs       = rasterio.crs.CRS.from_wkt(
            "GEOGCS[\"WGS 84\",DATUM[\"WGS_1984\","
            "SPHEROID[\"WGS 84\",6378137,298.257223563]],"
            "PRIMEM[\"Greenwich\",0],"
            "UNIT[\"degree\",0.0174532925199433]]"
        ),
        transform = transform,
        nodata    = nodata,
    ) as dst:
        dst.write(data_f32, 1)


def export_geotiffs(ds: xr.Dataset, age: str, tif_dir: Path) -> int:
    lon_name = next((c for c in ("longitude", "lon", "x") if c in ds.coords), "longitude")
    lat_name = next((c for c in ("latitude",  "lat", "y") if c in ds.coords), "latitude")

    lons = ds[lon_name].values
    lats = ds[lat_name].values

    n_written = 0

    for var in ds.data_vars:
        da = ds[var]

        if "month" in da.dims:
            da = da.mean(dim="month")

        if "level" in da.dims:
            for lev_val in ds["level"].values:
                lev_da    = da.sel(level=lev_val)
                lev_label = format_level(float(lev_val))

                if da.dims[-2:] != (lat_name, lon_name):
                    lev_da = lev_da.transpose(lat_name, lon_name)

                out_path = tif_dir / f"{var}_lev{lev_label}_{age}.tif"
                write_geotiff(lev_da.values, lats, lons, out_path)
                n_written += 1

        else:
            if da.dims != (lat_name, lon_name):
                da = da.transpose(lat_name, lon_name)

            out_path = tif_dir / f"{var}_{age}.tif"
            write_geotiff(da.values, lats, lons, out_path)
            n_written += 1

    return n_written


def process_folder(input_dir: Path, output_dir: Path) -> None:
    nc_files = sorted(input_dir.glob("*.nc"))
    if not nc_files:
        print(f"ERROR: no .nc files found in {input_dir}")
        sys.exit(1)

    nc_out_dir  = output_dir
    tif_out_dir = output_dir / "geotiff"
    nc_out_dir.mkdir(parents=True, exist_ok=True)
    tif_out_dir.mkdir(parents=True, exist_ok=True)

    ok     = 0
    errors = 0

    for nc_path in nc_files:
        age    = parse_age(nc_path)
        out_nc = nc_out_dir / nc_path.name

        try:
            ds       = xr.open_dataset(str(nc_path))
            ds_fixed = fix_longitude(ds)
            ds.close()

            lon = ds_fixed["longitude"].values
            print(f"  [{age}]  lon fixed: {lon[0]:.3f} .. {lon[-1]:.3f}")

            ds_fixed.to_netcdf(str(out_nc))

            n = export_geotiffs(ds_fixed, age, tif_out_dir)
            print(f"  [{age}]  {n} GeoTIFF(s) written\n")

            ok += 1

        except Exception as exc:
            import traceback
            print(f"  ERROR [{age}] {nc_path.name}: {exc}")
            traceback.print_exc()
            errors += 1

    print(f"Finished. OK: {ok}  Errors: {errors}")


def main() -> None:
    config = load_config()

    parser = argparse.ArgumentParser(
        description="Fix longitude and export GeoTIFFs from palaeoclimate NetCDF files."
    )
    parser.add_argument(
        "--input",  "-i",
        default=config.get("input_dir"),
        help="Input directory containing raw .nc files (overrides config.json)",
    )
    parser.add_argument(
        "--output", "-o",
        default=config.get("output_dir"),
        help="Output directory for corrected files (overrides config.json)",
    )
    args = parser.parse_args()

    if not args.input:
        print("ERROR: no input_dir set. Add it to config.json or pass --input.")
        sys.exit(1)
    if not args.output:
        print("ERROR: no output_dir set. Add it to config.json or pass --output.")
        sys.exit(1)

    input_dir  = Path(args.input)
    output_dir = Path(args.output)

    if not input_dir.exists():
        print(f"ERROR: input directory not found: {input_dir}")
        sys.exit(1)

    process_folder(input_dir, output_dir)


if __name__ == "__main__":
    main()
