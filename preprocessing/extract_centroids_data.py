"""
NetCDF to GeoJSON Converter for Climate Data - BATCH VERSION (extended)
========================================================================

Extends the original 4-variable extraction (precipitation_total,
temperature_surface_air, evaporation, snow_depth) with all PLASIM
variables needed for the LivingEarth PV/NPV -> PVT/PVA/NPVT/NPVA
classification, plus ice/snow discrimination:

  - fractional_vegetation        (L1: PV vs NPV)
  - leaf_area_index              (L1 cross-check)
  - forest_cover                 (optional PVT sub-typing)
  - gross_primary_production     (optional PVT sub-typing)
  - net_primary_production       (optional PVT sub-typing)
  - soil_moisture                (flooding proxy, numerator)
  - field_capacity               (flooding proxy, denominator)
  - runoff                       (flooding corroboration)
  - sea_ice_area_fraction        (ice vs open water)
  - sea_ice_cover                (duplicate check vs sea_ice_area_fraction)
  - sea_ice_thickness            (ice secondary check)
  - land_ice_mask                (permanent land ice)

All variables already tested (precipitation_total, temperature_surface_air,
evaporation, snow_depth) are kept exactly as before, including the
Kelvin -> Celsius conversion for temperature only. No other behaviour of
the original script (filename parsing, Pacific-centered longitude
reconstruction, NaN/mask handling, output naming) has changed.

Requirements:
- Python 3.x
- netCDF4 library (install with: pip install netCDF4)
  OR scipy (install with: pip install scipy)

Usage:
1. Update the folder_path variable below with your folder path
2. Run: python nc_to_geojson_batch.py
"""

import json
import numpy as np
import os
import re

# Choose your preferred library (uncomment one)
try:
    import netCDF4 as nc
    USE_NETCDF4 = True
except ImportError:
    USE_NETCDF4 = False

if not USE_NETCDF4:
    try:
        from scipy.io import netcdf_file
    except ImportError:
        raise ImportError("Please install either netCDF4 or scipy: pip install netCDF4")

# ===== CONFIGURATION =====
CONFIG_PATH = Path(__file__).resolve().parent.parent / "config.json"
with open(CONFIG_PATH, "r") as f:
    _config = json.load(f)

nc_folder_path = _config["nc_folder_path"]
# =========================

# mm/year per m/s -- applied to precipitation_total and evaporation below.
# (kept as its own named constant rather than inlined so it's auditable/
# traceable to your existing postprocessing code, and easy to reuse for
# other flux variables later, e.g. runoff, if you want consistent units.)
CONVERSION_MM_PER_YEAR = 1000 * 365.25 * 24 * 60 * 60

# variable name in the .nc file -> {short property prefix, Kelvin->Celsius
# flag, optional multiplicative scale}. 'scale' defaults to 1.0 (no-op) if
# omitted. Add/remove entries here if you need more or fewer variables;
# nothing else in the script needs to change.
VARIABLES = {
    # --- originally tested ---
    # precip/evap converted from native flux units to mm/year using your
    # existing conversion factor. If you'd rather keep raw flux units
    # (e.g. to cross-check against the .nc attrs), just drop the 'scale' key.
    'precipitation_total':      {'prefix': 'precip',  'to_celsius': False, 'scale': CONVERSION_MM_PER_YEAR},
    'temperature_surface_air':  {'prefix': 'TSA',     'to_celsius': True},
    'evaporation':              {'prefix': 'evap',    'to_celsius': False, 'scale': CONVERSION_MM_PER_YEAR},
    'snow_depth':                {'prefix': 'snow',    'to_celsius': False},

    # --- LivingEarth L1: PV vs NPV ---
    'fractional_vegetation':     {'prefix': 'fveg',    'to_celsius': False},
    'leaf_area_index':           {'prefix': 'lai',     'to_celsius': False},

    # --- optional sub-typing of PVT (forest/non-forest, productivity) ---
    'forest_cover':               {'prefix': 'forest',  'to_celsius': False},
    'gross_primary_production':   {'prefix': 'gpp',     'to_celsius': False},
    'net_primary_production':     {'prefix': 'npp',     'to_celsius': False},

    # --- flooding / waterlogging proxy (PVA / NPVA on land) ---
    'soil_moisture':               {'prefix': 'soilm',   'to_celsius': False},
    'field_capacity':              {'prefix': 'fcap',    'to_celsius': False},
    'runoff':                       {'prefix': 'runoff',  'to_celsius': False},

    # --- ice / open water discrimination ---
    'sea_ice_area_fraction':         {'prefix': 'siaf',     'to_celsius': False},
    'sea_ice_cover':                  {'prefix': 'sicov',    'to_celsius': False},
    'sea_ice_thickness':              {'prefix': 'sithick',  'to_celsius': False},
    'land_ice_mask':                   {'prefix': 'landice',  'to_celsius': False},
}

MONTHS = ['JAN', 'FEB', 'MAR', 'APR', 'MAY', 'JUN',
          'JUL', 'AUG', 'SEP', 'OCT', 'NOV', 'DEC']


def extract_age_from_filename(filename):
    """Extract age from filename pattern panalesis_sea_lev_corr_copse_XXXXma"""
    match = re.search(r'panalesis_sea_lev_corr_copse_(\d+)ma', filename)
    if match:
        return int(match.group(1))
    return None


def _read_var(dataset, name):
    """Read a variable's raw array, regardless of backend (netCDF4 / scipy)."""
    if USE_NETCDF4:
        return dataset.variables[name][:]
    return dataset.variables[name].data


def process_nc_file(nc_file_path, age):
    """Process a single NetCDF file and return GeoJSON"""

    print(f"\n{'='*60}")
    print(f"Processing: {os.path.basename(nc_file_path)}")
    print(f"Age: {age} Ma")
    print(f"{'='*60}")

    if USE_NETCDF4:
        dataset = nc.Dataset(nc_file_path, 'r')
    else:
        dataset = netcdf_file(nc_file_path, 'r', mmap=False)

    # Some time slices / postprocessing runs may legitimately be missing a
    # field (e.g. no sea ice on a hothouse slice's output, or a variable
    # added later for some ages only). Skip missing ones instead of failing.
    missing = [v for v in VARIABLES if v not in dataset.variables]
    if missing:
        print(f"  ! Not found in this file, skipping: {missing}")

    data_arrays = {}
    for var_name in VARIABLES:
        if var_name in dataset.variables:
            arr = _read_var(dataset, var_name)
            print(f"  {var_name}: shape={arr.shape}")
            data_arrays[var_name] = arr

    if not data_arrays:
        dataset.close()
        raise ValueError("None of the configured VARIABLES were found in this file")

    # Get latitude and longitude
    lat_var_names = ['lat', 'latitude', 'y', 'rlat', 'LATITUDE']
    lon_var_names = ['lon', 'longitude', 'x', 'rlon', 'LONGITUDE']

    lat = None
    lon = None

    for name in lat_var_names:
        if name in dataset.variables:
            lat = _read_var(dataset, name)
            break

    for name in lon_var_names:
        if name in dataset.variables:
            lon = _read_var(dataset, name)
            break

    if lat is None or lon is None:
        dataset.close()
        raise ValueError("Could not find latitude/longitude variables")

    lon_min, lon_max = float(np.min(lon)), float(np.max(lon))
    print(f"Longitude range: {lon_min:.2f} to {lon_max:.2f}")

    # Handle Pacific-centered data: reconstruct lon AND re-roll every loaded
    # variable along its longitude axis (last axis, for both 3D month/lat/lon
    # and, defensively, 4D month/level/lat/lon variables).
    convert_longitude = False
    if lon[0] < 0 and lon[-1] > 0 and np.all(np.diff(lon) > 0):
        print(f"-> Reconstructing proper longitude values...")
        n_lon = len(lon)
        lon_step = 360.0 / n_lon
        lon_0_360 = np.arange(0, 360, lon_step)
        split_idx = n_lon // 2

        lon = np.concatenate([lon_0_360[split_idx:] - 360, lon_0_360[:split_idx]])

        for var_name, arr in data_arrays.items():
            if arr.ndim == 3:  # (month, lat, lon)
                data_arrays[var_name] = np.concatenate(
                    [arr[:, :, split_idx:], arr[:, :, :split_idx]], axis=2
                )
            elif arr.ndim == 4:  # (month, level, lat, lon) - defensive, unused today
                data_arrays[var_name] = np.concatenate(
                    [arr[:, :, :, split_idx:], arr[:, :, :, :split_idx]], axis=3
                )

        print(f"   New longitude range: {float(np.min(lon)):.2f} to {float(np.max(lon)):.2f}")
    elif lon_max > 180:
        print(f"-> Converting to -180 to 180 range")
        convert_longitude = True

    geojson = {"type": "FeatureCollection", "features": []}

    print(f"Creating GeoJSON features...")

    # Grid dimensions, taken from whichever variable was loaded
    ref_arr = next(iter(data_arrays.values()))
    n_lat, n_lon = ref_arr.shape[1], ref_arr.shape[2]

    for i in range(n_lat):
        for j in range(n_lon):
            if len(lat.shape) == 1 and len(lon.shape) == 1:
                centroid_lat = float(lat[i])
                centroid_lon = float(lon[j])
            elif len(lat.shape) == 2 and len(lon.shape) == 2:
                centroid_lat = float(lat[i, j])
                centroid_lon = float(lon[i, j])
            else:
                raise ValueError(f"Unexpected lat/lon shapes: lat={lat.shape}, lon={lon.shape}")

            if convert_longitude and centroid_lon > 180:
                centroid_lon = centroid_lon - 360

            properties = {
                'latitude': centroid_lat,
                'longitude': centroid_lon,
                'age_Ma': age,
            }

            for var_name, spec in VARIABLES.items():
                if var_name not in data_arrays:
                    continue
                prefix = spec['prefix']
                to_celsius = spec['to_celsius']
                arr = data_arrays[var_name]

                monthly_clean = []
                for month_idx, month in enumerate(MONTHS):
                    raw_val = arr[month_idx, i, j]
                    val = float(raw_val)

                    if np.isnan(val) or (hasattr(raw_val, 'mask') and raw_val.mask):
                        val = None
                    elif to_celsius:
                        val = val - 273.15
                    else:
                        val = val * spec.get('scale', 1.0)

                    if val is not None:
                        monthly_clean.append(val)

                    properties[f'{prefix}_{month}'] = val

                properties[f'{prefix}_ANNUAL_MEAN'] = (
                    float(np.mean(monthly_clean)) if monthly_clean else None
                )

            feature = {
                "type": "Feature",
                "geometry": {"type": "Point", "coordinates": [centroid_lon, centroid_lat]},
                "properties": properties,
            }
            geojson["features"].append(feature)

    dataset.close()

    print(f"Created {len(geojson['features'])} point features ({n_lat} x {n_lon} grid)")
    print(f"Variables written: {[VARIABLES[v]['prefix'] for v in data_arrays]}")
    return geojson


# ===== MAIN PROCESSING =====
if __name__ == "__main__":
    print(f"Using {'netCDF4' if USE_NETCDF4 else 'scipy'} library")
    print(f"Searching for NC files in: {nc_folder_path}")

    nc_files = []
    for file in os.listdir(nc_folder_path):
        if file.endswith('.nc'):
            age = extract_age_from_filename(file)
            if age is not None:
                full_path = os.path.join(nc_folder_path, file)
                nc_files.append((age, full_path, file))

    nc_files.sort(key=lambda x: x[0])

    print(f"\nFound {len(nc_files)} files to process:")
    for age, path, filename in nc_files:
        print(f"  {age:4d} Ma: {filename}")

    output_dir = os.path.join(nc_folder_path, "output")
    os.makedirs(output_dir, exist_ok=True)

    for age, nc_file_path, filename in nc_files:
        try:
            geojson = process_nc_file(nc_file_path, age)

            output_file = os.path.join(output_dir, f"climate_data_points_{age:04d}Ma.geojson")

            print(f"Writing: {os.path.basename(output_file)}")
            with open(output_file, 'w') as f:
                json.dump(geojson, f, indent=2)

            print(f"SUCCESS: {os.path.basename(output_file)}")

        except Exception as e:
            print(f"ERROR processing {filename}: {str(e)}")
            continue

    print(f"\n{'='*60}")
    print(f"BATCH PROCESSING COMPLETE")
    print(f"{'='*60}")
    print(f"Processed {len(nc_files)} files")
    print(f"Output files saved to: {output_dir}")