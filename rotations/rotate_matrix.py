"""
Plate-tectonic pixel rotation and age-to-age change matrix.

Rotates map pixels (in ESRI:54034 metres) back/forward through geological
time using the per-plate Euler pole rotations stored in a GeoPackage, and
can build a full source-code -> destination-code change matrix across an
entire raster grid.

Each layer in the GeoPackage, e.g. "Plates006preRot000", encodes how to
rotate the plate geometries FROM age 6 Ma TO age 0 Ma. The layer whose name
starts with "Plates{age:03d}preRot" is the one that applies to a pixel
currently expressed at that age.

Single-pixel workflow (rotate_pixel / rotate_raster_pixel):
    1. Convert the input pixel from ESRI:54034 (metres) to lon/lat (EPSG:4326).
    2. Load the layer matching the requested age.
    3. Find which plate polygon contains the point.
    4. Read that plate's Euler pole (EuLong, EuLat, EuAng).
    5. Rotate the point on the sphere around that pole by that angle
       (Rodrigues' rotation formula - the standard plate-tectonics
       convention: right-hand rotation about the pole vector).
    6. Convert the rotated lon/lat back to ESRI:54034 metres.

Whole-grid workflow (build_change_matrix):
    Same idea, but every step is a vectorized numpy/rasterio array
    operation over all pixels at once instead of a per-pixel Python loop
    - rasterize plate polygons onto the grid, rotate every pixel's lon/lat
    about its own plate's pole in one batch, then look up the destination
    raster's value at each rotated position. Fast enough for full grids
    (e.g. 1273 x 4008 -> ~5.1M pixels in single-digit seconds).

The map rasters live under:
    <RASTER_BASE_DIR>\\<2000 + age>\\level4_out_LE_DT.tif
e.g. age=6 Ma -> ...\\test_processing\\output\\2006\\level4_out_LE_DT.tif

Requires: geopandas, shapely, pyproj, numpy, pandas, fiona, rasterio
"""

import re
from dataclasses import dataclass
from pathlib import Path

import fiona
import geopandas as gpd
import numpy as np
import pandas as pd
import json
from pathlib import Path

pd.set_option("display.max_columns", None)
pd.set_option("display.max_rows", None)
pd.set_option("display.width", None)
import rasterio
from pyproj import CRS, Transformer
from rasterio.features import rasterize
from shapely.geometry import Point

# ============================================================================
# Configuration
# ============================================================================
CONFIG_PATH = Path(__file__).resolve().parent.parent / "config.json"
with open(CONFIG_PATH, "r") as f:
    _config = json.load(f)

GPKG_PATH = _config["plates_gpkg"]

RASTER_BASE_DIR = _config["LE_outputs"]
RASTER_FILENAME = "level4_out_LE_DT.tif"
RASTER_AGE_OFFSET = 2000  # folder name = RASTER_AGE_OFFSET + age (Ma)

ESRI_54034 = "ESRI:54034"
EPSG_4326 = "EPSG:4326"

LAYER_RE = re.compile(r"^Plates(\d+)preRot(\d+)$")

# Flip to -1.0 if rotations come out mirrored relative to what you expect -
# the sign convention for EuAng depends on how the source data was produced.
ANGLE_SIGN = 1.0

BAND = 4
VALID_CODES = [19, 20, 21, 55, 94, 98, 99, 105, 255]
NODATA_CODE = 255

UNMAPPED_LABEL = "unmapped_plate"   # pixel not inside any plate polygon
OFFGRID_LABEL = "off_grid"          # rotated position falls outside target raster

_ESRI_54034_CRS = CRS.from_user_input(ESRI_54034)
_to_lonlat = Transformer.from_crs(ESRI_54034, EPSG_4326, always_xy=True)
_to_meters = Transformer.from_crs(EPSG_4326, ESRI_54034, always_xy=True)


def _crs_matches_54034(raster_crs) -> bool:
    """True if raster_crs is the same projection as ESRI:54034, even if it
    lacks the ESRI:54034 authority tag (e.g. a raw WKT written by some GIS
    tools omits the registry code while keeping identical parameters)."""
    if raster_crs is None:
        return False
    try:
        return CRS.from_user_input(raster_crs) == _ESRI_54034_CRS
    except Exception:
        return False


# ============================================================================
# Layer discovery
# ============================================================================

def build_layer_lookup(gpkg_path: str) -> dict:
    """Map 'from age' (Ma, as int) -> layer name, by parsing layer names."""
    lookup = {}
    for name in fiona.listlayers(gpkg_path):
        m = LAYER_RE.match(name)
        if m:
            from_age = int(m.group(1))
            lookup[from_age] = name
    return lookup


def get_layer_for_age(gpkg_path: str, age: float) -> str:
    lookup = build_layer_lookup(gpkg_path)
    age_int = int(round(age))
    if age_int not in lookup:
        available = ", ".join(str(a) for a in sorted(lookup))
        raise ValueError(
            f"No layer found starting from age {age} Ma. "
            f"Available 'from' ages are: {available}"
        )
    return lookup[age_int]


# ============================================================================
# Coordinate transforms: ESRI:54034 (metres) <-> EPSG:4326 (lon/lat)
# ============================================================================

def meters_to_lonlat(x: float, y: float) -> tuple:
    lon, lat = _to_lonlat.transform(x, y)
    return lon, lat


def lonlat_to_meters(lon: float, lat: float) -> tuple:
    x, y = _to_meters.transform(lon, lat)
    return x, y


# ============================================================================
# Spherical Euler-pole rotation (Rodrigues' rotation formula)
# ============================================================================

def lonlat_to_xyz(lon_deg: float, lat_deg: float) -> np.ndarray:
    lon = np.radians(lon_deg)
    lat = np.radians(lat_deg)
    x = np.cos(lat) * np.cos(lon)
    y = np.cos(lat) * np.sin(lon)
    z = np.sin(lat)
    return np.array([x, y, z])


def xyz_to_lonlat(vec: np.ndarray) -> tuple:
    x, y, z = vec
    lat = np.degrees(np.arcsin(np.clip(z, -1.0, 1.0)))
    lon = np.degrees(np.arctan2(y, x))
    return lon, lat


def rotate_point(lon: float, lat: float, pole_lon: float, pole_lat: float,
                  angle_deg: float) -> tuple:
    """Rotate (lon, lat) about the Euler pole (pole_lon, pole_lat) by
    angle_deg degrees, using Rodrigues' rotation formula on the unit sphere."""
    p = lonlat_to_xyz(lon, lat)
    k = lonlat_to_xyz(pole_lon, pole_lat)  # rotation axis (unit vector)
    theta = np.radians(ANGLE_SIGN * angle_deg)

    p_rot = (
        p * np.cos(theta)
        + np.cross(k, p) * np.sin(theta)
        + k * np.dot(k, p) * (1 - np.cos(theta))
    )
    p_rot = p_rot / np.linalg.norm(p_rot)
    return xyz_to_lonlat(p_rot)


# ============================================================================
# Vectorized version of the same rotation, for whole arrays of points at once
# (each point rotated about its own per-pixel pole/angle)
# ============================================================================

def lonlat_to_xyz_arr(lon_deg, lat_deg):
    lon = np.radians(lon_deg)
    lat = np.radians(lat_deg)
    x = np.cos(lat) * np.cos(lon)
    y = np.cos(lat) * np.sin(lon)
    z = np.sin(lat)
    return np.stack([x, y, z], axis=-1)  # (N, 3)


def xyz_to_lonlat_arr(vec):
    x, y, z = vec[..., 0], vec[..., 1], vec[..., 2]
    lat = np.degrees(np.arcsin(np.clip(z, -1.0, 1.0)))
    lon = np.degrees(np.arctan2(y, x))
    return lon, lat


def rotate_points_vectorized(lon, lat, pole_lon, pole_lat, angle_deg):
    """Rotate arrays of points, each about its OWN pole/angle (per-pixel,
    since different pixels can belong to different plates)."""
    p = lonlat_to_xyz_arr(lon, lat)              # (N, 3)
    k = lonlat_to_xyz_arr(pole_lon, pole_lat)    # (N, 3)
    theta = np.radians(ANGLE_SIGN * angle_deg)[:, None]  # (N, 1)

    dot_kp = np.einsum("ij,ij->i", k, p)[:, None]
    cross_kp = np.cross(k, p)

    p_rot = p * np.cos(theta) + cross_kp * np.sin(theta) + k * dot_kp * (1 - np.cos(theta))
    p_rot = p_rot / np.linalg.norm(p_rot, axis=1, keepdims=True)
    return xyz_to_lonlat_arr(p_rot)


# ============================================================================
# Raster access (maps stored per-age as <2000+age>\level4_out_LE_DT.tif)
# ============================================================================

def get_raster_path(age: float) -> str:
    """Build the path to the map raster for a given age, e.g. age=6 ->
    <RASTER_BASE_DIR>\\2006\\level4_out_LE_DT.tif"""
    folder = str(RASTER_AGE_OFFSET + int(round(age)))
    return str(Path(RASTER_BASE_DIR) / folder / RASTER_FILENAME)


def raster_pixel_to_xy(raster_path: str, row: int, col: int) -> tuple:
    """Return the (x, y) map-CRS coordinate of a pixel's centre, and warn
    if the raster's CRS isn't equivalent to ESRI:54034 as expected."""
    with rasterio.open(raster_path) as src:
        if not _crs_matches_54034(src.crs):
            print(f"  [warning] raster CRS is {src.crs}, not equivalent to {ESRI_54034}. "
                  f"Coordinates below are in the raster's native CRS.")
        x, y = src.xy(row, col)  # pixel centre, in the raster's CRS units
    return x, y


def xy_to_raster_pixel(raster_path: str, x: float, y: float) -> tuple:
    """Return the (row, col) of the raster pixel containing map coordinate (x, y)."""
    with rasterio.open(raster_path) as src:
        row, col = src.index(x, y)
    return row, col


def sample_raster_value(raster_path: str, x: float, y: float, band: int = 1):
    """Read the raster value at map coordinate (x, y)."""
    with rasterio.open(raster_path) as src:
        for val in src.sample([(x, y)], indexes=band):
            return val[0]


def read_band(raster_path: str, band: int = BAND):
    with rasterio.open(raster_path) as src:
        if not _crs_matches_54034(src.crs):
            print(f"  [warning] {raster_path} CRS is not equivalent to {ESRI_54034}: {src.crs}")
        array = src.read(band)
        transform = src.transform
        shape = (src.height, src.width)
    return array, transform, shape


def pixel_centers(shape, transform):
    """Vectorized (x, y) map-CRS coordinates for every pixel centre."""
    rows, cols = np.indices(shape)
    xs, ys = transform * (cols.ravel() + 0.5, rows.ravel() + 0.5)
    return np.asarray(xs), np.asarray(ys)


def xy_to_rowcol(x, y, transform, shape):
    """Vectorized inverse: map-CRS (x, y) -> (row, col) in a grid, plus a
    boolean mask of which points actually fall inside that grid."""
    inv = ~transform
    cols, rows = inv * (x, y)
    rows = np.floor(rows).astype(np.int64)
    cols = np.floor(cols).astype(np.int64)
    in_bounds = (rows >= 0) & (rows < shape[0]) & (cols >= 0) & (cols < shape[1])
    return rows, cols, in_bounds


# ============================================================================
# Plate lookup (single point)
# ============================================================================

@dataclass
class PlateMatch:
    plate_name: str
    eu_long: float
    eu_lat: float
    eu_ang: float
    row: "gpd.GeoSeries"


def find_plate(gdf: gpd.GeoDataFrame, lon: float, lat: float) -> PlateMatch:
    point = Point(lon, lat)
    hits = gdf[gdf.geometry.contains(point)]
    if hits.empty:
        # Fall back to intersects (handles points exactly on a boundary)
        hits = gdf[gdf.geometry.intersects(point)]
    if hits.empty:
        raise ValueError(f"No plate polygon contains point (lon={lon}, lat={lat})")
    if len(hits) > 1:
        print(f"  [warning] {len(hits)} overlapping plates matched; using the first one.")
    row = hits.iloc[0]
    return PlateMatch(
        plate_name=row["PlateName"],
        eu_long=row["EuLong"],
        eu_lat=row["EuLat"],
        eu_ang=row["EuAng"],
        row=row,
    )


# ============================================================================
# Single-pixel rotation
# ============================================================================

def rotate_pixel(gpkg_path: str, age: float, x_54034: float, y_54034: float,
                  verbose: bool = True) -> tuple:
    """
    Rotate a pixel given in ESRI:54034 metres, currently expressed at `age`
    Ma, using that age's plate Euler pole. Returns (new_x, new_y) in
    ESRI:54034 metres.
    """
    layer_name = get_layer_for_age(gpkg_path, age)
    gdf = gpd.read_file(gpkg_path, layer=layer_name)

    lon, lat = meters_to_lonlat(x_54034, y_54034)
    plate = find_plate(gdf, lon, lat)

    new_lon, new_lat = rotate_point(lon, lat, plate.eu_long, plate.eu_lat, plate.eu_ang)
    new_x, new_y = lonlat_to_meters(new_lon, new_lat)

    if verbose:
        print(f"Layer            : {layer_name}")
        print(f"Input (m, 54034) : ({x_54034}, {y_54034})")
        print(f"Input (lon,lat)  : ({lon:.4f}, {lat:.4f})")
        print(f"Plate            : {plate.plate_name}")
        print(f"Euler pole       : lon={plate.eu_long}, lat={plate.eu_lat}, angle={plate.eu_ang}")
        print(f"Rotated (lon,lat): ({new_lon:.4f}, {new_lat:.4f})")
        print(f"Rotated (m,54034): ({new_x:.2f}, {new_y:.2f})")

    return new_x, new_y


def rotate_raster_pixel(age: float, row: int, col: int, gpkg_path: str = GPKG_PATH,
                         verbose: bool = True) -> tuple:
    """
    Convenience wrapper: given the age (which selects both the raster
    folder and the rotation layer) and a (row, col) pixel index into that
    age's raster, look up the pixel's map coordinate, rotate it, and return
    the new (x, y) in ESRI:54034 metres.
    """
    raster_path = get_raster_path(age)
    x, y = raster_pixel_to_xy(raster_path, row, col)
    if verbose:
        print(f"Raster           : {raster_path}")
        print(f"Pixel (row,col)  : ({row}, {col}) -> (x,y) = ({x:.2f}, {y:.2f})")
    return rotate_pixel(gpkg_path, age, x, y, verbose=verbose)


# ============================================================================
# Whole-grid rotation + change matrix
# ============================================================================

def build_change_matrix(age_from: float, age_to: float = 0, band: int = BAND,
                         gpkg_path: str = GPKG_PATH, save_csv: bool = True):
    print(f"Building change matrix: {age_from} Ma -> {age_to} Ma, band {band}")

    src_path = get_raster_path(age_from)
    dst_path = get_raster_path(age_to)
    print(f"  source raster: {src_path}")
    print(f"  target raster: {dst_path}")

    src_array, src_transform, src_shape = read_band(src_path, band)
    dst_array, dst_transform, dst_shape = read_band(dst_path, band)

    n_pixels = src_array.size
    src_codes = src_array.ravel()

    # 1. Pixel centres of the source grid, in map CRS.
    xs, ys = pixel_centers(src_shape, src_transform)

    # 2. To lon/lat.
    lon, lat = _to_lonlat.transform(xs, ys)
    lon = np.asarray(lon)
    lat = np.asarray(lat)

    # 3. Rasterize the source age's plates onto the source grid -> plate id per pixel.
    layer_name = get_layer_for_age(gpkg_path, age_from)
    gdf = gpd.read_file(gpkg_path, layer=layer_name).reset_index(drop=True)
    # The polygons are in EPSG:4326 (lon/lat) but the raster's transform is in
    # ESRI:54034 (metres) - rasterize() needs both in the same CRS, so reproject.
    gdf_projected = gdf.to_crs(ESRI_54034)
    shapes = list(zip(gdf_projected.geometry, gdf_projected.index))
    plate_id_grid = rasterize(
        shapes=shapes, out_shape=src_shape, transform=src_transform,
        fill=-1, dtype="int32",
    )
    plate_id = plate_id_grid.ravel()
    mapped_mask = plate_id >= 0

    print(f"  {mapped_mask.sum():,} / {n_pixels:,} pixels fall inside a plate polygon")

    # 4. Per-pixel Euler pole, only where mapped.
    eu_long_vals = gdf["EuLong"].to_numpy()
    eu_lat_vals = gdf["EuLat"].to_numpy()
    eu_ang_vals = gdf["EuAng"].to_numpy()

    pole_lon = np.full(n_pixels, np.nan)
    pole_lat = np.full(n_pixels, np.nan)
    pole_ang = np.full(n_pixels, np.nan)
    pole_lon[mapped_mask] = eu_long_vals[plate_id[mapped_mask]]
    pole_lat[mapped_mask] = eu_lat_vals[plate_id[mapped_mask]]
    pole_ang[mapped_mask] = eu_ang_vals[plate_id[mapped_mask]]

    # 5. Rotate mapped pixels.
    new_lon = np.full(n_pixels, np.nan)
    new_lat = np.full(n_pixels, np.nan)
    new_lon[mapped_mask], new_lat[mapped_mask] = rotate_points_vectorized(
        lon[mapped_mask], lat[mapped_mask],
        pole_lon[mapped_mask], pole_lat[mapped_mask], pole_ang[mapped_mask],
    )

    # 6. Back to ESRI:54034, then to (row, col) in the TARGET grid.
    new_x = np.full(n_pixels, np.nan)
    new_y = np.full(n_pixels, np.nan)
    new_x[mapped_mask], new_y[mapped_mask] = _to_meters.transform(
        new_lon[mapped_mask], new_lat[mapped_mask]
    )

    dst_rows = np.full(n_pixels, -1, dtype=np.int64)
    dst_cols = np.full(n_pixels, -1, dtype=np.int64)
    in_bounds = np.zeros(n_pixels, dtype=bool)
    r, c, ib = xy_to_rowcol(new_x[mapped_mask], new_y[mapped_mask], dst_transform, dst_shape)
    dst_rows[mapped_mask] = r
    dst_cols[mapped_mask] = c
    in_bounds[mapped_mask] = ib

    print(f"  {in_bounds.sum():,} / {mapped_mask.sum():,} mapped pixels rotate to within the target grid")

    # 7. Destination code per pixel: real code where in bounds, else a label.
    dest_code = np.empty(n_pixels, dtype=object)
    dest_code[:] = UNMAPPED_LABEL
    dest_code[mapped_mask] = OFFGRID_LABEL
    valid = mapped_mask & in_bounds
    dest_code[valid] = dst_array[dst_rows[valid], dst_cols[valid]]

    # 8. Cross-tabulate.
    df = pd.DataFrame({"source_code": src_codes, "dest_code": dest_code})
    matrix = pd.crosstab(df["source_code"], df["dest_code"])

    # Reorder so the known codes come first, in the given order, followed by
    # the unmapped/off_grid buckets and anything unexpected.
    ordered_cols = [c for c in VALID_CODES if c in matrix.columns]
    extra_cols = [c for c in matrix.columns if c not in VALID_CODES]
    matrix = matrix[ordered_cols + extra_cols]

    ordered_rows = [r for r in VALID_CODES if r in matrix.index]
    extra_rows = [r for r in matrix.index if r not in VALID_CODES]
    matrix = matrix.loc[ordered_rows + extra_rows]

    print("\nChange matrix (rows = source code at "
          f"{age_from} Ma, columns = destination code at {age_to} Ma):\n")
    print(matrix)

    if save_csv:
        out_path = Path(f"change_matrix_{age_from}Ma_to_{age_to}Ma_band{band}.csv")
        matrix.to_csv(out_path)
        print(f"\nSaved to {out_path.resolve()}")

    return matrix


# ============================================================================
# Entry point
# ============================================================================
if __name__ == "__main__":
    # Your list of years as strings
    years_list = [
        "2000", "2006", "2011", "2015", "2020", "2033", "2040", "2048",
        "2056", "2068", "2094", "2100", "2113", "2120", "2133", "2140",
        "2154", "2165", "2180", "2200", "2210", "2220", "2230", "2240",
        "2250", "2270", "2290", "2300", "2315", "2331", "2350", "2370",
        "2383", "2393", "2408", "2420", "2444", "2463", "2475", "2489",
        "2500", "2518", "2535", "2545"
    ]

    # Convert to integers first for easier math, then to float Ma
    ages_int = [int(y) for y in years_list]

    # Iterate up to the second-to-last element
    for i in range(len(ages_int) - 1):
        year_to = ages_int[i]
        year_from = ages_int[i + 1]

        # Convert to Ma (e.g., 2006 -> 6.0)
        age_to = float(year_to - RASTER_AGE_OFFSET)
        age_from = float(year_from - RASTER_AGE_OFFSET)

        print(f"\n--- Processing Transition: {year_from} ({age_from} Ma) -> {year_to} ({age_to} Ma) ---")

        try:
            build_change_matrix(age_from=age_from, age_to=age_to)
        except Exception as e:
            print(f"Error processing {age_from} -> {age_to}: {e}")