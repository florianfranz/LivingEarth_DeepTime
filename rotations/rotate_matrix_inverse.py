"""
Complete Plate Tectonic Change & Subduction Matrix Generator (FINAL FIX v2).

Key Logic:
- Uses TWO distinct GeoPackage layers for comparison:
  1. SOURCE LAYER (age_from): Defines plate positions at T1 and poles to T2.
  2. DEST LAYER (age_to): Defines plate positions at T2 (the ground truth).
- For 0 Ma, loads the specific 'Plates_000.gpkg'.
- Fixes array broadcasting errors by ensuring all masks are full-size.
"""

import re
import json
from pathlib import Path

import fiona
import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from pyproj import CRS, Transformer
from rasterio.features import rasterize
from shapely.ops import transform as shapely_transform

pd.set_option("display.max_columns", None)
pd.set_option("display.max_rows", None)
pd.set_option("display.width", None)

# ============================================================================
# Configuration
# ============================================================================
CONFIG_PATH = Path(__file__).resolve().parent.parent / "config.json"
with open(CONFIG_PATH, "r") as f:
    _config = json.load(f)

GPKG_BASE_DIR = Path(_config["plates_gpkg"]).parent
GPKG_MAIN = _config["plates_gpkg"]
RASTER_BASE_DIR = _config["LE_outputs"]
RASTER_FILENAME = "level4_out_LE_DT.tif"
RASTER_AGE_OFFSET = 2000

# Specific GPKG for 0 Ma (Final State)
GPKG_0MA = str(GPKG_BASE_DIR / "Plates_000.gpkg")

ESRI_54034 = "ESRI:54034"
EPSG_4326 = "EPSG:4326"

LAYER_RE = re.compile(r"^Plates(\d+)preRot(\d+)$")

ANGLE_SIGN = 1.0
BAND = 4
VALID_CODES = [20, 21, 56, 94, 98, 99, 106, 105, 255]

UNMAPPED_LABEL = "unmapped_plate"
OFFGRID_LABEL = "off_grid"

_ESRI_54034_CRS = CRS.from_user_input(ESRI_54034)
_to_lonlat = Transformer.from_crs(ESRI_54034, EPSG_4326, always_xy=True)
_to_meters = Transformer.from_crs(EPSG_4326, ESRI_54034, always_xy=True)


# ============================================================================
# Helper Functions
# ============================================================================

def get_gpkg_path_for_age(age_to: float):
    """Finds the GPKG and Layer name that defines the topology AT age_to."""
    if int(round(age_to)) == 0:
        # Special case for 0 Ma
        gpkg_path = GPKG_0MA
        layers = fiona.listlayers(gpkg_path)
        if not layers:
            raise ValueError(f"No layers found in {gpkg_path}")
        return gpkg_path, layers[0]

    # Search main GPKG for a layer starting with age_to
    for name in fiona.listlayers(GPKG_MAIN):
        m = LAYER_RE.match(name)
        if m:
            from_age = int(m.group(1))
            if from_age == int(round(age_to)):
                return GPKG_MAIN, name

    raise ValueError(f"Could not find a layer starting at age {age_to} Ma.")


def get_layer_for_transition(age_from: float, age_to: float):
    """Finds the GPKG and Layer for the transition FROM age_from TO age_to."""
    target_from = int(round(age_from))
    target_to = int(round(age_to))

    for name in fiona.listlayers(GPKG_MAIN):
        m = LAYER_RE.match(name)
        if m:
            from_age = int(m.group(1))
            to_age = int(m.group(2))
            if from_age == target_from and to_age == target_to:
                return GPKG_MAIN, name

    raise ValueError(f"Could not find layer for transition {age_from} -> {age_to}")


def _crs_matches_54034(raster_crs) -> bool:
    if raster_crs is None: return False
    try:
        return CRS.from_user_input(raster_crs) == _ESRI_54034_CRS
    except Exception:
        return False


def lonlat_to_xyz_arr(lon_deg, lat_deg):
    lon = np.radians(lon_deg)
    lat = np.radians(lat_deg)
    x = np.cos(lat) * np.cos(lon)
    y = np.cos(lat) * np.sin(lon)
    z = np.sin(lat)
    return np.stack([x, y, z], axis=-1)


def xyz_to_lonlat_arr(vec):
    x, y, z = vec[..., 0], vec[..., 1], vec[..., 2]
    lat = np.degrees(np.arcsin(np.clip(z, -1.0, 1.0)))
    lon = np.degrees(np.arctan2(y, x))
    return lon, lat


def rotate_points_vectorized(lon, lat, pole_lon, pole_lat, angle_deg):
    lon = np.atleast_1d(lon)
    lat = np.atleast_1d(lat)
    pole_lon = np.atleast_1d(pole_lon)
    pole_lat = np.atleast_1d(pole_lat)
    angle_deg = np.atleast_1d(angle_deg)

    p = lonlat_to_xyz_arr(lon, lat)
    k = lonlat_to_xyz_arr(pole_lon, pole_lat)
    theta = np.radians(ANGLE_SIGN * angle_deg)[:, None]

    dot_kp = np.einsum("ij,ij->i", k, p)[:, None]
    cross_kp = np.cross(k, p)

    p_rot = p * np.cos(theta) + cross_kp * np.sin(theta) + k * dot_kp * (1 - np.cos(theta))
    p_rot = p_rot / np.linalg.norm(p_rot, axis=1, keepdims=True)
    return xyz_to_lonlat_arr(p_rot)


def get_raster_path(age: float) -> str:
    folder = str(RASTER_AGE_OFFSET + int(round(age)))
    return str(Path(RASTER_BASE_DIR) / folder / RASTER_FILENAME)


def read_band(raster_path: str, band: int = BAND):
    with rasterio.open(raster_path) as src:
        if not _crs_matches_54034(src.crs):
            print(f"  [warning] {raster_path} CRS mismatch: {src.crs}")
        array = src.read(band)
        transform = src.transform
        shape = (src.height, src.width)
    return array, transform, shape


def pixel_centers(shape, transform):
    rows, cols = np.indices(shape)
    xs, ys = transform * (cols.ravel() + 0.5, rows.ravel() + 0.5)
    return np.asarray(xs), np.asarray(ys)


def xy_to_rowcol(x, y, transform, shape):
    inv = ~transform
    cols, rows = inv * (x, y)
    rows = np.floor(rows).astype(np.int64)
    cols = np.floor(cols).astype(np.int64)
    in_bounds = (rows >= 0) & (rows < shape[0]) & (cols >= 0) & (cols < shape[1])
    return rows, cols, in_bounds


# ============================================================================
# PASS 1: INVERSE MATRIX (Gains)
# ============================================================================

def build_change_matrix_inverse(age_from: float, age_to: float = 0, band: int = BAND, save_csv: bool = True):
    print(f"\n--- PASS 1: INVERSE (Gains) ---")
    print(f"Building INVERSE change matrix: {age_from} Ma -> {age_to} Ma")

    src_path = get_raster_path(age_from)
    dst_path = get_raster_path(age_to)

    src_array, src_transform, src_shape = read_band(src_path, band)
    dst_array, dst_transform, dst_shape = read_band(dst_path, band)

    n_dst_pixels = dst_array.size
    dst_codes = dst_array.ravel()

    xs, ys = pixel_centers(dst_shape, dst_transform)
    lon, lat = _to_lonlat.transform(xs, ys)
    lon = np.asarray(lon)
    lat = np.asarray(lat)

    # Get Source Layer
    src_gpkg, src_layer = get_layer_for_transition(age_from, age_to)
    gdf = gpd.read_file(src_gpkg, layer=src_layer).reset_index(drop=True)

    # Rotate Polygons Forward
    rotated_geoms = []
    for _, row in gdf.iterrows():
        def _rot(x, y, z=None):
            nl, nlat = rotate_points_vectorized(x, y, np.full_like(x, row["EuLong"]), np.full_like(x, row["EuLat"]),
                                                np.full_like(x, row["EuAng"]))
            return (nl, nlat) if z is None else (nl, nlat, z)

        rotated_geoms.append(shapely_transform(_rot, row.geometry))

    gdf_rotated = gdf.copy()
    gdf_rotated["geometry"] = rotated_geoms
    gdf_rotated = gdf_rotated.set_crs(EPSG_4326, allow_override=True)
    gdf_rotated_proj = gdf_rotated.to_crs(ESRI_54034)

    shapes = list(zip(gdf_rotated_proj.geometry, gdf_rotated_proj.index))
    plate_id_grid = rasterize(shapes=shapes, out_shape=dst_shape, transform=dst_transform, fill=-1, dtype="int32")
    plate_id = plate_id_grid.ravel()
    mapped_mask = plate_id >= 0

    print(f"  {mapped_mask.sum():,} / {n_dst_pixels:,} dest pixels covered by plates")

    eu_long_vals = gdf["EuLong"].to_numpy()
    eu_lat_vals = gdf["EuLat"].to_numpy()
    eu_ang_vals = gdf["EuAng"].to_numpy()

    pole_lon = np.full(n_dst_pixels, np.nan)
    pole_lat = np.full(n_dst_pixels, np.nan)
    pole_ang = np.full(n_dst_pixels, np.nan)

    valid_plate_indices = plate_id[mapped_mask]
    if len(valid_plate_indices) > 0:
        pole_lon[mapped_mask] = eu_long_vals[valid_plate_indices]
        pole_lat[mapped_mask] = eu_lat_vals[valid_plate_indices]
        pole_ang[mapped_mask] = eu_ang_vals[valid_plate_indices]

    new_lon = np.full(n_dst_pixels, np.nan)
    new_lat = np.full(n_dst_pixels, np.nan)
    if mapped_mask.any():
        new_lon[mapped_mask], new_lat[mapped_mask] = rotate_points_vectorized(
            lon[mapped_mask], lat[mapped_mask],
            pole_lon[mapped_mask], pole_lat[mapped_mask], -pole_ang[mapped_mask],
        )

    new_x = np.full(n_dst_pixels, np.nan)
    new_y = np.full(n_dst_pixels, np.nan)
    if mapped_mask.any():
        new_x[mapped_mask], new_y[mapped_mask] = _to_meters.transform(
            new_lon[mapped_mask], new_lat[mapped_mask]
        )

    src_rows = np.full(n_dst_pixels, -1, dtype=np.int64)
    src_cols = np.full(n_dst_pixels, -1, dtype=np.int64)
    in_bounds = np.zeros(n_dst_pixels, dtype=bool)

    if mapped_mask.any():
        r, c, ib = xy_to_rowcol(new_x[mapped_mask], new_y[mapped_mask], src_transform, src_shape)
        src_rows[mapped_mask] = r
        src_cols[mapped_mask] = c
        in_bounds[mapped_mask] = ib

    source_code = np.empty(n_dst_pixels, dtype=object)
    source_code[:] = UNMAPPED_LABEL
    source_code[mapped_mask] = OFFGRID_LABEL
    valid = mapped_mask & in_bounds
    source_code[valid] = src_array[src_rows[valid], src_cols[valid]]

    df = pd.DataFrame({"source_code": source_code, "dest_code": dst_codes})
    matrix = pd.crosstab(df["source_code"], df["dest_code"])

    ordered_rows = [r for r in VALID_CODES if r in matrix.index]
    extra_rows = [r for r in matrix.index if r not in VALID_CODES]
    matrix = matrix.loc[ordered_rows + extra_rows]

    ordered_cols = [c for c in VALID_CODES if c in matrix.columns]
    extra_cols = [c for c in matrix.columns if c not in VALID_CODES]
    matrix = matrix[ordered_cols + extra_cols]

    print("\nInverse Matrix (Gains): Rows=Source, Cols=Dest")
    print(matrix)

    if save_csv:
        out_path = Path(f"change_matrix_inverse_{age_from}Ma_to_{age_to}Ma_band{band}.csv")
        matrix.to_csv(out_path)
        print(f"Saved to {out_path}")

    return matrix


# ============================================================================
# PASS 2: FORWARD MATRIX (Losses) - FIXED BROADCASTING
# ============================================================================

def build_subduction_matrix(age_from: float, age_to: float = 0, band: int = BAND, save_csv: bool = True):
    print(f"\n--- PASS 2: FORWARD (Losses) ---")
    print(f"Building SUBDUCTION matrix: {age_from} Ma -> {age_to} Ma")
    print(f"  Comparing Source ({age_from} Ma) vs Destination Topology ({age_to} Ma)")

    src_path = get_raster_path(age_from)
    dst_path = get_raster_path(age_to)

    src_array, src_transform, src_shape = read_band(src_path, band)
    dst_array, dst_transform, dst_shape = read_band(dst_path, band)

    n_src_pixels = src_array.size

    # 1. Source pixel centers
    src_xs, src_ys = pixel_centers(src_shape, src_transform)
    src_lon, src_lat = _to_lonlat.transform(src_xs, src_ys)
    src_lon = np.asarray(src_lon)
    src_lat = np.asarray(src_lat)

    # 2. Load SOURCE Layer
    src_gpkg, src_layer = get_layer_for_transition(age_from, age_to)
    print(f"  Source Layer: {src_layer} from {src_gpkg}")
    gdf_src = gpd.read_file(src_gpkg, layer=src_layer).reset_index(drop=True)

    if 'PlateName' not in gdf_src.columns:
        raise ValueError("Source layer missing 'PlateName'.")

    gdf_src_proj = gdf_src.to_crs(ESRI_54034)
    shapes_src = list(zip(gdf_src_proj.geometry, gdf_src.index))
    src_plate_grid = rasterize(shapes=shapes_src, out_shape=src_shape, transform=src_transform, fill=-1, dtype="int32")
    src_plate_ids = src_plate_grid.ravel()
    src_index_to_name = gdf_src['PlateName'].to_numpy()

    # 3. Load DESTINATION Layer (Ground Truth)
    dst_gpkg_path, dst_layer_name = get_gpkg_path_for_age(age_to)
    print(f"  Dest Layer ({age_to} Ma): {dst_layer_name} from {dst_gpkg_path}")
    gdf_dst_truth = gpd.read_file(dst_gpkg_path, layer=dst_layer_name).reset_index(drop=True)

    gdf_dst_truth_proj = gdf_dst_truth.to_crs(ESRI_54034)
    if 'PlateName' not in gdf_dst_truth.columns:
        col_candidates = [c for c in gdf_dst_truth.columns if 'plate' in c.lower() or 'name' in c.lower()]
        if col_candidates:
            gdf_dst_truth = gdf_dst_truth.rename(columns={col_candidates[0]: 'PlateName'})
        else:
            raise ValueError("Destination layer missing PlateName.")

    shapes_dst = list(zip(gdf_dst_truth_proj.geometry, gdf_dst_truth.index))
    dst_plate_grid_truth = rasterize(shapes=shapes_dst, out_shape=dst_shape, transform=dst_transform, fill=-1,
                                     dtype="int32")
    dst_index_to_name_truth = gdf_dst_truth['PlateName'].to_numpy()

    # 4. Assign poles from SOURCE
    pole_lon = np.full(n_src_pixels, np.nan)
    pole_lat = np.full(n_src_pixels, np.nan)
    pole_ang = np.full(n_src_pixels, np.nan)

    mapped_src = src_plate_ids >= 0
    valid_plate_indices = src_plate_ids[mapped_src]
    if len(valid_plate_indices) > 0:
        pole_lon[mapped_src] = gdf_src["EuLong"].to_numpy()[valid_plate_indices]
        pole_lat[mapped_src] = gdf_src["EuLat"].to_numpy()[valid_plate_indices]
        pole_ang[mapped_src] = gdf_src["EuAng"].to_numpy()[valid_plate_indices]

    # 5. Rotate FORWARD
    fwd_lon = np.full(n_src_pixels, np.nan)
    fwd_lat = np.full(n_src_pixels, np.nan)

    if mapped_src.any():
        fwd_lon[mapped_src], fwd_lat[mapped_src] = rotate_points_vectorized(
            src_lon[mapped_src], src_lat[mapped_src],
            pole_lon[mapped_src], pole_lat[mapped_src], pole_ang[mapped_src]
        )

    # 6. Check Destination Location (FIXED BROADCASTING)
    fwd_x = np.full(n_src_pixels, np.nan)
    fwd_y = np.full(n_src_pixels, np.nan)

    # Initialize full-size masks and arrays
    in_bounds = np.zeros(n_src_pixels, dtype=bool)
    r_fwd_full = np.full(n_src_pixels, -1, dtype=np.int64)
    c_fwd_full = np.full(n_src_pixels, -1, dtype=np.int64)

    if mapped_src.any():
        fwd_x[mapped_src], fwd_y[mapped_src] = _to_meters.transform(fwd_lon[mapped_src], fwd_lat[mapped_src])

        # Calculate for mapped pixels only
        r_mapped, c_mapped, ib_mapped = xy_to_rowcol(fwd_x[mapped_src], fwd_y[mapped_src], dst_transform, dst_shape)

        # Fill full-size arrays
        in_bounds[mapped_src] = ib_mapped
        r_fwd_full[mapped_src] = r_mapped
        c_fwd_full[mapped_src] = c_mapped

    # 7. Determine Fate
    fate = np.full(n_src_pixels, fill_value="unknown", dtype=object)
    fate[~mapped_src] = "unmapped_source"

    plate_change_mask = np.zeros(n_src_pixels, dtype=bool)

    # Identify valid hits (Mapped AND In Bounds)
    valid_hits_mask = mapped_src & in_bounds
    valid_hits_indices = np.where(valid_hits_mask)[0]

    if len(valid_hits_indices) > 0:
        rows_hit = r_fwd_full[valid_hits_indices]
        cols_hit = c_fwd_full[valid_hits_indices]

        dest_pids_truth = dst_plate_grid_truth[rows_hit, cols_hit]
        source_pids = src_plate_ids[valid_hits_indices]

        src_names = src_index_to_name[source_pids]
        dst_names_truth = dst_index_to_name_truth[dest_pids_truth]

        is_no_data = (dest_pids_truth == -1)
        is_mismatch = (src_names != dst_names_truth)

        plate_change_mask[valid_hits_indices] = is_no_data | is_mismatch

    # Assign Fates
    fate[mapped_src & ~in_bounds] = "subducted_offgrid"
    fate[mapped_src & in_bounds & plate_change_mask] = "subducted_collision"
    fate[mapped_src & in_bounds & ~plate_change_mask] = "survived"

    # 8. Cross-tabulate
    df = pd.DataFrame({"source_code": src_array.ravel(), "fate": fate})
    matrix = pd.crosstab(df["source_code"], df["fate"])

    ordered_rows = [r for r in VALID_CODES if r in matrix.index]
    extra_rows = [r for r in matrix.index if r not in VALID_CODES]
    matrix = matrix.loc[ordered_rows + extra_rows]

    ordered_cols = ["survived", "subducted_offgrid", "subducted_collision", "unmapped_source"]
    existing_cols = [c for c in ordered_cols if c in matrix.columns]
    extra_cols = [c for c in matrix.columns if c not in ordered_cols]
    matrix = matrix[existing_cols + extra_cols]

    print("\nSubduction Matrix (Losses): Rows=Source, Cols=Fate")
    print(matrix)

    total_lost = matrix.get('subducted_offgrid', 0).sum() + matrix.get('subducted_collision', 0).sum()
    total_survived = matrix.get('survived', 0).sum()
    print(f"\nSummary: Survived={total_survived:,}, Lost={total_lost:,}")

    if save_csv:
        out_path = Path(f"subduction_matrix_{age_from}Ma_to_{age_to}Ma_band{band}.csv")
        matrix.to_csv(out_path)
        print(f"Saved to {out_path}")

    return matrix


# ============================================================================
# Entry Point
# ============================================================================
if __name__ == "__main__":
    years_list = [
        "2000", "2006", "2011", "2015", "2020", "2033", "2040", "2048",
        "2056", "2068", "2094", "2100", "2113", "2120", "2133", "2140",
        "2154", "2165", "2180", "2200", "2210", "2220", "2230", "2240",
        "2250", "2270", "2290", "2300", "2315", "2331", "2350", "2370",
        "2383", "2393", "2408", "2420", "2444", "2463", "2475", "2489",
        "2500", "2518", "2535", "2545"
    ]
    ages_int = [int(y) for y in years_list]

    for i in range(len(ages_int) - 1):
        year_to = ages_int[i]
        year_from = ages_int[i + 1]
        age_to = float(year_to - RASTER_AGE_OFFSET)
        age_from = float(year_from - RASTER_AGE_OFFSET)

        print(f"\n{'=' * 60}")
        print(f"PROCESSING: {year_from} ({age_from} Ma) -> {year_to} ({age_to} Ma)")
        print(f"{'=' * 60}")

        try:
            build_change_matrix_inverse(age_from=age_from, age_to=age_to)
            build_subduction_matrix(age_from=age_from, age_to=age_to)
            break  # Test first step
        except Exception as e:
            print(f"Error: {e}")
            import traceback

            traceback.print_exc()