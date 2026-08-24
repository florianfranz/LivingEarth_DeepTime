import json
import re
from pathlib import Path
import geopandas as gpd
import numpy as np
import rasterio
import fiona
import pandas as pd
from shapely.geometry import Point
from rasterio.features import rasterize
# Added import to ensure we can create a robust CRS object
from pyproj import CRS

# ============================================================================
# Configuration
# ============================================================================
CONFIG_PATH = Path(__file__).resolve().parent.parent / "config.json"
with open(CONFIG_PATH, "r") as f:
    _config = json.load(f)

GPKG_PATH = _config["plates_gpkg"]
RASTER_BASE_DIR = _config["LE_outputs"]
RASTER_FILENAME = "level4_out_LE_DT.tif"
RASTER_AGE_OFFSET = 2000

ESRI_54034 = "ESRI:54034"
# CHANGE: Define as a pyproj.CRS object immediately to force database lookup now,
# not during the loop where it might fail.
EPSG_4326 = CRS.from_epsg(4326)

LAYER_RE = re.compile(r"^Plates(\d+)preRot(\d+)$")

ANGLE_SIGN = 1.0
BAND = 4
VALID_CODES = [20, 21, 56, 94, 98, 99, 106, 105, 255]

LABEL_NEW_CRUST = "new_crust"
LABEL_SUBDUCTED = "subducted"
LABEL_OFFGRID = "off_grid"

# ============================================================================
# Land-cover overlap priority
# ============================================================================
# When two or more source pixels rotate forward and land in the same
# destination cell, only one "survives" (keeps the real destination
# land-cover code); the rest are marked LABEL_SUBDUCTED.
#
# Priority reflects convergent-margin physical behaviour:
#   - Oceanic classes (liquid water, sea-ice) subduct first (lowest priority).
#   - Vegetation and wet soil are intermediate.
#   - Bare surface / snow represent uplifted collision-zone terrain and
#     therefore survive preferentially (highest priority).
#   - No Data (255) carries no real land-cover information and is never
#     preferred as a survivor (lowest priority of all).
#
# Lower number = higher priority = more likely to survive an overlap.
# Ties within the same priority tier fall back to first-occurrence
# (row-major scan order of the source raster).
SOURCE_CODE_PRIORITY = {
    94: 0, 105: 0,          # Bare surface, Snow -- highest priority
    98: 1,                  # Wet soil
    20: 2, 21: 2, 56: 2,    # Terrestrial / aquatic vegetation
    99: 3, 106: 3,          # Water: liquid, Water: sea-ice -- lowest (subduct first)
    255: 4,                 # No Data -- never preferred
}


# ============================================================================
# Robust Rotation Logic (Fixed Dimensions)
# ============================================================================

def wrap_lon_180(lon_arr):
    """Wrap array of longitudes to [-180, 180]."""
    return (lon_arr + 180) % 360 - 180


def rotate_points_vectorized(lon, lat, pole_lon, pole_lat, angle_deg):
    """
    Rotates arrays of points, each with its own pole.
    Inputs: lon, lat, pole_lon, pole_lat, angle_deg are all arrays of shape (N,)
    """
    # Ensure inputs are numpy arrays
    lon = np.asarray(lon)
    lat = np.asarray(lat)
    pole_lon = np.asarray(pole_lon)
    pole_lat = np.asarray(pole_lat)
    angle_deg = np.asarray(angle_deg)

    # Convert to radians
    lon_r = np.radians(lon)
    lat_r = np.radians(lat)

    # To XYZ (N, 3)
    x = np.cos(lat_r) * np.cos(lon_r)
    y = np.cos(lat_r) * np.sin(lon_r)
    z = np.sin(lat_r)
    p = np.stack([x, y, z], axis=1)  # Shape (N, 3)

    # Pole Vector (N, 3)
    k_lon = np.radians(pole_lon)
    k_lat = np.radians(pole_lat)
    k_x = np.cos(k_lat) * np.cos(k_lon)
    k_y = np.cos(k_lat) * np.sin(k_lon)
    k_z = np.sin(k_lat)
    k = np.stack([k_x, k_y, k_z], axis=1)  # Shape (N, 3)

    # Angle (N,)
    theta = np.radians(ANGLE_SIGN * angle_deg)
    cos_t = np.cos(theta)
    sin_t = np.sin(theta)
    one_minus_cos = 1.0 - cos_t

    # Rodrigues Formula (Vectorized)
    # cross product of (N,3) and (N,3) -> (N,3)
    cross_kp = np.cross(k, p)

    # dot product of (N,3) and (N,3) -> (N,)
    dot_kp = np.sum(k * p, axis=1)

    # Term 3: k * (dot_kp * one_minus_cos)
    # k: (N,3), dot_kp: (N,) -> need to expand dot_kp to (N,1) for broadcasting
    term3 = k * (dot_kp * one_minus_cos)[:, None]

    p_rot = (p * cos_t[:, None]) + (cross_kp * sin_t[:, None]) + term3

    # Normalize
    norms = np.linalg.norm(p_rot, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    p_rot = p_rot / norms

    # To Lon/Lat
    r_lon = np.degrees(np.arctan2(p_rot[:, 1], p_rot[:, 0]))
    r_lat = np.degrees(np.arcsin(np.clip(p_rot[:, 2], -1.0, 1.0)))

    # CRITICAL: Wrap to [-180, 180]
    r_lon_wrapped = wrap_lon_180(r_lon)

    return r_lon_wrapped, r_lat

GPKG_AGE_OVERRIDE = {
    331: 330,
}

def get_layer_for_age(gpkg_path, age):
    age_int = int(round(age))
    age_int = GPKG_AGE_OVERRIDE.get(age_int, age_int)
    for name in fiona.listlayers(gpkg_path):
        m = LAYER_RE.match(name)
        if m and int(m.group(1)) == age_int:
            return name
    raise ValueError(f"No layer found for age {age}")




def get_raster_path(age):
    folder = str(RASTER_AGE_OFFSET + int(round(age)))
    return str(Path(RASTER_BASE_DIR) / folder / RASTER_FILENAME)


# ============================================================================
# Matrix Builder
# ============================================================================

def build_change_matrix(age_from, age_to, band=BAND, save_csv=True):
    print(f"\n{'=' * 60}")
    print(f"BUILDING MATRIX: {age_from} Ma -> {age_to} Ma")
    print(f"{'=' * 60}")

    # 1. Load Rasters
    src_path = get_raster_path(age_from)
    dst_path = get_raster_path(age_to)

    try:
        with rasterio.open(src_path) as src:
            src_array = src.read(band)
            src_transform = src.transform
            src_shape = src.shape
            src_crs = src.crs

        with rasterio.open(dst_path) as dst:
            dst_array = dst.read(band)
            dst_transform = dst.transform
            dst_shape = dst.shape
    except Exception as e:
        print(f"Error loading rasters: {e}")
        return

    n_pixels = src_array.size
    src_codes = src_array.ravel()
    valid_mask = np.isin(src_codes, VALID_CODES)

    # 2. Get Pixel Centers (Lon/Lat)
    # Create grid of row/col indices
    rows_idx, cols_idx = np.indices(src_shape)

    # Transform to Lon/Lat using rasterio's xy (vectorized)
    # xy returns tuples, we need to unzip
    # Faster: use transform directly
    xs_flat, ys_flat = src_transform * (cols_idx.ravel() + 0.5, rows_idx.ravel() + 0.5)

    # Create GeoDataFrame for CRS transform
    pts_gdf = gpd.GeoDataFrame(geometry=gpd.points_from_xy(xs_flat, ys_flat), crs=src_crs)

    # Using the pre-initialized CRS object avoids the string lookup error in loops
    pts_latlon = pts_gdf.to_crs(EPSG_4326)

    lons = pts_latlon.geometry.x.values
    lats = pts_latlon.geometry.y.values

    # 3. Rasterize Source Plates (to get Plate ID per pixel)
    layer_name = get_layer_for_age(GPKG_PATH, age_from)
    src_gdf = gpd.read_file(GPKG_PATH, layer=layer_name)
    src_gdf_projected = src_gdf.to_crs(src_crs)  # Match raster CRS

    shapes = [(geom, idx) for idx, geom in enumerate(src_gdf_projected.geometry)]
    plate_id_grid = rasterize(shapes, out_shape=src_shape, transform=src_transform, fill=-1, dtype='int32')
    plate_ids = plate_id_grid.ravel()

    mapped_mask = (plate_ids >= 0) & valid_mask
    print(f"Mapped pixels: {mapped_mask.sum()} / {valid_mask.sum()}")

    # 4. Prepare Poles (Per-Pixel)
    # Initialize with NaN
    pole_lons = np.full(n_pixels, np.nan)
    pole_lats = np.full(n_pixels, np.nan)
    pole_angles = np.full(n_pixels, np.nan)

    # Map poles to pixels based on plate_id
    # Extract pole data from GeoDataFrame
    gdf_pole_lons = src_gdf['EuLong'].values
    gdf_pole_lats = src_gdf['EuLat'].values
    gdf_pole_angles = src_gdf['EuAng'].values

    # Vectorized mapping: use plate_ids as indices
    # Only for mapped pixels
    valid_plate_ids = plate_ids[mapped_mask]

    pole_lons[mapped_mask] = gdf_pole_lons[valid_plate_ids]
    pole_lats[mapped_mask] = gdf_pole_lats[valid_plate_ids]
    pole_angles[mapped_mask] = gdf_pole_angles[valid_plate_ids]

    # 5. Rotate (Vectorized)
    rot_lons = np.full(n_pixels, np.nan)
    rot_lats = np.full(n_pixels, np.nan)

    valid_indices = np.where(mapped_mask)[0]

    if len(valid_indices) > 0:
        # Extract arrays for valid pixels
        lons_v = lons[valid_indices]
        lats_v = lats[valid_indices]
        plon_v = pole_lons[valid_indices]
        plat_v = pole_lats[valid_indices]
        pang_v = pole_angles[valid_indices]

        r_lon, r_lat = rotate_points_vectorized(lons_v, lats_v, plon_v, plat_v, pang_v)

        rot_lons[valid_indices] = r_lon
        rot_lats[valid_indices] = r_lat

    # 6. Map Rotated Points to Destination Grid
    # Convert Rotated Lon/Lat back to ESRI:54034 (or whatever dst CRS is)
    # Assuming dst CRS is same as src (ESRI:54034)

    # Using the pre-initialized CRS object here as well
    rot_pts_gdf = gpd.GeoDataFrame(geometry=gpd.points_from_xy(rot_lons, rot_lats), crs=EPSG_4326)
    rot_pts_proj = rot_pts_gdf.to_crs(src_crs)  # Use src_crs which should be ESRI:54034

    rot_xs = rot_pts_proj.geometry.x.values
    rot_ys = rot_pts_proj.geometry.y.values

    # Calculate Row/Col in Destination
    inv_transform = ~dst_transform
    cols, rows = inv_transform * (rot_xs, rot_ys)
    rows = np.floor(rows).astype(np.int64)
    cols = np.floor(cols).astype(np.int64)

    in_bounds = (rows >= 0) & (rows < dst_shape[0]) & (cols >= 0) & (cols < dst_shape[1])

    # 7. Calculate Hit Counts (To detect Overlaps/Subduction)
    # Only count valid, in-bounds pixels
    valid_dest_mask = mapped_mask & in_bounds
    valid_rows = rows[valid_dest_mask]
    valid_cols = cols[valid_dest_mask]

    flat_dest_indices = valid_rows * dst_shape[1] + valid_cols

    # Count hits per destination cell
    hit_counts = np.bincount(flat_dest_indices, minlength=dst_shape[0] * dst_shape[1]).reshape(dst_shape)

    # 8. Build Destination Codes
    dest_codes = np.full(n_pixels, LABEL_OFFGRID, dtype=object)

    # We need to know for each pixel, how many hits its destination cell has
    # Create a map of (row, col) -> hit_count
    # We can just index hit_counts with rows/cols
    pixel_hit_counts = np.zeros(n_pixels, dtype=int)
    pixel_hit_counts[valid_dest_mask] = hit_counts[rows[valid_dest_mask], cols[valid_dest_mask]]

    # 9. Assign Codes based on Hit Counts
    # Strategy:
    # - If hit_count == 1: Normal -> Dest Code
    # - If hit_count > 1: Overlap -> Subducted (except one survivor, chosen by
    #   source land-cover priority, see SOURCE_CODE_PRIORITY)
    # - If hit_count == 0: (Should not happen for valid_dest_mask)

    # Identify overlaps
    is_overlap = (pixel_hit_counts > 1) & valid_dest_mask
    is_normal = (pixel_hit_counts == 1) & valid_dest_mask

    # Normal pixels: Assign destination code
    normal_indices = np.where(is_normal)[0]
    for i in normal_indices:
        r, c = rows[i], cols[i]
        dest_codes[i] = dst_array[r, c]

    # Overlap pixels: Mark as SUBDUCTED initially
    overlap_indices = np.where(is_overlap)[0]
    dest_codes[overlap_indices] = LABEL_SUBDUCTED

    # Select Survivors for overlaps: pick the pixel with the highest-priority
    # source_code for each destination cell (physically-motivated priority,
    # instead of arbitrary row-major first-occurrence).
    if len(overlap_indices) > 0:
        overlap_rows = rows[overlap_indices]
        overlap_cols = cols[overlap_indices]
        keys = overlap_rows * dst_shape[1] + overlap_cols

        overlap_src_codes = src_codes[overlap_indices]
        # Map each source code to its priority rank (unlisted codes get the worst rank)
        priority_vals = np.array([
            SOURCE_CODE_PRIORITY.get(c, len(SOURCE_CODE_PRIORITY)) for c in overlap_src_codes
        ])

        df_overlap = pd.DataFrame({
            'key': keys,
            'priority': priority_vals,
            'global_idx': overlap_indices
        })

        # Stable sort: within equal priority, first-occurrence (row-major) wins the tie
        df_overlap = df_overlap.sort_values(['key', 'priority'], kind='stable')
        survivor_rows = df_overlap.drop_duplicates(subset='key', keep='first')
        survivor_global_indices = survivor_rows['global_idx'].values

        # Update survivors to have the actual destination code
        for i in survivor_global_indices:
            r, c = rows[i], cols[i]
            dest_codes[i] = dst_array[r, c]

    # 10. Handle Gaps (New Crust)
    # Find destination cells with hit_count == 0
    gap_mask = (hit_counts == 0)
    gap_rows, gap_cols = np.where(gap_mask)

    new_crust_entries = []
    for r, c in zip(gap_rows, gap_cols):
        code = dst_array[r, c]
        if code in VALID_CODES:
            new_crust_entries.append({'source_code': LABEL_NEW_CRUST, 'dest_code': code})

    # 11. Build DataFrame & Matrix
    df_rows = []
    # Only include pixels that have a valid destination assignment (not OFFGRID)
    final_valid_mask = (dest_codes != LABEL_OFFGRID)

    # Extract arrays for final valid pixels
    final_indices = np.where(final_valid_mask)[0]
    if len(final_indices) > 0:
        df_rows = pd.DataFrame({
            'source_code': src_codes[final_indices],
            'dest_code': dest_codes[final_indices]
        })
    else:
        df_rows = pd.DataFrame(columns=['source_code', 'dest_code'])

    # Add New Crust entries
    if new_crust_entries:
        df_nc = pd.DataFrame(new_crust_entries)
        df_rows = pd.concat([df_rows, df_nc], ignore_index=True)

    if df_rows.empty:
        print("No valid transitions found.")
        return

    matrix = pd.crosstab(df_rows['source_code'], df_rows['dest_code'])

    # Reorder
    ordered_cols = [c for c in VALID_CODES if c in matrix.columns] + [LABEL_SUBDUCTED, LABEL_NEW_CRUST, LABEL_OFFGRID]
    matrix = matrix[[c for c in ordered_cols if c in matrix.columns]]

    ordered_rows = [r for r in VALID_CODES if r in matrix.index] + [LABEL_NEW_CRUST, LABEL_SUBDUCTED, LABEL_OFFGRID]
    matrix = matrix.loc[[r for r in ordered_rows if r in matrix.index]]

    print(f"Matrix Shape: {matrix.shape}")
    subducted_sum = matrix[LABEL_SUBDUCTED].sum() if LABEL_SUBDUCTED in matrix.columns else 0
    new_crust_sum = matrix.loc[LABEL_NEW_CRUST].sum() if LABEL_NEW_CRUST in matrix.index else 0
    print(f"Subducted Pixels: {subducted_sum}")
    print(f"New Crust Pixels: {new_crust_sum}")

    if save_csv:
        out_path = Path(f"change_matrix_{int(age_from)}Ma_to_{int(age_to)}Ma.csv")
        matrix.to_csv(out_path)
        print(f"Saved to {out_path}")

    return matrix


# ============================================================================
# Entry Point
# ============================================================================
if __name__ == "__main__":
    # Your list of years as strings
    years_list = [
        "2000","2006", "2011", "2015", "2020", "2033", "2040", "2048",
        "2056", "2068", "2094", "2100", "2113", "2120", "2133", "2140",
        "2154", "2165", "2180", "2200", "2210", "2220", "2230", "2240",
        "2250", "2270", "2290", "2300", "2315", "2331", "2350", "2370",
        "2383", "2393", "2408", "2420", "2444", "2463", "2475", "2489",
        "2500", "2518", "2535", "2545"
    ]

    years_list = [ "2315","2331",]

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