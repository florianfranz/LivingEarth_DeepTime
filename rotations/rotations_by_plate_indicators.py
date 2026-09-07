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

PALEO_FOLDER = _config["paleo_folder"]
PALEO_DEM_FILENAME_TEMPLATE = "palaeogeography_{age}.tif"
PALEO_DEM_BAND = 1
EARTH_RADIUS_KM = 6371.0

# --- Climate rasters (temperature / precipitation) ---
CLIMATE_FOLDER = _config["climate_output_folder"]
TEMPERATURE_FILENAME_TEMPLATE = "temperature_{age}_align.tif"
PRECIPITATION_FILENAME_TEMPLATE = "precipitation_{age}_align.tif"
CLIMATE_BAND = 1  # assumption: single-band rasters, same convention as PALEO_DEM_BAND -- adjust if not the case

ESRI_54034 = "ESRI:54034"
EPSG_4326 = CRS.from_epsg(4326)

LAYER_RE = re.compile(r"^Plates(\d+)preRot(\d+)$")

ANGLE_SIGN = 1.0
BAND = 4
VALID_CODES = [20, 21, 56, 94, 98, 99, 106, 105, 255]

LABEL_NEW_CRUST = "new_crust"
LABEL_SUBDUCTED = "subducted"
LABEL_OFFGRID = "off_grid"

STATUS_CODES = {"no_data": 0, "unmapped": 1, "off_grid": 2, "subducted": 3, "survivor": 4, "normal": 5}
RASTER_NODATA = -9999.0

# ============================================================================
# Land-cover overlap priority
# ============================================================================
SOURCE_CODE_PRIORITY = {
    94: 0, 105: 0,          # Bare surface, Snow -- highest priority
    98: 1,                  # Wet soil
    20: 2, 21: 2, 56: 2,    # Terrestrial / aquatic vegetation
    99: 3, 106: 3,          # Water: liquid, Water: sea-ice -- lowest (subduct first)
    255: 4,                 # No Data -- never preferred
}

GPKG_AGE_OVERRIDE = {
    331: 330,
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
    lon = np.asarray(lon)
    lat = np.asarray(lat)
    pole_lon = np.asarray(pole_lon)
    pole_lat = np.asarray(pole_lat)
    angle_deg = np.asarray(angle_deg)

    lon_r = np.radians(lon)
    lat_r = np.radians(lat)

    x = np.cos(lat_r) * np.cos(lon_r)
    y = np.cos(lat_r) * np.sin(lon_r)
    z = np.sin(lat_r)
    p = np.stack([x, y, z], axis=1)

    k_lon = np.radians(pole_lon)
    k_lat = np.radians(pole_lat)
    k_x = np.cos(k_lat) * np.cos(k_lon)
    k_y = np.cos(k_lat) * np.sin(k_lon)
    k_z = np.sin(k_lat)
    k = np.stack([k_x, k_y, k_z], axis=1)

    theta = np.radians(ANGLE_SIGN * angle_deg)
    cos_t = np.cos(theta)
    sin_t = np.sin(theta)
    one_minus_cos = 1.0 - cos_t

    cross_kp = np.cross(k, p)
    dot_kp = np.sum(k * p, axis=1)
    term3 = k * (dot_kp * one_minus_cos)[:, None]

    p_rot = (p * cos_t[:, None]) + (cross_kp * sin_t[:, None]) + term3

    norms = np.linalg.norm(p_rot, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    p_rot = p_rot / norms

    r_lon = np.degrees(np.arctan2(p_rot[:, 1], p_rot[:, 0]))
    r_lat = np.degrees(np.arcsin(np.clip(p_rot[:, 2], -1.0, 1.0)))

    r_lon_wrapped = wrap_lon_180(r_lon)

    return r_lon_wrapped, r_lat


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


def get_paleo_dem_path(age):
    # Same +2000 offset convention as get_raster_path (0 Ma -> "2000")
    year = RASTER_AGE_OFFSET + int(round(age))
    return str(Path(PALEO_FOLDER) / PALEO_DEM_FILENAME_TEMPLATE.format(age=year))


def get_temperature_path(age):
    # Same +2000 offset convention as get_raster_path / get_paleo_dem_path
    year = RASTER_AGE_OFFSET + int(round(age))
    return str(Path(CLIMATE_FOLDER) / TEMPERATURE_FILENAME_TEMPLATE.format(age=year))


def get_precipitation_path(age):
    year = RASTER_AGE_OFFSET + int(round(age))
    return str(Path(CLIMATE_FOLDER) / PRECIPITATION_FILENAME_TEMPLATE.format(age=year))


def sample_raster_at_points(xs, ys, transform, array, shape):
    """
    Sample a raster's values at arbitrary projected-CRS coordinates
    (same CRS as the raster — no reprojection here).
    Returns (values, in_bounds_mask); out-of-bounds points get NaN.
    """
    inv = ~transform
    cols, rows = inv * (xs, ys)
    rows = np.floor(rows).astype(np.int64)
    cols = np.floor(cols).astype(np.int64)
    in_bounds = (rows >= 0) & (rows < shape[0]) & (cols >= 0) & (cols < shape[1])
    values = np.full(len(xs), np.nan)
    values[in_bounds] = array[rows[in_bounds], cols[in_bounds]]
    return values, in_bounds


def haversine_km(lon1, lat1, lon2, lat2):
    lon1r, lat1r, lon2r, lat2r = map(np.radians, (lon1, lat1, lon2, lat2))
    dlon = lon2r - lon1r
    dlat = lat2r - lat1r
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1r) * np.cos(lat2r) * np.sin(dlon / 2.0) ** 2
    return 2 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def _write_multiband_raster(out_path, band_arrays, band_names, transform, crs, dtype='float32', nodata=RASTER_NODATA):
    with rasterio.open(
        out_path, 'w', driver='GTiff',
        height=band_arrays[0].shape[0], width=band_arrays[0].shape[1],
        count=len(band_arrays), dtype=dtype, crs=crs, transform=transform, nodata=nodata,
    ) as out:
        for i, (arr, name) in enumerate(zip(band_arrays, band_names), start=1):
            out.write(arr.astype(dtype), i)
            out.set_band_description(i, name)
    print(f"Saved raster to {out_path}")


# ============================================================================
# Matrix Builder (unchanged from the original script)
# ============================================================================

def build_change_matrix(age_from, age_to, band=BAND, save_csv=True):
    print(f"\n{'=' * 60}")
    print(f"BUILDING MATRIX: {age_from} Ma -> {age_to} Ma")
    print(f"{'=' * 60}")

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

    rows_idx, cols_idx = np.indices(src_shape)
    xs_flat, ys_flat = src_transform * (cols_idx.ravel() + 0.5, rows_idx.ravel() + 0.5)

    pts_gdf = gpd.GeoDataFrame(geometry=gpd.points_from_xy(xs_flat, ys_flat), crs=src_crs)
    pts_latlon = pts_gdf.to_crs(EPSG_4326)

    lons = pts_latlon.geometry.x.values
    lats = pts_latlon.geometry.y.values

    layer_name = get_layer_for_age(GPKG_PATH, age_from)
    src_gdf = gpd.read_file(GPKG_PATH, layer=layer_name)
    src_gdf_projected = src_gdf.to_crs(src_crs)

    shapes = [(geom, idx) for idx, geom in enumerate(src_gdf_projected.geometry)]
    plate_id_grid = rasterize(shapes, out_shape=src_shape, transform=src_transform, fill=-1, dtype='int32')
    plate_ids = plate_id_grid.ravel()

    mapped_mask = (plate_ids >= 0) & valid_mask
    print(f"Mapped pixels: {mapped_mask.sum()} / {valid_mask.sum()}")

    pole_lons = np.full(n_pixels, np.nan)
    pole_lats = np.full(n_pixels, np.nan)
    pole_angles = np.full(n_pixels, np.nan)

    gdf_pole_lons = src_gdf['EuLong'].values
    gdf_pole_lats = src_gdf['EuLat'].values
    gdf_pole_angles = src_gdf['EuAng'].values

    valid_plate_ids = plate_ids[mapped_mask]

    pole_lons[mapped_mask] = gdf_pole_lons[valid_plate_ids]
    pole_lats[mapped_mask] = gdf_pole_lats[valid_plate_ids]
    pole_angles[mapped_mask] = gdf_pole_angles[valid_plate_ids]

    rot_lons = np.full(n_pixels, np.nan)
    rot_lats = np.full(n_pixels, np.nan)

    valid_indices = np.where(mapped_mask)[0]

    if len(valid_indices) > 0:
        lons_v = lons[valid_indices]
        lats_v = lats[valid_indices]
        plon_v = pole_lons[valid_indices]
        plat_v = pole_lats[valid_indices]
        pang_v = pole_angles[valid_indices]

        r_lon, r_lat = rotate_points_vectorized(lons_v, lats_v, plon_v, plat_v, pang_v)

        rot_lons[valid_indices] = r_lon
        rot_lats[valid_indices] = r_lat

    rot_pts_gdf = gpd.GeoDataFrame(geometry=gpd.points_from_xy(rot_lons, rot_lats), crs=EPSG_4326)
    rot_pts_proj = rot_pts_gdf.to_crs(src_crs)

    rot_xs = rot_pts_proj.geometry.x.values
    rot_ys = rot_pts_proj.geometry.y.values

    inv_transform = ~dst_transform
    cols, rows = inv_transform * (rot_xs, rot_ys)
    rows = np.floor(rows).astype(np.int64)
    cols = np.floor(cols).astype(np.int64)

    in_bounds = (rows >= 0) & (rows < dst_shape[0]) & (cols >= 0) & (cols < dst_shape[1])

    valid_dest_mask = mapped_mask & in_bounds
    valid_rows = rows[valid_dest_mask]
    valid_cols = cols[valid_dest_mask]

    flat_dest_indices = valid_rows * dst_shape[1] + valid_cols

    hit_counts = np.bincount(flat_dest_indices, minlength=dst_shape[0] * dst_shape[1]).reshape(dst_shape)

    dest_codes = np.full(n_pixels, LABEL_OFFGRID, dtype=object)

    pixel_hit_counts = np.zeros(n_pixels, dtype=int)
    pixel_hit_counts[valid_dest_mask] = hit_counts[rows[valid_dest_mask], cols[valid_dest_mask]]

    is_overlap = (pixel_hit_counts > 1) & valid_dest_mask
    is_normal = (pixel_hit_counts == 1) & valid_dest_mask

    normal_indices = np.where(is_normal)[0]
    for i in normal_indices:
        r, c = rows[i], cols[i]
        dest_codes[i] = dst_array[r, c]

    overlap_indices = np.where(is_overlap)[0]
    dest_codes[overlap_indices] = LABEL_SUBDUCTED

    if len(overlap_indices) > 0:
        overlap_rows = rows[overlap_indices]
        overlap_cols = cols[overlap_indices]
        keys = overlap_rows * dst_shape[1] + overlap_cols

        overlap_src_codes = src_codes[overlap_indices]
        priority_vals = np.array([
            SOURCE_CODE_PRIORITY.get(c, len(SOURCE_CODE_PRIORITY)) for c in overlap_src_codes
        ])

        df_overlap = pd.DataFrame({
            'key': keys,
            'priority': priority_vals,
            'global_idx': overlap_indices
        })

        df_overlap = df_overlap.sort_values(['key', 'priority'], kind='stable')
        survivor_rows = df_overlap.drop_duplicates(subset='key', keep='first')
        survivor_global_indices = survivor_rows['global_idx'].values

        for i in survivor_global_indices:
            r, c = rows[i], cols[i]
            dest_codes[i] = dst_array[r, c]

    gap_mask = (hit_counts == 0)
    gap_rows, gap_cols = np.where(gap_mask)

    new_crust_entries = []
    for r, c in zip(gap_rows, gap_cols):
        code = dst_array[r, c]
        if code in VALID_CODES:
            new_crust_entries.append({'source_code': LABEL_NEW_CRUST, 'dest_code': code})

    final_valid_mask = (dest_codes != LABEL_OFFGRID)

    final_indices = np.where(final_valid_mask)[0]
    if len(final_indices) > 0:
        df_rows = pd.DataFrame({
            'source_code': src_codes[final_indices],
            'dest_code': dest_codes[final_indices]
        })
    else:
        df_rows = pd.DataFrame(columns=['source_code', 'dest_code'])

    if new_crust_entries:
        df_nc = pd.DataFrame(new_crust_entries)
        df_rows = pd.concat([df_rows, df_nc], ignore_index=True)

    if df_rows.empty:
        print("No valid transitions found.")
        return

    matrix = pd.crosstab(df_rows['source_code'], df_rows['dest_code'])

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
# Pixel History Builder (new)
# ============================================================================

def build_pixel_history(age_from, age_to, band=BAND, save_parquet=False,
                         save_raster='multiband', raster_out_dir="."):
    """
    save_raster: 'multiband' (one GeoTIFF, 5 bands), 'separate' (one GeoTIFF
    per variable), or None to skip raster output.

    Rasters are written on the SOURCE grid (age_from's transform/shape) —
    each output pixel is "this location, at age_from, is about to move by
    [delta_lat / distance / delta_alt / delta_temp / delta_precip] by age_to".
    Since every source pixel keeps its own cell (no rasterization needed —
    arrays already reshape directly to src_shape), there's no
    overlap-resolution step here, unlike the destination-grid land-cover
    matrix. Cells with no plate/no valid source code carry the nodata value
    (-9999).
    """
    print(f"\n{'=' * 60}")
    print(f"PIXEL HISTORY: {age_from} Ma -> {age_to} Ma")
    print(f"{'=' * 60}")

    src_path = get_raster_path(age_from)
    dst_path = get_raster_path(age_to)

    with rasterio.open(src_path) as src:
        src_array = src.read(band)
        src_transform = src.transform
        src_shape = src.shape
        src_crs = src.crs

    with rasterio.open(dst_path) as dst:
        dst_array = dst.read(band)
        dst_transform = dst.transform
        dst_shape = dst.shape

    n_pixels = src_array.size
    src_codes = src_array.ravel()
    valid_mask = np.isin(src_codes, VALID_CODES)

    rows_idx, cols_idx = np.indices(src_shape)
    xs_flat, ys_flat = src_transform * (cols_idx.ravel() + 0.5, rows_idx.ravel() + 0.5)

    pts_gdf = gpd.GeoDataFrame(geometry=gpd.points_from_xy(xs_flat, ys_flat), crs=src_crs)
    pts_latlon = pts_gdf.to_crs(EPSG_4326)
    lons = pts_latlon.geometry.x.values
    lats = pts_latlon.geometry.y.values

    layer_name = get_layer_for_age(GPKG_PATH, age_from)
    src_gdf = gpd.read_file(GPKG_PATH, layer=layer_name)
    src_gdf_projected = src_gdf.to_crs(src_crs)

    shapes = [(geom, idx) for idx, geom in enumerate(src_gdf_projected.geometry)]
    plate_id_grid = rasterize(shapes, out_shape=src_shape, transform=src_transform, fill=-1, dtype='int32')
    plate_ids = plate_id_grid.ravel()

    mapped_mask = (plate_ids >= 0) & valid_mask

    pole_lons = np.full(n_pixels, np.nan)
    pole_lats = np.full(n_pixels, np.nan)
    pole_angles = np.full(n_pixels, np.nan)

    gdf_pole_lons = src_gdf['EuLong'].values
    gdf_pole_lats = src_gdf['EuLat'].values
    gdf_pole_angles = src_gdf['EuAng'].values

    valid_plate_ids = plate_ids[mapped_mask]
    pole_lons[mapped_mask] = gdf_pole_lons[valid_plate_ids]
    pole_lats[mapped_mask] = gdf_pole_lats[valid_plate_ids]
    pole_angles[mapped_mask] = gdf_pole_angles[valid_plate_ids]

    rot_lons = np.full(n_pixels, np.nan)
    rot_lats = np.full(n_pixels, np.nan)
    valid_indices = np.where(mapped_mask)[0]

    if len(valid_indices) > 0:
        r_lon, r_lat = rotate_points_vectorized(
            lons[valid_indices], lats[valid_indices],
            pole_lons[valid_indices], pole_lats[valid_indices], pole_angles[valid_indices]
        )
        rot_lons[valid_indices] = r_lon
        rot_lats[valid_indices] = r_lat

    rot_pts_gdf = gpd.GeoDataFrame(geometry=gpd.points_from_xy(rot_lons, rot_lats), crs=EPSG_4326)
    rot_pts_proj = rot_pts_gdf.to_crs(src_crs)
    rot_xs = rot_pts_proj.geometry.x.values
    rot_ys = rot_pts_proj.geometry.y.values

    inv_dst = ~dst_transform
    cols, rows = inv_dst * (rot_xs, rot_ys)
    rows = np.floor(rows).astype(np.int64)
    cols = np.floor(cols).astype(np.int64)
    in_bounds = (rows >= 0) & (rows < dst_shape[0]) & (cols >= 0) & (cols < dst_shape[1])
    valid_dest_mask = mapped_mask & in_bounds

    dest_land_cover = np.full(n_pixels, np.nan)
    dest_land_cover[valid_dest_mask] = dst_array[rows[valid_dest_mask], cols[valid_dest_mask]]

    flat_dest_indices = rows[valid_dest_mask] * dst_shape[1] + cols[valid_dest_mask]
    hit_counts = np.bincount(flat_dest_indices, minlength=dst_shape[0] * dst_shape[1]).reshape(dst_shape)
    pixel_hit_counts = np.zeros(n_pixels, dtype=int)
    pixel_hit_counts[valid_dest_mask] = hit_counts[rows[valid_dest_mask], cols[valid_dest_mask]]

    status = np.full(n_pixels, "unmapped", dtype=object)
    status[~valid_mask] = "no_data"
    status[mapped_mask] = "off_grid"          # mapped to a plate but rotated outside dst grid
    status[valid_dest_mask] = "normal"
    is_overlap = (pixel_hit_counts > 1) & valid_dest_mask
    status[is_overlap] = "subducted"

    if is_overlap.sum() > 0:
        overlap_indices = np.where(is_overlap)[0]
        keys = rows[overlap_indices] * dst_shape[1] + cols[overlap_indices]
        priority_vals = np.array([
            SOURCE_CODE_PRIORITY.get(c, len(SOURCE_CODE_PRIORITY)) for c in src_codes[overlap_indices]
        ])
        df_overlap = pd.DataFrame({'key': keys, 'priority': priority_vals, 'global_idx': overlap_indices})
        df_overlap = df_overlap.sort_values(['key', 'priority'], kind='stable')
        survivors = df_overlap.drop_duplicates(subset='key', keep='first')['global_idx'].values
        status[survivors] = "survivor"

    status_code = np.vectorize(STATUS_CODES.get)(status).astype('uint8')

    # --- Altitude (paleo-DEM) ---
    dem_from_path = get_paleo_dem_path(age_from)
    dem_to_path = get_paleo_dem_path(age_to)

    with rasterio.open(dem_from_path) as dem_from:
        dem_from_array = dem_from.read(PALEO_DEM_BAND)
        dem_from_transform = dem_from.transform
        dem_from_shape = dem_from.shape

    with rasterio.open(dem_to_path) as dem_to:
        dem_to_array = dem_to.read(PALEO_DEM_BAND)
        dem_to_transform = dem_to.transform
        dem_to_shape = dem_to.shape

    orig_alt, _ = sample_raster_at_points(xs_flat, ys_flat, dem_from_transform, dem_from_array, dem_from_shape)

    dest_alt = np.full(n_pixels, np.nan)
    _vals, _ib = sample_raster_at_points(rot_xs[valid_dest_mask], rot_ys[valid_dest_mask],
                                          dem_to_transform, dem_to_array, dem_to_shape)
    dest_alt[np.where(valid_dest_mask)[0]] = _vals

    delta_alt = dest_alt - orig_alt

    # --- Temperature ---
    temp_from_path = get_temperature_path(age_from)
    temp_to_path = get_temperature_path(age_to)

    with rasterio.open(temp_from_path) as temp_from:
        temp_from_array = temp_from.read(CLIMATE_BAND)
        temp_from_transform = temp_from.transform
        temp_from_shape = temp_from.shape

    with rasterio.open(temp_to_path) as temp_to:
        temp_to_array = temp_to.read(CLIMATE_BAND)
        temp_to_transform = temp_to.transform
        temp_to_shape = temp_to.shape

    orig_temp, _ = sample_raster_at_points(xs_flat, ys_flat, temp_from_transform, temp_from_array, temp_from_shape)

    dest_temp = np.full(n_pixels, np.nan)
    _vals, _ib = sample_raster_at_points(rot_xs[valid_dest_mask], rot_ys[valid_dest_mask],
                                          temp_to_transform, temp_to_array, temp_to_shape)
    dest_temp[np.where(valid_dest_mask)[0]] = _vals

    delta_temp = dest_temp - orig_temp

    # --- Precipitation ---
    precip_from_path = get_precipitation_path(age_from)
    precip_to_path = get_precipitation_path(age_to)

    with rasterio.open(precip_from_path) as precip_from:
        precip_from_array = precip_from.read(CLIMATE_BAND)
        precip_from_transform = precip_from.transform
        precip_from_shape = precip_from.shape

    with rasterio.open(precip_to_path) as precip_to:
        precip_to_array = precip_to.read(CLIMATE_BAND)
        precip_to_transform = precip_to.transform
        precip_to_shape = precip_to.shape

    orig_precip, _ = sample_raster_at_points(xs_flat, ys_flat, precip_from_transform, precip_from_array, precip_from_shape)

    dest_precip = np.full(n_pixels, np.nan)
    _vals, _ib = sample_raster_at_points(rot_xs[valid_dest_mask], rot_ys[valid_dest_mask],
                                          precip_to_transform, precip_to_array, precip_to_shape)
    dest_precip[np.where(valid_dest_mask)[0]] = _vals

    delta_precip = dest_precip - orig_precip

    delta_lat_signed = rot_lats - lats
    delta_lat_poleward = np.abs(rot_lats) - np.abs(lats)  # positive = moved toward pole
    distance_km = haversine_km(lons, lats, rot_lons, rot_lats)

    keep = mapped_mask
    idx = np.where(keep)[0]

    pixel_df = pd.DataFrame({
        'pixel_idx': idx,
        'row': rows_idx.ravel()[idx],
        'col': cols_idx.ravel()[idx],
        'orig_lon': lons[idx],
        'orig_lat': lats[idx],
        'rot_lon': rot_lons[idx],
        'rot_lat': rot_lats[idx],
        'delta_lat_signed': delta_lat_signed[idx],
        'delta_lat_poleward': delta_lat_poleward[idx],
        'distance_km': distance_km[idx],
        'orig_alt': orig_alt[idx],
        'dest_alt': dest_alt[idx],
        'delta_alt': delta_alt[idx],
        'orig_temp': orig_temp[idx],
        'dest_temp': dest_temp[idx],
        'delta_temp': delta_temp[idx],
        'orig_precip': orig_precip[idx],
        'dest_precip': dest_precip[idx],
        'delta_precip': delta_precip[idx],
        'source_code': src_codes[idx],
        'dest_code': dest_land_cover[idx],
        'status': status[idx],
    })

    print(f"Pixel history rows: {len(pixel_df)} "
          f"(normal={ (status[idx]=='normal').sum() }, "
          f"survivor={ (status[idx]=='survivor').sum() }, "
          f"subducted={ (status[idx]=='subducted').sum() }, "
          f"off_grid={ (status[idx]=='off_grid').sum() }, "
          f"unmapped={ (status[idx]=='unmapped').sum() })")

    if save_parquet:
        out_path = Path(f"pixel_history_{int(age_from)}Ma_to_{int(age_to)}Ma.parquet")
        pixel_df.to_parquet(out_path, index=False)
        print(f"Saved to {out_path}")

    if save_raster in ('multiband', 'separate'):
        band_names = [
            'delta_lat_signed', 'delta_lat_poleward', 'distance_km',
            'delta_alt', 'delta_temp', 'delta_precip',
            'status_code', 'dest_code',
        ]
        band_arrays = [
            np.where(np.isnan(delta_lat_signed), RASTER_NODATA, delta_lat_signed).reshape(src_shape),
            np.where(np.isnan(delta_lat_poleward), RASTER_NODATA, delta_lat_poleward).reshape(src_shape),
            np.where(np.isnan(distance_km), RASTER_NODATA, distance_km).reshape(src_shape),
            np.where(np.isnan(delta_alt), RASTER_NODATA, delta_alt).reshape(src_shape),
            np.where(np.isnan(delta_temp), RASTER_NODATA, delta_temp).reshape(src_shape),
            np.where(np.isnan(delta_precip), RASTER_NODATA, delta_precip).reshape(src_shape),
            status_code.reshape(src_shape).astype('float32'),
            np.where(np.isnan(dest_land_cover), RASTER_NODATA, dest_land_cover).reshape(src_shape),
        ]

        if save_raster == 'multiband':
            out_path = Path(raster_out_dir) / f"pixel_history_{int(age_from)}Ma_to_{int(age_to)}Ma.tif"
            _write_multiband_raster(out_path, band_arrays, band_names, src_transform, src_crs)
        else:
            for arr, name in zip(band_arrays, band_names):
                out_path = Path(raster_out_dir) / f"pixel_history_{name}_{int(age_from)}Ma_to_{int(age_to)}Ma.tif"
                _write_multiband_raster(out_path, [arr], [name], src_transform, src_crs)

    return pixel_df


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

        print(f"\n--- Processing Transition: {year_from} ({age_from} Ma) -> {year_to} ({age_to} Ma) ---")

        try:
            build_change_matrix(age_from=age_from, age_to=age_to)
        except Exception as e:
            print(f"Error processing change matrix {age_from} -> {age_to}: {e}")

        try:
            build_pixel_history(age_from=age_from, age_to=age_to)
        except Exception as e:
            print(f"Error processing pixel history {age_from} -> {age_to}: {e}")