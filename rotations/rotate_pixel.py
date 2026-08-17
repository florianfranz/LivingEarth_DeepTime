import re
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import geopandas as gpd
import rasterio
from rasterio.plot import show as rshow
from pyproj import Transformer
from shapely.geometry import Point
import fiona

# --------------------------------------------------------------------------
# Configuration & Paths
# --------------------------------------------------------------------------

GPKG_PATH = r"C:\Users\franzisf\PycharmProjects\LivingEarth_DeepTime\rotations\PlatesxxxpreRotyyy_geom_valid_harmo.gpkg"

# Updated Palaeogeography Path
PALAEO_BASE_DIR = r"C:\Users\franzisf\Documents\PANALESIS_Atlas\Output\Palaeogeography"
PALAEO_FILENAME_TEMPLATE = "palaeogeography_{}.tif"

RASTER_BASE_DIR = r"C:\Users\franzisf\PycharmProjects\LivingEarth_DeepTime\test_processing\output"
RASTER_FILENAME = "level4_out_LE_DT.tif"
RASTER_AGE_OFFSET = 2000

ESRI_54034 = "ESRI:54034"
EPSG_4326 = "EPSG:4326"

LAYER_RE = re.compile(r"^Plates(\d+)preRot(\d+)$")
ANGLE_SIGN = 1.0

# --------------------------------------------------------------------------
# Custom Palaeogeography Colormap
# --------------------------------------------------------------------------

def create_palaeo_cmap():
    """Creates the LinearSegmentedColormap and Normalize object based on your specific entries."""
    colormap_entries = [
        (9000, "#ffffff", "9000"),
        (7000, "#cecece", "7000"),
        (5000, "#a1a1a1", "5000"),
        (3000, "#821e1e", "3000"),
        (2000, "#a34400", "2000"),
        (1000, "#e8d67d", "1000"),
        (200, "#107b30", "200"),
        (0, "#006147", "0"),
        (-100, "#b0e2ff", "-100"),
        (-500, "#87cefa", "-500"),
        (-2000, "#188ccd", "-2000"),
        (-4000, "#136ca0", "-4000"),
        (-7000, "#003266", "-7000"),
        (-9000, "#001e64", "-9000"),
        (-11000, "#000050", "-11000"),
    ]

    # Sort by elevation value
    colormap_entries_sorted = sorted(colormap_entries, key=lambda x: x[0])
    elevations = [e[0] for e in colormap_entries_sorted]
    colors = [e[1] for e in colormap_entries_sorted]

    vmin, vmax = elevations[0], elevations[-1]

    # Normalize positions (0.0 to 1.0)
    norm_values = [(e - vmin) / (vmax - vmin) for e in elevations]

    # Create colormap
    cmap = mcolors.LinearSegmentedColormap.from_list(
        "palaeogeography",
        list(zip(norm_values, colors))
    )

    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)

    return cmap, norm, vmin, vmax

# Pre-build the colormap objects
PALAEO_CMAP, PALAEO_NORM, PALAEO_VMIN, PALAEO_VMAX = create_palaeo_cmap()

# --------------------------------------------------------------------------
# Helper Functions (Existing Logic)
# --------------------------------------------------------------------------

def build_layer_lookup(gpkg_path: str) -> dict:
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
        raise ValueError(f"No layer found starting from age {age} Ma. Available: {available}")
    return lookup[age_int]

_to_lonlat = Transformer.from_crs(ESRI_54034, EPSG_4326, always_xy=True)
_to_meters = Transformer.from_crs(EPSG_4326, ESRI_54034, always_xy=True)

def meters_to_lonlat(x: float, y: float) -> tuple:
    return _to_lonlat.transform(x, y)

def lonlat_to_meters(lon: float, lat: float) -> tuple:
    return _to_meters.transform(lon, lat)

def lonlat_to_xyz(lon_deg: float, lat_deg: float) -> np.ndarray:
    lon = np.radians(lon_deg)
    lat = np.radians(lat_deg)
    return np.array([
        np.cos(lat) * np.cos(lon),
        np.cos(lat) * np.sin(lon),
        np.sin(lat)
    ])

def xyz_to_lonlat(vec: np.ndarray) -> tuple:
    x, y, z = vec
    lat = np.degrees(np.arcsin(np.clip(z, -1.0, 1.0)))
    lon = np.degrees(np.arctan2(y, x))
    return lon, lat

def rotate_point(lon: float, lat: float, pole_lon: float, pole_lat: float, angle_deg: float) -> tuple:
    p = lonlat_to_xyz(lon, lat)
    k = lonlat_to_xyz(pole_lon, pole_lat)
    theta = np.radians(ANGLE_SIGN * angle_deg)
    p_rot = (
        p * np.cos(theta)
        + np.cross(k, p) * np.sin(theta)
        + k * np.dot(k, p) * (1 - np.cos(theta))
    )
    p_rot = p_rot / np.linalg.norm(p_rot)
    return xyz_to_lonlat(p_rot)

def get_palaeo_raster_path(age: float) -> str:
    year = 2000 + int(round(age))
    filename = PALAEO_FILENAME_TEMPLATE.format(year)
    return str(Path(PALAEO_BASE_DIR) / filename)

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

# --------------------------------------------------------------------------
# Visualization Function
# --------------------------------------------------------------------------

def plot_rotation_on_map(age, original_xy, rotated_xy, plate_name):
    """
    Loads the palaeogeography raster, applies the custom colormap,
    and overlays the rotation vector.
    """
    x1, y1 = original_xy
    x2, y2 = rotated_xy

    # 1. Locate Raster
    raster_path = get_palaeo_raster_path(age)

    if not Path(raster_path).exists():
        print(f"ERROR: Map file not found at: {raster_path}")
        print("Check PALAEO_BASE_DIR and file naming (e.g., palaeogeography_2006.tif)")
        return

    # 2. Setup Plot
    fig, ax = plt.subplots(figsize=(12, 9))

    # 3. Read and Display Raster
    with rasterio.open(raster_path) as src:
        # Read the first band
        raster_data = src.read(1).astype(float)

        # Handle NoData
        nodata = src.nodata
        if nodata is not None:
            raster_data[raster_data == nodata] = np.nan

        # Define extent for plotting (left, right, bottom, top)
        # rasterio bounds: left, bottom, right, top
        bounds = src.bounds
        extent = [bounds.left, bounds.right, bounds.bottom, bounds.top]

        # Plot using imshow with custom cmap and norm
        # origin='upper' ensures Y axis matches standard map orientation
        img = ax.imshow(
            raster_data,
            cmap=PALAEO_CMAP,
            norm=PALAEO_NORM,
            extent=extent,
            origin='upper',
            alpha=0.9,
            zorder=1
        )

    # 4. Calculate Displacement
    dx = x2 - x1
    dy = y2 - y1
    dist_m = np.sqrt(dx**2 + dy**2)
    dist_km = dist_m / 1000.0

    # 5. Overlay Vector
    # Scale arrowhead relative to distance (5% of length)
    head_w = dist_m * 0.05 if dist_m > 0 else 50000
    head_l = dist_m * 0.05 if dist_m > 0 else 50000

    ax.arrow(x1, y1, dx, dy,
             head_width=head_w,
             head_length=head_l,
             fc='red', ec='red',
             linewidth=3,
             length_includes_head=True,
             zorder=10,
             label=f'Movement ({dist_km:.1f} km)')

    # 6. Overlay Points
    ax.plot(x1, y1, 'bo', markersize=10, zorder=11, label=f'Start ({age} Ma)')
    ax.plot(x2, y2, 'go', markersize=10, zorder=11, label=f'End (240 Ma)')

    # 7. Colorbar
    cbar = plt.colorbar(img, ax=ax, shrink=0.8)
    cbar.set_label("Elevation (m)")

    # 8. Formatting
    year = 2000 + int(round(age))
    ax.set_title(f"Palaeogeography Reconstruction ({age} Ma)\nPlate: {plate_name}", fontsize=14)
    ax.set_xlabel("X (ESRI:54034 meters)")
    ax.set_ylabel("Y (ESRI:54034 meters)")

    ax.set_aspect('equal')
    ax.grid(True, linestyle='--', alpha=0.5, zorder=5)
    ax.legend(loc='best')

    # 9. Auto-Zoom to Area of Interest
    # Prevents showing the whole world if the movement is local
    padding = max(dist_m * 1.5, 1500000) # 1.5x movement or min 1000km
    ax.set_xlim(x1 - padding, x2 + padding)
    ax.set_ylim(min(y1, y2) - padding, max(y1, y2) + padding)

    plt.show()

# --------------------------------------------------------------------------
# Main Execution
# --------------------------------------------------------------------------

if __name__ == "__main__":
    # Configuration
    age_ma = 250
    x_in, y_in = -9_001_000.0, -0_501_000.0  # ESRI:54034 meters

    # 1. Perform Rotation Logic
    layer_name = get_layer_for_age(GPKG_PATH, age_ma)
    gdf = gpd.read_file(GPKG_PATH, layer=layer_name)

    lon, lat = meters_to_lonlat(x_in, y_in)
    plate = find_plate(gdf, lon, lat)

    new_lon, new_lat = rotate_point(lon, lat, plate.eu_long, plate.eu_lat, plate.eu_ang)
    x_out, y_out = lonlat_to_meters(new_lon, new_lat)

    print(f"Plate: {plate.plate_name}")
    print(f"Original: ({x_in:,.0f}, {y_in:,.0f})")
    print(f"Rotated : ({x_out:,.0f}, {y_out:,.0f})")

    raster_path = get_palaeo_raster_path(age_ma)
    print(f"Loading map: {raster_path}")

    # 2. Plot
    plot_rotation_on_map(
        age=age_ma,
        original_xy=(x_in, y_in),
        rotated_xy=(x_out, y_out),
        plate_name=plate.plate_name
    )