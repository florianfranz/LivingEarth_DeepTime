import re
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import geopandas as gpd
import rasterio
from pyproj import Transformer
from shapely.geometry import Point
import fiona
import json
from pathlib import Path

# --------------------------------------------------------------------------
# Configuration & Paths
# --------------------------------------------------------------------------
CONFIG_PATH = Path(__file__).resolve().parent.parent / "config.json"
with open(CONFIG_PATH, "r") as f:
    _config = json.load(f)

GPKG_PATH = _config["plates_gpkg"]
PALAEO_BASE_DIR = _config["paleo_folder"]
PALAEO_FILENAME_TEMPLATE = "palaeogeography_{}.tif"

ESRI_54034 = "ESRI:54034"
EPSG_4326 = "EPSG:4326"
LAYER_RE = re.compile(r"^Plates(\d+)preRot(\d+)$")
ANGLE_SIGN = 1.0

# South America Extent (ESRI:54034 meters)
SA_EXTENT = {
    "north": 0,
    "south": -2000000,
    "west": -10000000,
    "east": -6000000
}

# Target Pixel Configuration
TARGET_AGE = 6
PIXEL_X_6MA = -8473561.0
PIXEL_Y_6MA = -1310551.0


# --------------------------------------------------------------------------
# Custom Palaeogeography Colormap
# --------------------------------------------------------------------------

def create_palaeo_cmap():
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
    colormap_entries_sorted = sorted(colormap_entries, key=lambda x: x[0])
    elevations = [e[0] for e in colormap_entries_sorted]
    colors = [e[1] for e in colormap_entries_sorted]
    vmin, vmax = elevations[0], elevations[-1]
    norm_values = [(e - vmin) / (vmax - vmin) for e in elevations]
    cmap = mcolors.LinearSegmentedColormap.from_list("palaeogeography", list(zip(norm_values, colors)))
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
    return cmap, norm


PALAEO_CMAP, PALAEO_NORM = create_palaeo_cmap()


# --------------------------------------------------------------------------
# Core Logic Functions
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
        raise ValueError(f"No layer found for age {age} Ma.")
    return lookup[age_int]


_to_lonlat = Transformer.from_crs(ESRI_54034, EPSG_4326, always_xy=True)
_to_meters = Transformer.from_crs(EPSG_4326, ESRI_54034, always_xy=True)


def meters_to_lonlat(x, y): return _to_lonlat.transform(x, y)


def lonlat_to_meters(lon, lat): return _to_meters.transform(lon, lat)


def lonlat_to_xyz(lon_deg, lat_deg):
    lon, lat = np.radians(lon_deg), np.radians(lat_deg)
    return np.array([np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)])


def xyz_to_lonlat(vec):
    x, y, z = vec
    lat = np.degrees(np.arcsin(np.clip(z, -1.0, 1.0)))
    lon = np.degrees(np.arctan2(y, x))
    return lon, lat


def rotate_point(lon, lat, pole_lon, pole_lat, angle_deg):
    p = lonlat_to_xyz(lon, lat)
    k = lonlat_to_xyz(pole_lon, pole_lat)
    theta = np.radians(ANGLE_SIGN * angle_deg)
    p_rot = (p * np.cos(theta) + np.cross(k, p) * np.sin(theta) + k * np.dot(k, p) * (1 - np.cos(theta)))
    return xyz_to_lonlat(p_rot / np.linalg.norm(p_rot))


def get_palaeo_raster_path(age: float) -> str:
    year = 2000 + int(round(age))
    return str(Path(PALAEO_BASE_DIR) / PALAEO_FILENAME_TEMPLATE.format(year))


def find_plate(gdf, lon, lat):
    point = Point(lon, lat)
    hits = gdf[gdf.geometry.contains(point)]
    if hits.empty: hits = gdf[gdf.geometry.intersects(point)]
    if hits.empty: raise ValueError(f"No plate found at ({lon}, {lat})")
    row = hits.iloc[0]
    return row["PlateName"], row["EuLong"], row["EuLat"], row["EuAng"]


# --------------------------------------------------------------------------
# Main Execution & Plotting
# --------------------------------------------------------------------------

if __name__ == "__main__":
    # 1. Calculate Rotation
    print(f"--- Calculating Rotation for Age {TARGET_AGE} Ma ---")
    layer_name = get_layer_for_age(GPKG_PATH, TARGET_AGE)
    gdf = gpd.read_file(GPKG_PATH, layer=layer_name)

    # Convert start point to Lon/Lat to find plate
    lon_6, lat_6 = meters_to_lonlat(PIXEL_X_6MA, PIXEL_Y_6MA)
    plate_name, eu_lon, eu_lat, eu_ang = find_plate(gdf, lon_6, lat_6)

    # Rotate to get 0 Ma position
    lon_0, lat_0 = rotate_point(lon_6, lat_6, eu_lon, eu_lat, eu_ang)
    pixel_x_0ma, pixel_y_0ma = lonlat_to_meters(lon_0, lat_0)

    print(f"Plate: {plate_name}")
    print(f"Start (6 Ma):  ({PIXEL_X_6MA:,.0f}, {PIXEL_Y_6MA:,.0f})")
    print(f"End   (0 Ma):  ({pixel_x_0ma:,.0f}, {pixel_y_0ma:,.0f})")

    # Displacement calculation
    dx = pixel_x_0ma - PIXEL_X_6MA
    dy = pixel_y_0ma - PIXEL_Y_6MA
    dist_km = np.sqrt(dx ** 2 + dy ** 2) / 1000.0
    print(f"Displacement:  {dist_km:.2f} km")

    # 2. Prepare Plot
    fig, axes = plt.subplots(1, 2, figsize=(16, 80))

    # --- LEFT MAP: 6 Ma ---
    ax6 = axes[0]
    r_path_6 = get_palaeo_raster_path(6)

    if Path(r_path_6).exists():
        with rasterio.open(r_path_6) as src:
            data = src.read(1).astype(float)
            if src.nodata is not None: data[data == src.nodata] = np.nan
            bounds = src.bounds
            extent = [bounds.left, bounds.right, bounds.bottom, bounds.top]

            ax6.imshow(data, cmap=PALAEO_CMAP, norm=PALAEO_NORM, extent=extent, origin='upper', alpha=0.9)

            # Plot the pixel at its 6 Ma location
            ax6.plot(PIXEL_X_6MA, PIXEL_Y_6MA, 'o', color='blue', markersize=12,
                     markeredgecolor='white', markeredgewidth=2, label='Original Location')

            ax6.set_xlim(SA_EXTENT["west"], SA_EXTENT["east"])
            ax6.set_ylim(SA_EXTENT["south"], SA_EXTENT["north"])
            ax6.set_title(f"Palaeogeography: 6 Ma\nPlate: {plate_name}", fontsize=14)
            ax6.set_xlabel("X (ESRI:54034 m)")
            ax6.set_ylabel("Y (ESRI:54034 m)")
            ax6.set_aspect('equal')
            ax6.grid(True, linestyle='--', alpha=0.5)
            ax6.legend(loc='upper right')
            ax6.text(0.02, 0.98, f"Coords: ({PIXEL_X_6MA / 1e6:.2f}M, {PIXEL_Y_6MA / 1e6:.2f}M)",
                     transform=ax6.transAxes, fontsize=10, verticalalignment='top',
                     bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    else:
        ax6.text(0.5, 0.5, f"Map not found:\n{r_path_6}", transform=ax6.transAxes, ha='center', va='center')

    # --- RIGHT MAP: 0 Ma ---
    ax0 = axes[1]
    r_path_0 = get_palaeo_raster_path(0)

    if Path(r_path_0).exists():
        with rasterio.open(r_path_0) as src:
            data = src.read(1).astype(float)
            if src.nodata is not None: data[data == src.nodata] = np.nan
            bounds = src.bounds
            extent = [bounds.left, bounds.right, bounds.bottom, bounds.top]

            ax0.imshow(data, cmap=PALAEO_CMAP, norm=PALAEO_NORM, extent=extent, origin='upper', alpha=0.9)

            # 1. Plot the NEW position (Rotated) in RED
            ax0.plot(pixel_x_0ma, pixel_y_0ma, 'o', color='red', markersize=12,
                     markeredgecolor='white', markeredgewidth=2, label='Rotated')

            # 2. Plot the ORIGINAL position (Static) in BLUE (Hollow)
            # This shows where the coordinate was relative to the modern map
            ax0.plot(PIXEL_X_6MA, PIXEL_Y_6MA, 'o', color='blue', markersize=10, markeredgewidth=2,
                     label='Original Position')

            # Optional: Draw a line between them on the 0 Ma map to show shift
            ax0.plot([PIXEL_X_6MA, pixel_x_0ma], [PIXEL_Y_6MA, pixel_y_0ma],
                     color='gray', linestyle=':', linewidth=1.5, alpha=0.7)

            ax0.set_xlim(SA_EXTENT["west"], SA_EXTENT["east"])
            ax0.set_ylim(SA_EXTENT["south"], SA_EXTENT["north"])
            ax0.set_title(f"Palaeogeography: 0 Ma (Modern)\nDisplacement: {dist_km:.0f} km", fontsize=14)
            ax0.set_xlabel("X (ESRI:54034 m)")
            ax0.set_ylabel("Y (ESRI:54034 m)")
            ax0.set_aspect('equal')
            ax0.grid(True, linestyle='--', alpha=0.5)
            ax0.legend(loc='upper right')
            ax0.text(0.02, 0.98, f"New: ({pixel_x_0ma / 1e6:.2f}M, {pixel_y_0ma / 1e6:.2f}M)",
                     transform=ax0.transAxes, fontsize=10, verticalalignment='top',
                     bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    else:
        ax0.text(0.5, 0.5, f"Map not found:\n{r_path_0}", transform=ax0.transAxes, ha='center', va='center')


    plt.tight_layout(pad=5)
    plt.show()