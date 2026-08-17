import requests
from xml.etree import ElementTree as ET
from rasterio.io import MemoryFile
import numpy as np
from skimage import measure
from shapely.geometry import Polygon
import pandas as pd
import geopandas as gpd
from rasterio.features import rasterize
import matplotlib.pyplot as plt
import os
from scipy.interpolate import PchipInterpolator

def get_available_ages_web(
        layer_name,
        base_url="https://geoserver.panalesis.org/geoserver/",
        workspace="panalesis_atlas",
        service_type="WCS",
        service_version="1.0.0"
):
    describe_url = (
        f"{base_url}{workspace}/{service_type.lower()}"
        f"?service={service_type}&version={service_version}"
        f"&request=DescribeCoverage&coverage={workspace}:{layer_name}"
    )
    response = requests.get(describe_url)
    root = ET.fromstring(response.content)

    years = sorted(set(
        int(el.text.split("-")[0])
        for el in root.iter()
        if el.tag.endswith('timePosition') and el.text
    ))
    geological_ages = [year - 2000 for year in years]

    return geological_ages


def construct_wcs_url(layer_name,
                      geological_age,
                      base_url="https://geoserver.panalesis.org/geoserver/",
                      workspace="panalesis_atlas",
                      service_type="WCS",
                      service_version="1.0.0",
                      request_type="GetCoverage",
                      crs="EPSG:54034",
                      bbox="-20037508.34,-6363885.33,20037508.34,6363885.33",
                      resx="10000",
                      resy="10000",
                      format="GEOTIFF"):
    time = f"{geological_age + 2000:04d}-01-01T00:00:00.000Z"

    url = (
        rf"{base_url}{workspace}/{service_type.lower()}?"
        rf"service={service_type}&"
        rf"version={service_version}&"
        rf"request={request_type}&"
        rf"coverage={workspace}:{layer_name}&"
        rf"crs={crs}&"
        rf"bbox={bbox}&"
        rf"resx={resx}&"
        rf"resy={resy}&"
        rf"time={time}&"
        rf"format={format}"
    )
    return url

def load_tiff_web(url):
    """
    Fetch a GeoTIFF from a WCS URL and return the data and associated metadata,
    in the same format as load_tiff().
    """
    response = requests.get(url)
    response.raise_for_status()  # raise an error if the request failed

    with MemoryFile(response.content) as memfile:
        with memfile.open() as src:
            data = src.read()
            metadata = src.meta

    return data, metadata

def plot_mask(mask, title):
    plt.figure(figsize=(10, 6))
    plt.imshow(mask, cmap='Greys', interpolation='none')
    plt.title(title)
    plt.xlabel("Longitude (meters)")
    plt.ylabel("Latitude (meters)")
    plt.tight_layout()
    plt.show()

def get_land_area(data,transform, plot=False):
    """
    Process the input raster to calculate an area after reprojection.

    Returns:
    - tuple: (area in square meters, volume in cubic meters).
    """

    pixel_area = abs(transform[0] * transform[4])  # width * height of a pixel in meters
    land_mask = data[0] >= 0  # Mask the pixels where the elevation is above or equal to sea level
    if plot == True:
        plot_mask(land_mask, "Land (z >= 0m)")

    # Calculate area (count of pixels * pixel area)
    land_area = np.sum(land_mask) * pixel_area
    total_area = data[0].size * pixel_area

    return land_area, total_area

def close_contour(contour):
    """
    Ensures that a contour line is closed to form a valid polygon.
    For contours spanning near the antimeridian, close them by adding intermediate points
    at the antimeridian (-180 or 180) to form a properly closed polygon.
    """
    if len(contour) < 3:
        return None  # Not enough points to form a polygon

    # Get first and last points
    first_lon, first_lat = contour[0]
    last_lon, last_lat = contour[-1]

    # Determine the pole latitude based on the hemisphere of the contour
    pole_lat = -6360338.2938046408817172 if last_lat < 0 else 6360338.2938046408817172

    # Define longitude threshold for detecting near-antimeridian points
    lon_threshold = 19982002.33 #based on a 179.5 threshold for a 180 longitude value, rescaled to WCEA (World Cylindrical Equal Area) projection values

    # Check if the contour spans the antimeridian
    if first_lon <= -lon_threshold and last_lon >= lon_threshold:
        # First point near -180 and last point near 180
        contour.append((20037508.3427892439067364, last_lat))  # Add point at the WCEA projection east border with last latitude
        contour.append((20037508.3427892439067364, pole_lat))  # Add point at the WCEA projection east border and the pole
        contour.append((-20037508.3427892439067364, pole_lat))  # Add point at the WCEA projection west border and the pole
        contour.append((-20037508.3427892439067364, first_lat))  # Add point at the WCEA projection wast border with first latitude

    elif first_lon >= lon_threshold and last_lon <= -lon_threshold:
        # First point near 180 and last point near -180
        contour.append((-20037508.3427892439067364, last_lat))  # Add point at WCEA projection wast border with last latitude
        contour.append((-20037508.3427892439067364, pole_lat))  # Add point at WCEA projection wast border and the pole
        contour.append((20037508.3427892439067364, pole_lat))  # Add point at WCEA projection east border and the pole
        contour.append((20037508.3427892439067364, first_lat))  # Add point at WCEA projection east border with first latitude

    # Ensure the contour is a closed loop
    if contour[0] != contour[-1]:
        contour.append(contour[0])

    return contour

def create_contours(data, transform, level):
    """
    Process the input raster to create coastline contours.

    Returns:
    - a GeoDataFrame (gdf) with all contour features, or an empty GeoDataFrame if no contours are found.
    """
    polygons = []
    contours = measure.find_contours(data[0], level)

    if not contours:
        return gpd.GeoDataFrame(geometry=[], crs='ESRI:54034')

    for contour in contours:
        transformed_contour = []
        for point in contour:
            x, y = transform * (point[1], point[0])  # Transform to geographic coordinates
            transformed_contour.append((x, y))

        closed_contour = close_contour(transformed_contour)
        if closed_contour:
            polygon = Polygon(closed_contour)
            if polygon.is_valid:
                polygons.append({'geometry': polygon, 'level': level})

    if not polygons:
        return gpd.GeoDataFrame(geometry=[], crs='ESRI:54034')

    gdf = gpd.GeoDataFrame(polygons)
    gdf = gdf.set_geometry('geometry')
    gdf.set_crs('ESRI:54034', allow_override=True)

    return gdf

def get_total_length(gdf):
    """
    Calculates the total length of all features in the GeoDataFrame.

    Parameters:
        gdf (GeoDataFrame): A GeoDataFrame with polygon geometries.

    Returns:
        float: Total perimeter length in meters.
    """
    # Ensure geometry is valid and in a projected CRS
    if gdf.crs is None:
        gdf.set_crs('ESRI:54034', allow_override=True)

    # Convert polygons to boundaries and calculate lengths
    total_length = gdf.geometry.boundary.length.sum()

    return total_length


def filter_large_polygons(gdf, min_area_m2):
    """
    Filters polygons in the GeoDataFrame to keep only those with an area over the specified minimum area.

    Parameters:
        gdf (GeoDataFrame): A GeoDataFrame with polygon geometries.
        min_area_m2 (float): Minimum area in square meters.

    Returns:
        GeoDataFrame: Filtered GeoDataFrame with polygons having area over the specified minimum.
    """
    filtered_gdf = gdf[gdf.geometry.area >= min_area_m2]
    return filtered_gdf

def get_polar_mask(raster_meta, raster_shape):
    """
    Generate a binary mask where True indicates pixels either north of Polar_N or south of Polar_S.
    """
    latitudes_path = "reproj_latitudes.geojson"
    gdf = gpd.read_file(latitudes_path)

    required_lines = ['Polar_N', 'Polar_S']
    if not all(name in gdf['name'].values for name in required_lines):
        raise ValueError("GeoJSON must contain both 'Polar_N' and 'Polar_S' lines.")

    polar_mask = np.zeros(raster_shape, dtype=bool)

    for name in required_lines:
        geom = gdf[gdf['name'] == name].geometry
        if geom.empty:
            continue

        rasterized = rasterize(
            [(g, 1) for g in geom],
            out_shape=raster_shape,
            transform=raster_meta['transform'],
            fill=0,
            all_touched=True,
            dtype='uint8'
        )

        coords = np.indices(raster_shape)
        ys = coords[0]

        if name == 'Polar_N':
            rows = np.where(rasterized == 1)[0]
            if rows.size > 0:
                polar_mask |= ys < rows.min()
        elif name == 'Polar_S':
            rows = np.where(rasterized == 1)[0]
            if rows.size > 0:
                polar_mask |= ys > rows.max()

    return polar_mask

def get_polar_area(data, metadata, transform, plot=False):
    """
    Process the input raster to calculate total and polar land areas after reprojection.

    Returns:
    - tuple: (land area m², total area m², polar land area m²)
    """


    pixel_area = abs(transform[0] * transform[4])  # pixel width × height in meters
    elevation = data[0]

    # Land mask: elevation >= 0
    land_mask = elevation >= 0

    # Get polar region mask
    polar_mask = get_polar_mask(metadata, elevation.shape)

    # Combined mask: land AND within polar regions
    combined_mask = np.logical_and(land_mask, polar_mask)
    if plot == True:
        plot_mask(combined_mask, "Polar Land (Latitude > 60° N/S)")

    # Area calculations
    polar_land_area = np.sum(combined_mask) * pixel_area

    return polar_land_area

def get_temperate_mask(raster_meta, raster_shape):
    """
    Generate a binary mask where True indicates pixels between:
    - Temperate_N and Polar_N (Northern Hemisphere)
    - Polar_S and Temperate_S (Southern Hemisphere)
    """
    latitudes_path = "reproj_latitudes.geojson"
    gdf = gpd.read_file(latitudes_path)

    required_lines = ['Temperate_N', 'Polar_N', 'Temperate_S', 'Polar_S']
    if not all(name in gdf['name'].values for name in required_lines):
        raise ValueError("GeoJSON must contain all required latitude boundary lines.")

    line_rows = {}

    # Rasterize each line to find its row position
    for name in required_lines:
        geom = gdf[gdf['name'] == name].geometry
        if geom.empty:
            continue

        rasterized = rasterize(
            [(g, 1) for g in geom],
            out_shape=raster_shape,
            transform=raster_meta['transform'],
            fill=0,
            all_touched=True,
            dtype='uint8'
        )

        rows = np.where(rasterized == 1)[0]
        if rows.size > 0:
            line_rows[name] = rows.mean()  # average row if line spans multiple rows

    if len(line_rows) != 4:
        raise ValueError("Could not locate all required latitude boundaries in raster.")

    ys = np.indices(raster_shape)[0]
    temperate_mask_north = (ys >= line_rows['Polar_N']) & (ys <= line_rows['Temperate_N'])
    temperate_mask_south = (ys <= line_rows['Polar_S']) & (ys >= line_rows['Temperate_S'])

    temperate_mask = temperate_mask_north | temperate_mask_south

    return temperate_mask


def get_temperate_area(data, metadata, transform, plot=False):
    """
    Process the input raster to calculate total and temperate land areas after reprojection.

    Returns:
    - tuple: (land area m², total area m², temperate land area m²)
    """

    pixel_area = abs(transform[0] * transform[4])  # pixel width × height in meters
    elevation = data[0]

    # Land mask: elevation >= 0
    land_mask = elevation >= 0

    # Get temperate region mask
    temperate_mask = get_temperate_mask(metadata, elevation.shape)

    # Combined mask: land AND within temperate regions
    combined_mask = np.logical_and(land_mask, temperate_mask)
    if plot == True:
        plot_mask(combined_mask, "Temperate Land (23.5° < Latitude > 40° N/S)")

    # Area calculations
    total_area = elevation.size * pixel_area
    temperate_land_area = np.sum(combined_mask) * pixel_area

    return temperate_land_area

def get_subtropical_mask(raster_meta, raster_shape):
    """
    Generate a binary mask where True indicates pixels between:
    - Subtropical_N and Temperate_N (Northern Hemisphere)
    - Temperate_S and Subtropical_S(Southern Hemisphere)
    """
    latitudes_path = "reproj_latitudes.geojson"
    gdf = gpd.read_file(latitudes_path)

    required_lines = ['Subtropical_N', 'Temperate_N', 'Subtropical_S', 'Temperate_S']
    if not all(name in gdf['name'].values for name in required_lines):
        raise ValueError("GeoJSON must contain all required latitude boundary lines.")

    line_rows = {}

    # Rasterize each line to find its row position
    for name in required_lines:
        geom = gdf[gdf['name'] == name].geometry
        if geom.empty:
            continue

        rasterized = rasterize(
            [(g, 1) for g in geom],
            out_shape=raster_shape,
            transform=raster_meta['transform'],
            fill=0,
            all_touched=True,
            dtype='uint8'
        )

        rows = np.where(rasterized == 1)[0]
        if rows.size > 0:
            line_rows[name] = rows.mean()  # average row if line spans multiple rows

    if len(line_rows) != 4:
        raise ValueError("Could not locate all required latitude boundaries in raster.")

    ys = np.indices(raster_shape)[0]
    subtropical_mask_north = (ys >= line_rows['Temperate_N']) & (ys <= line_rows['Subtropical_N'])
    subtropical_mask_south = (ys <= line_rows['Temperate_S']) & (ys >= line_rows['Subtropical_S'])

    subtropical_mask = subtropical_mask_north | subtropical_mask_south

    return subtropical_mask


def get_subtropical_area(data, metadata, transform, plot=False):
    """
    Process the input raster to calculate total and subtropical land areas after reprojection.

    Returns:
    - tuple: (land area m², total area m², subtropical land area m²)
    """

    pixel_area = abs(transform[0] * transform[4])  # pixel width × height in meters
    elevation = data[0]

    # Land mask: elevation >= 0
    land_mask = elevation >= 0

    # Get subtropical region mask
    subtropical_mask = get_subtropical_mask(metadata, elevation.shape)

    # Combined mask: land AND within subtropical regions
    combined_mask = np.logical_and(land_mask, subtropical_mask)
    if plot == True:
        plot_mask(combined_mask,"Subtropical Land (23.5° < Latitude > 40° N/S)")


    # Area calculations
    subtropical_land_area = np.sum(combined_mask) * pixel_area

    return subtropical_land_area

def get_tropical_mask(raster_meta, raster_shape):
    """
    Generate a binary mask where True indicates pixels between:
    - Subtropical_S and Subtropical_N
    """
    latitudes_path = "reproj_latitudes.geojson"
    gdf = gpd.read_file(latitudes_path)

    required_lines = ['Subtropical_N', 'Subtropical_S']
    if not all(name in gdf['name'].values for name in required_lines):
        raise ValueError("GeoJSON must contain all required latitude boundary lines.")

    line_rows = {}

    # Rasterize each line to find its row position
    for name in required_lines:
        geom = gdf[gdf['name'] == name].geometry
        if geom.empty:
            continue

        rasterized = rasterize(
            [(g, 1) for g in geom],
            out_shape=raster_shape,
            transform=raster_meta['transform'],
            fill=0,
            all_touched=True,
            dtype='uint8'
        )

        rows = np.where(rasterized == 1)[0]
        if rows.size > 0:
            line_rows[name] = rows.mean()  # average row if line spans multiple rows

    if len(line_rows) != 2:
        raise ValueError("Could not locate all required latitude boundaries in raster.")

    ys = np.indices(raster_shape)[0]
    tropical_mask = (ys >= line_rows['Subtropical_N']) & (ys <= line_rows['Subtropical_S'])

    return tropical_mask


def get_tropical_area(data, metadata, transform, plot=False):
    """
    Process the input raster to calculate total and tropical land areas after reprojection.

    Returns:
    - tuple: (land area m², total area m², tropical land area m²)
    """

    pixel_area = abs(transform[0] * transform[4])  # pixel width × height in meters
    elevation = data[0]

    # Land mask: elevation >= 0
    land_mask = elevation >= 0

    # Get tropical region mask
    tropical_mask = get_tropical_mask(metadata, elevation.shape)

    # Combined mask: land AND within tropical regions
    combined_mask = np.logical_and(land_mask, tropical_mask)
    if plot == True:
        plot_mask(combined_mask, "Tropical Land (23.5° < Latitude > 40° N/S)")

    # Area calculations
    total_area = elevation.size * pixel_area
    tropical_land_area = np.sum(combined_mask) * pixel_area

    return tropical_land_area

def get_hemispheres_mask(raster_meta, raster_shape):
    """
    Generate a binary mask where True indicates pixels either north of Polar_N or south of Polar_S.
    """
    latitudes_path = "reproj_latitudes.geojson"
    gdf = gpd.read_file(latitudes_path)

    required_lines = ['Equator']
    if not all(name in gdf['name'].values for name in required_lines):
        raise ValueError("GeoJSON must contain both 'Polar_N' and 'Polar_S' lines.")

    northern_mask = np.zeros(raster_shape, dtype=bool)
    southern_mask = np.zeros(raster_shape, dtype=bool)

    for name in required_lines:
        geom = gdf[gdf['name'] == name].geometry
        if geom.empty:
            continue

        rasterized = rasterize(
            [(g, 1) for g in geom],
            out_shape=raster_shape,
            transform=raster_meta['transform'],
            fill=0,
            all_touched=True,
            dtype='uint8'
        )

        coords = np.indices(raster_shape)
        ys = coords[0]

        if name == 'Equator':
            rows = np.where(rasterized == 1)[0]
            if rows.size > 0:
                northern_mask |= ys <= rows.min()
                southern_mask |= ys > rows.max()

    return northern_mask, southern_mask

def get_hemispheres_area(data,metadata,transform,plot=False):
    """
    Process the input raster to calculate both hemispheres land areas after reprojection.

    Returns:
    - tuple: (northern and southern hemispheres land area m²)
    """
    pixel_area = abs(transform[0] * transform[4])  # pixel width × height in meters
    elevation = data[0]

    # Land mask: elevation >= 0
    land_mask = elevation >= 0

    # Get polar region mask
    northern_mask, southern_mask = get_hemispheres_mask(metadata, elevation.shape)

    # Combined mask: land AND within polar regions
    land_northern_mask = np.logical_and(land_mask, northern_mask)
    land_southern_mask = np.logical_and(land_mask, southern_mask)
    if plot == True:
        plot_mask(land_northern_mask, "Land Pixels in Northern Hemisphere")
        plot_mask(land_southern_mask, "Land Pixels in Southern Hemisphere")

    # Area calculations
    northern_land_area = np.sum(land_northern_mask) * pixel_area
    southern_land_area = np.sum(land_southern_mask) * pixel_area

    return northern_land_area,southern_land_area

def get_shelves_area(data, transform, plot=False):
    """
    Process the input raster to calculate continental shelves area after reprojection.

    Returns:
    - tuple: (area in square meters, volume in cubic meters).
    """

    pixel_area = abs(transform[0] * transform[4])  # width * height of a pixel in meters
    shelves_mask = (data[0] >= -300) & (data[0] < 0)  # Mask the pixels where the elevation is between -300 and 0m
    if plot == True:
        plot_mask(shelves_mask, "Continental Shelves (-300m >= z > 0m)")

    # Calculate area (count of pixels * pixel area)
    shelves_area = np.sum(shelves_mask) * pixel_area

    return shelves_area

def get_high_altitudes_area(data, transform, plot=False):
    """
    Process the input raster to calculate high altitudes (>3000m) area after reprojection.

    Returns:
    - tuple: (area in square meters).
    """

    pixel_area = abs(transform[0] * transform[4])  # width * height of a pixel in meters
    high_altitude_mask = data[0] >= 1000  # Mask the pixels where the elevation is above 3000m
    if plot == True:
        plot_mask(high_altitude_mask, "High Altitude Regions (z >=1000m)")

    # Calculate area (count of pixels * pixel area)
    high_altitude_area = np.sum(high_altitude_mask) * pixel_area
    total_area = data[0].size * pixel_area


    return high_altitude_area

def oceans_area_volume(data, transform):
    """
    Calculate the area and volume of pixels below sea level (elevation < 0).

    Parameters:
    - data (numpy.ndarray): The raster data array.
    - transform (Affine): The affine transformation of the raster (from rasterio).

    Returns:
    - tuple: (area in square meters, volume in cubic meters).
    """
    pixel_area = abs(transform[0] * transform[4])  # width * height of a pixel in meters
    mask = data < 0  # Mask the pixels where the elevation is below sea level

    # Calculate area (count of pixels * pixel area)
    area = np.sum(mask) * pixel_area

    # Calculate volume (sum of absolute elevation values * pixel area)
    volume = np.sum(np.abs(data[mask])) * pixel_area

    return area, volume


layer_name = "palaeogeography"
panalesis_ages = get_available_ages_web(layer_name)
ages = []
land_areas = []
total_coastline_lengths = []
continents_numbers = []
polar_land_areas = []
temperate_land_areas = []
subtropical_land_areas = []
tropical_land_areas = []
northern_land_areas = []
southern_land_areas = []
shelves_areas = []
high_altitudes_areas = []
ocean_areas = []
ocean_volumes = []
min_area_m2 = 7.5e12
for age in panalesis_ages:
    print(f"Processing {age} Ma")
    wcs_url = construct_wcs_url(layer_name, age)
    data, metadata = load_tiff_web(wcs_url)
    transform = metadata['transform']
    ages.append(age)

    land_area, total_area = get_land_area(data, transform, plot=False)
    land_areas.append(land_area)

    coastlines = create_contours(data, transform, 0)
    total_coastline_length = get_total_length(coastlines)
    total_coastline_lengths.append(total_coastline_length)

    large_polygons = filter_large_polygons(coastlines, min_area_m2)
    continents = len(large_polygons)
    continents_numbers.append(continents)

    polar_land_area = get_polar_area(data, metadata, transform, plot=False)
    polar_land_areas.append(polar_land_area)

    temperate_land_area = get_temperate_area(data, metadata, transform, plot=False)
    temperate_land_areas.append(temperate_land_area)

    subtropical_land_area = get_subtropical_area(data, metadata, transform, plot=False)
    subtropical_land_areas.append(subtropical_land_area)

    tropical_land_area = get_tropical_area(data, metadata, transform, plot=False)
    tropical_land_areas.append(tropical_land_area)

    northern_land_area, southern_land_area = get_hemispheres_area(data, metadata, transform, plot=False)
    northern_land_areas.append(northern_land_area)
    southern_land_areas.append(southern_land_area)

    shelves_area = get_shelves_area(data, transform, plot=False)
    shelves_areas.append(shelves_area)

    high_altitude_area = get_high_altitudes_area(data, transform, plot=False)
    high_altitudes_areas.append(high_altitude_area)

    ocean_area, ocean_volume = oceans_area_volume(data[0], transform)
    ocean_areas.append(ocean_area)
    ocean_volumes.append(ocean_volume)

# Sort everything together by age, once, after the loop
combined = sorted(
    zip(ages, land_areas, total_coastline_lengths, continents_numbers, polar_land_areas, temperate_land_areas,
        subtropical_land_areas, tropical_land_areas, northern_land_areas, southern_land_areas, shelves_areas,
        high_altitudes_areas, ocean_areas, ocean_volumes), key=lambda x: x[0])
ages, land_areas, total_coastline_lengths, continents_numbers, polar_land_areas, temperate_land_areas, subtropical_land_areas, tropical_land_areas, northern_land_areas, southern_land_areas, shelves_areas, high_altitudes_areas, ocean_areas, ocean_volumes = map(
    list, zip(*combined))
df = pd.DataFrame({
    'Age': ages,
    'Land_area': land_areas,
    'Coastline_length': total_coastline_lengths,
    'Continents_Number': continents_numbers,
    'Polar_Land_Area': polar_land_areas,
    'Temperate_Land_Area': temperate_land_areas,
    'Subtropical_Land_Area': subtropical_land_areas,
    'Tropical_Land_Area': tropical_land_areas,
    'Northern_Land_Area': northern_land_areas,
    'Southern_Land_Area': southern_land_areas,
    'Continental_Shelves_Area': shelves_areas,
    'High_Altitude_Area': high_altitudes_areas,
    'Ocean_Area': ocean_areas,
    'Ocean_Volume': ocean_volumes

})
print(df)

os.makedirs('output', exist_ok=True)
df.to_csv('output/geo_indicators.csv', index=False)



def plot_timeseries_simple(ages, metric, metric_name, title, color):
    ages_array = np.array(ages)
    x_smooth = np.linspace(ages_array.min(), ages_array.max(), 300)
    pchip = PchipInterpolator(ages, metric)
    y_smooth = pchip(x_smooth)
    plt.figure(figsize=(10, 6))
    plt.scatter(ages, metric, marker='o', linestyle='-', color=color)
    plt.plot(x_smooth, y_smooth, color=color, linewidth=2)
    plt.xlabel('Age (Ma)')
    plt.ylabel(metric_name)
    plt.title(title)
    plt.tight_layout()
    plt.gca().invert_xaxis()
    os.makedirs('output', exist_ok=True)
    filename = title.replace(' ', '_') + '.png'
    plt.savefig(os.path.join('output', filename), dpi=300)
    plt.show()


def plot_timeseries_double(ages, metric1, metric1_name, metric2, metric2_name, title):
    ages_array = np.array(ages)
    x_smooth = np.linspace(ages_array.min(), ages_array.max(), 300)
    pchip1 = PchipInterpolator(ages, metric1)
    pchip2 = PchipInterpolator(ages, metric2)
    y_smooth1 = pchip1(x_smooth)
    y_smooth2 = pchip2(x_smooth)
    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax2 = ax1.twinx()
    color1 = '#6D92C7'
    color2 = '#C94A65'
    ax1.scatter(ages, metric1, marker='o', color=color1, label=metric1_name)
    ax1.plot(x_smooth, y_smooth1, color=color1, linewidth=1.5)
    ax1.set_ylabel(metric1_name, color=color1)
    ax1.tick_params(axis='y', labelcolor=color1)
    ax2.scatter(ages, metric2, marker='o', color=color2, label=metric2_name)
    ax2.plot(x_smooth, y_smooth2, color=color2, linewidth=1.5)
    ax2.set_ylabel(metric2_name, color=color2)
    ax2.tick_params(axis='y', labelcolor=color2)
    ax1.set_xlabel('Age (Ma)')
    ax1.set_title(title)
    ax1.invert_xaxis()
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='best')
    plt.tight_layout()
    os.makedirs('output', exist_ok=True)
    filename = title.replace(' ', '_') + '.png'
    plt.savefig(os.path.join('output', filename), dpi=300)
    plt.show()

plot_timeseries_simple(ages, land_areas, 'Land Area (m²)', 'Land Area vs Age','darkgreen')
plot_timeseries_simple(ages, total_coastline_lengths, "Total Coastal Length (m)", "Coastal Length vs Age",'darkgrey')
plot_timeseries_simple(ages, continents_numbers, "Number of Continents", "Number of Continents vs Age",'black')
plot_timeseries_simple(ages, polar_land_areas, 'Polar Land Area (m²)', 'Polar Land Area vs Age','darkturquoise')
plot_timeseries_simple(ages, temperate_land_areas, 'Temperate Land Area (m²)','Temperate Land Area vs Age','greenyellow')
plot_timeseries_simple(ages, subtropical_land_areas, 'Subtropical Land Area (m²)', 'Subtropical Land Area vs Age','goldenrod')
plot_timeseries_simple(ages, tropical_land_areas, 'Tropical Land Area (m²)','Tropical Land Area vs Age','lightcoral')
plot_timeseries_double(ages,northern_land_areas, 'Northern Land',southern_land_areas, 'Southern Land','Northern and Southern Land Area vs Age')
plot_timeseries_simple(ages,shelves_areas, 'Shelves Area (m²)', 'Shelves Area vs Age','cornflowerblue')
plot_timeseries_simple(ages, high_altitudes_areas, 'High Altitude (z > 1000m)','High Altitude Regions','salmon')
plot_timeseries_simple(ages,ocean_areas, "Oceanic Area (m²)", "Oceanic Area vs Age",'darkblue')
plot_timeseries_simple(ages,ocean_volumes, "Oceanic Volume (m³)", "Oceanic Volume vs Age",'dodgerblue')