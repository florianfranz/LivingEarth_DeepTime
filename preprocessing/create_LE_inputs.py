import os
import shutil
import json
from pathlib import Path
import numpy as np
import processing
import rasterio
from qgis.core import QgsRasterLayer, QgsProject, QgsCoordinateReferenceSystem

# ===== CONFIG =====
# Load paths from config.json (not committed to git — see config.example.json)
CONFIG_PATH = Path(__file__).resolve().parent.parent / "config.json"
with open(CONFIG_PATH, "r") as f:
    _config = json.load(f)

climate_folder = _config["climate_folder"]
paleo_folder = _config["paleo_folder"]
output_base_folder = _config["output_base_folder"]

os.makedirs(output_base_folder, exist_ok=True)

ages = ["2000", "2006", "2011", "2015", "2020", "2033", "2040", "2048",
        "2056", "2068", "2094", "2100", "2113", "2120", "2133", "2140",
        "2154", "2165", "2180", "2200", "2210", "2220", "2230", "2240",
        "2250", "2270", "2290", "2300", "2315", "2331", "2350", "2370",
        "2383", "2393", "2408", "2420", "2444", "2463", "2475", "2489",
        "2500", "2518", "2535", "2545"]



# ==================

def load_layer(path, name):
    layer = QgsRasterLayer(path, name)
    if not layer.isValid():
        raise Exception(f"Invalid raster: {path}")
    QgsProject.instance().addMapLayer(layer, False)
    return layer


def remove_layer(layer):
    if layer:
        QgsProject.instance().removeMapLayer(layer.id())


def get_aligned_params(reference_layer):
    e = reference_layer.extent()
    crs_id = reference_layer.crs().authid()
    width = reference_layer.rasterUnitsPerPixelX()
    height = reference_layer.rasterUnitsPerPixelY()
    extent_str = f"{e.xMinimum()},{e.xMaximum()},{e.yMinimum()},{e.yMaximum()} [{crs_id}]"
    return extent_str, crs_id, width, height


def run_calc_aligned(layers, expression, extent, crs, res_x, res_y, output, nodata_val=-9999):
    """Runs raster calc forcing exact alignment. Does NOT delete output if it exists (handles overwrite internally)."""
    # We do NOT delete here. We let GDAL handle overwrite or we use a temp name if needed.
    # Actually, native:rastercalc usually overwrites fine if the file isn't open.

    processing.run("native:rastercalc", {
        'LAYERS': layers,
        'EXPRESSION': expression,
        'EXTENT': extent,
        'CELL_SIZE': f"{res_x},{res_y}",
        'CRS': QgsCoordinateReferenceSystem(crs),
        'NODATA': nodata_val,
        'OUTPUT': output
    })
    print(f"   Created: {os.path.basename(output)}")


def clean_no_data(src_path, target_dtype='uint8', fill_value=0):
    if not os.path.exists(src_path):
        return
    dst_path = src_path.replace(".tif", "_fixed.tif")

    with rasterio.open(src_path) as src:
        data = src.read(1)
        profile = src.profile
        nodata = src.nodata

    if nodata is not None:
        mask = (data == nodata) | (np.isnan(data))
        data[mask] = fill_value

    data = data.astype(target_dtype)
    profile.update(dtype=target_dtype, count=1, nodata=None)
    if 'scale' in profile: profile.pop('scale')
    if 'offset' in profile: profile.pop('offset')

    with rasterio.open(dst_path, 'w', **profile) as dst:
        dst.write(data, 1)

    if os.path.exists(src_path):
        try:
            os.remove(src_path)
        except:
            pass
    print(f"   Cleaned & Fixed: {os.path.basename(dst_path)}")
    return dst_path


# ===== MAIN =====

for age in ages:
    output_folder = os.path.join(output_base_folder, age)
    os.makedirs(output_folder, exist_ok=True)
    print(f"\n{'=' * 50}")
    print(f"Processing Age: {age}")
    print(f"{'=' * 50}")

    # ---- FILE PATHS ----
    soil_path = f"{climate_folder}/soil_moisture_{age}_align.tif"
    snow_path = f"{climate_folder}/snow_depth_{age}_align.tif"
    forest_path = f"{climate_folder}/forest_cover_{age}_align.tif"
    paleo_path = f"{paleo_folder}/palaeogeography_{age}.tif"
    fveg_path = f"{climate_folder}/fractional_vegetation_{age}_align.tif"
    ice_path = f"{climate_folder}/sea_ice_cover_{age}_align.tif"

    for p in [soil_path, snow_path, forest_path, paleo_path, fveg_path, ice_path]:
        if not os.path.exists(p):
            raise Exception(f"Missing file: {p}")

    # ---- LOAD LAYERS ----
    # NOTE: We do NOT load fveg_path here to prevent file locking!
    soil = load_layer(soil_path, f"soil_moisture_{age}")
    snow = load_layer(snow_path, f"snow_depth_{age}")
    forest = load_layer(forest_path, f"forest_cover_{age}")
    paleo = load_layer(paleo_path, f"palaeogeography_{age}")
    ice = load_layer(ice_path, f"sea_ice_cover_{age}")

    # ---- GET UNIFIED ALIGNMENT PARAMS ----
    extent, crs, res_x, res_y = get_aligned_params(paleo)
    print(f"   Target Alignment: Res={res_x:.4f}x{res_y:.4f}, CRS={crs}")

    try:
        # ========================
        # STEP 1: WATER STATE
        # ========================
        waterstt_temp = f"{output_folder}/waterstt_wat_cat_{age}.tif"
        waterstt_expr = (
            f'("sea_ice_cover_{age}@1" > 0.1) * 3 + '
            f'(("sea_ice_cover_{age}@1" <= 0.1) AND ("snow_depth_{age}@1" > 0.01) AND ("palaeogeography_{age}@1" >= 0)) * 2 + '
            f'(("sea_ice_cover_{age}@1" <= 0.1) AND ("palaeogeography_{age}@1" < 0)) * 1'
        )
        run_calc_aligned([ice, snow, paleo], waterstt_expr, extent, crs, res_x, res_y, waterstt_temp, nodata_val=-9999)
        clean_no_data(waterstt_temp, target_dtype='uint8', fill_value=0)

        # ========================
        # STEP 2: AQUATIC MASK
        # ========================
        aquatic_temp = f"{output_folder}/aquatic_wat_cat_{age}.tif"
        aquatic_expr = (
            f'(("palaeogeography_{age}@1" < 0) OR '
            f'("soil_moisture_{age}@1" > 0.4) OR '
            f'("snow_depth_{age}@1" > 0.01))'
        )
        run_calc_aligned([paleo, soil, snow], aquatic_expr, extent, crs, res_x, res_y, aquatic_temp, nodata_val=-9999)
        clean_no_data(aquatic_temp, target_dtype='uint8', fill_value=0)

        src_fixed = f"{output_folder}/aquatic_wat_cat_{age}_fixed.tif"
        dst_final = f"{output_folder}/aquatic_wat_cat_{age}.tif"
        if os.path.exists(src_fixed):
            if os.path.exists(dst_final):
                try:
                    os.remove(dst_final)
                except:
                    pass
            os.rename(src_fixed, dst_final)
            print(f"   [OK] aquatic_wat_cat_{age}.tif (Aligned)")

        # ========================
        # STEP 3: LIFEFORM (MASKED)
        # ========================
        veg_temp = f"{output_folder}/lifeform_veg_cat_{age}.tif"
        is_water_mask = (
            f'(("palaeogeography_{age}@1" < 0) OR '
            f'("sea_ice_cover_{age}@1" > 0.1) OR '
            f'("snow_depth_{age}@1" > 0.01))'
        )
        veg_expr_masked = (
            f'({is_water_mask}) * 0 + '
            f'(1 - ({is_water_mask})) * (2 - ("forest_cover_{age}@1" > 0.001))'
        )
        run_calc_aligned([paleo, ice, snow, forest], veg_expr_masked, extent, crs, res_x, res_y, veg_temp,
                         nodata_val=-9999)
        clean_no_data(veg_temp, target_dtype='uint8', fill_value=0)

        # ========================
        # STEP 4: FRACTIONAL VEGETATION (MASKED) - SIMPLIFIED
        # ========================
        fveg_final = f"{output_folder}/fractional_vegetation_{age}.tif"

        # We load fveg ONLY now, for this specific step, and remove it immediately after.
        # This ensures no long-term lock.
        fveg_layer = load_layer(fveg_path, f"fveg_{age}")

        fveg_expr_masked = (
            f'({is_water_mask}) * 0 + '
            f'(1 - ({is_water_mask})) * "fveg_{age}@1"'
        )

        # Write directly to the final file.
        # Since we didn't load it at the start of the script, it should not be locked by QGIS.
        # If it exists from a previous run, GDAL usually handles overwrite fine if not open in QGIS.
        # If error persists, we use a temp name and rename.
        fveg_temp_name = f"{output_folder}/fractional_vegetation_{age}_temp.tif"

        run_calc_aligned([paleo, ice, snow, fveg_layer], fveg_expr_masked, extent, crs, res_x, res_y, fveg_temp_name,
                         nodata_val=-9999)

        remove_layer(fveg_layer)  # Unlock immediately

        # Swap temp to final
        if os.path.exists(fveg_final):
            try:
                os.remove(fveg_final)
            except:
                os.rename(fveg_final, fveg_final + ".junk")
                os.remove(fveg_final + ".junk")

        os.rename(fveg_temp_name, fveg_final)

        # Clean NoData — read fully, close the handle, THEN write
        with rasterio.open(fveg_final) as src:
            data = src.read(1)
            profile = src.profile
            nodata = src.nodata

        if nodata is not None:
            data[np.isnan(data) | (data == nodata)] = 0.0
        profile.update(nodata=None)

        fveg_clean_temp = fveg_final.replace(".tif", "_clean_temp.tif")
        with rasterio.open(fveg_clean_temp, 'w', **profile) as dst:
            dst.write(data, 1)

        os.remove(fveg_final)
        os.rename(fveg_clean_temp, fveg_final)

        # ========================
        # STEP 5: BLANK RASTER
        # ========================
        blank_output = f"{output_base_folder}/blank_raster_all_zeroes.tif"
        if not os.path.exists(blank_output):
            processing.run("gdal:translate", {
                'INPUT': paleo_path,
                'OPTIONS': '-of GTiff -b 1 -ot Byte -a_nodata 0',
                'COPY_METADATA': False,
                'OUTPUT': blank_output
            })
            processing.run("native:rastercalc", {
                'LAYERS': [blank_output],
                'EXPRESSION': '0',
                'EXTENT': extent,
                'CELL_SIZE': f"{res_x},{res_y}",
                'CRS': QgsCoordinateReferenceSystem(crs),
                'OUTPUT': blank_output
            })
            print(f"   [OK] blank_raster_all_zeroes.tif")


    finally:
        for layer in [soil, snow, forest, paleo, ice]:
            remove_layer(layer)

print("\n✅ All inputs generated successfully.")