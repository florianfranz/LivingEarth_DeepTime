import rasterio
import numpy as np
import pandas as pd
import json
from pathlib import Path

# --- CONFIGURATION ---
age = "2000"
CONFIG_PATH = Path(__file__).resolve().parent.parent / "config.json"
with open(CONFIG_PATH, "r") as f:
    _config = json.load(f)


base_dir = _config["LE_inputs"]

input_veg_path = f"{base_dir}\\input\\{age}\\fractional_vegetation_{age}.tif"
output_l4_path = f"{base_dir}\\output\\{age}\\level4_out_LE_DT.tif"


def pixel_analysis(veg_path, l4_path):
    print(f"🔍 Analyzing Pixel Logic for Age {age}...")

    with rasterio.open(veg_path) as src_veg:
        veg_data = src_veg.read(1)
        veg_nodata = src_veg.nodata
        veg_profile = src_veg.profile

    with rasterio.open(l4_path) as src_l4:
        l4_data = src_l4.read(4)  # Band 4 is Level 4
        l4_nodata = src_l4.nodata

        # Also read Band 1 (Level 3) and Band 2 (Lifeform) for context
        l3_data = src_l4.read(1)
        l2_data = src_l4.read(2)

    # Ensure shapes match (crop to smallest if necessary)
    if veg_data.shape != l4_data.shape:
        min_h = min(veg_data.shape[0], l4_data.shape[0])
        min_w = min(veg_data.shape[1], l4_data.shape[1])
        veg_data = veg_data[:min_h, :min_w]
        l4_data = l4_data[:min_h, :min_w]
        l3_data = l3_data[:min_h, :min_w]
        l2_data = l2_data[:min_h, :min_w]
        print(f"⚠️  Shapes mismatched. Cropped to {min_h}x{min_w}")

    # --- SIMULATE THE THRESHOLD ---
    # This is what expstr: "(band > 0.1)" SHOULD produce
    # Result: 1 (Vegetation), 0 (No Vegetation), NaN (NoData)
    veg_mask = np.zeros_like(veg_data, dtype=int) - 1  # Default -1 (NoData)

    valid_veg_pixels = veg_data != veg_nodata if veg_nodata is not None else ~np.isnan(veg_data)

    # Apply threshold only to valid pixels
    veg_mask[valid_veg_pixels] = (veg_data[valid_veg_pixels] > 0.1).astype(int)

    # --- ANALYSIS ---
    # We want to check: When VegMask=1, do we get Valid Lifeform?
    # And: When VegMask=0, do we get Water/Bare classes?

    print("\n📊 CROSS-TABULATION: Input Vegetation vs. Output Level 3")
    print("(Level 3: 112/124=Veg, 216=Bare, 220=Water)")

    # Define masks
    mask_veg_present = (veg_mask == 1)
    mask_veg_absent = (veg_mask == 0)
    mask_veg_nodata = (veg_mask == -1)

    # Check Level 3 distribution for each mask
    def get_l3_dist(mask, name):
        l3_vals = l3_data[mask]
        if l3_vals.size == 0:
            return f"{name}: 0 pixels"
        unique, counts = np.unique(l3_vals, return_counts=True)
        top_3 = sorted(zip(unique, counts), key=lambda x: x[1], reverse=True)[:3]
        return f"{name} ({l3_vals.size:,} px): Top L3 classes = {top_3}"

    print(get_l3_dist(mask_veg_present, "🌱 Input Veg > 0.1"))
    print(get_l3_dist(mask_veg_absent, "🏜️  Input Veg <= 0.1"))
    print(get_l3_dist(mask_veg_nodata, "❓ Input Veg NoData"))

    # --- THE SMOKING GUN CHECK ---
    # Check Lifeform (Band 2) where Vegetation is Present
    # If Lifeform is 0 (NoData) where Veg is Present, the dependency chain is broken.
    print("\n🔍 DEPENDENCY CHECK: Lifeform vs. Vegetation")
    lifeform_when_veg = l2_data[mask_veg_present]
    if lifeform_when_veg.size > 0:
        lf_unique, lf_counts = np.unique(lifeform_when_veg, return_counts=True)
        lf_dist = dict(zip(lf_unique, lf_counts))
        print(f"   Where Input Veg > 0.1, Lifeform distribution: {lf_dist}")

        if 0 in lf_dist and lf_dist[0] > (lifeform_when_veg.size * 0.5):
            print("   🚨 CRITICAL: >50% of vegetated pixels have Lifeform=0 (NoData)!")
            print("      This confirms the classification rules are rejecting your input.")
        else:
            print("   ✅ Lifeform data looks valid where vegetation exists.")
    else:
        print("   🚨 No pixels found with Vegetation > 0.1! (Check your threshold or input range)")

    # --- CHECK ICE/WATER LOGIC ---
    print("\n🧊 ICE/WATER LOGIC CHECK")
    # Where Input Water State (Band 3) is 3 (Ice)
    # We need to read Band 3 from output
    waterstt_data = src_l4.read(3)
    waterstt_data = waterstt_data[:min_h, :min_w]  # Crop if needed

    mask_ice = (waterstt_data == 3)
    if np.sum(mask_ice) > 0:
        l3_ice = l3_data[mask_ice]
        l4_ice = l4_data[mask_ice]
        print(f"   Pixels with WaterState=3 (Ice): {np.sum(mask_ice):,}")
        print(f"   Their Level 3 classes: {dict(zip(*np.unique(l3_ice, return_counts=True)))}")
        print(f"   Their Level 4 classes: {dict(zip(*np.unique(l4_ice, return_counts=True)))}")
    else:
        print("   No Ice pixels found in Output Band 3.")


if __name__ == "__main__":
    pixel_analysis(input_veg_path, output_l4_path)