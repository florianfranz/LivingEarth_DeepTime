import rasterio
import numpy as np
import os
import json
from pathlib import Path

# --- CONFIGURATION ---
# Enter the age/folder name you want to inspect
age = 2000


CONFIG_PATH = Path(__file__).resolve().parent.parent / "config.json"
with open(CONFIG_PATH, "r") as f:
    _config = json.load(f)


base_dir = _config["LE_inputs"]


# Define file patterns (Adjust if your filenames differ slightly)
input_files = {
    "fractional_vegetation": f"{base_dir}\\{age}\\fractional_vegetation_{age}.tif",
    "aquatic_wat_cat": f"{base_dir}\\{age}\\aquatic_wat_cat_{age}.tif",
    "blank_raster_all_zeroes": f"{base_dir}\\blank_raster_all_zeroes.tif",
    "lifeform_veg_cat": f"{base_dir}\\{age}\\lifeform_veg_cat_{age}_fixed.tif",
    "waterstt_wat_cat": f"{base_dir}\\{age}\\waterstt_wat_cat_{age}_fixed.tif",
}


def analyze_layer(name, path):
    if not os.path.exists(path):
        print(f" FILE NOT FOUND: {name}")
        print(f"   Expected: {path}")
        return

    try:
        with rasterio.open(path) as src:
            band = src.read(1)
            nodata = src.nodata

            # Mask NoData safely
            if nodata is not None:
                valid_data = band[band != nodata]
            else:
                valid_data = band[~np.isnan(band)] if np.issubdtype(band.dtype, np.floating) else band

            if valid_data.size == 0:
                print(f"  {name}: Empty or all NoData.")
                return

            # Basic Stats
            dtype = valid_data.dtype
            min_val, max_val = valid_data.min(), valid_data.max()
            mean_val = valid_data.mean()
            unique_count = len(np.unique(valid_data))

            print(f"\n{'=' * 60}")
            print(f" FILE: {name}")
            print(f" Path: {path}")
            print(f"Type: {dtype} | Shape: {src.shape}")
            print(f" Range: [{min_val:.6f}, {max_val:.6f}] | Mean: {mean_val:.6f}")
            print(f" Unique Values: {unique_count:,}")

            # DECISION: Continuous vs Categorical
            # If many unique values OR float type with wide range -> Treat as Continuous
            is_continuous = (np.issubdtype(dtype, np.floating) and unique_count > 50) or (unique_count > 1000)

            if is_continuous:
                print("\n⚠️  MODE: CONTINUOUS DATA (Fractions/Probabilities)")
                print("   (Skipping unique value list to avoid spam. Showing histogram instead.)")

                # Generate a 10-bin histogram summary
                counts, bin_edges = np.histogram(valid_data, bins=10)
                total = valid_data.size

                print(f"\n   {'Value Range':<25} | {'Count':>10} | {'Percent':>8}")
                print("   " + "-" * 48)
                for i in range(len(counts)):
                    if counts[i] > 0:  # Only show bins with data
                        low, high = bin_edges[i], bin_edges[i + 1]
                        pct = (counts[i] / total) * 100
                        print(f"   {low:8.4f} to {high:8.4f} | {counts[i]:>10,} | {pct:7.2f}%")

                print(" NOTE: LCCS rules typically expect INTEGER classes (0, 1, 2).")
                print("      Feeding continuous floats (e.g., 0.4026) may cause unexpected Level 4 codes.")

            else:
                print(" MODE: CATEGORICAL DATA (Classes)")
                vals, counts = np.unique(valid_data, return_counts=True)

                # Sort by frequency (descending)
                sorted_idx = np.argsort(counts)[::-1]
                total = valid_data.size

                print(f"   {'Value':<12} | {'Count':>10} | {'Percent':>8}")
                print("   " + "-" * 35)

                # Show top 15 classes
                limit = min(15, len(vals))
                for i in range(limit):
                    idx = sorted_idx[i]
                    v, c = vals[idx], counts[idx]
                    pct = (c / total) * 100
                    # Format floats nicely if they are actually integers (e.g. 1.0 -> 1)
                    if isinstance(v, float) and v.is_integer():
                        v_str = str(int(v))
                    else:
                        v_str = f"{v:.4f}"

                    print(f"   {v_str:<12} | {c:>10,} | {pct:7.2f}%")

                if len(vals) > 15:
                    print(f"   ... ({len(vals) - 15} other minor classes omitted)")

    except Exception as e:
        print(f"ERROR reading {name}: {e}")


if __name__ == "__main__":
    print(f" Starting Read-Only Inspection for Age: {age}...")
    print("   (No files will be modified)")

    for name, path in input_files.items():
        analyze_layer(name, path)

    print(" Inspection Complete.")