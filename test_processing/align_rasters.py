"""
Align EVERY .tif in the climate "final" folder onto the FIXED reference grid
of palaeogeography_2000.tif (same CRS, transform/origin, resolution,
width/height) -- regardless of which age the climate file itself represents.

Since this is a global lon/lat grid, longitude is periodic: the source is
wrap-padded (columns copied from its opposite edge) before reprojecting, so
bilinear resampling can interpolate real values right across the antimeridian
seam instead of leaving NoData gaps there.

Output: <original_name>_align.tif written into the SAME climate folder,
alongside the originals (originals are never modified).
"""

import os
import rasterio
from rasterio.warp import reproject, Resampling
from affine import Affine
import numpy as np
import json
from pathlib import Path

# ===== CONFIG =====

CONFIG_PATH = Path(__file__).resolve().parent.parent / "config.json"
with open(CONFIG_PATH, "r") as f:
    _config = json.load(f)


climate_folder = _config["climate_output_folder"]
ref_path       = _config["paleo_folder"]

RESAMPLING = Resampling.bilinear  # continuous data -> bilinear (use .nearest for categorical layers)
WRAP_PAD_COLS = 10                # columns wrapped from each opposite edge


def align_one(src_path, ref_path, out_path):
    with rasterio.open(ref_path) as ref:
        dst_crs = ref.crs
        dst_transform = ref.transform
        dst_width = ref.width
        dst_height = ref.height

    with rasterio.open(src_path) as src:
        src_data = src.read(1)
        src_crs = src.crs
        src_transform = src.transform
        src_nodata = src.nodata
        profile = src.profile.copy()

    # ---- wrap-pad source in X so the antimeridian seam has real data ----
    pad = WRAP_PAD_COLS
    src_data_padded = np.concatenate(
        [src_data[:, -pad:], src_data, src_data[:, :pad]], axis=1
    )
    src_transform_padded = src_transform * Affine.translation(-pad, 0)

    # ---- destination ----
    dst_nodata = src_nodata if src_nodata is not None else -9999.0
    dst_data = np.full((dst_height, dst_width), dst_nodata, dtype=src_data.dtype)

    reproject(
        source=src_data_padded,
        destination=dst_data,
        src_transform=src_transform_padded,
        src_crs=src_crs,
        src_nodata=src_nodata,
        dst_transform=dst_transform,
        dst_crs=dst_crs,
        dst_nodata=dst_nodata,
        resampling=RESAMPLING,
    )

    profile.update({
        "crs": dst_crs,
        "transform": dst_transform,
        "width": dst_width,
        "height": dst_height,
        "nodata": dst_nodata,
    })

    with rasterio.open(out_path, "w", **profile) as dst:
        dst.write(dst_data, 1)

    remaining_mask = (dst_data == dst_nodata) if not np.isnan(dst_nodata) else np.isnan(dst_data)
    n_remaining = int(remaining_mask.sum())
    return n_remaining, dst_data.size


def main():
    if not os.path.exists(ref_path):
        raise FileNotFoundError(f"Reference raster not found: {ref_path}")

    all_tifs = [
        f for f in os.listdir(climate_folder)
        if f.lower().endswith(".tif") and not f.lower().endswith("_align.tif")
    ]
    all_tifs.sort()

    print(f"Reference grid: {ref_path}")
    print(f"Found {len(all_tifs)} .tif file(s) in {climate_folder}\n")

    results = []
    skipped = []

    for fname in all_tifs:
        src_path = os.path.join(climate_folder, fname)
        out_name = fname[:-4] + "_align.tif"
        out_path = os.path.join(climate_folder, out_name)

        try:
            n_remaining, total = align_one(src_path, ref_path, out_path)
            pct = 100 * n_remaining / total
            status = "OK" if n_remaining == 0 else f"WARNING: {n_remaining} NoData px remain ({pct:.4f}%)"
            print(f"  [{status}] {fname} -> {out_name}")
            results.append((fname, out_name, n_remaining))
        except Exception as e:
            skipped.append((fname, f"error: {e}"))
            print(f"  [FAILED] {fname}: {e}")

    print(f"\n{'=' * 60}")
    print(f"Done. Aligned: {len(results)}   Skipped/Failed: {len(skipped)}")
    if skipped:
        print("\nSkipped/failed files:")
        for fname, reason in skipped:
            print(f"  - {fname}: {reason}")
    remaining_gaps = [r for r in results if r[2] > 0]
    if remaining_gaps:
        print("\nFiles with remaining NoData after wrap-fill (investigate WRAP_PAD_COLS or source data):")
        for fname, out_name, n in remaining_gaps:
            print(f"  - {out_name}: {n} pixels")


if __name__ == "__main__":
    main()