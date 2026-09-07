"""
Standalone analysis script — no imports from the preprocessing script.

Reads the outputs already on disk from the preprocessing run:
  - the original L4 land-cover rasters (for source_code)
  - the 6-band pixel_history_{age}Ma_to_{age}Ma.tif rasters (for
    delta_lat_signed, delta_lat_poleward, distance_km, delta_alt,
    status_code, dest_code)

Does nothing geospatial — no rotation, no reprojection, no plate
rasterization. Just reads arrays, compares each land-cover transition
against a same-step stable baseline, prints the resulting tables, and
saves them as CSV.

Edit the CONFIG block below to match your paths/conventions if they
differ from the preprocessing script.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from scipy.stats import mannwhitneyu

# ============================================================================
# CONFIG — copied/inlined so this script has no dependency on the
# preprocessing script. Keep these in sync manually if you change either.
# ============================================================================
CONFIG_PATH = Path(__file__).resolve().parent.parent / "config.json"
with open(CONFIG_PATH, "r") as f:
    _config = json.load(f)

RASTER_BASE_DIR = _config["LE_outputs"]
RASTER_FILENAME = "level4_out_LE_DT.tif"
RASTER_AGE_OFFSET = 2000
BAND = 4

PIXEL_HISTORY_DIR = r"C:\Users\franzisf\PycharmProjects\LivingEarth_DeepTime\rotations"

STATUS_CODES = {"no_data": 0, "unmapped": 1, "off_grid": 2, "subducted": 3, "survivor": 4, "normal": 5}
RASTER_NODATA = -9999.0

MIN_GROUP_SIZE = 5
CLIMATE_STATUS_CODES = {STATUS_CODES["normal"], STATUS_CODES["survivor"]}
TECTONIC_STATUS_CODES = {STATUS_CODES["subducted"], STATUS_CODES["off_grid"],
                          STATUS_CODES["no_data"], STATUS_CODES["unmapped"]}
DELTA_VARS = ["delta_alt", "delta_lat_signed", "delta_lat_poleward", "distance_km"]

YEARS_LIST = [
    "2000", "2006", "2011", "2015", "2020", "2033", "2040", "2048",
    "2056", "2068", "2094", "2100", "2113", "2120", "2133", "2140",
    "2154", "2165", "2180", "2200", "2210", "2220", "2230", "2240",
    "2250", "2270", "2290", "2300", "2315", "2331", "2350", "2370",
    "2383", "2393", "2408", "2420", "2444", "2463", "2475", "2489",
    "2500", "2518", "2535", "2545"
]


# ============================================================================
# Path helpers
# ============================================================================

def get_raster_path(age):
    folder = str(RASTER_AGE_OFFSET + int(round(age)))
    return str(Path(RASTER_BASE_DIR) / folder / RASTER_FILENAME)


def get_pixel_history_path(age_from, age_to):
    return Path(PIXEL_HISTORY_DIR) / f"pixel_history_{int(age_from)}Ma_to_{int(age_to)}Ma.tif"


# ============================================================================
# Loading (pure array reads — no recomputation)
# ============================================================================

def load_transition_arrays(age_from, age_to, band=BAND):
    with rasterio.open(get_raster_path(age_from)) as src:
        source_code = src.read(band).ravel().astype('float64')

    ph_path = get_pixel_history_path(age_from, age_to)
    with rasterio.open(ph_path) as ph:
        if ph.count < 6:
            raise ValueError(
                f"{ph_path} only has {ph.count} bands — re-run the preprocessing "
                f"script's build_pixel_history with the 6-band version (needs dest_code)."
            )
        delta_lat_signed = ph.read(1).ravel()
        delta_lat_poleward = ph.read(2).ravel()
        distance_km = ph.read(3).ravel()
        delta_alt = ph.read(4).ravel()
        status_code = ph.read(5).ravel().astype(int)
        dest_code = ph.read(6).ravel()

    for arr in (delta_lat_signed, delta_lat_poleward, distance_km, delta_alt, dest_code):
        arr[arr == RASTER_NODATA] = np.nan

    return pd.DataFrame({
        'source_code': source_code,
        'dest_code': dest_code,
        'status_code': status_code,
        'delta_lat_signed': delta_lat_signed,
        'delta_lat_poleward': delta_lat_poleward,
        'distance_km': distance_km,
        'delta_alt': delta_alt,
    })


# ============================================================================
# Stats
# ============================================================================

def cliffs_delta(x, y):
    """Effect size in [-1, 1]: how much x tends to exceed y (0 = no difference)."""
    x, y = np.asarray(x), np.asarray(y)
    nx, ny = len(x), len(y)
    if nx == 0 or ny == 0:
        return np.nan
    ranks = pd.Series(np.concatenate([x, y])).rank().values
    u = ranks[:nx].sum() - nx * (nx + 1) / 2.0
    return (2 * u) / (nx * ny) - 1


def compare_to_baseline(climate_df):
    """One row per (source_code, dest_code) transition pair, compared
    against the same-step (source_code -> source_code) stable baseline."""
    results = []
    for source_code, src_grp in climate_df.groupby('source_code'):
        baseline = src_grp[src_grp['dest_code'] == source_code]
        if len(baseline) < MIN_GROUP_SIZE:
            continue
        for dest_code, transition_grp in src_grp.groupby('dest_code'):
            if dest_code == source_code or len(transition_grp) < MIN_GROUP_SIZE:
                continue
            row = {
                'source_code': source_code, 'dest_code': dest_code,
                'n_pixels': len(transition_grp), 'n_baseline': len(baseline),
            }
            for var in DELTA_VARS:
                a = transition_grp[var].dropna().values
                b = baseline[var].dropna().values
                if len(a) < MIN_GROUP_SIZE or len(b) < MIN_GROUP_SIZE:
                    row[f'{var}_median_transition'] = np.nan
                    row[f'{var}_median_baseline'] = np.nan
                    row[f'{var}_cliffs_delta'] = np.nan
                    row[f'{var}_pvalue'] = np.nan
                    continue
                row[f'{var}_median_transition'] = np.median(a)
                row[f'{var}_median_baseline'] = np.median(b)
                row[f'{var}_cliffs_delta'] = cliffs_delta(a, b)
                try:
                    _, p = mannwhitneyu(a, b, alternative='two-sided')
                except ValueError:
                    p = np.nan
                row[f'{var}_pvalue'] = p
            results.append(row)
    return pd.DataFrame(results)


def pool_across_steps(all_steps_df):
    """Pixel-count-weighted pooling across steps."""
    def wavg(g, col):
        w = g['n_pixels']
        vals = g[col]
        mask = vals.notna() & w.notna()
        return np.average(vals[mask], weights=w[mask]) if mask.sum() > 0 else np.nan

    rows = []
    for (source_code, dest_code), g in all_steps_df.groupby(['source_code', 'dest_code']):
        row = {
            'source_code': source_code, 'dest_code': dest_code,
            'n_steps': len(g), 'total_n_pixels': g['n_pixels'].sum(),
        }
        for var in DELTA_VARS:
            row[f'{var}_pooled_median_transition'] = wavg(g, f'{var}_median_transition')
            row[f'{var}_pooled_cliffs_delta'] = wavg(g, f'{var}_cliffs_delta')
        rows.append(row)
    return pd.DataFrame(rows).sort_values('total_n_pixels', ascending=False)


# ============================================================================
# Main — compute, print, save
# ============================================================================

def main():
    ages_int = [int(y) for y in YEARS_LIST]
    all_steps, tectonic_all = [], []

    for i in range(len(ages_int) - 1):
        year_to, year_from = ages_int[i], ages_int[i + 1]
        print(year_from, year_to)
        age_to = float(year_to - RASTER_AGE_OFFSET)
        age_from = float(year_from - RASTER_AGE_OFFSET)

        try:
            pixel_df = load_transition_arrays(age_from, age_to)
        except Exception as e:
            print(f"Skipping {age_from} -> {age_to}: {e}")
            continue

        climate_df = pixel_df[pixel_df['status_code'].isin(CLIMATE_STATUS_CODES)]
        step_result = compare_to_baseline(climate_df)
        step_result['age_from'] = age_from
        step_result['age_to'] = age_to
        all_steps.append(step_result)

        tect = pixel_df[pixel_df['status_code'].isin(TECTONIC_STATUS_CODES)] \
            .groupby(['source_code', 'status_code']).size().rename('n_pixels').reset_index()
        tect['age_from'] = age_from
        tect['age_to'] = age_to
        tectonic_all.append(tect)

    all_steps_df = pd.concat(all_steps, ignore_index=True)
    pooled_df = pool_across_steps(all_steps_df)
    tectonic_df = pd.concat(tectonic_all, ignore_index=True)

    print("\n=== transition_driver_per_step ===")
    print(all_steps_df)

    print("\n=== transition_driver_pooled ===")
    print(pooled_df)

    print("\n=== tectonic_transition_tally ===")
    print(tectonic_df)

    all_steps_df.to_csv("transition_driver_per_step.csv", index=False)
    pooled_df.to_csv("transition_driver_pooled.csv", index=False)
    tectonic_df.to_csv("tectonic_transition_tally.csv", index=False)
    print("\nSaved: transition_driver_per_step.csv, transition_driver_pooled.csv, tectonic_transition_tally.csv")


if __name__ == "__main__":
    main()