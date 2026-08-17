"""
Plot the evolution of Living Earth LCCS level4 class proportions
across all palaeogeographic/palaeoclimate time steps.

Reads each level4_out_LE_DT.tif, reads band 4 (the level4 code band),
tallies per-class pixel proportions, and plots a stacked-area chart
of how those proportions change through the age series.
"""

import rasterio
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json
import os
from pathlib import Path


CONFIG_PATH = Path(__file__).resolve().parent.parent / "config.json"
with open(CONFIG_PATH, "r") as f:
    _config = json.load(f)

le_outputs = _config["LE_outputs"]

ages = ["2000", "2006", "2011", "2015", "2020", "2033", "2040", "2048",
        "2056", "2068", "2094", "2100", "2113", "2120", "2133", "2140",
        "2154", "2165", "2180", "2200", "2210", "2220", "2230", "2240",
        "2250", "2270", "2290", "2300", "2315", "2331", "2350", "2370",
        "2383", "2393", "2408", "2420", "2444", "2463", "2475", "2489",
        "2500", "2518", "2535", "2545"]

# Label + colour for each level4 code, taken directly from your QGIS
# .qml paletted colorPalette entries, so the chart matches your map
# styling. Code 19 doesn't appear in your .qml (it wasn't present in
# your rendered rasters yet) - it's given a placeholder shade in the
# same green family as 20/21; change LEVEL4_STYLE[19] if you'd rather
# use something else once/if it shows up in your data.
LEVEL4_STYLE = {
    19:  ("Nat. Terrestrial Veg.: Generic",       "#4d7a1a"),  # placeholder - not in your .qml
    20:  ("Nat. Terrestrial Veg.: Woody",          "#009100"),
    21:  ("Nat. Terrestrial Veg.: Herbaceous",     "#79de13"),
    55:  ("Nat. Aquatic Veg.: Generic",           "#9acdb9"),
    56:  ("Nat Aquatic Veg.: Woody",               "#67897b"),
    94:  ("Nat. Bare Surface",                      "#daa520"),
    98:  ("Water: Sea-ice",                       "#a6bddb"),
    99:  ("Water: Liquid",                         "#30b2ef"),
    105: ("Water: Snow",                           "#efffff"),
    255: ("No Data",                               "#cccccc"),  # placeholder - not in your .qml
}

BASE_PATH = r"C:\Users\franzisf\PycharmProjects\LivingEarth_DeepTime\test_processing\output\{age}\level4_out_LE_DT.tif"
LEVEL4_BAND = 4  # confirmed from your earlier band descriptions dump

records = []
for age in ages:
    l4_raster = os.path.join(le_outputs,age,"level4_out_LE_DT.tif")
    with rasterio.open(l4_raster) as src:
        band = src.read(LEVEL4_BAND)
        vals, counts = np.unique(band, return_counts=True)
        total = counts.sum()
        geol_age = int(age) -2000
        for v, c in zip(vals, counts):
            records.append({
                "age": geol_age,
                "code": int(v),
                "count": int(c),
                "pct": c / total * 100,
            })

df = pd.DataFrame(records)

# --- Optional: convert the folder-name "age" to Ma before present ---
# If these are template time steps offset by +2000 (as suggested by the
# "2000" folder corresponding to classification_time 2000-12-31 in your
# YAML), uncomment the next line to get a geologically meaningful x-axis:
# df["age"] = df["age"] - 2000

# Pivot into age x code table of proportions, ordered chronologically
pivot = df.pivot(index="age", columns="code", values="pct").fillna(0)
pivot = pivot.sort_index()

# Build the colour list (in column order) BEFORE renaming columns to
# labels, so each area matches the colour you use in QGIS
color_list = [
    LEVEL4_STYLE.get(c, (f"code {c}", "#999999"))[1]  # unseen codes fall back to grey
    for c in pivot.columns
]
pivot = pivot.rename(columns=lambda c: LEVEL4_STYLE.get(c, (f"code {c}", None))[0])

# --- Plot ---
fig, ax = plt.subplots(figsize=(12, 4))
pivot.plot.area(ax=ax, color=color_list, linewidth=0)

ax.set_xlabel("Age (Ma)")
ax.set_ylabel("Proportion of pixels (%)")
ax.set_ylim(0, 100)
ax.set_xlim(0,545)

ax.legend(loc="center left", bbox_to_anchor=(1.0, 0.5), fontsize=8, title="Level4 class")
plt.tight_layout()
ax.invert_xaxis()

out_path = "level4_class_evolution.png"
plt.savefig(out_path, dpi=150)
print(f"Saved chart to {out_path}")
plt.show()

# Also dump the raw proportion table to CSV for your own record/QC
pivot.to_csv("level4_class_proportions_by_age.csv")
print("Saved table to level4_class_proportions_by_age.csv")