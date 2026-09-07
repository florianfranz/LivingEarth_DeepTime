import rasterio
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
from matplotlib.gridspec import GridSpec
import json
import os
from pathlib import Path


CONFIG_PATH = Path(__file__).resolve().parent.parent / "config.json"
with open(CONFIG_PATH, "r") as f:
    _config = json.load(f)
le_outputs = _config["LE_outputs"]

# --- Configuration ---
# Change this to any 6 folder names in your series
AGES = ["2000", "2056", "2120", "2250", "2444", "2545"]
LEVEL4_BAND = 4  # The band containing the classification codes

# --- Legend & Style (Exactly as defined in your previous script) ---
# Format: Code: (Label, Hex Color)
LEVEL4_STYLE = {
    20:  ("Nat. Terrestrial Veg.: Woody",          "#009100"),
    21:  ("Nat. Terrestrial Veg.: Herbaceous",     "#79de13"),
    56:  ("Nat Aquatic Veg.: Woody",               "#67897b"),
    94:  ("Nat. Bare Surface",                      "#daa520"),
    98:  ("Water: Wet Soil",                       "#8a6a96"),
    99:  ("Water: Liquid",                         "#30b2ef"),
    105: ("Water: Snow",                           "#efffff"),
    106: ("Water: Sea-ice",                         "#a6bddb"),
    255: ("No Data",                               "#cccccc"),  # placeholder - not in your .qml
}

# Precompute code -> index mapping, colors, labels (shared across all panels)
sorted_codes = sorted(LEVEL4_STYLE.keys())
code_to_index = {code: idx for idx, code in enumerate(sorted_codes)}
colors = [LEVEL4_STYLE[code][1] for code in sorted_codes]
labels = [LEVEL4_STYLE[code][0] for code in sorted_codes]
cmap = ListedColormap(colors)


def load_plot_data(age_str):
    """Read a single age's raster and re-index it onto the shared class codes."""
    file_path = os.path.join(le_outputs, age_str, "level4_out_LE_DT.tif")

    if not os.path.exists(file_path):
        print(f"Error: File not found at {file_path}")
        return None

    print(f"Processing age: {age_str} -> {file_path}")

    with rasterio.open(file_path) as src:
        data = src.read(LEVEL4_BAND)

    plot_data = np.full(data.shape, -1, dtype=float)  # -1 for nodata/unmapped
    for code, idx in code_to_index.items():
        plot_data[data == code] = idx

    return plot_data


def plot_grid(ages):
    if len(ages) != 6:
        raise ValueError(f"Expected exactly 6 ages, got {len(ages)}: {ages}")

    # 4 rows x 2 cols: rows 0-2 are the maps, row 3 (spanning both cols) is the legend.
    # height_ratios makes the legend row shorter than the map rows.
    fig = plt.figure(figsize=(10, 7.5))
    gs = GridSpec(
        4, 2, figure=fig, height_ratios=[1, 1, 1, 0.3],
        hspace=0.04, wspace=0.02,
        left=0.01, right=0.99, top=0.97, bottom=0.01,
    )

    for i, age_str in enumerate(ages):
        row, col = divmod(i, 2)
        ax = fig.add_subplot(gs[row, col])

        age_ma = int(age_str) - 2000
        age_label = f"{age_ma} Ma"

        plot_data = load_plot_data(age_str)
        if plot_data is None:
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_edgecolor('black')
                spine.set_linewidth(1.2)
            ax.text(0.02, 0.5, f"{age_label} (missing)", transform=ax.transAxes,
                    fontsize=9, va='center', ha='left')
            continue

        ax.imshow(plot_data, cmap=cmap, vmin=0, vmax=len(sorted_codes) - 1, interpolation='nearest')
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_edgecolor('black')
            spine.set_linewidth(1.2)

        # Age label as a text box in the top-left corner
        ax.text(
            0.02, 0.96, age_label, transform=ax.transAxes,
            fontsize=10, va='top', ha='left',
            bbox=dict(boxstyle="square,pad=0.25", facecolor="white", edgecolor="black", linewidth=0.8),
        )

    # Legend axis spans the full width of the 4th row
    legend_ax = fig.add_subplot(gs[3, :])
    legend_ax.axis('off')

    legend_handles = [Patch(facecolor=color, label=label) for color, label in zip(colors, labels)]
    legend_ax.legend(
        handles=legend_handles,
        loc="center",
        ncol=4,
        fontsize=9,
        title="Level 4 Class",
        frameon=True,
        borderaxespad=0,
    )

    out_name = "level4_map_grid.png"
    plt.savefig(out_name, dpi=300, bbox_inches='tight', pad_inches=0.05)
    print(f"Map saved to {out_name}")
    plt.show()


if __name__ == "__main__":
    plot_grid(AGES)