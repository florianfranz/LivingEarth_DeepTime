import rasterio
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import json
import os
from pathlib import Path


CONFIG_PATH = Path(__file__).resolve().parent.parent / "config.json"
with open(CONFIG_PATH, "r") as f:
    _config = json.load(f)
le_outputs = _config["LE_outputs"]

# --- Configuration ---
AGE = "2120"  # Change this to any folder name in your series (e.g., "2056", "2100")
LEVEL4_BAND = 4  # The band containing the classification codes

# --- Legend & Style (Exactly as defined in your previous script) ---
# Format: Code: (Label, Hex Color)
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


def plot_single_map(age_str):
    # 1. Construct File Path
    file_path = os.path.join(le_outputs,age_str,"level4_out_LE_DT.tif")

    if not os.path.exists(file_path):
        print(f"Error: File not found at {file_path}")
        return

    print(f"Processing age: {age_str} -> {file_path}")

    # 2. Read Data
    with rasterio.open(file_path) as src:
        data = src.read(LEVEL4_BAND)
        # Use the transform and extent for accurate geospatial plotting if needed,
        # but for a simple array plot, we can just use the array shape.
        extent = [src.bounds.left, src.bounds.right, src.bounds.bottom, src.bounds.top]

    # 3. Prepare Colormap and Norm
    # We need a colormap that maps specific integer values to specific colors.
    # Matplotlib's ListedColormap works on an index basis (0, 1, 2...),
    # so we must map our specific codes (19, 20, 94...) to indices.

    # Get all unique codes present in the style dictionary, sorted
    sorted_codes = sorted(LEVEL4_STYLE.keys())

    # Create a mapping from Code -> Index
    code_to_index = {code: idx for idx, code in enumerate(sorted_codes)}

    # Extract colors and labels in the same order
    colors = [LEVEL4_STYLE[code][1] for code in sorted_codes]
    labels = [LEVEL4_STYLE[code][0] for code in sorted_codes]

    # Create the colormap
    cmap = ListedColormap(colors)

    # Define normalization:
    # We map the data values such that the smallest code maps to 0, next to 1, etc.
    # However, since our data contains gaps (e.g., 19, then 20, then 55),
    # we need a custom norm or we need to re-index the data array.
    # Re-indexing the array is safer for discrete classes.

    # Create an empty array for the plotted indices
    plot_data = np.full(data.shape, -1, dtype=float)  # -1 for nodata/unmapped

    for code, idx in code_to_index.items():
        plot_data[data == code] = idx

    # 4. Plotting
    fig, ax = plt.subplots(figsize=(10, 3))

    # Plot the re-indexed data
    # vmin/vmax ensure the colormap stretches exactly across our defined classes
    im = ax.imshow(plot_data, cmap=cmap, vmin=0, vmax=len(sorted_codes) - 1, interpolation='nearest')

    # Title
    ax.axis('off')  # Hide axes for a clean map look

    # 5. Create Custom Legend
    # We create patch handles manually to match the specific labels and colors
    from matplotlib.patches import Patch
    legend_handles = [Patch(facecolor=color, label=label) for color, label in zip(colors, labels)]

    # Filter out "No Data" from the main legend if desired, or keep it.
    # Keeping it here for completeness.

    ax.legend(
        handles=legend_handles,
        loc="center left",
        bbox_to_anchor=(1.0, 0.5),
        fontsize=9,
        title="Level 4 Class",
        frameon=True
    )

    plt.tight_layout()

    # Save and Show
    out_name = f"level4_map_age_{age_str}.png"
    plt.savefig(out_name, dpi=300, bbox_inches='tight')
    print(f"Map saved to {out_name}")
    plt.show()


if __name__ == "__main__":
    plot_single_map(AGE)