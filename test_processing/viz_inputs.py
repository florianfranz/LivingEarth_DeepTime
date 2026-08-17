import os
import numpy as np
import rasterio
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.patches import Patch
import json
from pathlib import Path

# ===== CONFIG =====
age = "2120"

CONFIG_PATH = Path(__file__).resolve().parent.parent / "config.json"
with open(CONFIG_PATH, "r") as f:
    _config = json.load(f)


base_dir = _config["LE_inputs"]
folder = os.path.join(base_dir,age)

# Each entry: (title, filename, kind, categories)
# kind = "categorical" -> discrete legend built from categories
# kind = "continuous"  -> colorbar with a label
layers = [
    (
        "Water state",
        f"waterstt_wat_cat_{age}_fixed.tif",
        "categorical",
        {
            0: ("Land (dry)", "#c2a679"),
            1: ("Open water (ice-free)", "#3182bd"),
            2: ("Snow-covered land", "#e5f5f9"),
            3: ("Sea ice", "#a6bddb"),
        },
    ),
    (
        "Aquatic mask",
        f"aquatic_wat_cat_{age}.tif",
        "categorical",
        {
            0: ("Not aquatic", "#c2a679"),
            1: ("Aquatic (ocean / wet soil / snow)", "#3182bd"),
        },
    ),
    (
        "Lifeform / vegetation category",
        f"lifeform_veg_cat_{age}_fixed.tif",
        "categorical",
        {
            0: ("Water / ice / snow (masked)", "#3182bd"),
            1: ("Forested land", "#31a354"),
            2: ("Non-forested land", "#e5c185"),
        },
    ),
    (
        "Fractional vegetation",
        f"fractional_vegetation_{age}.tif",
        "continuous",
        "Fractional vegetation cover (0-1)",
    ),
]

fig, axes = plt.subplots(2, 2, figsize=(13, 11))
axes = axes.flatten()

for ax, (title, fname, kind, spec) in zip(axes, layers):
    path = os.path.join(folder, fname)
    if not os.path.exists(path):
        ax.set_title(f"{title}\n[MISSING FILE]")
        ax.axis("off")
        continue

    with rasterio.open(path) as src:
        data = src.read(1, masked=True)

    if kind == "categorical":
        cat_values = sorted(spec.keys())
        colors = [spec[v][1] for v in cat_values]
        labels = [spec[v][0] for v in cat_values]

        cmap = ListedColormap(colors)
        bounds = cat_values + [cat_values[-1] + 1]
        norm = BoundaryNorm(bounds, cmap.N)

        ax.imshow(data, cmap=cmap, norm=norm)
        handles = [Patch(facecolor=c, edgecolor="black", label=l) for c, l in zip(colors, labels)]
        ax.legend(
            handles=handles,
            loc="upper center",
            bbox_to_anchor=(0.5, -0.02),
            ncol=1,
            fontsize=8,
            frameon=False,
        )
    else:
        im = ax.imshow(data, cmap="YlGn", vmin=0, vmax=1)
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label(spec, fontsize=8)

    ax.set_title(title)
    ax.axis("off")

plt.tight_layout(rect=[0, 0, 1, 0.96])

output_path = os.path.join(folder, f"outputs_plot_{age}.png")
plt.savefig(output_path, dpi=150, bbox_inches="tight")
plt.show()

print(f"Saved plot to {output_path}")