"""
Sankey flow diagram of Level-4 land-cover code transitions between two ages,
built from the change-matrix CSV produced by build_change_matrix() in
plate_rotation.py.

Each flow's width is proportional to the pixel count moving from a source
code (left) to a destination code (right); flows are colored by their
source code using LEVEL4_STYLE, and nodes are colored the same way.

Usage:
    python plot_change_sankey.py "change_matrix_6Ma_to_0Ma_band4.csv"
    python plot_change_sankey.py "path\\to\\matrix.csv" --age-from 6 --age-to 0
    python plot_change_sankey.py "path\\to\\matrix.csv" --min-count 500

Requires: pandas, plotly
    pip install plotly
"""

import argparse
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go

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

# Labels/colors for the two non-land-cover buckets build_change_matrix adds.
EXTRA_STYLE = {
    "off_grid":       ("Rotated off grid",  "#777777"),
    "unmapped_plate": ("No plate match",    "#333333"),
}

DEFAULT_COLOR = "#bbbbbb"


def label_and_color(code) -> tuple:
    """Return (display label, hex color) for a source/dest code - handles
    both numeric LEVEL4 codes and the off_grid/unmapped_plate labels."""
    try:
        code_int = int(code)
    except (TypeError, ValueError):
        code_int = None

    if code_int is not None and code_int in LEVEL4_STYLE:
        name, color = LEVEL4_STYLE[code_int]
        return f"{code_int} {name}", color

    key = str(code)
    if key in EXTRA_STYLE:
        return EXTRA_STYLE[key]

    return str(code), DEFAULT_COLOR


def hex_to_rgba(hex_color: str, alpha: float = 0.45) -> str:
    hex_color = hex_color.lstrip("#")
    r, g, b = (int(hex_color[i:i + 2], 16) for i in (0, 2, 4))
    return f"rgba({r},{g},{b},{alpha})"


def build_sankey(csv_path: str, age_from, age_to, min_count: int = 0,
                  out_html: str = None):
    matrix = pd.read_csv(csv_path, index_col=0)

    source_codes = list(matrix.index)
    dest_codes = list(matrix.columns)

    source_meta = {code: label_and_color(code) for code in source_codes}
    dest_meta = {code: label_and_color(code) for code in dest_codes}

    node_labels = (
        [f"{source_meta[c][0]}  ({age_from} Ma)" for c in source_codes]
        + [f"{dest_meta[c][0]}  ({age_to} Ma)" for c in dest_codes]
    )
    node_colors = (
        [source_meta[c][1] for c in source_codes]
        + [dest_meta[c][1] for c in dest_codes]
    )

    src_idx = {code: i for i, code in enumerate(source_codes)}
    dst_idx = {code: len(source_codes) + j for j, code in enumerate(dest_codes)}

    link_source, link_target, link_value, link_color = [], [], [], []
    for src in source_codes:
        for dst in dest_codes:
            value = matrix.loc[src, dst]
            if pd.isna(value) or value <= min_count:
                continue
            link_source.append(src_idx[src])
            link_target.append(dst_idx[dst])
            link_value.append(int(value))
            link_color.append(hex_to_rgba(source_meta[src][1]))

    if not link_value:
        raise ValueError(
            "No flows to plot - either the matrix is empty or --min-count "
            "filtered everything out."
        )

    fig = go.Figure(data=[go.Sankey(
        arrangement="snap",
        node=dict(
            pad=18,
            thickness=18,
            line=dict(color="black", width=0.4),
            label=node_labels,
            color=node_colors,
        ),
        link=dict(
            source=link_source,
            target=link_target,
            value=link_value,
            color=link_color,
        ),
    )])

    fig.update_layout(
        title_text=f"Level-4 land-cover transitions: {age_from} Ma \u2192 {age_to} Ma",
        font_size=13,
        width=1150,
        height=700,
    )

    if out_html is None:
        out_html = f"{Path(csv_path).stem}_sankey.html"

    fig.write_html(out_html)
    print(f"Saved interactive Sankey diagram to {Path(out_html).resolve()}")
    return fig


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Build a Sankey diagram from a change-matrix CSV.")
    parser.add_argument(
        "csv_path", nargs="?", default="change_matrix_6Ma_to_0Ma_band4.csv",
        help="Path to the change-matrix CSV (default: %(default)s)")
    parser.add_argument("--age-from", default=6, help="Source age label (default: 6)")
    parser.add_argument("--age-to", default=0, help="Destination age label (default: 0)")
    parser.add_argument(
        "--min-count", type=int, default=0,
        help="Hide flows with this many pixels or fewer (default: 0 = show all)")
    args = parser.parse_args()

    build_sankey(args.csv_path, args.age_from, args.age_to, min_count=args.min_count)