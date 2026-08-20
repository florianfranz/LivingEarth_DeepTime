"""
Sankey flow diagram of Level-4 land-cover code transitions between two ages,
built from the change-matrix CSV produced by build_change_matrix() in
plate_rotation.py.

NEW: Batch mode now also generates a summary time-series plot showing the
PERCENTAGE of changed pixels between each reconstruction step.

Usage:
    python plot_change_sankey.py                 # Batch mode (all pairs + timeseries)
    python plot_change_sankey.py "matrix.csv"    # Single mode
"""

import argparse
import sys
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

AGE_FROM = 11
AGE_TO = 6

LEVEL4_STYLE = {
    20:  ("Nat. Terrestrial Veg.: Woody",          "#009100"),
    21:  ("Nat. Terrestrial Veg.: Herbaceous",     "#79de13"),
    56:  ("Nat Aquatic Veg.: Woody",               "#67897b"),
    94:  ("Nat. Bare Surface",                      "#daa520"),
    98:  ("Water: Wet Soil",                       "#8a6a96"),
    99:  ("Water: Liquid",                         "#30b2ef"),
    105: ("Water: Snow",                           "#efffff"),
    106: ("Water: Sea-ice",                         "#a6bddb"),
    255: ("No Data",                               "#cccccc"),
}

EXTRA_STYLE = {
    "off_grid":       ("Rotated off grid",  "#777777"),
    "unmapped_plate": ("No plate match",    "#333333"),
}

DEFAULT_COLOR = "#bbbbbb"

# The list of years for batch processing
BATCH_YEARS_STR = [
    "2000", "2006", "2011", "2015", "2020", "2033", "2040", "2048",
    "2056", "2068", "2094", "2100", "2113", "2120", "2133", "2140",
    "2154", "2165", "2180", "2200", "2210", "2220", "2230", "2240",
    "2250", "2270", "2290", "2300", "2315", "2331", "2350", "2370",
    "2383", "2393", "2408", "2420", "2444", "2463", "2475", "2489",
    "2500", "2518", "2535", "2545"
]

RASTER_AGE_OFFSET = 2000

# ---------------------------------------------------------------------------
# Helper Functions
# ---------------------------------------------------------------------------

def default_csv_name(age_from, age_to) -> str:
    return f"change_matrix_{age_from}Ma_to_{age_to}Ma_band4.csv"

def label_and_color(code) -> tuple:
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

def _codes_match(src, dst) -> bool:
    try:
        return int(src) == int(dst)
    except (TypeError, ValueError):
        return str(src) == str(dst)

def build_sankey(csv_path: str, age_from, age_to, min_count: int = 0,
                  out_html: str = None, include_unchanged: bool = False,
                  return_stats: bool = False):
    """
    Added return_stats: if True, returns (fig, changed_count, total_count)
    """
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
    unchanged_pixels = 0
    changed_pixels = 0

    for src in source_codes:
        for dst in dest_codes:
            value = matrix.loc[src, dst]
            if pd.isna(value) or value <= 0:
                continue

            if _codes_match(src, dst):
                unchanged_pixels += int(value)
                if not include_unchanged:
                    continue
            else:
                changed_pixels += int(value)

            if value <= min_count:
                continue

            link_source.append(src_idx[src])
            link_target.append(dst_idx[dst])
            link_value.append(int(value))
            link_color.append(hex_to_rgba(source_meta[src][1]))

    if not link_value and not return_stats:
        raise ValueError(
            "No flows to plot - either the matrix is empty, --min-count "
            "filtered everything out, or every pixel was unchanged."
        )

    total_pixels = unchanged_pixels + changed_pixels

    # If only stats are needed (for the time series) and no flows to plot,
    # we still return the counts.
    if not link_value and return_stats:
        pass
    elif not link_value:
        raise ValueError("No flows to plot.")

    # Only build the figure if we have links or we are forced to (though empty sankey is useless)
    fig = None
    if link_value:
        pct_changed = (100 * changed_pixels / total_pixels) if total_pixels else 0.0
        print(f"  Changed pixels: {changed_pixels:,} ({pct_changed:.2f}% of total)")

        title_suffix = "" if include_unchanged else " (changed pixels only)"

        fig = go.Figure(data=[go.Sankey(
            arrangement="snap",
            node=dict(
                pad=18, thickness=18,
                line=dict(color="black", width=0.4),
                label=node_labels, color=node_colors,
            ),
            link=dict(
                source=link_source, target=link_target,
                value=link_value, color=link_color,
            ),
        )])

        fig.update_layout(
            title_text=f"Level-4 land-cover transitions: {age_from} Ma \u2192 {age_to} Ma{title_suffix}",
            font_size=13, width=1150, height=700,
        )

        if out_html is None:
            suffix = "_sankey" if include_unchanged else "_sankey_changed"
            out_html = f"{Path(csv_path).stem}{suffix}.html"

        fig.write_html(out_html)
        print(f"  Saved: {Path(out_html).resolve()}")

    if return_stats:
        return fig, changed_pixels, total_pixels
    return fig

def plot_change_timeseries(stats_list, out_path="change_timeseries_summary.html"):
    """
    Creates a time series plot of the PERCENTAGE of pixels changed.
    stats_list: List of dicts {'age_from': float, 'age_to': float, 'changed': int, 'total': int}
    """
    if not stats_list:
        print("No data available to plot time series.")
        return

    df = pd.DataFrame(stats_list)

    # Percentage of pixels changed at each step (guard against total == 0)
    df['pct_changed'] = df.apply(
        lambda row: (100 * row['changed'] / row['total']) if row['total'] else 0.0,
        axis=1
    )

    # We plot against the 'age_from' (the starting point of the transition)
    # Sort by age to ensure line continuity
    df = df.sort_values(by='age_from')

    # Create the plot
    fig = go.Figure()

    # Add trace for Percent Changed
    fig.add_trace(go.Scatter(
        x=df['age_from'],
        y=df['pct_changed'],
        mode='lines+markers',
        name='% Pixels Changed',
        line=dict(color='#d62728', width=3),
        marker=dict(size=8),
        hovertemplate='<b>%{x:.1f} Ma</b><br>Changed: %{y:.2f}%<extra></extra>'
    ))

    # Layout
    fig.update_layout(
        title="Land-Cover Change Magnitude Through Geological Time",
        xaxis_title="Age (Ma) - Transition Starting Point",
        yaxis_title="Pixels Changed (%)",
        yaxis=dict(ticksuffix="%"),
        hovermode="x unified",
        template="plotly_white",
        width=1000,
        height=600,
    )

    # GEOLOGICAL TIME AXIS:
    # Data runs 0 -> 545 in age_from. To show Oldest (left) -> Youngest (right),
    # the x-axis range is inverted (max age on the left, 0 on the right).
    max_age = df['age_from'].max()
    min_age = df['age_from'].min()

    fig.update_xaxes(range=[max_age + 5, min_age - 5], title_text="Age (Ma)")

    fig.write_html(out_path)
    print(f"\n--- Summary Time Series Saved to: {Path(out_path).resolve()} ---")

def plot_change_rate_timeseries(stats_list, out_path="change_rate_timeseries_summary.html"):
    """
    Creates a time series plot of the RATE of change: percentage of pixels
    changed, normalised by the duration (in Ma) of each transition step.
    This corrects for reconstruction steps of uneven length (e.g. a 6 Myr
    step vs. a 13 Myr step aren't directly comparable in raw % terms).

    stats_list: List of dicts {'age_from': float, 'age_to': float, 'changed': int, 'total': int}
    """
    if not stats_list:
        print("No data available to plot rate time series.")
        return

    df = pd.DataFrame(stats_list)

    # Duration of each step in Ma (age_from is older, age_to is younger)
    df['duration'] = df['age_from'] - df['age_to']

    df['pct_changed'] = df.apply(
        lambda row: (100 * row['changed'] / row['total']) if row['total'] else 0.0,
        axis=1
    )

    # Rate = % changed per Myr of the step. Guard against zero-length steps.
    df['pct_per_myr'] = df.apply(
        lambda row: (row['pct_changed'] / row['duration']) if row['duration'] else 0.0,
        axis=1
    )

    df = df.sort_values(by='age_from')

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=df['age_from'],
        y=df['pct_per_myr'],
        mode='lines+markers',
        name='% Changed per Myr',
        line=dict(color='#1f77b4', width=3),
        marker=dict(size=8),
        customdata=df['duration'],
        hovertemplate=(
            '<b>%{x:.1f} Ma</b><br>Rate: %{y:.3f} %/Myr'
            '<br>Step length: %{customdata:.1f} Myr<extra></extra>'
        )
    ))

    fig.update_layout(
        title="Land-Cover Change Rate Through Geological Time (duration-normalised)",
        xaxis_title="Age (Ma) - Transition Starting Point",
        yaxis_title="Pixels Changed per Myr (%/Myr)",
        hovermode="x unified",
        template="plotly_white",
        width=1000,
        height=600,
    )

    max_age = df['age_from'].max()
    min_age = df['age_from'].min()
    fig.update_xaxes(range=[max_age + 5, min_age - 5], title_text="Age (Ma)")

    fig.write_html(out_path)
    print(f"--- Rate Time Series (duration-normalised) Saved to: {Path(out_path).resolve()} ---")

# ---------------------------------------------------------------------------
# Main Execution
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Check if user provided command line arguments for single mode
    if len(sys.argv) > 1:
        parser = argparse.ArgumentParser(description="Build a Sankey diagram from a change-matrix CSV.")
        parser.add_argument("csv_path", nargs="?", default=None, help="Path to the change-matrix CSV")
        parser.add_argument("--age-from", default=AGE_FROM, type=float)
        parser.add_argument("--age-to", default=AGE_TO, type=float)
        parser.add_argument("--min-count", type=int, default=0, help="Hide flows with <= this many pixels")
        parser.add_argument("--include-unchanged", action="store_true", help="Plot diagonal (unchanged pixels)")
        args = parser.parse_args()

        csv_path = args.csv_path or default_csv_name(args.age_from, args.age_to)

        try:
            build_sankey(csv_path, args.age_from, args.age_to,
                         min_count=args.min_count, include_unchanged=args.include_unchanged)
        except FileNotFoundError:
            print(f"Error: File not found: {csv_path}")
        except ValueError as e:
            print(f"Error: {e}")

    else:
        # BATCH MODE
        print(f"--- BATCH MODE: Processing {len(BATCH_YEARS_STR)-1} transitions ---\n")

        ages_int = [int(y) for y in BATCH_YEARS_STR]
        stats_collection = []

        for i in range(len(ages_int) - 1):
            year_to = ages_int[i]
            year_from = ages_int[i + 1]

            age_to = float(year_to - RASTER_AGE_OFFSET)
            age_from = float(year_from - RASTER_AGE_OFFSET)

            csv_filename = default_csv_name(age_from, age_to)

            print(f"\n[{i+1}/{len(ages_int)-1}] Transition: {int(age_from)} Ma -> {int(age_to)} Ma")

            if not Path(csv_filename).exists():
                print(f"  [SKIP] CSV not found: {csv_filename}")
                continue

            try:
                # Run sankey generation and capture stats
                fig, changed, total = build_sankey(
                    csv_path=csv_filename,
                    age_from=age_from,
                    age_to=age_to,
                    min_count=0,
                    include_unchanged=False,
                    return_stats=True
                )

                # Store data for the time series
                stats_collection.append({
                    'age_from': age_from,
                    'age_to': age_to,
                    'changed': changed,
                    'total': total
                })

            except Exception as e:
                print(f"  [ERROR] Failed: {e}")

        # Generate the Time Series Plots if we have data
        if stats_collection:
            plot_change_timeseries(stats_collection)          # raw % changed per step
            plot_change_rate_timeseries(stats_collection)      # % changed per Myr (duration-normalised)
        else:
            print("\nNo data collected. Skipping time series plots.")

        print("\n--- Batch processing complete ---")