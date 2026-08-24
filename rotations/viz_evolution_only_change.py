"""
Sankey flow diagram of Level-4 land-cover code transitions between two ages,
INCLUDING 'new_crust' (source) and 'subducted' (sink) nodes.

Filename matching fixed to: change_matrix_{age_from}Ma_to_{age_to}Ma.csv
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

# Standard Land Cover Codes
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

# Special Tectonic Nodes
TECTONIC_STYLE = {
    "new_crust":   (" Diverging Plates", "#ff4500"),      # Orange-Red
    "subducted":   ("Converging Plates", "#191970"),     # Midnight Blue
    "off_grid":    ("Rotated off grid",     "#777777"),
    "unmapped_plate": ("No plate match",    "#333333"),
}

DEFAULT_COLOR = "#bbbbbb"

# Define the years you want to process
BATCH_YEARS_STR =[
        "2000","2006", "2011", "2015", "2020", "2033", "2040", "2048",
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
    """
    Generates the CSV filename matching the output of build_change_matrix().
    Format: change_matrix_11Ma_to_6Ma.csv
    """
    return f"change_matrix_{int(age_from)}Ma_to_{int(age_to)}Ma.csv"

def label_and_color(code) -> tuple:
    """Returns (Label, Color) for a given code."""
    # 1. Check Land Cover Codes
    try:
        code_int = int(code)
        if code_int in LEVEL4_STYLE:
            name, color = LEVEL4_STYLE[code_int]
            return f"{code_int} {name}", color
    except (TypeError, ValueError):
        pass

    # 2. Check Special Tectonic/Extra Codes
    key = str(code).lower()
    if key in TECTONIC_STYLE:
        return TECTONIC_STYLE[key]

    # Fallback
    if key in ["off_grid", "unmapped_plate"]:
        return TECTONIC_STYLE.get(key, (str(code), DEFAULT_COLOR))

    return str(code), DEFAULT_COLOR

def hex_to_rgba(hex_color: str, alpha: float = 0.6) -> str:
    """Converts hex to rgba for semi-transparent flows."""
    hex_color = hex_color.lstrip("#")
    if len(hex_color) == 3:
        hex_color = "".join([c*2 for c in hex_color])
    try:
        r, g, b = (int(hex_color[i:i + 2], 16) for i in (0, 2, 4))
        return f"rgba({r},{g},{b},{alpha})"
    except ValueError:
        return "rgba(100,100,100,0.6)"

def _codes_match(src, dst) -> bool:
    """Checks if source and destination are the same type."""
    # New Crust and Subducted never match anything
    src_str = str(src).lower()
    dst_str = str(dst).lower()

    if src_str in ["new_crust", "subducted"] or dst_str in ["new_crust", "subducted"]:
        return False

    try:
        return int(src) == int(dst)
    except (TypeError, ValueError):
        return src_str == dst_str


def build_sankey(csv_path: str, age_from, age_to, min_count: int = 0,
                 out_html: str = None, include_unchanged: bool = False,
                 return_stats: bool = False):
    try:
        matrix = pd.read_csv(csv_path, index_col=0)
    except FileNotFoundError:
        raise FileNotFoundError(f"Matrix file not found: {csv_path}")

    # DEBUG: Print exact index and columns to check for whitespace/case issues
    print(f"  [DEBUG] Source Index: {list(matrix.index)}")
    print(f"  [DEBUG] Dest Columns: {list(matrix.columns)}")

    source_codes = list(matrix.index)
    dest_codes = list(matrix.columns)

    source_meta = {code: label_and_color(code) for code in source_codes}
    dest_meta = {code: label_and_color(code) for code in dest_codes}

    # DEBUG: Check if special keys exist
    has_new_crust = any(str(c).lower() == 'new_crust' for c in source_codes)
    has_subducted = any(str(c).lower() == 'subducted' for c in dest_codes)
    if not has_new_crust:
        print("  [WARNING] 'new_crust' not found in source index! Check CSV spelling.")
    if not has_subducted:
        print("  [WARNING] 'subducted' not found in dest columns! Check CSV spelling.")

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
    new_crust_total = 0
    subducted_total = 0

    for src in source_codes:
        for dst in dest_codes:
            value = matrix.loc[src, dst]
            if pd.isna(value) or value <= 0:
                continue

            val_int = int(value)
            src_str = str(src).lower().strip()
            dst_str = str(dst).lower().strip()

            # Track Tectonic Totals
            if src_str == "new_crust":
                new_crust_total += val_int
            if dst_str == "subducted":
                subducted_total += val_int

            # Check for Unchanged
            if _codes_match(src, dst):
                unchanged_pixels += val_int
                if not include_unchanged:
                    continue
            else:
                changed_pixels += val_int

            # CRITICAL FIX: Do NOT filter out tectonic flows with min_count
            # If it's new_crust or subducted, always include it regardless of size
            is_tectonic = (src_str == "new_crust" or dst_str == "subducted")

            if not is_tectonic and value <= min_count:
                continue

            link_source.append(src_idx[src])
            link_target.append(dst_idx[dst])
            link_value.append(val_int)
            link_color.append(hex_to_rgba(source_meta[src][1], alpha=0.6))

    if not link_value and not return_stats:
        raise ValueError("No flows to plot.")

    total_pixels = unchanged_pixels + changed_pixels

    fig = None
    if link_value:
        pct_changed = (100 * changed_pixels / total_pixels) if total_pixels else 0.0
        print(
            f"  Changed: {changed_pixels:,} ({pct_changed:.2f}%) | New Crust: {new_crust_total:,} | Subducted: {subducted_total:,}")
        print(f"  Total Links Created: {len(link_value)}")

        title_suffix = "" if include_unchanged else " (changed + tectonic flows)"

        fig = go.Figure(data=[go.Sankey(
            arrangement="snap",
            node=dict(
                pad=20,
                thickness=20,
                line=dict(color="black", width=0.5),
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
            title_text=f"Land-Cover & Tectonic Flux: {age_from} Ma → {age_to} Ma{title_suffix}",
            font_size=12,
            width=1200,
            height=800,
            hoverlabel=dict(font_size=12)
        )

        if out_html is None:
            suffix = "_sankey_full" if include_unchanged else "_sankey_tectonic"
            out_html = f"{Path(csv_path).stem}{suffix}.html"

        fig.write_html(out_html)
        print(f"  Saved: {Path(out_html).resolve()}")

    if return_stats:
        return fig, changed_pixels, total_pixels, new_crust_total, subducted_total
    return fig

def plot_change_timeseries(stats_list, out_path="change_timeseries_summary.html"):
    """Plots % Changed, New Crust, and Subducted over time (absolute, per-step %)."""
    if not stats_list:
        print("No data available.")
        return

    df = pd.DataFrame(stats_list)
    df = df.sort_values(by='age_from')

    # Calculate percentages
    df['pct_changed'] = (100 * df['changed'] / df['total']).fillna(0)
    df['pct_new_crust'] = (100 * df['new_crust'] / df['total']).fillna(0)
    df['pct_subducted'] = (100 * df['subducted'] / df['total']).fillna(0)

    fig = go.Figure()

    # Trace 1: Changed Pixels
    fig.add_trace(go.Scatter(
        x=df['age_from'], y=df['pct_changed'],
        mode='lines+markers', name='Land-Cover Change',
        line=dict(color='#d62728', width=3)
    ))

    # Trace 2: New Crust
    fig.add_trace(go.Scatter(
        x=df['age_from'], y=df['pct_new_crust'],
        mode='lines+markers', name=' Diverging Plates (Source)',
        line=dict(color='#ff7f0e', width=3, dash='dot')
    ))

    # Trace 3: Collision
    fig.add_trace(go.Scatter(
        x=df['age_from'], y=df['pct_subducted'],
        mode='lines+markers', name=' Converging Plates (Sink)',
        line=dict(color='#1f77b4', width=3, dash='dash')
    ))

    fig.update_layout(
        title="Tectonic & Land-Cover Flux Through Geological Time (per transition, absolute %)",
        xaxis_title="Age (Ma) - Transition Starting Point",
        yaxis_title="Pixels (%)",
        yaxis=dict(ticksuffix="%"),
        hovermode="x unified",
        template="plotly_white",
        width=1100,
        height=650,
    )

    # Invert X for Geological Time
    max_age = df['age_from'].max()
    min_age = df['age_from'].min()
    fig.update_xaxes(range=[max_age + 5, min_age - 5])

    fig.write_html(out_path)
    print(f"\n--- Time Series (absolute) Saved to: {Path(out_path).resolve()} ---")


def plot_change_timeseries_normalized(stats_list, out_path="change_timeseries_normalized.html"):
    """
    Plots % Changed, New Crust, and Subducted PER MILLION YEARS, i.e. each
    step's absolute % is divided by that step's duration (age_from - age_to).

    This corrects for the irregular spacing of BATCH_YEARS_STR: a transition
    spanning 13 Myr will otherwise show more accumulated change than one
    spanning 5 Myr purely because it had more time, not because the rate of
    change was actually higher. Dividing by duration turns the per-step
    percentages into a rate (%/Myr) that is comparable across steps of
    different lengths.
    """
    if not stats_list:
        print("No data available.")
        return

    df = pd.DataFrame(stats_list)
    df = df.sort_values(by='age_from')

    # Duration of each transition step, in Myr. age_from > age_to by construction.
    df['duration'] = (df['age_from'] - df['age_to']).abs()
    # Guard against zero-duration rows (shouldn't occur, but avoid div/0)
    safe_duration = df['duration'].replace(0, pd.NA)

    df['pct_changed'] = (100 * df['changed'] / df['total']).fillna(0)
    df['pct_new_crust'] = (100 * df['new_crust'] / df['total']).fillna(0)
    df['pct_subducted'] = (100 * df['subducted'] / df['total']).fillna(0)

    df['pct_changed_per_myr'] = (df['pct_changed'] / safe_duration).fillna(0)
    df['pct_new_crust_per_myr'] = (df['pct_new_crust'] / safe_duration).fillna(0)
    df['pct_subducted_per_myr'] = (df['pct_subducted'] / safe_duration).fillna(0)

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=df['age_from'], y=df['pct_changed_per_myr'],
        mode='lines+markers', name='Land-Cover Change (rate)',
        line=dict(color='#d62728', width=3),
        customdata=df['duration'],
        hovertemplate="Age: %{x} Ma<br>Rate: %{y:.3f} %/Myr<br>Step length: %{customdata:.0f} Myr<extra></extra>"
    ))

    fig.add_trace(go.Scatter(
        x=df['age_from'], y=df['pct_new_crust_per_myr'],
        mode='lines+markers', name=' Diverging Plates (rate)',
        line=dict(color='#ff7f0e', width=3, dash='dot'),
        customdata=df['duration'],
        hovertemplate="Age: %{x} Ma<br>Rate: %{y:.3f} %/Myr<br>Step length: %{customdata:.0f} Myr<extra></extra>"
    ))

    fig.add_trace(go.Scatter(
        x=df['age_from'], y=df['pct_subducted_per_myr'],
        mode='lines+markers', name=' Converging Plates (rate)',
        line=dict(color='#1f77b4', width=3, dash='dash'),
        customdata=df['duration'],
        hovertemplate="Age: %{x} Ma<br>Rate: %{y:.3f} %/Myr<br>Step length: %{customdata:.0f} Myr<extra></extra>"
    ))

    # Secondary trace showing step length itself, so uneven spacing is visible
    fig.add_trace(go.Bar(
        x=df['age_from'], y=df['duration'],
        name='Step length (Myr)',
        marker=dict(color='rgba(150,150,150,0.35)'),
        yaxis='y2',
    ))

    fig.update_layout(
        title="Tectonic & Land-Cover Flux Through Geological Time (normalized, %/Myr)",
        xaxis_title="Age (Ma) - Transition Starting Point",
        yaxis=dict(title="Rate (% / Myr)"),
        yaxis2=dict(
            title="Step length (Myr)",
            overlaying='y',
            side='right',
            showgrid=False,
        ),
        hovermode="x unified",
        template="plotly_white",
        width=1100,
        height=650,
        barmode='overlay',
    )

    max_age = df['age_from'].max()
    min_age = df['age_from'].min()
    fig.update_xaxes(range=[max_age + 5, min_age - 5])

    fig.write_html(out_path)
    print(f"--- Time Series (normalized, %/Myr) Saved to: {Path(out_path).resolve()} ---")

# ---------------------------------------------------------------------------
# Main Execution
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    if len(sys.argv) > 1:
        # Single Mode
        parser = argparse.ArgumentParser()
        parser.add_argument("csv_path", nargs="?", default=None)
        parser.add_argument("--age-from", default=AGE_FROM, type=float)
        parser.add_argument("--age-to", default=AGE_TO, type=float)
        parser.add_argument("--min-count", type=int, default=0)
        parser.add_argument("--include-unchanged", action="store_true")
        args = parser.parse_args()

        csv_path = args.csv_path or default_csv_name(args.age_from, args.age_to)
        try:
            build_sankey(csv_path, args.age_from, args.age_to,
                         min_count=args.min_count, include_unchanged=args.include_unchanged)
        except Exception as e:
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

            # Uses the FIXED filename generator
            csv_filename = default_csv_name(age_from, age_to)

            print(f"\n[{i+1}/{len(ages_int)-1}] Transition: {int(age_from)} Ma -> {int(age_to)} Ma")
            print(f"  Looking for: {csv_filename}")

            if not Path(csv_filename).exists():
                print(f"  [SKIP] CSV not found: {csv_filename}")
                continue

            try:
                # FIXED: Unpack 5 values now
                fig, changed, total, new_crust, subducted = build_sankey(
                    csv_path=csv_filename,
                    age_from=age_from,
                    age_to=age_to,
                    min_count=0,
                    include_unchanged=False,
                    return_stats=True
                )

                stats_collection.append({
                    'age_from': age_from,
                    'age_to': age_to,
                    'changed': changed,
                    'total': total,
                    'new_crust': new_crust,
                    'subducted': subducted
                })

            except Exception as e:
                print(f"  [ERROR] Failed: {e}")
                import traceback
                traceback.print_exc()

        if stats_collection:
            plot_change_timeseries(stats_collection)
            plot_change_timeseries_normalized(stats_collection)
        else:
            print("\nNo data collected. Skipping time series plots.")

        print("\n--- Batch processing complete ---")