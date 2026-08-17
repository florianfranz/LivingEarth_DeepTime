import re
from pathlib import Path
import pandas as pd
import rasterio
import json
import os
from pathlib import Path


CONFIG_PATH = Path(__file__).resolve().parent.parent / "config.json"
with open(CONFIG_PATH, "r") as f:
    _config = json.load(f)




# ============================================================================
# Configuration
# ============================================================================

GEO_INDICATORS_CSV = r"../geo_indicators/output/geo_indicators.csv"
MATRIX_DIR = r"../rotations"
OUTPUT_DIR = Path("output")

RASTER_BASE_DIR = _config["LE_outputs"]
RASTER_FILENAME = "level4_out_LE_DT.tif"
RASTER_AGE_OFFSET = 2000

TOP_N_PIXEL_ROWS = 15
AGE_MATCH_TOL = 0.5

# Updated labels for Land Cover Classes
LEVEL4_STYLE = {
    19: "Terr. Veg.: Generic",
    20: "Terr. Veg.: Woody",
    21: "Terr. Veg.: Herbaceous",
    55: "Aqu. Veg.: Generic",
    56: "Aqu. Veg.: Woody",
    94: "Bare Surface",
    98: "Sea-ice",
    99: "Water",
    105: "Snow",
    255: "No Data",
}

# Mapping for Geo Indicators
GEO_INDICATOR_LABELS = {
    "Ocean_Volume": "Ocean volume",
    "Tropical_Land_Area": "Tropical land",
    "Land_area": "Total land",
    "Ocean_Area": "Ocean area",
    "Southern_Land_Area": "Southern land",
    "Continental_Shelves_Area": "Cont. shelves",
    "Temperate_Land_Area": "Temperate land",
    "Polar_Land_Area": "Polar land",
    "High_Altitude_Area": "High altitude",
    "Northern_Land_Area": "Northern land",
    "Subtropical_Land_Area": "Subtropical land",
    "Coastline_length": "Coast length",
    "Sea_Level_dSL": "Sea level",
    "Continents_Number": "Continents",
    "BQART_TSS_est": "Sed. flux BQART",
    "ROBART_TSS_est": "Sed. flux ROBART",
    "CO2_ppm": "CO2",
}

# --- Thematic Grouping Definitions ---

# Group 1: Broad State (Land vs Ocean vs NoData)
THEME_BROAD_LAND = {"Bare Surface", "Snow", "Terr. Veg.: Generic", "Terr. Veg.: Woody", "Terr. Veg.: Herbaceous"}
THEME_BROAD_OCEAN = {"Aqu. Veg.: Generic", "Aqu. Veg.: Woody", "Sea-ice", "Water"}
THEME_BROAD_NODATA = {"No Data"}

# Group 2: Vegetation Status (Strictly Land Only)
THEME_VEG_VEGETATED = {"Terr. Veg.: Generic", "Terr. Veg.: Woody", "Terr. Veg.: Herbaceous"}
THEME_VEG_NON_VEGETATED = {"Bare Surface", "Snow"}
# Everything else is ignored for this specific table

MATRIX_FILENAME_RE = re.compile(r"change_matrix_([\d.]+)Ma_to_([\d.]+)Ma_band(\d+)\.csv")


# ============================================================================
# Helpers
# ============================================================================

def get_theme_broad(label: str) -> str:
    if label in THEME_BROAD_LAND:
        return "Land"
    elif label in THEME_BROAD_OCEAN:
        return "Ocean"
    elif label in THEME_BROAD_NODATA or "off_grid" in label or "unmapped" in label:
        return "No Data"
    else:
        return "Other"


def get_theme_veg_strict(label: str) -> str:
    """
    Returns 'Vegetated Land' or 'Non-Vegetated Land'.
    Returns None for everything else (Ocean, Ice, No Data, etc.) to facilitate strict filtering.
    """
    if label in THEME_VEG_VEGETATED:
        return "Vegetated Land"
    elif label in THEME_VEG_NON_VEGETATED:
        return "Non-Vegetated Land"
    else:
        return None


def label_for(code) -> str:
    try:
        code_int = int(float(code))
        return LEVEL4_STYLE.get(code_int, f"code_{code_int}")
    except (ValueError, TypeError):
        return str(code)


_pixel_area_cache = {}


def get_pixel_area_m2(age: float) -> float:
    key = round(age, 1)
    if key in _pixel_area_cache:
        return _pixel_area_cache[key]
    folder = str(RASTER_AGE_OFFSET + int(round(age)))
    raster_path = Path(RASTER_BASE_DIR) / folder / RASTER_FILENAME
    try:
        with rasterio.open(raster_path) as src:
            transform = src.transform
        area = abs(transform.a * transform.e)
        _pixel_area_cache[key] = area
        return area
    except Exception:
        return 1.0


def find_matrix_files(matrix_dir: str):
    by_pair = {}
    for path in Path(matrix_dir).glob("change_matrix_*.csv"):
        m = MATRIX_FILENAME_RE.match(path.name)
        if not m:
            continue
        age_from, age_to, band = float(m.group(1)), float(m.group(2)), int(m.group(3))
        by_pair.setdefault((age_from, age_to), []).append((band, path))

    found = []
    for (age_from, age_to), candidates in by_pair.items():
        if len(candidates) > 1:
            candidates_sorted = sorted(candidates, key=lambda c: c[1].stat().st_mtime, reverse=True)
            kept_band, kept_path = candidates_sorted[0]
            print(
                f"  [warning] {len(candidates)} files found for {age_from} Ma -> {age_to} Ma - keeping {kept_path.name}")
        else:
            kept_band, kept_path = candidates[0]
        found.append((age_from, age_to, kept_band, kept_path))

    found.sort(key=lambda t: t[0], reverse=True)
    return found


# ============================================================================
# Data Processing
# ============================================================================

def process_transitions(path: Path, pixel_area_m2: float, top_n: int):
    raw = pd.read_csv(path, index_col=0)
    raw.index.name = "source_code"
    long = raw.reset_index().melt(id_vars="source_code", var_name="dest_code", value_name="pixel_count")

    # Remove no-change pixels
    long = long[(long["source_code"].astype(str) != long["dest_code"].astype(str)) & (long["pixel_count"] > 0)]

    if long.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    long["source_label"] = long["source_code"].apply(label_for)
    long["dest_label"] = long["dest_code"].apply(label_for)
    long["area_km2"] = long["pixel_count"] * pixel_area_m2 / 1e6

    # --- 1. Broad Thematic (Land/Ocean/NoData) ---
    long["source_theme_broad"] = long["source_label"].apply(get_theme_broad)
    long["dest_theme_broad"] = long["dest_label"].apply(get_theme_broad)

    # Filter: Only keep changes where theme changes (Land->Ocean, etc.)
    broad_mask = long["source_theme_broad"] != long["dest_theme_broad"]
    long_broad = long[broad_mask].copy()

    thematic_broad = long_broad.groupby(["source_theme_broad", "dest_theme_broad"], as_index=False)["area_km2"].sum()
    thematic_broad = thematic_broad.sort_values(by=["source_theme_broad", "area_km2"], ascending=[True, False])

    # --- 2. Vegetation Status Transitions (STRICT: Veg <-> Non-Veg ONLY) ---
    # Apply strict mapping
    long["source_theme_veg"] = long["source_label"].apply(get_theme_veg_strict)
    long["dest_theme_veg"] = long["dest_label"].apply(get_theme_veg_strict)

    # CRITICAL FILTER:
    # 1. Drop any row where Source is NOT land (i.e., is None)
    # 2. Drop any row where Dest is NOT land (i.e., is None)
    # 3. Drop any row where Source == Dest (e.g., Veg -> Veg)
    veg_mask = (
            long["source_theme_veg"].notna() &
            long["dest_theme_veg"].notna() &
            (long["source_theme_veg"] != long["dest_theme_veg"])
    )

    long_veg = long[veg_mask].copy()

    # Aggregate
    if not long_veg.empty:
        thematic_veg = long_veg.groupby(["source_theme_veg", "dest_theme_veg"], as_index=False)["area_km2"].sum()
        thematic_veg = thematic_veg.sort_values(by=["source_theme_veg", "area_km2"], ascending=[True, False])
    else:
        thematic_veg = pd.DataFrame()

    # --- 3. Top N Pixel Details (Based on Broad Changes only) ---
    if long_broad.empty:
        top_details = pd.DataFrame()
    else:
        top_details = long_broad.nlargest(top_n, "area_km2")
        top_details = top_details.sort_values(by=["source_label", "area_km2"], ascending=[True, False])

    return thematic_broad, thematic_veg, top_details


def match_age_row(df: pd.DataFrame, age: float) -> pd.Series:
    diffs = (df["Age"] - age).abs()
    idx = diffs.idxmin()
    if diffs.loc[idx] > AGE_MATCH_TOL:
        raise ValueError(f"No geo-indicator row found within {AGE_MATCH_TOL} Ma of {age} Ma")
    return df.loc[idx]


def indicator_deltas(df: pd.DataFrame, age_from: float, age_to: float) -> pd.DataFrame:
    row_from = match_age_row(df, age_from)
    row_to = match_age_row(df, age_to)
    numeric_cols = [c for c in df.columns if c != "Age"]

    records = []
    for col in numeric_cols:
        v_from, v_to = row_from[col], row_to[col]
        delta = v_to - v_from
        pct = (delta / v_from * 100) if v_from not in (0, None) else float("nan")
        display_name = GEO_INDICATOR_LABELS.get(col, col)
        records.append({
            "indicator": display_name,
            "value_from": v_from,
            "value_to": v_to,
            "delta": delta,
            "pct_change": pct,
            "delta_abs": abs(delta)
        })

    out = pd.DataFrame(records)
    out = out.sort_values(by=["indicator", "delta_abs"], ascending=[True, False])
    return out.drop(columns=["delta_abs"])


# ============================================================================
# Markdown Generation
# ============================================================================

def fmt_num(x):
    if pd.isna(x):
        return "n/a"
    elif -10000 <= x < 10000:
        return round(x, 2)
    else:
        return f"{x:,.3e}"


def fmt_pct(x):
    return f"{x:.2f}%" if pd.notna(x) else "n/a"


def thematic_table(df: pd.DataFrame, col_src: str, col_dest: str) -> list:
    lines = ["| From | To | Area (km²) |", "|-----|-----|---:|"]
    for _, r in df.iterrows():
        lines.append(f"| {r[col_src]} | {r[col_dest]} | {int(r['area_km2']):,} |")
    return lines


def matrix_table(df: pd.DataFrame) -> list:
    if df.empty:
        return ["*No significant cross-boundary pixel transitions found.*"]
    lines = ["| Source | Destination | Pixels | Area (km²) |", "|-----|-----|---:|---:|"]
    for _, r in df.iterrows():
        lines.append(
            f"| {r['source_label']} | {r['dest_label']} | {int(r['pixel_count']):,} | {int(r['area_km2']):,} |")
    return lines


def indicator_table(df: pd.DataFrame) -> list:
    lines = ["| Indicator | Value @From | Value @To | Delta | % Change |", "|-----|---:|---:|---:|---:|"]
    for _, r in df.iterrows():
        lines.append(
            f"| {r['indicator']} | {fmt_num(r['value_from'])} | {fmt_num(r['value_to'])} | {fmt_num(r['delta'])} | {fmt_pct(r['pct_change'])} |")
    return lines


# ============================================================================
# Main
# ============================================================================

# ============================================================================
# Main
# ============================================================================

def main():
    OUTPUT_DIR.mkdir(exist_ok=True)
    geo_df = pd.read_csv(GEO_INDICATORS_CSV)

    matrix_files = find_matrix_files(MATRIX_DIR)
    if not matrix_files:
        print(f"No change_matrix_*.csv files found in {MATRIX_DIR}")
        return

    # YAML Header
    yaml_header = [
        "---",
        "title: 'Phanerozoic Transition Report'",
        "format:",
        "  pdf:",
        "    pdf-engine: pdflatex",
        "    fontsize: 8pt",
        "    geometry:",
        "      - margin=0.5in",
        "    table-caption-location: margin",
        "---",
        ""
    ]

    lines = yaml_header
    lines.append("This report summarizes Earth surface transitions during the Phanerozoic.")
    lines.append("Only transitions involving a change in state are shown.")
    lines.append("")
    lines.append("**Thematic Definitions:**")
    lines.append("- **Broad:** Land (Veg+Bare+Snow), Ocean (Water+Ice+Aquatic Veg), No Data.")
    # CHANGED: Unicode arrow replaced with ASCII
    lines.append("- **Vegetation:** Strictly exchanges between **Vegetated Land** (Terr. Veg) and **Non-Vegetated Land** (Bare+Snow). Ocean/No Data transitions are excluded from this table.")
    lines.append("")

    for age_from, age_to, band, path in matrix_files:
        print(f"Processing transition {age_from} Ma -> {age_to} Ma")
        lines.append(f"## {age_from} Ma -> {age_to} Ma")
        lines.append("")

        pixel_area_m2 = get_pixel_area_m2(age_from)

        # Generate Data
        thematic_broad_df, thematic_veg_df, top_details_df = process_transitions(path, pixel_area_m2, TOP_N_PIXEL_ROWS)

        # 1. Broad Thematic Table
        # CHANGED: Unicode arrow replaced with ASCII
        lines.append("### 1. Broad Thematic Transitions (Land <-> Ocean <-> No Data)")
        lines.append("")
        if not thematic_broad_df.empty:
            lines.extend(thematic_table(thematic_broad_df, "source_theme_broad", "dest_theme_broad"))
        else:
            lines.append("*No cross-boundary thematic changes detected.*")
        lines.append("")

        # 2. Vegetation Status Table (STRICT)
        # CHANGED: Unicode arrow replaced with ASCII
        lines.append("### 2. Vegetation Status Transitions (Vegetated <-> Non-Vegetated)")
        lines.append("")
        if not thematic_veg_df.empty:
            lines.extend(thematic_table(thematic_veg_df, "source_theme_veg", "dest_theme_veg"))
        else:
            lines.append("*No vegetation status changes detected (strictly land-to-land).*")
        lines.append("")

        # 3. Top Pixel Changes
        lines.append(f"### 3. Top {TOP_N_PIXEL_ROWS} Specific Class Conversions")
        lines.append("")
        lines.extend(matrix_table(top_details_df))
        lines.append("")

        # 4. Geo Indicators
        try:
            deltas = indicator_deltas(geo_df, age_from, age_to)
            lines.append("### 4. Geo-Indicator Deltas")
            lines.append("")
            lines.extend(indicator_table(deltas))
        except ValueError as e:
            lines.append(f"*No matching geo-indicator rows found: {e}*")

        # Page Break
        lines.append("")
        lines.append("\\newpage")
        lines.append("")

    # Write file
    output_path = OUTPUT_DIR / "transition_report.md"
    output_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nDone. Wrote {output_path.resolve()}")
    print("Next step: Run 'quarto render transition_report.md --to pdf' in the output directory.")

if __name__ == "__main__":
    main()