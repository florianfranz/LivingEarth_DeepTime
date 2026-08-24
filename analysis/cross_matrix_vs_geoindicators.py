import re
from pathlib import Path
import pandas as pd
import rasterio
import json

# ============================================================================
# Configuration
# ============================================================================

CONFIG_PATH = Path(__file__).resolve().parent.parent / "config.json"
with open(CONFIG_PATH, "r") as f:
    _config = json.load(f)

GEO_INDICATORS_CSV = r"../geo_indicators/output/geo_indicators.csv"
MATRIX_DIR = r"../rotations"
OUTPUT_DIR = Path("output")

RASTER_BASE_DIR = _config["LE_outputs"]
RASTER_FILENAME = "level4_out_LE_DT.tif"
RASTER_AGE_OFFSET = 2000

TOP_N_PIXEL_ROWS = 15
AGE_MATCH_TOL = 0.5

LEVEL4_STYLE = {
    20: "Terr. Veg.: Woody",
    21: "Terr. Veg.: Herbaceous",
    56: "Aqu. Veg.: Woody",
    94: "Bare Surface",
    98: "Wet Soil",
    99: "Water",
    105: "Snow",
    106: "Sea-ice",
    255: "No Data"
}

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

# Theme Definitions
THEME_BROAD_LAND = {"Terr. Veg.: Woody", "Terr. Veg.: Herbaceous", "Bare Surface", "Snow", "Wet Soil",
                    "Aqu. Veg.: Woody"}
THEME_BROAD_OCEAN = {"Water", "Sea-ice"}
THEME_BROAD_NODATA = {"No Data", "unmapped_plate", "off_grid"}

THEME_VEG_VEGETATED = {"Terr. Veg.: Woody", "Terr. Veg.: Herbaceous", "Aqu. Veg.: Woody"}
THEME_VEG_NON_VEGETATED = {"Bare Surface", "Snow", "Wet Soil"}

# UPDATED: Regex now looks for 'inverse' in the filename
MATRIX_FILENAME_RE = re.compile(r"change_matrix_([\d.]+)Ma_to_([\d.]+)Ma.csv")


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


def get_theme_veg_strict(label: str):
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
        area = float(abs(transform.a * transform.e))
        _pixel_area_cache[key] = area
        return area
    except Exception:
        return 1.0


def find_matrix_files(matrix_dir: str):
    by_pair = {}
    # Deterministic file discovery
    all_paths = sorted(Path(matrix_dir).glob("change_matrix_*.csv"))

    for path in all_paths:
        m = MATRIX_FILENAME_RE.match(path.name)
        if not m:
            continue
        age_from, age_to = float(m.group(1)), float(m.group(2))
        band= 4
        by_pair.setdefault((age_from, age_to), []).append((band, path))

    found = []
    for (age_from, age_to), candidates in by_pair.items():
        if len(candidates) > 1:
            candidates_sorted = sorted(candidates, key=lambda c: c[1].name)
            kept_band, kept_path = candidates_sorted[-1]
            print(
                f"  [warning] {len(candidates)} files found for {age_from} Ma -> {age_to} Ma - keeping {kept_path.name}")
        else:
            kept_band, kept_path = candidates[0]
        found.append((age_from, age_to, kept_band, kept_path))

    found.sort(key=lambda t: (t[0], t[1]))
    return found


# ============================================================================
# Data Processing
# ============================================================================

def process_transitions_gains(path: Path, pixel_area_m2: float, top_n: int):
    raw = pd.read_csv(path, index_col=0)
    raw.index.name = "source_code"

    # Unpivot to long format
    long = raw.reset_index().melt(id_vars="source_code", var_name="dest_code", value_name="pixel_count")
    long = long[long["pixel_count"] > 0]

    if long.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    # Normalize labels for logic
    long["source_norm"] = long["source_code"].apply(lambda x: str(x).lower().strip())
    long["dest_norm"] = long["dest_code"].apply(lambda x: str(x).lower().strip())

    # Keep original for display
    long["source_label"] = long["source_code"].apply(label_for)
    long["dest_label"] = long["dest_code"].apply(label_for)

    # Calculate Area in km²
    long["area_km2"] = long["pixel_count"] * pixel_area_m2 / 1e6

    # Define Sets for Logic
    # Note: We use the normalized strings or the original codes depending on how they appear in the CSV.
    # Assuming CSV has "new_crust" and "subducted" as text, and integers for others.

    # Helper to check membership safely
    def is_in_set(code, target_set):
        # Check direct match
        if code in target_set: return True
        # Check if it's a string representation of an int in the set
        try:
            return int(float(code)) in target_set
        except:
            return False

    LAND_CODES = {20, 21, 56, 94, 98, 105}  # Woody, Herb, Aqu Woody, Bare, Wet Soil, Snow
    OCEAN_CODES = {99, 106}  # Water, Sea-ice

    # Normalize sets for string comparison if needed
    LAND_STR = {str(c) for c in LAND_CODES}
    OCEAN_STR = {str(c) for c in OCEAN_CODES}

    def is_land(code):
        norm = str(code).lower().strip()
        return norm in LAND_STR or (norm.isdigit() and int(norm) in LAND_CODES)

    def is_ocean(code):
        norm = str(code).lower().strip()
        return norm in OCEAN_STR or (norm.isdigit() and int(norm) in OCEAN_CODES)

    def is_new_crust(code):
        norm = str(code).lower().strip()
        return "new crust" in norm or "🌋" in norm

    def is_subducted(code):
        norm = str(code).lower().strip()
        return "subducted" in norm or "🌊" in norm

    # Apply Flags
    long["is_source_land"] = long["source_code"].apply(is_land)
    long["is_dest_land"] = long["dest_code"].apply(is_land)

    long["is_source_ocean"] = long["source_code"].apply(is_ocean)
    long["is_dest_ocean"] = long["dest_code"].apply(is_ocean)

    long["is_source_new_crust"] = long["source_code"].apply(is_new_crust)
    long["is_dest_subducted"] = long["dest_code"].apply(is_subducted)

    # --- 1. Broad Gains & Losses (EXPLICIT LOGIC) ---
    records_broad = []

    # LAND GAIN:
    # 1. New Crust -> Land
    # 2. (Any Non-Land) -> Land (excluding New Crust->Land to avoid double counting if we separate them, but here we combine)
    # Actually, let's follow your prompt exactly:
    # "if new crust goes to any of the other labels... this is land gain"
    # "if new crust goes to 99/106... this is ocean gain"
    # So Land Gain = (New Crust -> Land) + (Non-Land/Non-NewCrust -> Land)

    land_gain_mask = (
            (long["is_source_new_crust"] & long["is_dest_land"]) |
            (~long["is_source_land"] & ~long["is_source_new_crust"] & long["is_dest_land"])
    )

    # LAND LOSS:
    # 1. Land -> Subducted
    # 2. Land -> (Any Non-Land/Non-Subducted)
    land_loss_mask = (
            (long["is_source_land"] & long["is_dest_subducted"]) |
            (long["is_source_land"] & ~long["is_dest_land"] & ~long["is_dest_subducted"])
    )

    if land_gain_mask.any():
        lp = long[land_gain_mask]["pixel_count"].sum()
        la = long[land_gain_mask]["area_km2"].sum()
        records_broad.append({"Metric": "Land Gain", "Pixels": lp, "Area_km2": la, "Type": "Gain"})

    if land_loss_mask.any():
        lp = long[land_loss_mask]["pixel_count"].sum()
        la = long[land_loss_mask]["area_km2"].sum()
        records_broad.append({"Metric": "Land Loss", "Pixels": lp, "Area_km2": la, "Type": "Loss"})

    # OCEAN GAIN:
    # 1. New Crust -> Ocean
    # 2. (Any Non-Ocean/Non-NewCrust) -> Ocean
    ocean_gain_mask = (
            (long["is_source_new_crust"] & long["is_dest_ocean"]) |
            (~long["is_source_ocean"] & ~long["is_source_new_crust"] & long["is_dest_ocean"])
    )

    # OCEAN LOSS:
    # 1. Ocean -> Subducted
    # 2. Ocean -> (Any Non-Ocean/Non-Subducted)
    ocean_loss_mask = (
            (long["is_source_ocean"] & long["is_dest_subducted"]) |
            (long["is_source_ocean"] & ~long["is_dest_ocean"] & ~long["is_dest_subducted"])
    )

    if ocean_gain_mask.any():
        op = long[ocean_gain_mask]["pixel_count"].sum()
        oa = long[ocean_gain_mask]["area_km2"].sum()
        records_broad.append({"Metric": "Ocean Gain", "Pixels": op, "Area_km2": oa, "Type": "Gain"})

    if ocean_loss_mask.any():
        op = long[ocean_loss_mask]["pixel_count"].sum()
        oa = long[ocean_loss_mask]["area_km2"].sum()
        records_broad.append({"Metric": "Ocean Loss", "Pixels": op, "Area_km2": oa, "Type": "Loss"})

    # TECTONIC TOTALS (For information)
    new_crust_mask = long["is_source_new_crust"]
    subducted_mask = long["is_dest_subducted"]

    if new_crust_mask.any():
        nc = long[new_crust_mask]["pixel_count"].sum()
        nca = long[new_crust_mask]["area_km2"].sum()
        records_broad.append({"Metric": "New Crust (Total Source)", "Pixels": nc, "Area_km2": nca, "Type": "Info"})

    if subducted_mask.any():
        sub = long[subducted_mask]["pixel_count"].sum()
        suba = long[subducted_mask]["area_km2"].sum()
        records_broad.append({"Metric": "Subducted (Total Sink)", "Pixels": sub, "Area_km2": suba, "Type": "Info"})

    df_broad = pd.DataFrame(records_broad)

    # Calculate Net Change Summary
    net_land = 0.0
    net_ocean = 0.0

    if not df_broad.empty:
        for _, r in df_broad.iterrows():
            if r['Type'] == 'Gain':
                val = r['Area_km2']
            elif r['Type'] == 'Loss':
                val = -r['Area_km2']
            else:
                val = 0

            if "Land" in r['Metric']:
                net_land += val
            if "Ocean" in r['Metric']:
                net_ocean += val

    summary_records = []
    if net_land != 0:
        summary_records.append({
            "Metric": "NET Land Change",
            "Pixels": int(round(net_land / (pixel_area_m2 / 1e6))),
            "Area_km2": int(round(net_land))
        })
    if net_ocean != 0:
        summary_records.append({
            "Metric": "NET Ocean Change",
            "Pixels": int(round(net_ocean / (pixel_area_m2 / 1e6))),
            "Area_km2": int(round(net_ocean))
        })

    df_summary = pd.DataFrame(summary_records)

    # --- 2. Vegetation Gains & Losses ---
    # (Logic remains similar, assuming New Crust/Subducted don't have vegetation status)
    records_veg = []

    # Define Veg Sets
    VEG_CODES = {20, 21, 56}
    NON_VEG_LAND_CODES = {94, 98, 105}

    def is_veg(code):
        try:
            return int(float(code)) in VEG_CODES
        except:
            return False

    def is_non_veg_land(code):
        try:
            return int(float(code)) in NON_VEG_LAND_CODES
        except:
            return False

    long["is_source_veg"] = long["source_code"].apply(is_veg)
    long["is_dest_veg"] = long["dest_code"].apply(is_veg)
    long["is_source_non_veg"] = long["source_code"].apply(is_non_veg_land)
    long["is_dest_non_veg"] = long["dest_code"].apply(is_non_veg_land)

    veg_gain_mask = (long["is_dest_veg"] & (~long["is_source_veg"]))
    veg_loss_mask = (long["is_source_veg"] & (~long["is_dest_veg"]))

    if veg_gain_mask.any():
        vp = long[veg_gain_mask]["pixel_count"].sum()
        va = long[veg_gain_mask]["area_km2"].sum()
        records_veg.append({"Metric": "Vegetation Gain", "Pixels": vp, "Area_km2": va, "Type": "Gain"})

    if veg_loss_mask.any():
        vp = long[veg_loss_mask]["pixel_count"].sum()
        va = long[veg_loss_mask]["area_km2"].sum()
        records_veg.append({"Metric": "Vegetation Loss", "Pixels": vp, "Area_km2": va, "Type": "Loss"})

    df_veg = pd.DataFrame(records_veg)

    # --- 3. Top N Specific Conversions ---
    # Filter: Only rows where Dest Theme != Source Theme
    # Re-calculate broad themes for filtering
    long["source_theme_broad"] = "Other"
    long.loc[long["is_source_land"], "source_theme_broad"] = "Land"
    long.loc[long["is_source_ocean"], "source_theme_broad"] = "Ocean"
    long.loc[long["is_source_new_crust"], "source_theme_broad"] = "New Crust"
    long.loc[long["is_dest_subducted"], "dest_theme_broad"] = "Subducted"  # Helper for dest

    # Re-do dest theme for filtering
    long["dest_theme_broad"] = "Other"
    long.loc[long["is_dest_land"], "dest_theme_broad"] = "Land"
    long.loc[long["is_dest_ocean"], "dest_theme_broad"] = "Ocean"
    long.loc[long["is_dest_subducted"], "dest_theme_broad"] = "Subducted"

    change_mask = long["source_theme_broad"] != long["dest_theme_broad"]
    long_changes = long[change_mask].copy()

    if long_changes.empty:
        top_details = pd.DataFrame()
    else:
        long_changes_sorted = long_changes.sort_values(by=["area_km2"], ascending=False)
        top_details = long_changes_sorted.head(top_n)
        top_details = top_details.sort_values(by=["source_label", "area_km2"], ascending=[True, False])

    return df_broad, df_summary, df_veg, top_details


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

    # Conversion factors
    KM2_FACTOR = 1e6  # m² to km²
    KM3_FACTOR = 1e9  # m³ to km³

    # Define which columns are Areas (m²) and which are Volumes (m³)
    area_columns = [
        "Tropical_Land_Area", "Land_area", "Ocean_Area", "Southern_Land_Area",
        "Continental_Shelves_Area", "Temperate_Land_Area", "Polar_Land_Area",
        "High_Altitude_Area", "Northern_Land_Area", "Subtropical_Land_Area",
        "Coastline_length"  # Technically length (m), convert to km? Let's keep consistent with area logic if needed.
        # For now, treating length as m -> km (divide 1e3) or leave?
        # To match the "Total Land" comparison, we focus on Area.
        # Let's convert Length to km as well for consistency if it's large.
    ]

    volume_columns = ["Ocean_Volume"]
    length_columns = ["Coastline_length"]  # Adding explicit handling for length

    records = []
    for col in numeric_cols:
        v_from, v_to = row_from[col], row_to[col]

        # Apply conversions based on column type
        if col in area_columns:
            v_from_c = v_from / KM2_FACTOR
            v_to_c = v_to / KM2_FACTOR
            delta_c = (v_to - v_from) / KM2_FACTOR
        elif col in volume_columns:
            v_from_c = v_from / KM3_FACTOR
            v_to_c = v_to / KM3_FACTOR
            delta_c = (v_to - v_from) / KM3_FACTOR
        elif col in length_columns:
            v_from_c = v_from / 1e3  # m to km
            v_to_c = v_to / 1e3
            delta_c = (v_to - v_from) / 1e3
        else:
            # No conversion for CO2, Sea Level (m), Continents, Sediment flux, etc.
            v_from_c = v_from
            v_to_c = v_to
            delta_c = v_to - v_from

        pct = (delta_c / v_from_c * 100) if v_from_c not in (0, None) else float("nan")
        display_name = GEO_INDICATOR_LABELS.get(col, col)

        records.append({
            "indicator": display_name,
            "value_from": v_from_c,
            "value_to": v_to_c,
            "delta": delta_c,
            "pct_change": pct,
            "delta_abs": abs(delta_c)
        })

    out = pd.DataFrame(records)
    out = out.sort_values(by=["indicator", "delta_abs", "delta"], ascending=[True, False, False])
    return out.drop(columns=["delta_abs"])


# ============================================================================
# Markdown Generation
# ============================================================================

def fmt_num(x):
    if pd.isna(x):
        return "n/a"
    elif isinstance(x, float) and (abs(x) > 1e6 or (abs(x) < 1e-2 and x != 0)):
        return f"{x:.3e}"
    elif -10000 <= x <= 10000:
        return round(x, 2)
    else:
        return f"{x:,.0f}"


def fmt_pct(x):
    return f"{x:.2f}%" if pd.notna(x) else "n/a"


def gains_table(df: pd.DataFrame) -> list:
    if df.empty:
        return ["*No changes detected.*"]
    lines = ["| Metric | Pixels | Area (km²) |", "|-----|---:|---:|"]
    for _, r in df.iterrows():
        metric_name = r['Metric']
        if r.get('Type') == 'Loss':
            metric_name = f"**{metric_name}**"
        lines.append(f"| {metric_name} | {int(r['Pixels']):,} | {int(r['Area_km2']):,} |")
    return lines


def net_summary_table(df: pd.DataFrame) -> list:
    if df.empty:
        return ["*No net change.*"]
    lines = ["| **Net Change** | **Pixels** | **Area (km²)** |", "|-----|---:|---:|"]
    for _, r in df.iterrows():
        val_str = f"{int(r['Area_km2']):,}"
        lines.append(f"| **{r['Metric']}** | {int(r['Pixels']):,} | {val_str} |")
    return lines


def matrix_table(df: pd.DataFrame) -> list:
    if df.empty:
        return ["*No significant cross-boundary pixel transitions found.*"]
    lines = ["| Source Class | Dest Class | Pixels | Area (km²) |", "|-----|-----|---:|---:|"]
    for _, r in df.iterrows():
        lines.append(
            f"| {r['source_label']} | {r['dest_label']} | {int(r['pixel_count']):,} | {int(r['area_km2']):,} |")
    return lines


def indicator_table(df: pd.DataFrame) -> list:
    # Updated header to reflect mixed units (km² for areas, original for others)
    lines = ["| Indicator | Value @From | Value @To | Delta | % Change |", "|-----|---:|---:|---:|---:|"]
    for _, r in df.iterrows():
        lines.append(
            f"| {r['indicator']} | {fmt_num(r['value_from'])} | {fmt_num(r['value_to'])} | {fmt_num(r['delta'])} | {fmt_pct(r['pct_change'])} |")
    return lines


# ============================================================================
# Main
# ============================================================================

def main():
    OUTPUT_DIR.mkdir(exist_ok=True)
    geo_df = pd.read_csv(GEO_INDICATORS_CSV)

    matrix_files = find_matrix_files(MATRIX_DIR)
    if not matrix_files:
        print(f"No change_matrix_inverse_*.csv files found in {MATRIX_DIR}")
        return

    yaml_header = [
        "---",
        "title: 'Phanerozoic Transition Report (Inverse Method)'",
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
    lines.append("This report summarizes Earth surface transitions using the **Inverse (Gather)** method.")
    lines.append("Metrics focus on **Net Gains** (e.g., new Land appearing, new Ocean appearing).")
    lines.append("All tables report both **Pixel Count** and **Area (km²)**.")
    lines.append("**Note:** Geo-Indicator values for Area have been converted to km² to match transition tables.")
    lines.append("")
    lines.append("**Thematic Definitions:**")
    lines.append("- **Land Gain**: Pixels becoming Land (from Ocean, No Data, or Unmapped).")
    lines.append("- **Ocean Gain**: Pixels becoming Ocean (from Land, No Data, or Unmapped).")
    lines.append("- **Vegetation Gain**: Pixels becoming Vegetated (from Non-Vegetated or Unmapped).")
    lines.append("")

    for age_from, age_to, band, path in matrix_files:
        print(f"Processing transition {age_from} Ma -> {age_to} Ma")
        lines.append(f"## {age_from} Ma -> {age_to} Ma")
        lines.append("")

        pixel_area_m2 = get_pixel_area_m2(age_from)
        df_broad, df_summary, df_veg, top_details = process_transitions_gains(path, pixel_area_m2, TOP_N_PIXEL_ROWS)

        # 1. Broad Gains & Losses
        lines.append("### 1. Broad Thematic Gains & Losses")
        lines.append("")
        lines.extend(gains_table(df_broad))
        lines.append("")

        # 1b. Net Summary
        lines.append("#### Net Change Summary (Should match Geo-Indicators)")
        lines.append("")
        lines.extend(net_summary_table(df_summary))
        lines.append("")

        # 2. Vegetation Gains & Losses
        lines.append("### 2. Vegetation Status Gains & Losses")
        lines.append("")
        lines.extend(gains_table(df_veg))
        lines.append("")

        # 3. Top N Specific Conversions
        lines.append(f"### 3. Top {TOP_N_PIXEL_ROWS} Specific Class Conversions")
        lines.append("")
        lines.extend(matrix_table(top_details))
        lines.append("")

        # 4. Indicators
        try:
            deltas = indicator_deltas(geo_df, age_from, age_to)
            lines.append("### 4. Geo-Indicator Deltas (Converted to km²/km)")
            lines.append("")
            lines.extend(indicator_table(deltas))
        except ValueError as e:
            lines.append(f"*No matching geo-indicator rows found: {e}*")

        lines.append("")
        lines.append("\\newpage")
        lines.append("")

    output_path = OUTPUT_DIR / "transition_report_inverse.md"
    output_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nDone. Wrote {output_path.resolve()}")


if __name__ == "__main__":
    main()