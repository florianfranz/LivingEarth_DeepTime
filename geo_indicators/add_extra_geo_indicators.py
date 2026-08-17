"""
Add sea-level change (dSL), sediment flux (BQART/RoBART), and CO2 (COPSE
max CO2) to the geo indicators CSV, all joined by age.

Inputs:
  SEA_LEVEL_TXT       Sea_level_v0_000-545.txt - columns:
                       age, initial_volume, initial_area, z_full_volume,
                       area_full_volume, dSL, subsidence
  SEDIMENT_FLUX_CSV   sediment_flux_series_D.csv - long-format columns:
                       series, series_label, age_Ma, model, TSS_low,
                       TSS_est, TSS_high, Q_discharge (model = bqart/robart)
  CO2_CSV             copse_maxCO2_atlas.csv - columns: age_ma, co2_ppm
  GEO_INDICATORS_CSV  the existing Age/Land_area/... CSV

Output:
  GEO_INDICATORS_CSV is overwritten in place with the new columns added.
  Safe to re-run any time an input file or the indicators CSV changes -
  any columns added by a previous run are dropped before re-joining, so
  it's always a clean re-join rather than a stale merge.

Requires: pandas
"""

import pandas as pd

# ============================================================================
# Configuration
# ============================================================================

SEA_LEVEL_TXT = "Sea_level_v0_000-545.txt"
SEDIMENT_FLUX_CSV = "sediment_flux_series_D.csv"
CO2_CSV = "copse_maxCO2_atlas.csv"
GEO_INDICATORS_CSV = "output/geo_indicators.csv"

AGE_MATCH_TOL = 0.5  # Ma tolerance when matching ages between files
SEA_LEVEL_COLUMN = "Sea_Level_dSL"
CO2_COLUMN = "CO2_ppm"


def load_sea_level():
    sea_level = pd.read_csv(SEA_LEVEL_TXT, skipinitialspace=True)
    sea_level.columns = [c.strip() for c in sea_level.columns]

    if "age" not in sea_level.columns or "dSL" not in sea_level.columns:
        raise ValueError(
            f"Expected 'age' and 'dSL' columns in {SEA_LEVEL_TXT}, "
            f"found: {list(sea_level.columns)}"
        )
    return sea_level[["age", "dSL"]].rename(
        columns={"age": "Age", "dSL": SEA_LEVEL_COLUMN}
    )


def load_sediment_flux():
    flux = pd.read_csv(SEDIMENT_FLUX_CSV)

    required = {"age_Ma", "model", "TSS_est"}
    missing = required - set(flux.columns)
    if missing:
        raise ValueError(
            f"Expected columns {sorted(required)} in {SEDIMENT_FLUX_CSV}, "
            f"missing: {sorted(missing)}"
        )

    flux = flux[flux["model"].isin(["bqart", "robart"])].copy()

    # Long -> wide: one row per age, one TSS_est column per model
    pivoted = flux.pivot(index="age_Ma", columns="model", values="TSS_est")
    pivoted.columns = [f"{model.upper()}_TSS_est" for model in pivoted.columns]
    pivoted = pivoted.reset_index().rename(columns={"age_Ma": "Age"})

    return pivoted


def load_co2():
    co2 = pd.read_csv(CO2_CSV)
    co2.columns = [c.strip() for c in co2.columns]

    if "age_ma" not in co2.columns or "co2_ppm" not in co2.columns:
        raise ValueError(
            f"Expected 'age_ma' and 'co2_ppm' columns in {CO2_CSV}, "
            f"found: {list(co2.columns)}"
        )
    return co2[["age_ma", "co2_ppm"]].rename(
        columns={"age_ma": "Age", "co2_ppm": CO2_COLUMN}
    )


def asof_join(base, other, tol=AGE_MATCH_TOL):
    """Nearest-age join with tolerance, preserving base's row order."""
    base = base.copy()
    base["Age"] = base["Age"].astype(float)
    other = other.copy()
    other["Age"] = other["Age"].astype(float)

    merged = pd.merge_asof(
        base.sort_values("Age"),
        other.sort_values("Age"),
        on="Age",
        direction="nearest",
        tolerance=tol,
    ).sort_index()
    return merged


def report_missing(df, column, source_label):
    n_missing = df[column].isna().sum()
    if n_missing:
        missing_ages = df.loc[df[column].isna(), "Age"].tolist()
        print(f"[warning] {n_missing} age(s) had no match within "
              f"{AGE_MATCH_TOL} Ma in {source_label}: {missing_ages}")


def main():
    geo_df = pd.read_csv(GEO_INDICATORS_CSV)
    geo_df["Age"] = geo_df["Age"].astype(float)

    sea_level = load_sea_level()
    sediment_flux = load_sediment_flux()
    co2 = load_co2()

    # Drop any columns added by a previous run so re-running is always a
    # clean re-join, not a collision with stale data already in the CSV.
    cols_to_add = (
        [SEA_LEVEL_COLUMN]
        + [c for c in sediment_flux.columns if c != "Age"]
        + [CO2_COLUMN]
    )
    geo_df = geo_df.drop(columns=[c for c in cols_to_add if c in geo_df.columns])

    merged = asof_join(geo_df, sea_level)
    report_missing(merged, SEA_LEVEL_COLUMN, SEA_LEVEL_TXT)

    merged = asof_join(merged, sediment_flux)
    flux_cols = [c for c in sediment_flux.columns if c != "Age"]
    for col in flux_cols:
        report_missing(merged, col, SEDIMENT_FLUX_CSV)

    merged = asof_join(merged, co2)
    report_missing(merged, CO2_COLUMN, CO2_CSV)

    merged.to_csv(GEO_INDICATORS_CSV, index=False)
    print(f"Added sea-level, sediment flux (BQART/RoBART), and CO2 columns; "
          f"wrote {GEO_INDICATORS_CSV}")
    print(merged[["Age", SEA_LEVEL_COLUMN] + flux_cols + [CO2_COLUMN]].to_string(index=False))


if __name__ == "__main__":
    main()