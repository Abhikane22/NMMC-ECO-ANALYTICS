
import pandas as pd
import numpy as np

from pathlib import Path

from config import (
    PROJECT_ROOT,
    REPORTS_DIR,
    YEARS,
)

LULC_FILE = (
    REPORTS_DIR / "NMMC_Ward_Level_LULC_Statistics.csv"
)

ECO_FILE = (
    REPORTS_DIR / "NMMC_Ward_Level_Ecological_Statistics.csv"
)

CLASS_NAMES = [
    "Water",
    "Vegetation",
    "Built-up",
    "Barren",
    "Coastal Wetlands",
]

INDICATORS = [
    "ndvi", "ndbi", "ndmi", "ndwi",
    "mndwi", "lst_c", "utfvi",
]


def check(condition, description):
    status = "PASS" if condition else "FAIL"
    print(f"[{status}] {description}")
    return bool(condition)


def main():
    print("\nNMMC PHASE 1 — FINAL VALIDATION")
    print("=" * 65)

    lulc = pd.read_csv(LULC_FILE)
    eco = pd.read_csv(ECO_FILE)

    checks = []

    # Record counts and unique wards
    checks.append(check(
        len(lulc) == 444,
        "LULC contains 444 ward-year records"
    ))

    checks.append(check(
        len(eco) == 444,
        "Ecological data contains 444 ward-year records"
    ))

    checks.append(check(
        lulc["ward_id"].nunique() == 111
        and eco["ward_id"].nunique() == 111,
        "Both datasets contain 111 unique wards"
    ))

    checks.append(check(
        set(lulc["year"].unique()) == set(YEARS)
        and set(eco["year"].unique()) == set(YEARS),
        "All four study years are present"
    ))

    checks.append(check(
        not lulc.duplicated(["year", "ward_id"]).any()
        and not eco.duplicated(["year", "ward_id"]).any(),
        "No duplicate ward-year records"
    ))

    # Verify the datasets cover the same ward-year combinations.
    lulc_keys = set(zip(lulc.year, lulc.ward_id))
    eco_keys = set(zip(eco.year, eco.ward_id))

    checks.append(check(
        lulc_keys == eco_keys,
        "LULC and ecological records align"
    ))

    # LULC area accounting
    class_columns = [
        f"{name}_area_km2" for name in CLASS_NAMES
    ]

    lulc_difference = (
        lulc["valid_raster_area_km2"]
        - lulc[class_columns].sum(axis=1)
    )

    checks.append(check(
        np.allclose(lulc_difference, 0, atol=1e-8),
        "LULC class areas equal valid raster area"
    ))

    # Ecological coverage and numeric values
    for indicator in INDICATORS:
        coverage = f"{indicator}_coverage_pct"
        mean_column = f"{indicator}_mean"

        checks.append(check(
            eco[coverage].between(0, 100.01).all(),
            f"{indicator.upper()} coverage within valid range"
        ))

        checks.append(check(
            np.isfinite(eco[mean_column]).all(),
            f"{indicator.upper()} means contain finite values"
        ))

    # Area-weighted summaries (ward aggregation).
    # Small ward overlaps mean these totals can include
    # negligible double-counting; see boundary audit.
    summary_rows = []

    for year, group in eco.groupby("year"):
        row = {"year": int(year)}

        for indicator in INDICATORS:
            weights = group[
                f"{indicator}_valid_area_km2"
            ]

            row[f"{indicator}_area_weighted_mean"] = np.average(
                group[f"{indicator}_mean"],
                weights=weights,
            )

        summary_rows.append(row)

    summary = pd.DataFrame(summary_rows)

    summary_path = (
        REPORTS_DIR / "NMMC_Phase1_Ecological_Summary.csv"
    )

    summary.to_csv(summary_path, index=False)

    print("\n" + "=" * 65)
    print("AREA-WEIGHTED ECOLOGICAL SUMMARY")
    print("=" * 65)
    print(summary.round(5).to_string(index=False))

    passed = sum(checks)
    failed = len(checks) - passed

    print("\n" + "=" * 65)
    print(f"VALIDATION RESULTS: {passed} PASS, {failed} FAIL")
    print("Summary file:", summary_path)

    if failed:
        raise SystemExit("Phase 1 validation requires investigation.")

    print("\nPHASE 1 DATA VALIDATION: SUCCESS")


if __name__ == "__main__":
    main()
