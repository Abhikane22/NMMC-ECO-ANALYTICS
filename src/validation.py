
from pathlib import Path
import json

import numpy as np
import pandas as pd
import rasterio

from src.config import (
    YEARS,
    LULC_DIR,
    LULC_CLASSES,
    REPORTS_DIR,
    ECOLOGICAL_DIR,
)


# =========================================================
# 1. CONFIGURATION
# =========================================================

LULC_FILES = {
    1994: LULC_DIR / "NMMC_LULC_1994_PRELIMINARY.tif",
    2004: LULC_DIR / "NMMC_LULC_2004_PRELIMINARY.tif",
    2014: LULC_DIR / "NMMC_LULC_2014_PRELIMINARY.tif",
    2024: LULC_DIR / "NMMC_2024_LULC_SCIENTIFIC_FINAL.tif",
}

GEE_AREA_FILE = (
    LULC_DIR / "NMMC_2024_LULC_AREA_FINAL_2.csv"
)

ANNUAL_LULC_FILE = (
    REPORTS_DIR / "NMMC_LULC_annual_area_statistics.csv"
)

TRANSITION_FILE = (
    REPORTS_DIR / "NMMC_LULC_all_transition_details.csv"
)

CLASS_CHANGES_FILE = (
    REPORTS_DIR / "NMMC_LULC_class_gains_losses.csv"
)

ECOLOGICAL_ANNUAL_FILE = (
    REPORTS_DIR / "ecological_annual_raster_statistics.csv"
)

EXPECTED_CLASSES = set(LULC_CLASSES.keys())
NODATA_VALUE = 255

TOLERANCE = 1e-5

REPORTS_DIR.mkdir(parents=True, exist_ok=True)


# =========================================================
# 2. VALIDATION RESULT COLLECTOR
# =========================================================

class ValidationReport:

    def __init__(self):

        self.results = []

    def add(
        self,
        category,
        test,
        status,
        message,
    ):

        self.results.append({
            "category": category,
            "test": test,
            "status": status,
            "message": str(message),
        })

        print(
            f"[{status}] {category} | {test}: {message}"
        )

    def dataframe(self):

        return pd.DataFrame(self.results)

    def summary(self):

        df = self.dataframe()

        if df.empty:
            return {
                "PASS": 0,
                "WARNING": 0,
                "FAIL": 0,
                "INFO": 0,
            }

        return {
            status: int((df["status"] == status).sum())
            for status in ["PASS", "WARNING", "FAIL", "INFO"]
        }


# =========================================================
# 3. RASTER VALIDATION
# =========================================================

def validate_lulc_rasters(report):

    print("\n" + "=" * 65)
    print("1. LULC RASTER VALIDATION")
    print("=" * 65)

    metadata = {}
    rasters = {}

    reference = None

    for year in YEARS:

        path = LULC_FILES[year]

        if not path.exists():

            report.add(
                "Raster",
                f"{year}_file_exists",
                "FAIL",
                f"Missing file: {path}",
            )

            continue

        try:

            with rasterio.open(path) as src:

                data = src.read(1, masked=True)

                info = {
                    "crs": src.crs,
                    "transform": src.transform,
                    "width": src.width,
                    "height": src.height,
                    "resolution": src.res,
                    "nodata": src.nodata,
                    "bounds": src.bounds,
                }

            metadata[year] = info
            rasters[year] = data

            valid = data.compressed()

            actual_classes = set(
                np.unique(valid).astype(int).tolist()
            )

            unexpected = actual_classes - EXPECTED_CLASSES

            if valid.size == 0:

                report.add(
                    "Raster",
                    f"{year}_valid_pixels",
                    "FAIL",
                    "Raster contains no valid pixels.",
                )

            else:

                report.add(
                    "Raster",
                    f"{year}_valid_pixels",
                    "PASS",
                    f"{valid.size:,} valid pixels.",
                )

            if unexpected:

                report.add(
                    "Raster",
                    f"{year}_class_codes",
                    "FAIL",
                    f"Unexpected codes: {unexpected}",
                )

            else:

                report.add(
                    "Raster",
                    f"{year}_class_codes",
                    "PASS",
                    f"Observed classes: {sorted(actual_classes)}",
                )

            if info["nodata"] != NODATA_VALUE:

                report.add(
                    "Raster",
                    f"{year}_nodata",
                    "WARNING",
                    f"Expected {NODATA_VALUE}; found {info['nodata']}.",
                )

            else:

                report.add(
                    "Raster",
                    f"{year}_nodata",
                    "PASS",
                    f"NoData = {NODATA_VALUE}.",
                )

            if reference is None:

                reference = info

            else:

                checks = {
                    "CRS": info["crs"] == reference["crs"],
                    "Transform": (
                        info["transform"] == reference["transform"]
                    ),
                    "Width": info["width"] == reference["width"],
                    "Height": info["height"] == reference["height"],
                    "Bounds": info["bounds"] == reference["bounds"],
                }

                for name, passed in checks.items():

                    report.add(
                        "Alignment",
                        f"{year}_{name}",
                        "PASS" if passed else "FAIL",
                        (
                            "Matches reference grid."
                            if passed
                            else "Does not match reference grid."
                        ),
                    )

        except Exception as exc:

            report.add(
                "Raster",
                f"{year}_read",
                "FAIL",
                str(exc),
            )

    return rasters, metadata


# =========================================================
# 4. ANNUAL LULC STATISTICS VALIDATION
# =========================================================

def validate_annual_lulc(report, rasters):

    print("\n" + "=" * 65)
    print("2. ANNUAL LULC STATISTICS VALIDATION")
    print("=" * 65)

    if not ANNUAL_LULC_FILE.exists():

        report.add(
            "Annual LULC",
            "report_exists",
            "FAIL",
            f"Missing file: {ANNUAL_LULC_FILE}",
        )

        return None

    annual = pd.read_csv(ANNUAL_LULC_FILE)

    required = {
        "year",
        "class_code",
        "pixel_count",
        "area_km2",
        "percentage",
    }

    missing = required - set(annual.columns)

    if missing:

        report.add(
            "Annual LULC",
            "required_columns",
            "FAIL",
            f"Missing columns: {missing}",
        )

        return None

    report.add(
        "Annual LULC",
        "report_read",
        "PASS",
        f"Loaded {len(annual)} records.",
    )

    for year in YEARS:

        if year not in rasters:
            continue

        subset = annual[annual["year"] == year]

        if len(subset) != len(EXPECTED_CLASSES):

            report.add(
                "Annual LULC",
                f"{year}_record_count",
                "FAIL",
                f"Expected 5 class records; found {len(subset)}.",
            )

            continue

        observed_codes = set(
            subset["class_code"].astype(int)
        )

        if observed_codes != EXPECTED_CLASSES:

            report.add(
                "Annual LULC",
                f"{year}_class_records",
                "FAIL",
                f"Unexpected class records: {observed_codes}",
            )

        else:

            report.add(
                "Annual LULC",
                f"{year}_class_records",
                "PASS",
                "All five class records present.",
            )

        percentage_sum = subset["percentage"].sum()

        if np.isclose(
            percentage_sum,
            100.0,
            atol=0.01,
        ):

            report.add(
                "Annual LULC",
                f"{year}_percentage_sum",
                "PASS",
                f"Percentage sum = {percentage_sum:.4f}%.",
            )

        else:

            report.add(
                "Annual LULC",
                f"{year}_percentage_sum",
                "FAIL",
                f"Percentage sum = {percentage_sum:.4f}%.",
            )

        raster = rasters[year]

        valid_values = raster.compressed()

        for code in EXPECTED_CLASSES:

            expected_count = int(
                np.count_nonzero(valid_values == code)
            )

            row = subset[
                subset["class_code"] == code
            ]

            if row.empty:
                continue

            reported_count = int(
                row.iloc[0]["pixel_count"]
            )

            passed = expected_count == reported_count

            report.add(
                "Annual LULC",
                f"{year}_class_{code}_pixel_count",
                "PASS" if passed else "FAIL",
                (
                    f"Raster={expected_count}, "
                    f"report={reported_count}."
                ),
            )

    return annual


# =========================================================
# 5. TRANSITION MATRIX VALIDATION
# =========================================================

def validate_transitions(report, rasters):

    print("\n" + "=" * 65)
    print("3. TRANSITION MATRIX VALIDATION")
    print("=" * 65)

    if not TRANSITION_FILE.exists():

        report.add(
            "Transitions",
            "report_exists",
            "FAIL",
            f"Missing file: {TRANSITION_FILE}",
        )

        return None

    transitions = pd.read_csv(TRANSITION_FILE)

    required = {
        "start_year",
        "end_year",
        "from_class_code",
        "to_class_code",
        "pixel_count",
        "area_km2",
    }

    missing = required - set(transitions.columns)

    if missing:

        report.add(
            "Transitions",
            "required_columns",
            "FAIL",
            f"Missing columns: {missing}",
        )

        return None

    report.add(
        "Transitions",
        "report_read",
        "PASS",
        f"Loaded {len(transitions)} transition records.",
    )

    periods = [
        (1994, 2004),
        (2004, 2014),
        (2014, 2024),
        (1994, 2024),
    ]

    for start_year, end_year in periods:

        if start_year not in rasters or end_year not in rasters:
            continue

        subset = transitions[
            (transitions["start_year"] == start_year)
            & (transitions["end_year"] == end_year)
        ]

        expected_pairs = len(EXPECTED_CLASSES) ** 2

        if len(subset) != expected_pairs:

            report.add(
                "Transitions",
                f"{start_year}_{end_year}_record_count",
                "FAIL",
                f"Expected {expected_pairs} pairs; found {len(subset)}.",
            )

            continue

        pair_keys = list(zip(
            subset["from_class_code"].astype(int),
            subset["to_class_code"].astype(int),
        ))

        if len(set(pair_keys)) != expected_pairs:

            report.add(
                "Transitions",
                f"{start_year}_{end_year}_unique_pairs",
                "FAIL",
                "Duplicate or missing transition pairs.",
            )

            continue

        old = rasters[start_year]
        new = rasters[end_year]

        old_values = np.asarray(old.data)
        new_values = np.asarray(new.data)

        common_mask = (
            ~np.ma.getmaskarray(old)
            & ~np.ma.getmaskarray(new)
            & np.isin(old_values, list(EXPECTED_CLASSES))
            & np.isin(new_values, list(EXPECTED_CLASSES))
        )

        old_common = old_values[common_mask]
        new_common = new_values[common_mask]

        total_common = int(common_mask.sum())

        reported_total = int(
            subset["pixel_count"].sum()
        )

        passed = total_common == reported_total

        report.add(
            "Transitions",
            f"{start_year}_{end_year}_common_pixels",
            "PASS" if passed else "FAIL",
            (
                f"Raster={total_common:,}, "
                f"transition report={reported_total:,}."
            ),
        )

        matrix_counts = np.zeros(
            (5, 5),
            dtype=np.int64,
        )

        for from_code in EXPECTED_CLASSES:

            for to_code in EXPECTED_CLASSES:

                matrix_counts[from_code, to_code] = np.count_nonzero(
                    (old_common == from_code)
                    & (new_common == to_code)
                )

        # Compare transition report counts with direct raster counts.
        for _, row in subset.iterrows():

            from_code = int(row["from_class_code"])
            to_code = int(row["to_class_code"])

            actual = int(
                matrix_counts[from_code, to_code]
            )

            reported = int(row["pixel_count"])

            passed = actual == reported

            report.add(
                "Transitions",
                (
                    f"{start_year}_{end_year}_"
                    f"{from_code}_to_{to_code}"
                ),
                "PASS" if passed else "FAIL",
                f"Raster={actual}, report={reported}.",
            )

        # Check row and column sums.
        row_totals = matrix_counts.sum(axis=1)
        col_totals = matrix_counts.sum(axis=0)

        expected_source = np.bincount(
            old_common.astype(int),
            minlength=5,
        )

        expected_destination = np.bincount(
            new_common.astype(int),
            minlength=5,
        )

        report.add(
            "Transitions",
            f"{start_year}_{end_year}_row_sums",
            (
                "PASS"
                if np.array_equal(row_totals, expected_source)
                else "FAIL"
            ),
            "Transition rows match source-class pixel counts.",
        )

        report.add(
            "Transitions",
            f"{start_year}_{end_year}_column_sums",
            (
                "PASS"
                if np.array_equal(col_totals, expected_destination)
                else "FAIL"
            ),
            "Transition columns match destination-class pixel counts.",
        )

    return transitions


# =========================================================
# 6. GAIN / LOSS VALIDATION
# =========================================================

def validate_class_changes(report):

    print("\n" + "=" * 65)
    print("4. CLASS GAIN / LOSS VALIDATION")
    print("=" * 65)

    if not CLASS_CHANGES_FILE.exists():

        report.add(
            "Gain/Loss",
            "report_exists",
            "FAIL",
            f"Missing file: {CLASS_CHANGES_FILE}",
        )

        return None

    changes = pd.read_csv(CLASS_CHANGES_FILE)

    required = {
        "start_year",
        "end_year",
        "class_code",
        "gain_km2",
        "loss_km2",
        "net_change_km2",
    }

    missing = required - set(changes.columns)

    if missing:

        report.add(
            "Gain/Loss",
            "required_columns",
            "FAIL",
            f"Missing columns: {missing}",
        )

        return None

    report.add(
        "Gain/Loss",
        "report_read",
        "PASS",
        f"Loaded {len(changes)} records.",
    )

    calculated_net = (
        changes["gain_km2"]
        - changes["loss_km2"]
    )

    difference = (
        calculated_net
        - changes["net_change_km2"]
    ).abs()

    passed = bool(
        (difference <= TOLERANCE).all()
    )

    report.add(
        "Gain/Loss",
        "gain_minus_loss",
        "PASS" if passed else "FAIL",
        (
            "Gain - Loss equals Net Change."
            if passed
            else "Gain/Loss arithmetic mismatch detected."
        ),
    )

    for start_year, end_year in [
        (1994, 2004),
        (2004, 2014),
        (2014, 2024),
        (1994, 2024),
    ]:

        subset = changes[
            (changes["start_year"] == start_year)
            & (changes["end_year"] == end_year)
        ]

        if len(subset) != len(EXPECTED_CLASSES):

            report.add(
                "Gain/Loss",
                f"{start_year}_{end_year}_class_count",
                "FAIL",
                f"Expected 5 classes; found {len(subset)}.",
            )

    return changes


# =========================================================
# 7. 2024 GEE AREA COMPARISON
# =========================================================

def validate_gee_comparison(report, annual):

    print("\n" + "=" * 65)
    print("5. 2024 GEE AREA COMPARISON")
    print("=" * 65)

    if not GEE_AREA_FILE.exists():

        report.add(
            "GEE Comparison",
            "csv_exists",
            "WARNING",
            "2024 GEE area CSV not found.",
        )

        return None

    gee = pd.read_csv(GEE_AREA_FILE)

    required = {
        "class_code",
        "area_km2",
        "percentage",
    }

    missing = required - set(gee.columns)

    if missing:

        report.add(
            "GEE Comparison",
            "required_columns",
            "FAIL",
            f"Missing columns: {missing}",
        )

        return None

    local = annual[
        annual["year"] == 2024
    ][
        ["class_code", "area_km2", "percentage"]
    ]

    comparison = local.merge(
        gee[
            ["class_code", "area_km2", "percentage"]
        ],
        on="class_code",
        suffixes=("_local", "_gee"),
        validate="one_to_one",
    )

    comparison["area_difference_km2"] = (
        comparison["area_km2_local"]
        - comparison["area_km2_gee"]
    )

    comparison["percentage_difference_points"] = (
        comparison["percentage_local"]
        - comparison["percentage_gee"]
    )

    comparison.to_csv(
        REPORTS_DIR / "NMMC_2024_GEE_area_comparison_validation.csv",
        index=False,
    )

    report.add(
        "GEE Comparison",
        "comparison_generated",
        "PASS",
        "Local and GEE area statistics compared.",
    )

    percentage_difference = (
        comparison["percentage_difference_points"].abs().max()
    )

    report.add(
        "GEE Comparison",
        "percentage_difference",
        "INFO",
        (
            f"Maximum class percentage difference: "
            f"{percentage_difference:.4f} percentage points. "
            "Investigate grid and boundary-area methodology."
        ),
    )

    print(
        comparison.round(4).to_string(index=False)
    )

    return comparison


# =========================================================
# 8. ECOLOGICAL REPORT VALIDATION
# =========================================================

def validate_ecological_reports(report):

    print("\n" + "=" * 65)
    print("6. ECOLOGICAL REPORT VALIDATION")
    print("=" * 65)

    if not ECOLOGICAL_ANNUAL_FILE.exists():

        report.add(
            "Ecological",
            "annual_report_exists",
            "WARNING",
            "Ecological annual statistics CSV not found.",
        )

        return

    annual = pd.read_csv(ECOLOGICAL_ANNUAL_FILE)

    expected_indicators = {
        "NDVI",
        "NDBI",
        "NDMI",
        "NDWI",
        "MNDWI",
        "LST_C",
        "UTFVI",
    }

    actual_indicators = set(
        annual["indicator"].unique()
    )

    missing = expected_indicators - actual_indicators

    if missing:

        report.add(
            "Ecological",
            "indicator_coverage",
            "WARNING",
            f"Missing indicators: {missing}",
        )

    else:

        report.add(
            "Ecological",
            "indicator_coverage",
            "PASS",
            "All seven ecological indicators present.",
        )

    expected_rows = len(YEARS) * len(expected_indicators)

    report.add(
        "Ecological",
        "annual_record_count",
        (
            "PASS"
            if len(annual) == expected_rows
            else "WARNING"
        ),
        f"Found {len(annual)} records; expected {expected_rows}.",
    )

    # Check whether NDBI and NDMI are redundant.
    for year in YEARS:

        subset = annual[
            annual["year"] == year
        ]

        ndbi = subset[
            subset["indicator"] == "NDBI"
        ]["mean"]

        ndmi = subset[
            subset["indicator"] == "NDMI"
        ]["mean"]

        if not ndbi.empty and not ndmi.empty:

            total = abs(
                float(ndbi.iloc[0]) + float(ndmi.iloc[0])
            )

            if total <= TOLERANCE:

                report.add(
                    "Ecological",
                    f"{year}_NDBI_NDMI_relationship",
                    "WARNING",
                    "NDBI and NDMI means are near exact negatives.",
                )


# =========================================================
# 9. SAVE FINAL VALIDATION REPORT
# =========================================================

def save_validation_report(report):

    results = report.dataframe()

    summary = report.summary()

    results_path = (
        REPORTS_DIR / "NMMC_validation_results.csv"
    )

    summary_path = (
        REPORTS_DIR / "NMMC_validation_summary.json"
    )

    results.to_csv(
        results_path,
        index=False,
    )

    overall_status = (
        "FAIL"
        if summary["FAIL"] > 0
        else "PASS"
    )

    summary_data = {
        "project": "NMMC LULC and Ecological Impact Analysis",
        "years": YEARS,
        "overall_status": overall_status,
        "summary": summary,
        "total_checks": len(results),
        "validation_scope": (
            "Internal data quality and analytical consistency. "
            "Does not establish independent historical classification accuracy."
        ),
    }

    with open(summary_path, "w") as file:

        json.dump(
            summary_data,
            file,
            indent=4,
        )

    print("\n" + "=" * 65)
    print("FINAL VALIDATION SUMMARY")
    print("=" * 65)

    print(f"Overall status: {overall_status}")

    for status, count in summary.items():

        print(f"{status}: {count}")

    print(f"\nTotal checks: {len(results)}")

    print(f"\nSaved: {results_path}")
    print(f"Saved: {summary_path}")

    return summary_data


# =========================================================
# 10. MAIN VALIDATION PIPELINE
# =========================================================

def run_validation():

    print("\n" + "=" * 65)
    print("NMMC PROJECT DATA VALIDATION")
    print("=" * 65)

    report = ValidationReport()

    rasters, metadata = validate_lulc_rasters(report)

    if len(rasters) == len(YEARS):

        annual = validate_annual_lulc(
            report,
            rasters,
        )

        validate_transitions(
            report,
            rasters,
        )

        validate_class_changes(report)

        if annual is not None:

            validate_gee_comparison(
                report,
                annual,
            )

    else:

        report.add(
            "Pipeline",
            "complete_raster_set",
            "FAIL",
            "Not all four LULC rasters could be loaded.",
        )

    validate_ecological_reports(report)

    return save_validation_report(report)


# =========================================================
# 11. EXECUTION
# =========================================================

if __name__ == "__main__":
    run_validation()
