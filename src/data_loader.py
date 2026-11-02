
from pathlib import Path
import json

import numpy as np
import pandas as pd
import rasterio

from src.config import (
    PROJECT_ROOT,
    LULC_DIR,
    ECOLOGICAL_DIR,
    STATISTICS_DIR,
    OUTPUT_DIR,
    REPORTS_DIR,
    YEARS,
)


# ============================================================
# EXPECTED DATASET CONFIGURATION
# ============================================================

OPTICAL_DIR = ECOLOGICAL_DIR / "optical_indices"
THERMAL_DIR = ECOLOGICAL_DIR / "thermal_indicators"

OPTICAL_BANDS = ["NDVI", "NDBI", "NDMI", "NDWI", "MNDWI"]
THERMAL_BANDS = ["LST_C", "UTFVI"]

EXPECTED_CRS = "EPSG:32643"
EXPECTED_NODATA = -9999.0

EXPECTED_RESOLUTIONS = {
    "optical": (30.0, 30.0),
    "thermal": (120.0, 120.0),
}

STATISTICS_FILE = (
    STATISTICS_DIR / "NMMC_Annual_Ecological_Statistics.csv"
)


# ============================================================
# GENERAL HELPERS
# ============================================================

def check(condition: bool, message: str, errors: list):
    """Add a validation error when a condition is false."""
    if not condition:
        errors.append(message)


def get_expected_raster_files():
    """Return the eight expected ecological raster paths."""

    files = []

    for year in YEARS:
        files.append({
            "year": year,
            "type": "optical",
            "path": OPTICAL_DIR / f"NMMC_Optical_Indices_{year}.tif",
            "bands": OPTICAL_BANDS,
            "resolution": EXPECTED_RESOLUTIONS["optical"],
        })

        files.append({
            "year": year,
            "type": "thermal",
            "path": THERMAL_DIR / f"NMMC_Thermal_Indicators_{year}.tif",
            "bands": THERMAL_BANDS,
            "resolution": EXPECTED_RESOLUTIONS["thermal"],
        })

    return files


# ============================================================
# RASTER VALIDATION
# ============================================================

def inspect_raster(spec: dict) -> dict:
    """Inspect raster metadata and validate actual pixel values."""

    path = spec["path"]
    errors = []
    warnings = []

    report = {
        "file": path.name,
        "year": spec["year"],
        "type": spec["type"],
        "exists": path.exists(),
        "valid": False,
        "errors": errors,
        "warnings": warnings,
    }

    if not path.exists():
        errors.append("File not found.")
        return report

    try:
        with rasterio.open(path) as src:

            report.update({
                "width": src.width,
                "height": src.height,
                "bands": src.count,
                "crs": str(src.crs),
                "resolution": list(src.res),
                "dtype": list(src.dtypes),
                "nodata": src.nodata,
                "bounds": list(src.bounds),
                "transform": list(src.transform),
                "band_descriptions": list(src.descriptions),
                "size_mb": round(
                    path.stat().st_size / (1024 ** 2), 3
                ),
            })

            # ------------------------------------------------
            # METADATA VALIDATION
            # ------------------------------------------------

            check(
                src.count == len(spec["bands"]),
                f"Expected {len(spec['bands'])} bands; found {src.count}.",
                errors,
            )

            check(
                str(src.crs) == EXPECTED_CRS,
                f"Expected CRS {EXPECTED_CRS}; found {src.crs}.",
                errors,
            )

            check(
                np.allclose(
                    src.res,
                    spec["resolution"],
                    atol=0.01,
                ),
                f"Unexpected resolution: {src.res}.",
                errors,
            )

            check(
                src.nodata is not None
                and np.isclose(src.nodata, EXPECTED_NODATA),
                f"Expected NoData {EXPECTED_NODATA}; found {src.nodata}.",
                errors,
            )

            check(
                all(dtype == "float32" for dtype in src.dtypes),
                f"Expected Float32 bands; found {src.dtypes}.",
                errors,
            )

            descriptions = list(src.descriptions)

            check(
                descriptions == spec["bands"],
                f"Expected band descriptions {spec['bands']}; "
                f"found {descriptions}.",
                errors,
            )

            # ------------------------------------------------
            # PIXEL VALIDATION
            # ------------------------------------------------

            band_reports = []

            for band_index, band_name in enumerate(
                spec["bands"], start=1
            ):

                data = src.read(band_index, masked=True)

                values = data.compressed()

                finite_mask = np.isfinite(values)

                finite_values = values[finite_mask]

                invalid_finite_count = int(
                    len(values) - len(finite_values)
                )

                band_report = {
                    "band": band_name,
                    "valid_pixels": int(len(values)),
                    "non_finite_valid_pixels": invalid_finite_count,
                    "nodata_pixels": int(
                        np.ma.getmaskarray(data).sum()
                    ),
                }

                check(
                    invalid_finite_count == 0,
                    f"{band_name}: non-finite values found in "
                    "unmasked pixels.",
                    errors,
                )

                if len(finite_values) > 0:

                    band_report.update({
                        "min": float(finite_values.min()),
                        "max": float(finite_values.max()),
                        "mean": float(finite_values.mean()),
                        "std_dev": float(finite_values.std()),
                    })

                    # Normalized indices can exceed [-1, 1]
                    # when scaled reflectance has negative values.
                    if band_name in OPTICAL_BANDS:

                        outside_range = int(
                            (
                                (finite_values < -1.000001)
                                | (finite_values > 1.000001)
                            ).sum()
                        )

                        if outside_range > 0:
                            warnings.append(
                                f"{band_name}: {outside_range} "
                                "valid pixels outside [-1, 1]. "
                                "Review denominator and reflectance."
                            )

                else:
                    errors.append(
                        f"{band_name}: no valid pixels found."
                    )

                band_reports.append(band_report)

            report["band_statistics"] = band_reports

            # ------------------------------------------------
            # WITHIN-RASTER MASK CONSISTENCY
            # ------------------------------------------------

            pixel_counts = [
                band["valid_pixels"]
                for band in band_reports
            ]

            report["valid_pixel_counts"] = pixel_counts

            check(
                len(set(pixel_counts)) == 1,
                "Different bands have different valid-pixel counts.",
                errors,
            )

            # ------------------------------------------------
            # NDBI / NDMI RELATIONSHIP
            # ------------------------------------------------

            if spec["type"] == "optical":

                ndbi = src.read(2, masked=True)
                ndmi = src.read(3, masked=True)

                common_mask = (
                    np.ma.getmaskarray(ndbi)
                    | np.ma.getmaskarray(ndmi)
                )

                ndbi_values = ndbi.data[~common_mask]
                ndmi_values = ndmi.data[~common_mask]

                if len(ndbi_values) > 0:

                    difference = np.abs(
                        ndbi_values + ndmi_values
                    )

                    max_difference = float(difference.max())

                    report["ndbi_ndmi_max_abs_sum"] = (
                        max_difference
                    )

                    if max_difference > 1e-5:
                        warnings.append(
                            "NDBI and NDMI are not exact negatives. "
                            "Review index calculation."
                        )

            report["valid"] = len(errors) == 0

    except Exception as error:
        errors.append(f"Raster inspection failed: {str(error)}")

    return report


# ============================================================
# CROSS-YEAR GRID CONSISTENCY
# ============================================================

def validate_yearly_consistency(raster_reports: list) -> list:
    """Check whether all years use consistent spatial grids."""

    errors = []

    for raster_type in ["optical", "thermal"]:

        group = [
            report for report in raster_reports
            if report["type"] == raster_type
            and report.get("exists")
            and report.get("width") is not None
        ]

        if len(group) != len(YEARS):
            errors.append(
                f"{raster_type}: expected {len(YEARS)} files; "
                f"found {len(group)}."
            )
            continue

        reference = group[0]

        for report in group[1:]:

            for field in [
                "width",
                "height",
                "crs",
                "resolution",
                "transform",
                "valid_pixel_counts",
            ]:

                if report.get(field) != reference.get(field):

                    # Float-valued metadata needs tolerant comparison.
                    if field in ["resolution", "transform"]:
                        if np.allclose(
                            report[field],
                            reference[field],
                            atol=1e-8,
                        ):
                            continue

                    errors.append(
                        f"{raster_type}: {field} differs between "
                        f"{reference['year']} and {report['year']}."
                    )

    return errors


# ============================================================
# CSV VALIDATION
# ============================================================

def inspect_statistics_csv() -> dict:
    """Validate the annual ecological statistics CSV."""

    errors = []
    warnings = []

    report = {
        "file": STATISTICS_FILE.name,
        "exists": STATISTICS_FILE.exists(),
        "valid": False,
        "errors": errors,
        "warnings": warnings,
    }

    if not STATISTICS_FILE.exists():
        errors.append("Statistics CSV not found.")
        return report

    try:
        df = pd.read_csv(STATISTICS_FILE)

        expected_columns = [
            "year",
            "LST_C_mean", "LST_C_min", "LST_C_max",
            "LST_C_stdDev", "LST_C_p5", "LST_C_p50",
            "LST_C_p95",
            "NDVI_mean", "NDVI_stdDev",
            "NDBI_mean", "NDBI_stdDev",
            "NDMI_mean", "NDMI_stdDev",
            "NDWI_mean", "NDWI_stdDev",
            "MNDWI_mean", "MNDWI_stdDev",
            "UTFVI_mean", "UTFVI_min",
            "UTFVI_max", "UTFVI_stdDev",
        ]

        missing_columns = [
            column for column in expected_columns
            if column not in df.columns
        ]

        if missing_columns:
            errors.append(
                f"Missing CSV columns: {missing_columns}"
            )

        actual_years = sorted(
            df["year"].astype(int).tolist()
        ) if "year" in df.columns else []

        expected_years = sorted(YEARS)

        check(
            actual_years == expected_years,
            f"Expected years {expected_years}; "
            f"found {actual_years}.",
            errors,
        )

        check(
            len(df) == len(YEARS),
            f"Expected {len(YEARS)} rows; found {len(df)}.",
            errors,
        )

        missing_values = int(df.isna().sum().sum())

        check(
            missing_values == 0,
            f"CSV contains {missing_values} missing values.",
            errors,
        )

        numeric_columns = df.select_dtypes(
            include=[np.number]
        ).columns

        non_finite_count = int(
            (~np.isfinite(df[numeric_columns])).sum().sum()
        )

        check(
            non_finite_count == 0,
            f"CSV contains {non_finite_count} non-finite values.",
            errors,
        )

        report.update({
            "rows": len(df),
            "columns": list(df.columns),
            "missing_values": missing_values,
            "non_finite_values": non_finite_count,
            "years": actual_years,
            "size_kb": round(
                STATISTICS_FILE.stat().st_size / 1024, 2
            ),
        })

        report["valid"] = len(errors) == 0

    except Exception as error:
        errors.append(f"CSV inspection failed: {str(error)}")

    return report


# ============================================================
# COMPLETE PROJECT VALIDATION
# ============================================================

def inspect_project() -> dict:
    """Run the complete ecological dataset quality-control audit."""

    print("\n" + "=" * 65)
    print("NMMC ECOLOGICAL DATASET VALIDATION")
    print("=" * 65)

    raster_reports = []

    # --------------------------------------------------------
    # A. RASTER VALIDATION
    # --------------------------------------------------------

    print("\n[1] GEOTIFF VALIDATION")

    for spec in get_expected_raster_files():

        report = inspect_raster(spec)
        raster_reports.append(report)

        status = "PASS" if report["valid"] else "FAIL"

        print(
            f"\n[{status}] {report['file']}"
        )

        if report.get("exists") and report.get("width"):

            print(
                f"  Dimensions: "
                f"{report['width']} x {report['height']}"
            )

            print(f"  Bands: {report['bands']}")
            print(f"  CRS: {report['crs']}")
            print(f"  Resolution: {report['resolution']}")
            print(f"  NoData: {report['nodata']}")

            for band in report.get("band_statistics", []):

                print(
                    f"  {band['band']}: "
                    f"min={band.get('min')}, "
                    f"max={band.get('max')}, "
                    f"mean={band.get('mean')}"
                )

        for error in report["errors"]:
            print(f"  ERROR: {error}")

        for warning in report["warnings"]:
            print(f"  WARNING: {warning}")

    # --------------------------------------------------------
    # B. CROSS-YEAR VALIDATION
    # --------------------------------------------------------

    print("\n[2] CROSS-YEAR GRID CONSISTENCY")

    consistency_errors = validate_yearly_consistency(
        raster_reports
    )

    if not consistency_errors:
        print("[PASS] Optical and thermal grids are consistent.")
    else:
        for error in consistency_errors:
            print(f"[FAIL] {error}")

    # --------------------------------------------------------
    # C. CSV VALIDATION
    # --------------------------------------------------------

    print("\n[3] ANNUAL STATISTICS CSV")

    csv_report = inspect_statistics_csv()

    print(
        "[PASS]" if csv_report["valid"] else "[FAIL]",
        csv_report["file"],
    )

    if csv_report.get("exists") and "rows" in csv_report:
        print(f"  Rows: {csv_report['rows']}")
        print(f"  Years: {csv_report['years']}")
        print(f"  Columns: {len(csv_report['columns'])}")
        print(
            f"  Missing values: "
            f"{csv_report['missing_values']}"
        )

    for error in csv_report["errors"]:
        print(f"  ERROR: {error}")

    # --------------------------------------------------------
    # D. FINAL SUMMARY
    # --------------------------------------------------------

    raster_passes = sum(
        report["valid"] for report in raster_reports
    )

    total_errors = (
        sum(len(report["errors"]) for report in raster_reports)
        + len(consistency_errors)
        + len(csv_report["errors"])
    )

    all_passed = (
        raster_passes == len(raster_reports)
        and not consistency_errors
        and csv_report["valid"]
    )

    final_report = {
        "project": "NMMC Multitemporal LULC and Ecological Study",
        "raster_files": raster_reports,
        "statistics_csv": csv_report,
        "consistency_errors": consistency_errors,
        "summary": {
            "raster_files_expected": len(raster_reports),
            "raster_files_passed": raster_passes,
            "csv_valid": csv_report["valid"],
            "total_errors": total_errors,
            "overall_pass": all_passed,
        },
    }

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    report_path = (
        REPORTS_DIR / "ecological_data_validation.json"
    )

    with open(report_path, "w", encoding="utf-8") as file:
        json.dump(final_report, file, indent=4)

    print("\n" + "=" * 65)
    print("VALIDATION SUMMARY")
    print("=" * 65)

    print(f"Raster files passed: {raster_passes}/{len(raster_reports)}")
    print(f"CSV valid: {csv_report['valid']}")
    print(f"Total errors: {total_errors}")
    print(f"Overall result: {'PASS' if all_passed else 'REVIEW REQUIRED'}")
    print(f"Report saved: {report_path}")

    print("=" * 65)

    return final_report


if __name__ == "__main__":
    inspect_project()
