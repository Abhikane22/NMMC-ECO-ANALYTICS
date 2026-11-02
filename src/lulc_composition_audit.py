
from pathlib import Path

import pandas as pd
import numpy as np


# ============================================================
# CONFIGURATION
# ============================================================

ROOT = Path(__file__).resolve().parent.parent

INPUT_FILE = (
    ROOT
    / "outputs"
    / "reports"
    / "NMMC_Ward_Level_LULC_Statistics_PRELIMINARY_ALL_YEARS.csv"
)

OUTPUT_DIR = ROOT / "outputs" / "reports"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

ID_COLUMN = "ward_id"
YEAR_COLUMN = "year"

EXPECTED_YEARS = [1994, 2004, 2014, 2024]

# Explicit mapping: class name -> actual area column.
CLASS_AREA_COLUMNS = {
    "water": "Water_area_km2",
    "vegetation": "Vegetation_area_km2",
    "builtup": "Built-up_area_km2",
    "barren": "Barren_area_km2",
    "wetlands": "Coastal Wetlands_area_km2",
}

VALID_AREA_COLUMN = "valid_raster_area_km2"


# ============================================================
# MAIN AUDIT
# ============================================================

def main():

    print("=" * 70)
    print("NMMC LULC COMPOSITIONAL INTEGRITY AUDIT")
    print("=" * 70)

    # --------------------------------------------------------
    # 1. LOAD DATA
    # --------------------------------------------------------

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Input file not found: {INPUT_FILE}"
        )

    df = pd.read_csv(INPUT_FILE)

    print("\nAvailable columns:")
    print(df.columns.tolist())

    required_columns = [
        ID_COLUMN,
        YEAR_COLUMN,
        VALID_AREA_COLUMN,
        *CLASS_AREA_COLUMNS.values(),
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}"
        )

    # --------------------------------------------------------
    # 2. BASIC DATA VALIDATION
    # --------------------------------------------------------

    df[ID_COLUMN] = pd.to_numeric(
        df[ID_COLUMN],
        errors="raise",
    )

    df[YEAR_COLUMN] = pd.to_numeric(
        df[YEAR_COLUMN],
        errors="raise",
    ).astype(int)

    print("\nDataset overview:")
    print("Rows:", len(df))
    print("Unique wards:", df[ID_COLUMN].nunique())
    print("Years:", sorted(df[YEAR_COLUMN].unique()))

    if df.duplicated([ID_COLUMN, YEAR_COLUMN]).any():
        raise ValueError(
            "Duplicate ward-year records found."
        )

    actual_years = sorted(df[YEAR_COLUMN].unique())

    if actual_years != EXPECTED_YEARS:
        raise ValueError(
            f"Expected years {EXPECTED_YEARS}, "
            f"but found {actual_years}"
        )

    expected_rows = (
        df[ID_COLUMN].nunique() * len(EXPECTED_YEARS)
    )

    if len(df) != expected_rows:
        raise ValueError(
            f"Incomplete ward-year dataset. "
            f"Expected {expected_rows} rows, found {len(df)}."
        )

    # --------------------------------------------------------
    # 3. VALIDATE CLASS AREA COLUMNS
    # --------------------------------------------------------

    print("\nDetected class-area columns:")

    for class_name, column in CLASS_AREA_COLUMNS.items():
        print(f"  {class_name}: {column}")

        df[column] = pd.to_numeric(
            df[column],
            errors="coerce",
        )

    df[VALID_AREA_COLUMN] = pd.to_numeric(
        df[VALID_AREA_COLUMN],
        errors="coerce",
    )

    numeric_columns = [
        *CLASS_AREA_COLUMNS.values(),
        VALID_AREA_COLUMN,
    ]

    if df[numeric_columns].isna().any().any():
        raise ValueError(
            "Missing or non-numeric class/valid area values found."
        )

    if not np.isfinite(
        df[numeric_columns].to_numpy(dtype=float)
    ).all():
        raise ValueError(
            "Infinite values found in area columns."
        )

    if (df[numeric_columns] < 0).any().any():
        raise ValueError(
            "Negative area values found."
        )

    # --------------------------------------------------------
    # 4. FIVE-CLASS AREA CLOSURE PER WARD-YEAR
    # --------------------------------------------------------

    df["five_class_total_km2"] = df[
        list(CLASS_AREA_COLUMNS.values())
    ].sum(axis=1)

    df["area_closure_error_km2"] = (
        df["five_class_total_km2"]
        - df[VALID_AREA_COLUMN]
    )

    df["absolute_area_closure_error_km2"] = (
        df["area_closure_error_km2"].abs()
    )

    closure_tolerance = 1e-8

    df["area_closure_pass"] = (
        df["absolute_area_closure_error_km2"]
        <= closure_tolerance
    )

    print("\n" + "-" * 70)
    print("FIVE-CLASS AREA CLOSURE")
    print("-" * 70)

    print(
        "Maximum absolute area closure error (km²):",
        df["absolute_area_closure_error_km2"].max(),
    )

    print(
        "Mean absolute area closure error (km²):",
        df["absolute_area_closure_error_km2"].mean(),
    )

    print(
        "Ward-year records passing closure:",
        df["area_closure_pass"].sum(),
        "/",
        len(df),
    )

    # Save ward-year area closure audit.
    area_closure_file = (
        OUTPUT_DIR
        / "NMMC_LULC_Five_Class_Area_Closure_Audit.csv"
    )

    df[
        [
            ID_COLUMN,
            YEAR_COLUMN,
            "five_class_total_km2",
            VALID_AREA_COLUMN,
            "area_closure_error_km2",
            "absolute_area_closure_error_km2",
            "area_closure_pass",
        ]
    ].to_csv(
        area_closure_file,
        index=False,
    )

    # --------------------------------------------------------
    # 5. CALCULATE CLASS AREA CHANGES: 1994-2024
    # --------------------------------------------------------

    print("\n" + "-" * 70)
    print("CLASS AREA CHANGE: 1994-2024")
    print("-" * 70)

    data_1994 = (
        df[df[YEAR_COLUMN] == 1994]
        .set_index(ID_COLUMN)
    )

    data_2024 = (
        df[df[YEAR_COLUMN] == 2024]
        .set_index(ID_COLUMN)
    )

    wards_1994 = set(data_1994.index)
    wards_2024 = set(data_2024.index)

    if wards_1994 != wards_2024:
        raise ValueError(
            "Ward IDs differ between 1994 and 2024."
        )

    if data_1994.index.has_duplicates:
        raise ValueError("Duplicate 1994 ward IDs.")

    if data_2024.index.has_duplicates:
        raise ValueError("Duplicate 2024 ward IDs.")

    common_wards = sorted(wards_1994)

    rows = []

    for class_name, area_column in CLASS_AREA_COLUMNS.items():

        for ward_id in common_wards:

            area_1994 = data_1994.loc[
                ward_id, area_column
            ]

            area_2024 = data_2024.loc[
                ward_id, area_column
            ]

            change = area_2024 - area_1994

            rows.append({
                ID_COLUMN: ward_id,
                "class": class_name,
                "area_1994_km2": area_1994,
                "area_2024_km2": area_2024,
                "change_km2": change,
            })

    changes = pd.DataFrame(rows)

    changes_file = (
        OUTPUT_DIR
        / "NMMC_LULC_Five_Class_Change_Audit.csv"
    )

    changes.to_csv(
        changes_file,
        index=False,
    )

    # --------------------------------------------------------
    # 6. COMPOSITIONAL CLOSURE PER WARD
    # --------------------------------------------------------

    closure = (
        changes
        .groupby(ID_COLUMN)["change_km2"]
        .sum()
        .rename("five_class_net_change_km2")
        .reset_index()
    )

    # Independently calculate the difference in valid area.
    valid_area_change = (
        data_2024[VALID_AREA_COLUMN]
        - data_1994[VALID_AREA_COLUMN]
    )

    valid_area_change = (
        valid_area_change
        .rename("valid_area_change_km2")
        .reset_index()
    )

    closure = closure.merge(
        valid_area_change,
        on=ID_COLUMN,
        how="left",
        validate="one_to_one",
    )

    closure["closure_difference_km2"] = (
        closure["five_class_net_change_km2"]
        - closure["valid_area_change_km2"]
    )

    closure["absolute_closure_difference_km2"] = (
        closure["closure_difference_km2"].abs()
    )

    closure["closure_pass"] = (
        closure["absolute_closure_difference_km2"]
        <= closure_tolerance
    )

    closure_file = (
        OUTPUT_DIR
        / "NMMC_LULC_Compositional_Closure_Audit.csv"
    )

    closure.to_csv(
        closure_file,
        index=False,
    )

    print("\nFive-class net change summary (km²):")
    print(
        closure["five_class_net_change_km2"]
        .describe()
        .round(10)
    )

    print(
        "\nMaximum absolute five-class net change (km²):",
        closure["five_class_net_change_km2"].abs().max(),
    )

    print(
        "Mean absolute five-class net change (km²):",
        closure["five_class_net_change_km2"].abs().mean(),
    )

    print(
        "\nMaximum absolute difference between "
        "five-class net change and valid-area change (km²):",
        closure["absolute_closure_difference_km2"].max(),
    )

    print(
        "Wards passing compositional closure:",
        closure["closure_pass"].sum(),
        "/",
        len(closure),
    )

    # --------------------------------------------------------
    # 7. CLASS-WISE CHANGE SUMMARY
    # --------------------------------------------------------

    print("\n" + "-" * 70)
    print("CLASS-WISE CHANGE SUMMARY")
    print("-" * 70)

    class_summary = (
        changes
        .groupby("class")["change_km2"]
        .agg(
            mean="mean",
            std="std",
            min="min",
            max="max",
            total="sum",
        )
        .round(8)
    )

    print(class_summary)

    class_summary_file = (
        OUTPUT_DIR
        / "NMMC_LULC_Class_Change_Summary_1994_2024.csv"
    )

    class_summary.to_csv(class_summary_file)

    # --------------------------------------------------------
    # 8. FINAL AUDIT STATUS
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("FINAL AUDIT STATUS")
    print("=" * 70)

    checks = {
        "No duplicate ward-year records": (
            not df.duplicated(
                [ID_COLUMN, YEAR_COLUMN]
            ).any()
        ),
        "Expected years present": (
            actual_years == EXPECTED_YEARS
        ),
        "All five class-area columns available": (
            all(
                column in df.columns
                for column in CLASS_AREA_COLUMNS.values()
            )
        ),
        "No missing or invalid area values": (
            np.isfinite(
                df[numeric_columns].to_numpy(dtype=float)
            ).all()
        ),
        "Five-class area closure": (
            bool(df["area_closure_pass"].all())
        ),
        "All wards match between 1994 and 2024": (
            wards_1994 == wards_2024
        ),
        "Five-class compositional closure": (
            bool(closure["closure_pass"].all())
        ),
    }

    for check_name, passed in checks.items():

        status = "PASS" if passed else "FAIL"

        print(f"[{status}] {check_name}")

    passed_count = sum(checks.values())
    total_checks = len(checks)

    print(
        f"\nOverall: {passed_count}/{total_checks} checks passed."
    )

    print("\nGenerated files:")

    print(area_closure_file)
    print(changes_file)
    print(closure_file)
    print(class_summary_file)

    print("\nAudit completed.")


if __name__ == "__main__":
    main()
