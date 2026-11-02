
import numpy as np
import pandas as pd

from scipy.stats import spearmanr, pearsonr
from scipy.stats import false_discovery_control

from config import REPORTS_DIR


# ---------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------

INPUT_FILE = (
    REPORTS_DIR
    / "NMMC_Ward_Level_LULC_Statistics_PRELIMINARY_ALL_YEARS.csv"
)

ECOLOGICAL_FILE = (
    REPORTS_DIR
    / "NMMC_Ward_Level_Ecological_Statistics.csv"
)

OUTPUT_DIR = REPORTS_DIR

START_YEAR = 1994
END_YEAR = 2024
EXPECTED_WARDS = 111


# ---------------------------------------------------------
# LOAD AND VALIDATE DATA
# ---------------------------------------------------------

def load_2024_change_indicators():

    lulc = pd.read_csv(INPUT_FILE)
    ecological = pd.read_csv(ECOLOGICAL_FILE)

    lulc["ward_id"] = lulc["ward_id"].astype(str)
    ecological["ward_id"] = ecological["ward_id"].astype(str)

    # Validate ward-year records.
    if lulc.duplicated(["ward_id", "year"]).any():
        raise ValueError("Duplicate LULC ward-year records.")

    if ecological.duplicated(["ward_id", "year"]).any():
        raise ValueError("Duplicate ecological ward-year records.")

    if not {START_YEAR, END_YEAR}.issubset(set(lulc["year"])):
        raise ValueError("Missing required LULC years.")

    if not {START_YEAR, END_YEAR}.issubset(set(ecological["year"])):
        raise ValueError("Missing required ecological years.")

    # -----------------------------------------------------
    # LULC TEMPORAL CHANGES
    # -----------------------------------------------------

    lulc_columns = [
        "Built-up_area_km2",
        "Vegetation_area_km2",
        "Coastal Wetlands_area_km2",
    ]

    lulc_pivots = {}

    for column in lulc_columns:

        pivot = lulc.pivot(
            index="ward_id",
            columns="year",
            values=column,
        )

        lulc_pivots[column] = pivot

    # Consistent convention:
    # Change = END_YEAR - START_YEAR

    builtup_change = (
        lulc_pivots["Built-up_area_km2"][END_YEAR]
        - lulc_pivots["Built-up_area_km2"][START_YEAR]
    )

    vegetation_change = (
        lulc_pivots["Vegetation_area_km2"][END_YEAR]
        - lulc_pivots["Vegetation_area_km2"][START_YEAR]
    )

    wetland_change = (
        lulc_pivots["Coastal Wetlands_area_km2"][END_YEAR]
        - lulc_pivots["Coastal Wetlands_area_km2"][START_YEAR]
    )

    # -----------------------------------------------------
    # ECOLOGICAL TEMPORAL CHANGES
    # -----------------------------------------------------

    ecological_columns = [
        "lst_c_mean",
        "ndvi_mean",
    ]

    ecological_pivots = {}

    for column in ecological_columns:

        pivot = ecological.pivot(
            index="ward_id",
            columns="year",
            values=column,
        )

        ecological_pivots[column] = pivot

    lst_change = (
        ecological_pivots["lst_c_mean"][END_YEAR]
        - ecological_pivots["lst_c_mean"][START_YEAR]
    )

    ndvi_change = (
        ecological_pivots["ndvi_mean"][END_YEAR]
        - ecological_pivots["ndvi_mean"][START_YEAR]
    )

    # -----------------------------------------------------
    # COMBINE INDICATORS
    # -----------------------------------------------------

    result = pd.DataFrame({
        "builtup_change_km2": builtup_change,
        "vegetation_change_km2": vegetation_change,
        "wetland_change_km2": wetland_change,
        "lst_change_c": lst_change,
        "ndvi_change": ndvi_change,
    })

    result.index.name = "ward_id"
    result = result.reset_index()

    # -----------------------------------------------------
    # VALIDATION
    # -----------------------------------------------------

    if len(result) != EXPECTED_WARDS:
        raise ValueError(
            f"Expected {EXPECTED_WARDS} wards, "
            f"found {len(result)}."
        )

    if result["ward_id"].duplicated().any():
        raise ValueError("Duplicate ward IDs in final dataset.")

    if result.isna().any().any():
        raise ValueError("Missing values in change indicators.")

    numeric_values = (
        result.select_dtypes(include="number").to_numpy()
    )

    if not np.isfinite(numeric_values).all():
        raise ValueError("Non-finite values detected.")

    print("\nInput validation: PASS")
    print("Ward count:", len(result))
    print(
        f"Change period: {START_YEAR}–{END_YEAR}"
    )
    print("Change convention: END_YEAR - START_YEAR")

    return result


# ---------------------------------------------------------
# CORRELATION ANALYSIS
# ---------------------------------------------------------

def calculate_correlations(data):

    variables = [
        "builtup_change_km2",
        "vegetation_change_km2",
        "wetland_change_km2",
        "lst_change_c",
        "ndvi_change",
    ]

    records = []

    for i, first in enumerate(variables):

        for second in variables[i + 1:]:

            x = data[first].to_numpy(dtype=float)
            y = data[second].to_numpy(dtype=float)

            pearson = pearsonr(x, y)
            spearman = spearmanr(x, y)

            records.append({
                "variable_1": first,
                "variable_2": second,
                "pearson_r": pearson.statistic,
                "pearson_p": pearson.pvalue,
                "spearman_rho": spearman.statistic,
                "spearman_p": spearman.pvalue,
                "sample_size": len(data),
            })

    results = pd.DataFrame(records)

    # -----------------------------------------------------
    # MULTIPLE TESTING CORRECTION
    # -----------------------------------------------------

    # Ten pairwise Spearman tests.
    results["spearman_fdr_p"] = false_discovery_control(
        results["spearman_p"].to_numpy(),
        method="bh",
    )

    # Ten pairwise Pearson tests.
    results["pearson_fdr_p"] = false_discovery_control(
        results["pearson_p"].to_numpy(),
        method="bh",
    )

    results["spearman_significant_fdr_05"] = (
        results["spearman_fdr_p"] < 0.05
    )

    results["pearson_significant_fdr_05"] = (
        results["pearson_fdr_p"] < 0.05
    )

    return results


# ---------------------------------------------------------
# CHANGE DIRECTION SUMMARY
# ---------------------------------------------------------

def calculate_change_summary(data):

    variables = [
        "builtup_change_km2",
        "vegetation_change_km2",
        "wetland_change_km2",
        "lst_change_c",
        "ndvi_change",
    ]

    records = []

    for variable in variables:

        values = data[variable]

        records.append({
            "indicator": variable,
            "mean_change": values.mean(),
            "median_change": values.median(),
            "minimum_change": values.min(),
            "maximum_change": values.max(),
            "wards_with_increase": int((values > 0).sum()),
            "wards_with_decrease": int((values < 0).sum()),
            "wards_unchanged": int(
                np.isclose(values, 0, atol=1e-10).sum()
            ),
            "standard_deviation": values.std(),
        })

    return pd.DataFrame(records)


# ---------------------------------------------------------
# MAIN
# ---------------------------------------------------------

def main():

    print("\nNMMC ECOLOGICAL IMPACT ANALYSIS")
    print("=" * 70)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load indicators.
    data = load_2024_change_indicators()

    # Descriptive statistics.
    print("\nCHANGE INDICATOR DESCRIPTIVE STATISTICS")
    print("-" * 70)

    print(
        data.drop(columns="ward_id")
        .describe()
        .round(4)
        .to_string()
    )

    # Change direction summary.
    change_summary = calculate_change_summary(data)

    print("\nCHANGE DIRECTION SUMMARY")
    print("-" * 70)

    print(
        change_summary.round(4).to_string(index=False)
    )

    # Correlations.
    results = calculate_correlations(data)

    print("\nSPEARMAN CORRELATION RESULTS")
    print("-" * 70)

    print(
        results[
            [
                "variable_1",
                "variable_2",
                "spearman_rho",
                "spearman_p",
                "spearman_fdr_p",
                "spearman_significant_fdr_05",
            ]
        ].round(5).to_string(index=False)
    )

    print("\nPEARSON CORRELATION RESULTS")
    print("-" * 70)

    print(
        results[
            [
                "variable_1",
                "variable_2",
                "pearson_r",
                "pearson_p",
                "pearson_fdr_p",
                "pearson_significant_fdr_05",
            ]
        ].round(5).to_string(index=False)
    )

    # -----------------------------------------------------
    # SAVE REPORTS
    # -----------------------------------------------------

    output_data = (
        OUTPUT_DIR
        / "NMMC_Ecological_Impact_Ward_Indicators.csv"
    )

    output_corr = (
        OUTPUT_DIR
        / "NMMC_Ecological_Impact_Correlation_Results.csv"
    )

    output_summary = (
        OUTPUT_DIR
        / "NMMC_Ecological_Impact_Change_Summary.csv"
    )

    data.to_csv(output_data, index=False)
    results.to_csv(output_corr, index=False)
    change_summary.to_csv(output_summary, index=False)

    print("\n" + "=" * 70)
    print("ECOLOGICAL IMPACT ANALYSIS COMPLETED")
    print("=" * 70)

    print("\nGenerated reports:")
    print(output_data)
    print(output_corr)
    print(output_summary)

    print("\nValidation: PASS")


if __name__ == "__main__":
    main()
