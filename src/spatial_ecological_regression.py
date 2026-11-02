from pathlib import Path
import sys
import warnings

import numpy as np
import pandas as pd
import geopandas as gpd

import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor

from libpysal.weights import Queen
from esda.moran import Moran

try:
    import spreg
except ImportError:
    raise SystemExit(
        "Missing spreg. Install it using: python -m pip install spreg"
    )


# ============================================================
# 1. PROJECT CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

BOUNDARY_FILE = (
    PROJECT_ROOT
    / "data/boundaries/NMMC_Ward_Boundaries_2024_CLEAN.geojson"
)

INDICATOR_FILE = (
    PROJECT_ROOT
    / "outputs/reports/NMMC_Ecological_Impact_Ward_Indicators.csv"
)

OUTPUT_DIR = PROJECT_ROOT / "outputs/reports"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

ID_COLUMN = "ward_id"

TARGET = "lst_change_c"

PREDICTORS = [
    "builtup_change_km2",
    "vegetation_change_km2",
    "wetland_change_km2",
]

RANDOM_PERMUTATIONS = 999
SIGNIFICANCE_LEVEL = 0.05


# ============================================================
# 2. LOAD AND VALIDATE DATA
# ============================================================

def normalize_ward_id(value):
    """Normalize numeric-looking ward IDs without changing their identity."""

    if pd.isna(value):
        return None

    value = str(value).strip()

    try:
        numeric_value = float(value)

        if numeric_value.is_integer():
            return str(int(numeric_value))

    except ValueError:
        pass

    return value


def load_data():

    print("\n[1/8] Loading ward boundaries and indicators...")

    if not BOUNDARY_FILE.exists():
        raise FileNotFoundError(
            f"Boundary file not found: {BOUNDARY_FILE}"
        )

    if not INDICATOR_FILE.exists():
        raise FileNotFoundError(
            f"Indicator file not found: {INDICATOR_FILE}"
        )

    wards = gpd.read_file(BOUNDARY_FILE)

    indicators = pd.read_csv(INDICATOR_FILE)

    if "ward_lgd_c" not in wards.columns:
        raise ValueError(
            "Boundary file does not contain ward_lgd_c."
        )

    required_columns = [ID_COLUMN, TARGET] + PREDICTORS

    missing_columns = [
        column
        for column in required_columns
        if column not in indicators.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing indicator columns: {missing_columns}"
        )

    wards["ward_id"] = wards["ward_lgd_c"].apply(
        normalize_ward_id
    )

    indicators["ward_id"] = indicators[ID_COLUMN].apply(
        normalize_ward_id
    )

    if wards["ward_id"].duplicated().any():
        raise ValueError("Duplicate ward IDs in boundary file.")

    if indicators["ward_id"].duplicated().any():
        raise ValueError("Duplicate ward IDs in indicator file.")

    boundary_ids = set(wards["ward_id"])
    indicator_ids = set(indicators["ward_id"])

    missing_from_indicators = boundary_ids - indicator_ids
    missing_from_boundaries = indicator_ids - boundary_ids

    if missing_from_indicators or missing_from_boundaries:

        raise ValueError(
            "Ward ID mismatch detected.\n"
            f"Missing from indicators: "
            f"{sorted(missing_from_indicators)[:15]}\n"
            f"Missing from boundaries: "
            f"{sorted(missing_from_boundaries)[:15]}"
        )

    # Join attributes to geometries using normalized ward IDs.
    model_gdf = wards[
        ["ward_id", wards.geometry.name]
    ].merge(
        indicators[
            ["ward_id", TARGET] + PREDICTORS
        ],
        on="ward_id",
        how="inner",
        validate="one_to_one",
        sort=False,
    )

    model_gdf = gpd.GeoDataFrame(
        model_gdf,
        geometry=wards.geometry.name,
        crs=wards.crs,
    )

    if len(model_gdf) != 111:
        raise ValueError(
            f"Expected 111 wards, found {len(model_gdf)}."
        )

    if not model_gdf.geometry.is_valid.all():
        raise ValueError(
            "Invalid geometries detected in the regression dataset."
        )

    if model_gdf.geometry.is_empty.any():
        raise ValueError(
            "Empty geometries detected in the regression dataset."
        )

    for column in [TARGET] + PREDICTORS:

        model_gdf[column] = pd.to_numeric(
            model_gdf[column],
            errors="coerce",
        )

    if model_gdf[[TARGET] + PREDICTORS].isna().any().any():
        raise ValueError(
            "Missing or non-numeric values found in model variables."
        )

    if not np.isfinite(
        model_gdf[[TARGET] + PREDICTORS].to_numpy(dtype=float)
    ).all():
        raise ValueError(
            "Infinite values found in model variables."
        )

    print("Boundary wards:", len(wards))
    print("Indicator wards:", len(indicators))
    print("Matched wards:", len(model_gdf))
    print("Ward ID alignment: PASS")
    print("Geometry validation: PASS")
    print("Missing values: 0")

    return model_gdf


# ============================================================
# 3. DESCRIPTIVE STATISTICS
# ============================================================

def calculate_descriptive_statistics(df):

    print("\n[2/8] Calculating descriptive statistics...")

    columns = [TARGET] + PREDICTORS

    result = df[columns].describe().T

    result.to_csv(
        OUTPUT_DIR
        / "NMMC_Spatial_Regression_Descriptive_Statistics.csv"
    )

    print(result)

    return result


# ============================================================
# 4. QUEEN CONTIGUITY SPATIAL WEIGHTS
# ============================================================

def build_spatial_weights(gdf):

    print("\n[3/8] Building Queen contiguity weights...")

    weights = Queen.from_dataframe(
        gdf,
        ids="ward_id",
    )

    if len(weights.id_order) != len(gdf):
        raise ValueError(
            "Spatial weights do not contain all wards."
        )

    if weights.islands:
        raise ValueError(
            f"Isolated wards detected: {weights.islands}"
        )

    weights.transform = "R"

    print("Number of wards:", weights.n)
    print("Connected components:", weights.n_components)
    print("Islands:", len(weights.islands))
    print("Mean neighbours:", round(weights.mean_neighbors, 3))
    print("Weight transformation: Row-standardized")

    return weights


# ============================================================
# 5. MULTICOLLINEARITY: VIF
# ============================================================

def calculate_vif(df):

    print("\n[4/8] Checking multicollinearity...")

    x = df[PREDICTORS].astype(float)

    x_with_constant = sm.add_constant(
        x,
        has_constant="add",
    )

    rows = []

    for index, column in enumerate(x_with_constant.columns):

        if column == "const":
            continue

        vif = variance_inflation_factor(
            x_with_constant.to_numpy(),
            index,
        )

        rows.append({
            "variable": column,
            "VIF": float(vif),
        })

    vif_df = pd.DataFrame(rows)

    vif_df.to_csv(
        OUTPUT_DIR
        / "NMMC_Spatial_Regression_VIF.csv",
        index=False,
    )

    print(vif_df.to_string(index=False))

    print(
        "\nInterpretation guide:"
        "\nVIF < 5: generally acceptable"
        "\nVIF 5–10: investigate"
        "\nVIF > 10: serious multicollinearity concern"
    )

    return vif_df


# ============================================================
# 6. OLS REGRESSION
# ============================================================

def fit_ols(df):

    print("\n[5/8] Fitting Ordinary Least Squares...")

    y = df[TARGET].astype(float)

    x = df[PREDICTORS].astype(float)

    x = sm.add_constant(
        x,
        has_constant="add",
    )

    # HC3 standard errors provide a heteroskedasticity-robust
    # baseline for coefficient inference.
    model = sm.OLS(y, x).fit(cov_type="HC3")

    print(model.summary())

    with open(
        OUTPUT_DIR / "NMMC_OLS_Regression_Summary.txt",
        "w",
    ) as file:

        file.write(model.summary().as_text())

    coefficients = pd.DataFrame({
        "variable": model.params.index,
        "coefficient": model.params.values,
        "robust_std_error": model.bse.values,
        "t_statistic": model.tvalues.values,
        "p_value": model.pvalues.values,
        "ci_lower": model.conf_int()[0].values,
        "ci_upper": model.conf_int()[1].values,
    })

    coefficients.to_csv(
        OUTPUT_DIR
        / "NMMC_OLS_Regression_Coefficients.csv",
        index=False,
    )

    residuals = np.asarray(model.resid).flatten()

    return model, residuals


# ============================================================
# 7. SPATIAL AUTOCORRELATION OF OLS RESIDUALS
# ============================================================

def test_residual_moran(residuals, weights):

    print("\n[6/8] Testing spatial autocorrelation in OLS residuals...")

    moran = Moran(
        residuals,
        weights,
        transformation="R",
        permutations=RANDOM_PERMUTATIONS,
    )

    results = {
        "moran_i": float(moran.I),
        "expected_i": float(moran.EI),
        "permutation_p_value": float(moran.p_sim),
        "z_sim": float(moran.z_sim),
        "permutations": RANDOM_PERMUTATIONS,
    }

    print("Moran's I:", round(results["moran_i"], 6))
    print("Expected I:", round(results["expected_i"], 6))
    print(
        "Permutation p-value:",
        round(results["permutation_p_value"], 6),
    )
    print("Permutation z-score:", round(results["z_sim"], 6))

    pd.DataFrame([results]).to_csv(
        OUTPUT_DIR
        / "NMMC_OLS_Residual_Morans_I.csv",
        index=False,
    )

    return results


# ============================================================
# 8. SPATIAL REGRESSION MODELS
# ============================================================

def fit_spatial_models(df, weights):

    print("\n[7/8] Fitting spatial regression models...")

    y = df[[TARGET]].to_numpy(dtype=float)

    x = df[PREDICTORS].to_numpy(dtype=float)

    models = {}

    # Spatial Lag Model
    print("\n--- Spatial Lag Model (SAR) ---")

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")

        lag_model = spreg.ML_Lag(
            y,
            x,
            w=weights,
            method="ord",
            name_y=TARGET,
            name_x=PREDICTORS,
        )

    models["SAR"] = lag_model

    for warning in caught:
        print("SAR warning:", str(warning.message))

    print(lag_model.summary)

    # Spatial Error Model
    print("\n--- Spatial Error Model (SEM) ---")

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")

        error_model = spreg.ML_Error(
            y,
            x,
            w=weights,
            method="ord",
            name_y=TARGET,
            name_x=PREDICTORS,
        )

    models["SEM"] = error_model

    for warning in caught:
        print("SEM warning:", str(warning.message))

    print(error_model.summary)

    return models


# ============================================================
# 9. SAVE SPATIAL MODEL COEFFICIENTS AND RESIDUAL DIAGNOSTICS
# ============================================================

def save_spatial_model_results(models, weights):

    print("\n[8/8] Exporting spatial model results...")

    coefficient_rows = []
    diagnostic_rows = []

    for model_name, model in models.items():

        names = list(model.name_x)
        betas = np.asarray(model.betas).flatten()
        standard_errors = np.asarray(model.std_err).flatten()

        if len(names) != len(betas):
            names = [
                f"parameter_{i}"
                for i in range(len(betas))
            ]

        for i, name in enumerate(names):

            coefficient_rows.append({
                "model": model_name,
                "variable": name,
                "coefficient": float(betas[i]),
                "standard_error": (
                    float(standard_errors[i])
                    if i < len(standard_errors)
                    else np.nan
                ),
            })

        residuals = np.asarray(model.u).flatten()

        if len(residuals) != weights.n:
            raise ValueError(
                f"{model_name}: residual count does not match "
                "the number of wards."
            )

        residual_moran = Moran(
            residuals,
            weights,
            transformation="R",
            permutations=RANDOM_PERMUTATIONS,
        )

        diagnostic_rows.append({
            "model": model_name,
            "log_likelihood": float(model.logll),
            "moran_i_residuals": float(residual_moran.I),
            "residual_moran_p_value": float(residual_moran.p_sim),
            "residual_moran_z": float(residual_moran.z_sim),
            "permutations": RANDOM_PERMUTATIONS,
        })

    pd.DataFrame(coefficient_rows).to_csv(
        OUTPUT_DIR
        / "NMMC_Spatial_Regression_Coefficients.csv",
        index=False,
    )

    pd.DataFrame(diagnostic_rows).to_csv(
        OUTPUT_DIR
        / "NMMC_Spatial_Regression_Diagnostics.csv",
        index=False,
    )

    print("\nSpatial regression coefficients:")
    print(pd.DataFrame(coefficient_rows).to_string(index=False))

    print("\nSpatial model residual diagnostics:")
    print(pd.DataFrame(diagnostic_rows).to_string(index=False))


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 65)
    print("NMMC ECOLOGICAL IMPACT — SPATIAL REGRESSION ANALYSIS")
    print("=" * 65)

    gdf = load_data()

    calculate_descriptive_statistics(gdf)

    weights = build_spatial_weights(gdf)

    vif_df = calculate_vif(gdf)

    ols_model, ols_residuals = fit_ols(gdf)

    ols_moran = test_residual_moran(
        ols_residuals,
        weights,
    )

    spatial_models = fit_spatial_models(
        gdf,
        weights,
    )

    save_spatial_model_results(
        spatial_models,
        weights,
    )

    print("\n" + "=" * 65)
    print("SPATIAL REGRESSION ANALYSIS COMPLETED")
    print("=" * 65)

    print("\nOLS R-squared:", round(ols_model.rsquared, 5))
    print("OLS adjusted R-squared:", round(ols_model.rsquared_adj, 5))

    print(
        "OLS residual Moran's I:",
        round(ols_moran["moran_i"], 5),
    )

    print(
        "OLS residual Moran's p-value:",
        round(ols_moran["permutation_p_value"], 5),
    )

    print("\nVIF assessment:")

    for _, row in vif_df.iterrows():

        status = (
            "HIGH"
            if row["VIF"] >= 10
            else "INVESTIGATE"
            if row["VIF"] >= 5
            else "ACCEPTABLE"
        )

        print(
            f'{row["variable"]}: '
            f'VIF={row["VIF"]:.3f} [{status}]'
        )

    print("\nGenerated files in:")
    print(OUTPUT_DIR)


if __name__ == "__main__":
    main()