
from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import geopandas as gpd

import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor

from libpysal.weights import Queen, Rook
from esda.moran import Moran
import spreg


# ============================================================
# 1. CONFIGURATION
# ============================================================

ROOT = Path(__file__).resolve().parent.parent

BOUNDARY_FILE = (
    ROOT / "data/boundaries/"
    "NMMC_Ward_Boundaries_2024_CLEAN.geojson"
)

INDICATOR_FILE = (
    ROOT / "outputs/reports/"
    "NMMC_Ecological_Impact_Ward_Indicators.csv"
)

OUTPUT_DIR = ROOT / "outputs/reports"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TARGET = "lst_change_c"

ABSOLUTE_PREDICTORS = [
    "builtup_change_km2",
    "vegetation_change_km2",
    "wetland_change_km2",
]

NORMALIZED_PREDICTORS = [
    "builtup_change_pct_points",
    "vegetation_change_pct_points",
    "wetland_change_pct_points",
]

PERMUTATIONS = 999


# ============================================================
# 2. DATA LOADING AND VALIDATION
# ============================================================

def normalize_id(value):

    if pd.isna(value):
        return None

    value = str(value).strip()

    try:
        number = float(value)

        if number.is_integer():
            return str(int(number))

    except ValueError:
        pass

    return value


def load_data():

    print("\n[1/6] Loading and validating data...")

    # --------------------------------------------------------
    # Load source files
    # --------------------------------------------------------

    wards = gpd.read_file(BOUNDARY_FILE)
    indicators = pd.read_csv(INDICATOR_FILE)

    if "ward_lgd_c" not in wards.columns:
        raise ValueError("ward_lgd_c not found in boundaries.")

    # --------------------------------------------------------
    # Normalize ward IDs
    # --------------------------------------------------------

    wards["ward_id"] = wards["ward_lgd_c"].apply(normalize_id)
    indicators["ward_id"] = indicators["ward_id"].apply(normalize_id)

    if wards["ward_id"].isna().any():
        raise ValueError("Missing ward IDs in boundary file.")

    if indicators["ward_id"].isna().any():
        raise ValueError("Missing ward IDs in indicator file.")

    if wards["ward_id"].duplicated().any():
        raise ValueError("Duplicate boundary ward IDs.")

    if indicators["ward_id"].duplicated().any():
        raise ValueError("Duplicate indicator ward IDs.")

    # --------------------------------------------------------
    # Verify ward ID alignment
    # --------------------------------------------------------

    boundary_ids = set(wards["ward_id"])
    indicator_ids = set(indicators["ward_id"])

    if boundary_ids != indicator_ids:

        missing_from_indicators = boundary_ids - indicator_ids
        missing_from_boundaries = indicator_ids - boundary_ids

        raise ValueError(
            "Ward ID mismatch between files.\n"
            f"Missing from indicators: "
            f"{sorted(missing_from_indicators)[:15]}\n"
            f"Missing from boundaries: "
            f"{sorted(missing_from_boundaries)[:15]}"
        )

    # --------------------------------------------------------
    # Project geometries and calculate ward areas
    # --------------------------------------------------------

    wards = wards.to_crs("EPSG:32643")

    wards["ward_area_km2"] = (
        wards.geometry.area / 1_000_000
    )

    if not wards.geometry.is_valid.all():
        raise ValueError("Invalid ward geometries detected.")

    if wards.geometry.is_empty.any():
        raise ValueError("Empty ward geometries detected.")

    if (wards["ward_area_km2"] <= 0).any():
        raise ValueError("Non-positive ward area detected.")

    # --------------------------------------------------------
    # Merge while PRESERVING geometry
    # --------------------------------------------------------

    geometry_column = wards.geometry.name

    area_data = wards[
        [
            "ward_id",
            "ward_area_km2",
            geometry_column,
        ]
    ]

    df = area_data.merge(
        indicators,
        on="ward_id",
        how="inner",
        validate="one_to_one",
    )

    # Explicitly construct GeoDataFrame.
    df = gpd.GeoDataFrame(
        df,
        geometry=geometry_column,
        crs=wards.crs,
    )

    # --------------------------------------------------------
    # Validate merged dataset
    # --------------------------------------------------------

    if len(df) != 111:
        raise ValueError(
            f"Expected 111 wards; found {len(df)}."
        )

    if df["ward_id"].duplicated().any():
        raise ValueError("Duplicate ward IDs after merging.")

    if not df.geometry.is_valid.all():
        raise ValueError("Invalid geometries after merging.")

    if df.geometry.is_empty.any():
        raise ValueError("Empty geometries after merging.")

    # --------------------------------------------------------
    # Validate numerical variables
    # --------------------------------------------------------

    required = [
        TARGET,
        *ABSOLUTE_PREDICTORS,
        "ward_area_km2",
    ]

    for column in required:

        df[column] = pd.to_numeric(
            df[column],
            errors="coerce",
        )

    if df[required].isna().any().any():
        raise ValueError(
            "Missing or non-numeric model values detected."
        )

    if not np.isfinite(
        df[required].to_numpy(dtype=float)
    ).all():

        raise ValueError(
            "Infinite values detected in model variables."
        )

    # --------------------------------------------------------
    # Calculate area-normalized changes
    # --------------------------------------------------------

    for source, destination in zip(
        ABSOLUTE_PREDICTORS,
        NORMALIZED_PREDICTORS,
    ):

        df[destination] = (
            df[source] / df["ward_area_km2"]
        ) * 100

    # --------------------------------------------------------
    # Stable ordering for spatial weights and model data
    # --------------------------------------------------------

    df = df.sort_values(
        "ward_id"
    ).reset_index(drop=True)

    # Reconstruct GeoDataFrame after sorting.
    df = gpd.GeoDataFrame(
        df,
        geometry=geometry_column,
        crs=wards.crs,
    )

    # --------------------------------------------------------
    # Final validation and output
    # --------------------------------------------------------

    if not isinstance(df, gpd.GeoDataFrame):
        raise TypeError("Merged dataset is not a GeoDataFrame.")

    if df.geometry.name != geometry_column:
        raise ValueError("Geometry column was not preserved.")

    print("Boundary wards:", len(wards))
    print("Indicator wards:", len(indicators))
    print("Matched wards:", len(df))

    print("Ward ID alignment: PASS")
    print("Geometry preservation: PASS")
    print("Geometry validation: PASS")
    print("Ward area validation: PASS")
    print("Missing values: 0")
    print("CRS:", df.crs)

    print("\nWard area statistics (km²):")
    print(df["ward_area_km2"].describe().round(4))

    print("\nNormalized predictor summary:")
    print(
        df[NORMALIZED_PREDICTORS]
        .describe()
        .round(4)
    )

    df.to_csv(
        OUTPUT_DIR
        / "NMMC_Ecological_Impact_Normalized_Indicators.csv",
        index=False,
    )

    print(
        "\nNormalized indicators exported successfully."
    )

    return df


# ============================================================
# 3. MULTICOLLINEARITY
# ============================================================

def calculate_vif(df, predictors, label):

    X = df[predictors].astype(float)

    X = sm.add_constant(X, has_constant="add")

    values = X.to_numpy()

    rows = []

    for i, column in enumerate(X.columns):

        if column == "const":
            continue

        rows.append({
            "model": label,
            "variable": column,
            "VIF": float(
                variance_inflation_factor(values, i)
            ),
        })

    return rows


# ============================================================
# 4. SPATIAL WEIGHTS
# ============================================================

def build_weights(gdf, weight_type):

    if weight_type == "Queen":

        weights = Queen.from_dataframe(
            gdf,
            ids="ward_id",
        )

    elif weight_type == "Rook":

        weights = Rook.from_dataframe(
            gdf,
            ids="ward_id",
        )

    else:
        raise ValueError("Unsupported spatial weights type.")

    if weights.n != len(gdf):
        raise ValueError("Spatial weights count mismatch.")

    if weights.islands:
        raise ValueError(
            f"{weight_type} has islands: {weights.islands}"
        )

    if weights.n_components != 1:
        raise ValueError(
            f"{weight_type} has disconnected components."
        )

    weights.transform = "R"

    print(
        f"{weight_type}:",
        weights.n,
        "wards |",
        weights.n_components,
        "component(s) |",
        round(weights.mean_neighbors, 3),
        "mean neighbours",
    )

    return weights


# ============================================================
# 5. EXACT SAR IMPACT POINT ESTIMATES
# ============================================================

def calculate_sar_impacts(model, weights, predictors):

    """
    Calculate exact average direct, indirect and total
    SAR impact point estimates:

        S = (I - rho W)^-1

    These are point estimates only. No standard errors or
    confidence intervals are calculated here.
    """

    n = weights.n

    W = weights.sparse.toarray()

    identity = np.eye(n)

    rho = float(np.asarray(model.rho).squeeze())

    spatial_multiplier = np.linalg.solve(
        identity - rho * W,
        identity,
    )

    direct_multiplier = (
        np.trace(spatial_multiplier) / n
    )

    total_multiplier = (
        spatial_multiplier.sum(axis=1).mean()
    )

    indirect_multiplier = (
        total_multiplier - direct_multiplier
    )

    names = list(model.name_x)
    betas = np.asarray(model.betas).flatten()

    coefficient_lookup = dict(zip(names, betas))

    results = []

    for predictor in predictors:

        if predictor not in coefficient_lookup:
            raise ValueError(
                f"Coefficient not found for {predictor}."
            )

        beta = float(coefficient_lookup[predictor])

        results.append({
            "variable": predictor,
            "coefficient": beta,
            "direct_impact": beta * direct_multiplier,
            "indirect_impact": beta * indirect_multiplier,
            "total_impact": beta * total_multiplier,
            "impact_type": "Exact point estimate; no CI",
        })

    return results


# ============================================================
# 6. FIT MODELS
# ============================================================

def run_model(df, predictors, data_type, weight_type):

    print("\n" + "=" * 65)
    print(f"{data_type} | {weight_type}")
    print("=" * 65)

    # Build weights and reorder observations to match id_order.
    weights = build_weights(
         df,
         weight_type,
         )

    indexed = df.set_index("ward_id")

    ordered = indexed.loc[
        list(weights.id_order)
    ].reset_index()

    y = ordered[[TARGET]].to_numpy(dtype=float)

    X = ordered[predictors].to_numpy(dtype=float)

    # OLS baseline for this predictor specification.
    ols_X = sm.add_constant(
        ordered[predictors].astype(float),
        has_constant="add",
    )

    ols = sm.OLS(
        ordered[TARGET].astype(float),
        ols_X,
    ).fit(cov_type="HC3")

    ols_residual_moran = Moran(
        np.asarray(ols.resid),
        weights,
        transformation="R",
        permutations=PERMUTATIONS,
    )

    # Spatial Lag model.
    with warnings.catch_warnings(record=True) as caught:

        warnings.simplefilter("always")

        sar = spreg.ML_Lag(
            y,
            X,
            w=weights,
            method="ord",
            name_y=TARGET,
            name_x=predictors,
        )

    for warning in caught:
        print("SAR warning:", str(warning.message))

    print("\nSAR summary:")
    print(sar.summary)

    residuals = np.asarray(sar.u).flatten()

    residual_moran = Moran(
        residuals,
        weights,
        transformation="R",
        permutations=PERMUTATIONS,
    )

    # Number of estimated parameters includes intercept,
    # explanatory variables and spatial lag parameter.
    parameter_count = len(np.asarray(sar.betas).flatten())

    aic = (
        -2 * float(sar.logll)
        + 2 * parameter_count
    )

    impacts = calculate_sar_impacts(
        sar,
        weights,
        predictors,
    )

    model_label = f"{data_type}_{weight_type}"

    diagnostics = {
        "model": model_label,
        "data_type": data_type,
        "weights": weight_type,
        "n": len(ordered),
        "ols_r_squared": float(ols.rsquared),
        "ols_adjusted_r_squared": float(ols.rsquared_adj),
        "ols_residual_moran_i": float(ols_residual_moran.I),
        "ols_residual_moran_p": float(ols_residual_moran.p_sim),
        "sar_pseudo_r_squared": float(sar.pr2),
        "sar_spatial_pseudo_r_squared": float(sar.pr2_e),
        "sar_log_likelihood": float(sar.logll),
        "sar_aic": aic,
        "rho": float(np.asarray(sar.rho).squeeze()),
        "sar_residual_moran_i": float(residual_moran.I),
        "sar_residual_moran_p": float(residual_moran.p_sim),
        "sar_residual_moran_z": float(residual_moran.z_sim),
    }

    # Coefficient estimates and standard errors.
    names = list(sar.name_x)
    betas = np.asarray(sar.betas).flatten()
    standard_errors = np.asarray(sar.std_err).flatten()
    z_values = np.asarray(sar.z_stat, dtype=float)

    coefficient_rows = []

    for i, name in enumerate(names):

        z_stat = float(z_values[i][0])
        p_value = float(z_values[i][1])

        coefficient_rows.append({
            "model": model_label,
            "variable": name,
            "coefficient": float(betas[i]),
            "standard_error": float(standard_errors[i]),
            "z_statistic": z_stat,
            "p_value": p_value,
        })

    vif_rows = calculate_vif(
        ordered,
        predictors,
        model_label,
    )

    for row in impacts:
        row["model"] = model_label

    print("\nOLS R-squared:", round(ols.rsquared, 4))
    print(
        "OLS residual Moran's I:",
        round(ols_residual_moran.I, 5),
        "| p:",
        round(ols_residual_moran.p_sim, 5),
    )

    print("SAR pseudo-R-squared:", round(sar.pr2, 4))
    print("SAR rho:", round(float(sar.rho), 5))
    print(
        "SAR residual Moran's I:",
        round(residual_moran.I, 5),
        "| p:",
        round(residual_moran.p_sim, 5),
    )

    return diagnostics, coefficient_rows, vif_rows, impacts


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 65)
    print("NMMC — AREA-NORMALIZED SPATIAL SENSITIVITY ANALYSIS")
    print("=" * 65)

    df = load_data()

    all_diagnostics = []
    all_coefficients = []
    all_vif = []
    all_impacts = []

    specifications = [
        ("ABSOLUTE", ABSOLUTE_PREDICTORS),
        ("NORMALIZED", NORMALIZED_PREDICTORS),
    ]

    weight_types = ["Queen", "Rook"]

    for data_type, predictors in specifications:

        for weight_type in weight_types:

            result = run_model(
                df,
                predictors,
                data_type,
                weight_type,
            )

            diagnostics, coefficients, vif, impacts = result

            all_diagnostics.append(diagnostics)
            all_coefficients.extend(coefficients)
            all_vif.extend(vif)
            all_impacts.extend(impacts)

    # Export combined results.
    diagnostics_df = pd.DataFrame(all_diagnostics)
    coefficients_df = pd.DataFrame(all_coefficients)
    vif_df = pd.DataFrame(all_vif)
    impacts_df = pd.DataFrame(all_impacts)

    diagnostics_df.to_csv(
        OUTPUT_DIR / "NMMC_SAR_Sensitivity_Diagnostics.csv",
        index=False,
    )

    coefficients_df.to_csv(
        OUTPUT_DIR / "NMMC_SAR_Sensitivity_Coefficients.csv",
        index=False,
    )

    vif_df.to_csv(
        OUTPUT_DIR / "NMMC_SAR_Sensitivity_VIF.csv",
        index=False,
    )

    impacts_df.to_csv(
        OUTPUT_DIR / "NMMC_SAR_Sensitivity_Exact_Impacts.csv",
        index=False,
    )

    print("\n" + "=" * 65)
    print("FINAL MODEL COMPARISON")
    print("=" * 65)

    display_columns = [
        "model",
        "ols_r_squared",
        "sar_pseudo_r_squared",
        "sar_aic",
        "rho",
        "sar_residual_moran_i",
        "sar_residual_moran_p",
    ]

    print(
        diagnostics_df[display_columns]
        .round(5)
        .to_string(index=False)
    )

    print("\nVIF results:")
    print(vif_df.round(4).to_string(index=False))

    print("\nExact SAR impact point estimates:")
    print(impacts_df.round(5).to_string(index=False))

    print("\nGenerated files:")
    print("1. NMMC_Ecological_Impact_Normalized_Indicators.csv")
    print("2. NMMC_SAR_Sensitivity_Diagnostics.csv")
    print("3. NMMC_SAR_Sensitivity_Coefficients.csv")
    print("4. NMMC_SAR_Sensitivity_VIF.csv")
    print("5. NMMC_SAR_Sensitivity_Exact_Impacts.csv")

    print("\nAnalysis completed.")


if __name__ == "__main__":
    main()
