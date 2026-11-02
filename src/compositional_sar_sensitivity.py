
from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import geopandas as gpd

from libpysal.weights import Queen, Rook
from esda import Moran
from spreg import ML_Lag


# ============================================================
# CONFIGURATION
# ============================================================

ROOT = Path(__file__).resolve().parent.parent

BOUNDARY_FILE = (
    ROOT
    / "data"
    / "boundaries"
    / "NMMC_Ward_Boundaries_2024_CLEAN.geojson"
)

LULC_FILE = (
    ROOT
    / "outputs"
    / "reports"
    / "NMMC_Ward_Level_LULC_Statistics_PRELIMINARY_ALL_YEARS.csv"
)

ECOLOGICAL_FILE = (
    ROOT
    / "outputs"
    / "reports"
    / "NMMC_Ecological_Impact_Ward_Indicators.csv"
)

OUTPUT_DIR = ROOT / "outputs" / "reports"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

WARD_ID = "ward_id"
BOUNDARY_WARD_ID = "ward_lgd_c"
LULC_WARD_ID = "ward_id"
YEAR_COLUMN = "year"

DEPENDENT_VARIABLE = "lst_change_c"

EXPECTED_WARDS = 111
YEARS = [1994, 2024]

CLASS_AREA_COLUMNS = {
    "water": "Water_area_km2",
    "vegetation": "Vegetation_area_km2",
    "builtup": "Built-up_area_km2",
    "barren": "Barren_area_km2",
    "wetlands": "Coastal Wetlands_area_km2",
}

RANDOM_SEED = 42
PERMUTATIONS = 999
CLOSURE_TOLERANCE = 1e-8


# ============================================================
# DATA PREPARATION
# ============================================================

def load_data():

    print("\n" + "=" * 70)
    print("LOADING AND PREPARING DATA")
    print("=" * 70)

       # --------------------------------------------------------
    # Load boundaries
    # --------------------------------------------------------

    boundaries = gpd.read_file(BOUNDARY_FILE)

    if BOUNDARY_WARD_ID not in boundaries.columns:
        raise ValueError(
            f"Boundary file is missing {BOUNDARY_WARD_ID}. "
            f"Available columns: {boundaries.columns.tolist()}"
        )

    # Standardize the boundary identifier.
    boundaries[WARD_ID] = pd.to_numeric(
        boundaries[BOUNDARY_WARD_ID],
        errors="raise",
    ).astype(int)

    if boundaries[WARD_ID].duplicated().any():
        raise ValueError("Duplicate ward IDs in boundary file.")

    boundaries = boundaries.to_crs("EPSG:32643")

    boundaries["ward_area_km2"] = (
        boundaries.geometry.area / 1_000_000
    )

    print("Boundary ward identifier:", BOUNDARY_WARD_ID)
    print("Standardized identifier:", WARD_ID)
    print("Boundary ward count:", len(boundaries))

    # --------------------------------------------------------
    # Load LULC statistics
    # --------------------------------------------------------

    lulc = pd.read_csv(LULC_FILE)

    required_lulc_columns = [
        LULC_WARD_ID,
        YEAR_COLUMN,
        *CLASS_AREA_COLUMNS.values(),
    ]

    missing = [
        column
        for column in required_lulc_columns
        if column not in lulc.columns
    ]

    if missing:
        raise ValueError(
            f"Missing LULC columns: {missing}"
        )

    lulc[LULC_WARD_ID] = pd.to_numeric(
        lulc[LULC_WARD_ID],
        errors="raise",
    ).astype(int)

    lulc[YEAR_COLUMN] = pd.to_numeric(
        lulc[YEAR_COLUMN],
        errors="raise",
    ).astype(int)

    if lulc.duplicated(
        [LULC_WARD_ID, YEAR_COLUMN]
    ).any():
        raise ValueError("Duplicate LULC ward-year records.")

    lulc = lulc[
        lulc[YEAR_COLUMN].isin(YEARS)
    ].copy()

    if set(lulc[YEAR_COLUMN].unique()) != set(YEARS):
        raise ValueError(
            f"LULC data must contain years {YEARS}"
        )

    for column in CLASS_AREA_COLUMNS.values():
        lulc[column] = pd.to_numeric(
            lulc[column],
            errors="coerce",
        )

    if lulc[
        list(CLASS_AREA_COLUMNS.values())
    ].isna().any().any():
        raise ValueError("Missing LULC class-area values.")

    # --------------------------------------------------------
    # Calculate normalized class changes
    # --------------------------------------------------------

    lulc_1994 = (
        lulc[lulc[YEAR_COLUMN] == 1994]
        .set_index(LULC_WARD_ID)
    )

    lulc_2024 = (
        lulc[lulc[YEAR_COLUMN] == 2024]
        .set_index(LULC_WARD_ID)
    )

    if set(lulc_1994.index) != set(lulc_2024.index):
        raise ValueError(
            "Ward IDs differ between 1994 and 2024."
        )

    class_changes = pd.DataFrame(
        index=pd.Index(
            sorted(lulc_1994.index),
            name=LULC_WARD_ID,
        )
    )

    for class_name, column in CLASS_AREA_COLUMNS.items():

        class_changes[f"{class_name}_change_km2"] = (
            lulc_2024.loc[
                class_changes.index, column
            ].to_numpy()
            -
            lulc_1994.loc[
                class_changes.index, column
            ].to_numpy()
        )

    # --------------------------------------------------------
    # Load ecological indicators
    # --------------------------------------------------------

    ecological = pd.read_csv(ECOLOGICAL_FILE)

    required_eco_columns = [
        WARD_ID,
        DEPENDENT_VARIABLE,
    ]

    missing = [
        column
        for column in required_eco_columns
        if column not in ecological.columns
    ]

    if missing:
        raise ValueError(
            f"Missing ecological columns: {missing}"
        )

    ecological[WARD_ID] = pd.to_numeric(
        ecological[WARD_ID],
        errors="raise",
    ).astype(int)

    if ecological[WARD_ID].duplicated().any():
        raise ValueError(
            "Duplicate ward records in ecological indicators."
        )

    ecological[DEPENDENT_VARIABLE] = pd.to_numeric(
        ecological[DEPENDENT_VARIABLE],
        errors="coerce",
    )

    if ecological[DEPENDENT_VARIABLE].isna().any():
        raise ValueError(
            "Missing LST change values."
        )

    # --------------------------------------------------------
    # Merge all sources
    # --------------------------------------------------------

    data = boundaries.merge(
        class_changes.reset_index(),
        left_on=WARD_ID,
        right_on=LULC_WARD_ID,
        how="inner",
        validate="one_to_one",
    )

    data = data.merge(
        ecological[
            [WARD_ID, DEPENDENT_VARIABLE]
        ],
        on=WARD_ID,
        how="inner",
        validate="one_to_one",
    )

    data = gpd.GeoDataFrame(
        data,
        geometry="geometry",
        crs="EPSG:32643",
    )

    data = data.sort_values(WARD_ID).set_index(WARD_ID)

    if len(data) != EXPECTED_WARDS:
        raise ValueError(
            f"Expected {EXPECTED_WARDS} wards; "
            f"found {len(data)}"
        )

    if data.geometry.isna().any():
        raise ValueError("Missing ward geometries.")

    if data.geometry.is_empty.any():
        raise ValueError("Empty ward geometries found.")

    if not data.geometry.is_valid.all():
        raise ValueError("Invalid ward geometries found.")

    # --------------------------------------------------------
    # Normalize class changes by ward area
    # --------------------------------------------------------

    for class_name in CLASS_AREA_COLUMNS:

        absolute_column = f"{class_name}_change_km2"

        normalized_column = (
            f"{class_name}_change_pct_points"
        )

        data[normalized_column] = (
            data[absolute_column]
            / data["ward_area_km2"]
        ) * 100

    predictor_columns = [
        f"{name}_change_pct_points"
        for name in CLASS_AREA_COLUMNS
    ]

    numeric_columns = [
        DEPENDENT_VARIABLE,
        "ward_area_km2",
        *predictor_columns,
    ]

    if not np.isfinite(
        data[numeric_columns].to_numpy(dtype=float)
    ).all():
        raise ValueError(
            "Non-finite values found in model variables."
        )

    # --------------------------------------------------------
    # Verify compositional closure
    # --------------------------------------------------------

    data["five_class_change_sum"] = data[
        predictor_columns
    ].sum(axis=1)

    maximum_closure_error = (
        data["five_class_change_sum"].abs().max()
    )

    print(
        "\nMaximum normalized five-class closure error:",
        maximum_closure_error,
    )

    if maximum_closure_error > CLOSURE_TOLERANCE:
        raise ValueError(
            "Five-class normalized changes do not "
            "satisfy compositional closure."
        )

    print("Ward count:", len(data))
    print("CRS:", data.crs)
    print("Dependent variable:", DEPENDENT_VARIABLE)
    print("Class predictors:", predictor_columns)

    # Save a clean, non-spatial CSV.
    export_columns = [
        "ward_area_km2",
        DEPENDENT_VARIABLE,
        *[
            f"{name}_change_km2"
            for name in CLASS_AREA_COLUMNS
        ],
        *predictor_columns,
        "five_class_change_sum",
    ]

    data[export_columns].to_csv(
        OUTPUT_DIR
        / "NMMC_Compositional_SAR_Model_Input.csv"
    )

    return data, predictor_columns


# ============================================================
# SPATIAL WEIGHTS
# ============================================================

def build_weights(data):

    print("\n" + "=" * 70)
    print("BUILDING SPATIAL WEIGHTS")
    print("=" * 70)

    weights = {}

    for name, weight_class in [
        ("Queen", Queen),
        ("Rook", Rook),
    ]:

        w = weight_class.from_dataframe(
            data,
            use_index=False,
        )

        if w.islands:
            raise ValueError(
                f"{name} weights contain islands: {w.islands}"
            )

        if w.n_components != 1:
            raise ValueError(
                f"{name} weights are disconnected."
            )

        w.transform = "R"

        weights[name] = w

        print(f"\n{name} weights")
        print("Wards:", w.n)
        print("Components:", w.n_components)
        print("Mean neighbours:", w.mean_neighbors)
        print("Minimum neighbours:", w.min_neighbors)
        print("Maximum neighbours:", w.max_neighbors)

    return weights


# ============================================================
# OLS BASELINE
# ============================================================

def calculate_ols_diagnostics(y, X, w):

    design = np.column_stack([
        np.ones(len(y)),
        X,
    ])

    beta = np.linalg.lstsq(
        design,
        y,
        rcond=None,
    )[0]

    predicted = design @ beta
    residuals = y - predicted

    ss_res = np.sum(residuals ** 2)
    ss_total = np.sum((y - y.mean()) ** 2)

    r_squared = (
        1 - ss_res / ss_total
        if ss_total > 0
        else np.nan
    )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        np.random.seed(RANDOM_SEED)

        moran = Moran(
            residuals,
            w,
            permutations=PERMUTATIONS,
        )

    return {
        "ols_r2": r_squared,
        "ols_residual_moran_i": moran.I,
        "ols_residual_moran_p": moran.p_sim,
    }


# ============================================================
# SAR MODEL
# ============================================================

def fit_sar_model(
    data,
    predictors,
    reference_class,
    weight_name,
    w,
):

    print("\n" + "-" * 70)
    print(
        f"REFERENCE: {reference_class.upper()} | "
        f"WEIGHTS: {weight_name.upper()}"
    )
    print("-" * 70)

    all_classes = list(CLASS_AREA_COLUMNS.keys())

    included_classes = [
        class_name
        for class_name in all_classes
        if class_name != reference_class
    ]

    included_predictors = [
        f"{name}_change_pct_points"
        for name in included_classes
    ]

    y = data[
        DEPENDENT_VARIABLE
    ].to_numpy(dtype=float).reshape(-1, 1)

    X = data[
        included_predictors
    ].to_numpy(dtype=float)

    if y.shape[0] != w.n:
        raise ValueError(
            "Dependent variable and weights have different sizes."
        )

    if X.shape[0] != w.n:
        raise ValueError(
            "Predictors and weights have different sizes."
        )

    if np.linalg.matrix_rank(
        np.column_stack([np.ones(len(X)), X])
    ) < X.shape[1] + 1:
        raise ValueError(
            f"Rank-deficient design matrix for "
            f"reference class {reference_class}."
        )

    ols = calculate_ols_diagnostics(
        y.flatten(),
        X,
        w,
    )

    model = ML_Lag(
        y,
        X,
        w=w,
        method="full",
        epsilon=1e-7,
        spat_diag=True,
        vm=True,
        name_y=DEPENDENT_VARIABLE,
        name_x=included_classes,
        name_w=weight_name,
    )

    rho = float(np.asarray(model.rho).flatten()[0])

    betas = np.asarray(model.betas).flatten()
    model_names = list(model.name_x)
    z_statistics = list(model.z_stat)

    coefficient_rows = []

    coefficient_lookup = {}

    for index, coefficient_name in enumerate(model_names):

        if index >= len(betas):
            continue

        estimate = float(betas[index])

        z_value = np.nan
        p_value = np.nan

        if index < len(z_statistics):
            z_value = float(z_statistics[index][0])
            p_value = float(z_statistics[index][1])

        coefficient_lookup[coefficient_name] = estimate

        coefficient_rows.append({
            "reference_class": reference_class,
            "weights": weight_name,
            "term": coefficient_name,
            "coefficient": estimate,
            "z_statistic": z_value,
            "p_value": p_value,
        })

    # --------------------------------------------------------
    # SAR residual Moran's I
    # --------------------------------------------------------

    residuals = np.asarray(
        model.u
    ).flatten()

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        np.random.seed(RANDOM_SEED)

        residual_moran = Moran(
            residuals,
            w,
            permutations=PERMUTATIONS,
        )

    # --------------------------------------------------------
    # Exact SAR impacts
    # --------------------------------------------------------

    # S = (I - rho W)^(-1)
    W = w.sparse.toarray()

    identity = np.eye(w.n)

    spatial_multiplier = np.linalg.solve(
        identity - rho * W,
        identity,
    )

    average_diagonal = (
        np.trace(spatial_multiplier) / w.n
    )

    average_row_sum = (
        spatial_multiplier.sum(axis=1).mean()
    )

    impact_rows = []

    for class_name in included_classes:

        predictor_name = f"{class_name}_change_pct_points"

        beta = coefficient_lookup.get(
            class_name,
            coefficient_lookup.get(
                predictor_name,
                np.nan,
            ),
        )

        if not np.isfinite(beta):
            # spreg may retain the supplied names in a different
            # format. The predictor order follows included_classes.
            predictor_position = included_classes.index(class_name)
            beta = float(betas[predictor_position + 1])

        direct = beta * average_diagonal
        total = beta * average_row_sum
        indirect = total - direct

        impact_rows.append({
            "reference_class": reference_class,
            "weights": weight_name,
            "predictor": class_name,
            "coefficient": beta,
            "direct_impact": direct,
            "indirect_impact": indirect,
            "total_impact": total,
        })

    diagnostics = {
        "reference_class": reference_class,
        "weights": weight_name,
        "n_wards": w.n,
        "n_predictors": X.shape[1],
        "ols_r2": ols["ols_r2"],
        "ols_residual_moran_i": ols[
            "ols_residual_moran_i"
        ],
        "ols_residual_moran_p": ols[
            "ols_residual_moran_p"
        ],
        "sar_pseudo_r2": float(model.pr2),
        "sar_aic": float(model.aic),
        "sar_schwarz": float(model.schwarz),
        "rho": rho,
        "sar_residual_moran_i": residual_moran.I,
        "sar_residual_moran_p": residual_moran.p_sim,
        "sar_log_likelihood": float(model.logll),
    }

    print("OLS R²:", diagnostics["ols_r2"])
    print("SAR pseudo-R²:", diagnostics["sar_pseudo_r2"])
    print("SAR AIC:", diagnostics["sar_aic"])
    print("Spatial rho:", rho)
    print(
        "SAR residual Moran's I:",
        residual_moran.I,
    )
    print(
        "SAR residual Moran p-value:",
        residual_moran.p_sim,
    )

    return (
        diagnostics,
        coefficient_rows,
        impact_rows,
        np.asarray(model.predy).flatten(),
    )


# ============================================================
# REFERENCE CLASS SENSITIVITY
# ============================================================

def main():

    print("=" * 70)
    print("NMMC COMPOSITIONAL SAR REFERENCE-CLASS SENSITIVITY")
    print("=" * 70)

    data, predictors = load_data()

    weights = build_weights(data)

    all_classes = list(CLASS_AREA_COLUMNS.keys())

    diagnostics_all = []
    coefficients_all = []
    impacts_all = []

    predictions = {}

    # --------------------------------------------------------
    # Fit 5 reference classes × 2 spatial weights
    # --------------------------------------------------------

    for weight_name, w in weights.items():

        for reference_class in all_classes:

            (
                diagnostics,
                coefficients,
                impacts,
                predicted,
            ) = fit_sar_model(
                data=data,
                predictors=predictors,
                reference_class=reference_class,
                weight_name=weight_name,
                w=w,
            )

            diagnostics_all.append(diagnostics)
            coefficients_all.extend(coefficients)
            impacts_all.extend(impacts)

            predictions[
                (weight_name, reference_class)
            ] = predicted

    diagnostics_df = pd.DataFrame(diagnostics_all)
    coefficients_df = pd.DataFrame(coefficients_all)
    impacts_df = pd.DataFrame(impacts_all)

    # --------------------------------------------------------
    # Compare model equivalence across reference classes
    # --------------------------------------------------------

    equivalence_rows = []

    for weight_name in weights:

        reference_prediction = predictions[
            (weight_name, all_classes[0])
        ]

        reference_diagnostic = diagnostics_df[
            (diagnostics_df["weights"] == weight_name)
            &
            (
                diagnostics_df["reference_class"]
                == all_classes[0]
            )
        ].iloc[0]

        for reference_class in all_classes:

            current_prediction = predictions[
                (weight_name, reference_class)
            ]

            current_diagnostic = diagnostics_df[
                (diagnostics_df["weights"] == weight_name)
                &
                (
                    diagnostics_df["reference_class"]
                    == reference_class
                )
            ].iloc[0]

            equivalence_rows.append({
                "weights": weight_name,
                "reference_class": reference_class,
                "max_absolute_prediction_difference": (
                    np.max(
                        np.abs(
                            current_prediction
                            - reference_prediction
                        )
                    )
                ),
                "aic_difference": (
                    current_diagnostic["sar_aic"]
                    - reference_diagnostic["sar_aic"]
                ),
                "rho_difference": (
                    current_diagnostic["rho"]
                    - reference_diagnostic["rho"]
                ),
                "pseudo_r2_difference": (
                    current_diagnostic["sar_pseudo_r2"]
                    - reference_diagnostic["sar_pseudo_r2"]
                ),
            })

    equivalence_df = pd.DataFrame(equivalence_rows)

    # --------------------------------------------------------
    # Save outputs
    # --------------------------------------------------------

    diagnostics_file = (
        OUTPUT_DIR
        / "NMMC_Compositional_SAR_Reference_Diagnostics.csv"
    )

    coefficients_file = (
        OUTPUT_DIR
        / "NMMC_Compositional_SAR_Reference_Coefficients.csv"
    )

    impacts_file = (
        OUTPUT_DIR
        / "NMMC_Compositional_SAR_Reference_Impacts.csv"
    )

    equivalence_file = (
        OUTPUT_DIR
        / "NMMC_Compositional_SAR_Reference_Equivalence.csv"
    )

    diagnostics_df.to_csv(
        diagnostics_file,
        index=False,
    )

    coefficients_df.to_csv(
        coefficients_file,
        index=False,
    )

    impacts_df.to_csv(
        impacts_file,
        index=False,
    )

    equivalence_df.to_csv(
        equivalence_file,
        index=False,
    )

    # --------------------------------------------------------
    # Final summaries
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("REFERENCE-CLASS MODEL DIAGNOSTICS")
    print("=" * 70)

    print(
        diagnostics_df[
            [
                "reference_class",
                "weights",
                "sar_aic",
                "sar_pseudo_r2",
                "rho",
                "sar_residual_moran_i",
                "sar_residual_moran_p",
            ]
        ].round(6).to_string(index=False)
    )

    print("\n" + "=" * 70)
    print("REFERENCE-CLASS EQUIVALENCE")
    print("=" * 70)

    print(
        equivalence_df.round(12).to_string(index=False)
    )

    print("\nMaximum prediction difference by weights:")

    print(
        equivalence_df.groupby("weights")[
            "max_absolute_prediction_difference"
        ].max()
    )

    print("\nMaximum absolute AIC difference by weights:")

    print(
        equivalence_df.groupby("weights")[
            "aic_difference"
        ].apply(lambda x: x.abs().max())
    )

    print("\n" + "=" * 70)
    print("OUTPUT FILES")
    print("=" * 70)

    for path in [
        diagnostics_file,
        coefficients_file,
        impacts_file,
        equivalence_file,
    ]:
        print(path)

    print("\nCompositional SAR sensitivity analysis completed.")


if __name__ == "__main__":
    main()
