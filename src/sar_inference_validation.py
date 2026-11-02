
from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import geopandas as gpd

from libpysal.weights import Queen, Rook
from spreg import ML_Lag


# ============================================================
# CONFIGURATION
# ============================================================

ROOT = Path(__file__).resolve().parent.parent

BOUNDARY_FILE = (
    ROOT / "data" / "boundaries"
    / "NMMC_Ward_Boundaries_2024_CLEAN.geojson"
)

LULC_FILE = (
    ROOT / "outputs" / "reports"
    / "NMMC_Ward_Level_LULC_Statistics_PRELIMINARY_ALL_YEARS.csv"
)

ECOLOGICAL_FILE = (
    ROOT / "outputs" / "reports"
    / "NMMC_Ecological_Impact_Ward_Indicators.csv"
)

OUTPUT_DIR = ROOT / "outputs" / "reports"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

BOUNDARY_ID = "ward_lgd_c"
WARD_ID = "ward_id"
YEAR = "year"
DEPENDENT = "lst_change_c"

CLASSES = {
    "water": "Water_area_km2",
    "vegetation": "Vegetation_area_km2",
    "builtup": "Built-up_area_km2",
    "barren": "Barren_area_km2",
    "wetlands": "Coastal Wetlands_area_km2",
}

ILR_BALANCES = [
    "ILR1_water_vs_rest",
    "ILR2_vegetation_vs_rest",
    "ILR3_builtup_vs_rest",
    "ILR4_barren_vs_wetlands",
]

ILR_BASIS = np.array([
    [
        np.sqrt(4 / 5),
        -1 / np.sqrt(20),
        -1 / np.sqrt(20),
        -1 / np.sqrt(20),
        -1 / np.sqrt(20),
    ],
    [
        0,
        np.sqrt(3 / 4),
        -1 / np.sqrt(12),
        -1 / np.sqrt(12),
        -1 / np.sqrt(12),
    ],
    [
        0,
        0,
        np.sqrt(2 / 3),
        -1 / np.sqrt(6),
        -1 / np.sqrt(6),
    ],
    [
        0,
        0,
        0,
        np.sqrt(1 / 2),
        -np.sqrt(1 / 2),
    ],
])

PIXEL_AREA_KM2 = 0.0009
REPLACEMENT_FACTORS = [0.5, 1.0, 2.0]

EXPECTED_WARDS = 111
RANDOM_SEED = 42

# Monte Carlo draws for impact confidence intervals.
SIMULATION_DRAWS = 10000
CONFIDENCE_LEVEL = 0.95

# Numerical validation tolerances.
COVARIANCE_SYMMETRY_TOL = 1e-7
COVARIANCE_EIGEN_TOL = 1e-8
METHOD_COMPARISON_TOL = 1e-4


# ============================================================
# DATA LOADING
# ============================================================

def load_source_data():

    print("\nLoading source datasets...")

    boundaries = gpd.read_file(BOUNDARY_FILE)

    if BOUNDARY_ID not in boundaries.columns:
        raise ValueError(
            f"Boundary identifier missing: {BOUNDARY_ID}"
        )

    boundaries[WARD_ID] = pd.to_numeric(
        boundaries[BOUNDARY_ID],
        errors="raise",
    ).astype(int)

    if boundaries[WARD_ID].duplicated().any():
        raise ValueError("Duplicate boundary ward IDs.")

    boundaries = boundaries.to_crs("EPSG:32643")

    boundaries["ward_area_km2"] = (
        boundaries.geometry.area / 1_000_000
    )

    if not boundaries.geometry.is_valid.all():
        raise ValueError("Invalid ward geometry found.")

    lulc = pd.read_csv(LULC_FILE)

    required = [
        WARD_ID,
        YEAR,
        *CLASSES.values(),
        "valid_raster_area_km2",
    ]

    missing = [
        c for c in required
        if c not in lulc.columns
    ]

    if missing:
        raise ValueError(
            f"Missing LULC columns: {missing}"
        )

    lulc[WARD_ID] = pd.to_numeric(
        lulc[WARD_ID],
        errors="raise",
    ).astype(int)

    lulc[YEAR] = pd.to_numeric(
        lulc[YEAR],
        errors="raise",
    ).astype(int)

    lulc = lulc[
        lulc[YEAR].isin([1994, 2024])
    ].copy()

    if lulc.duplicated([WARD_ID, YEAR]).any():
        raise ValueError("Duplicate LULC ward-year records.")

    numeric_cols = [
        *CLASSES.values(),
        "valid_raster_area_km2",
    ]

    for col in numeric_cols:
        lulc[col] = pd.to_numeric(
            lulc[col],
            errors="coerce",
        )

    if not np.isfinite(
        lulc[numeric_cols].to_numpy(dtype=float)
    ).all():
        raise ValueError("Invalid LULC area values.")

    if (lulc[numeric_cols] < 0).any().any():
        raise ValueError("Negative LULC area values.")

    ecological = pd.read_csv(ECOLOGICAL_FILE)

    if ecological[WARD_ID].duplicated().any():
        raise ValueError(
            "Duplicate ecological ward records."
        )

    ecological[WARD_ID] = pd.to_numeric(
        ecological[WARD_ID],
        errors="raise",
    ).astype(int)

    ecological[DEPENDENT] = pd.to_numeric(
        ecological[DEPENDENT],
        errors="coerce",
    )

    if ecological[DEPENDENT].isna().any():
        raise ValueError("Missing LST change values.")

    boundary_ids = set(boundaries[WARD_ID])

    for year in [1994, 2024]:

        year_ids = set(
            lulc.loc[lulc[YEAR] == year, WARD_ID]
        )

        if year_ids != boundary_ids:
            raise ValueError(
                f"Ward mismatch for LULC year {year}."
            )

    if set(ecological[WARD_ID]) != boundary_ids:
        raise ValueError(
            "Ecological and boundary ward IDs differ."
        )

    print("Boundary wards:", len(boundaries))
    print("LULC endpoint records:", len(lulc))
    print("Ecological wards:", len(ecological))

    if len(boundaries) != EXPECTED_WARDS:
        raise ValueError("Unexpected ward count.")

    return boundaries, lulc, ecological


# ============================================================
# PREPARE NORMALIZED CLASS CHANGES
# ============================================================

def prepare_normalized_data(
    boundaries,
    lulc,
    ecological,
):

    d94 = (
        lulc[lulc[YEAR] == 1994]
        .set_index(WARD_ID)
    )

    d24 = (
        lulc[lulc[YEAR] == 2024]
        .set_index(WARD_ID)
    )

    wards = sorted(d94.index)

    if set(wards) != set(d24.index):
        raise ValueError("Endpoint ward mismatch.")

    changes = pd.DataFrame(
        index=pd.Index(wards, name=WARD_ID)
    )

    for class_name, area_col in CLASSES.items():

        changes[f"{class_name}_change_km2"] = (
            d24.loc[wards, area_col].to_numpy()
            - d94.loc[wards, area_col].to_numpy()
        )

    # Verify area closure before normalization.
    class_change_sum = changes[
        [f"{name}_change_km2" for name in CLASSES]
    ].sum(axis=1)

    if class_change_sum.abs().max() > 1e-8:
        raise ValueError(
            "Five-class area change closure failed."
        )

    data = boundaries.merge(
        changes.reset_index(),
        on=WARD_ID,
        how="inner",
        validate="one_to_one",
    )

    for class_name in CLASSES:

        data[
            f"{class_name}_change_pct_points"
        ] = (
            data[f"{class_name}_change_km2"]
            / data["ward_area_km2"]
        ) * 100

    data = data.merge(
        ecological[[WARD_ID, DEPENDENT]],
        on=WARD_ID,
        how="inner",
        validate="one_to_one",
    )

    data = gpd.GeoDataFrame(
        data,
        geometry="geometry",
        crs="EPSG:32643",
    )

    return data.sort_values(WARD_ID).reset_index(drop=True)


# ============================================================
# ILR ZERO REPLACEMENT
# ============================================================

def replace_zeros(
    areas,
    valid_area,
    factor,
):

    composition = (
        np.asarray(areas, dtype=float)
        / float(valid_area)
    )

    if not np.isclose(
        composition.sum(),
        1.0,
        atol=1e-8,
    ):
        raise ValueError("Composition closure failed.")

    zero_mask = composition == 0
    zero_count = int(zero_mask.sum())

    if zero_count == 0:
        return composition

    positive = composition[~zero_mask]

    delta = min(
        factor * PIXEL_AREA_KM2 / valid_area,
        positive.min() * 0.5,
        0.25 / zero_count,
    )

    if delta <= 0 or not np.isfinite(delta):
        raise ValueError("Invalid zero replacement.")

    result = composition.copy()

    result[zero_mask] = delta

    result[~zero_mask] = (
        positive
        * (1 - zero_count * delta)
        / positive.sum()
    )

    if not np.isclose(
        result.sum(),
        1.0,
        atol=1e-10,
    ):
        raise ValueError("Replaced composition failed closure.")

    return result


def prepare_ilr_data(
    boundaries,
    lulc,
    ecological,
    factor,
):

    records = []

    for _, row in lulc.iterrows():

        areas = np.array([
            row[CLASSES[name]]
            for name in CLASSES
        ], dtype=float)

        valid_area = float(
            row["valid_raster_area_km2"]
        )

        composition = replace_zeros(
            areas,
            valid_area,
            factor,
        )

        balances = ILR_BASIS @ np.log(composition)

        record = {
            WARD_ID: int(row[WARD_ID]),
            YEAR: int(row[YEAR]),
        }

        for name, value in zip(
            ILR_BALANCES,
            balances,
        ):
            record[name] = float(value)

        records.append(record)

    ilr = pd.DataFrame(records)

    d94 = (
        ilr[ilr[YEAR] == 1994]
        .set_index(WARD_ID)
    )

    d24 = (
        ilr[ilr[YEAR] == 2024]
        .set_index(WARD_ID)
    )

    wards = sorted(d94.index)

    changes = pd.DataFrame(
        index=pd.Index(wards, name=WARD_ID)
    )

    for balance in ILR_BALANCES:

        changes[f"delta_{balance}"] = (
            d24.loc[wards, balance].to_numpy()
            - d94.loc[wards, balance].to_numpy()
        )

    data = boundaries.merge(
        changes.reset_index(),
        on=WARD_ID,
        how="inner",
        validate="one_to_one",
    )

    data = data.merge(
        ecological[[WARD_ID, DEPENDENT]],
        on=WARD_ID,
        how="inner",
        validate="one_to_one",
    )

    data = gpd.GeoDataFrame(
        data,
        geometry="geometry",
        crs="EPSG:32643",
    )

    return data.sort_values(WARD_ID).reset_index(drop=True)


# ============================================================
# SPATIAL WEIGHTS
# ============================================================

def build_weights(data):

    result = {}

    for name, cls in [
        ("Queen", Queen),
        ("Rook", Rook),
    ]:

        w = cls.from_dataframe(
            data,
            use_index=False,
        )

        if w.islands or w.n_components != 1:
            raise ValueError(
                f"{name} weights are not fully connected."
            )

        w.transform = "R"

        result[name] = w

    return result


# ============================================================
# COVARIANCE VALIDATION
# ============================================================

def inspect_covariance(model):

    beta = np.asarray(
        model.betas,
        dtype=float,
    ).flatten()

    covariance = np.asarray(
        model.vm,
        dtype=float,
    )

    result = {
        "parameter_count": len(beta),
        "covariance_shape": str(covariance.shape),
        "covariance_finite": False,
        "covariance_symmetric": False,
        "covariance_min_eigenvalue": np.nan,
        "covariance_max_eigenvalue": np.nan,
        "covariance_condition_number": np.nan,
        "covariance_positive_semidefinite": False,
        "standard_errors_finite": False,
        "covariance_usable_for_simulation": False,
        "failure_reason": "",
    }

    if covariance.shape != (
        len(beta),
        len(beta),
    ):
        result["failure_reason"] = (
            "Covariance shape does not match parameter vector."
        )
        return result

    if not np.isfinite(beta).all():
        result["failure_reason"] = (
            "Non-finite parameter estimate."
        )
        return result

    if not np.isfinite(covariance).all():
        result["failure_reason"] = (
            "Covariance contains non-finite values."
        )
        return result

    result["covariance_finite"] = True

    symmetry_error = np.max(
        np.abs(covariance - covariance.T)
    )

    result["covariance_symmetry_error"] = symmetry_error

    result["covariance_symmetric"] = (
        symmetry_error <= COVARIANCE_SYMMETRY_TOL
    )

    covariance = (
        covariance + covariance.T
    ) / 2

    eigenvalues = np.linalg.eigvalsh(covariance)

    minimum = float(eigenvalues.min())
    maximum = float(eigenvalues.max())

    result["covariance_min_eigenvalue"] = minimum
    result["covariance_max_eigenvalue"] = maximum

    if maximum > 0:
        result["covariance_condition_number"] = (
            np.linalg.cond(covariance)
        )

    scale = max(1.0, abs(maximum))

    psd = minimum >= -COVARIANCE_EIGEN_TOL * scale

    result["covariance_positive_semidefinite"] = psd

    standard_errors = np.sqrt(
        np.maximum(np.diag(covariance), 0)
    )

    result["standard_errors_finite"] = bool(
        np.isfinite(standard_errors).all()
    )

    usable = (
        result["covariance_symmetric"]
        and psd
        and result["standard_errors_finite"]
    )

    result["covariance_usable_for_simulation"] = usable

    if not usable:
        result["failure_reason"] = (
            "Covariance failed symmetry, PSD or standard-error checks."
        )

    return result


# ============================================================
# SAR IMPACT SIMULATION
# ============================================================


def simulate_impacts(
    model,
    w,
    predictor_names,
    specification,
    method,
):
    """
    Independently validates SAR impact calculations.

    Point estimates:
        Calculated directly from (I - rho W)^-1.

    Confidence intervals:
        Calculated using multivariate normal parameter simulation.

    Direct impact:
        beta * trace(S) / n

    Total impact:
        beta * mean(row sums of S)

    Indirect impact:
        Total - Direct

    where S = (I - rho W)^-1.
    """

    covariance_check = inspect_covariance(model)

    if not covariance_check[
        "covariance_usable_for_simulation"
    ]:
        print(
            "Impact simulation skipped:",
            covariance_check["failure_reason"],
        )
        return [], covariance_check

    # --------------------------------------------------------
    # 1. Extract fitted parameters and covariance
    # --------------------------------------------------------

    mean = np.asarray(
        model.betas,
        dtype=float,
    ).flatten()

    covariance = np.asarray(
        model.vm,
        dtype=float,
    )

    covariance = (
        covariance + covariance.T
    ) / 2

    n = w.n
    k = len(predictor_names)

    # Expected ordering:
    # intercept, X coefficients, spatial rho
    rho_index = k + 1

    if len(mean) != rho_index + 1:
        raise ValueError(
            "Unexpected ML_Lag parameter ordering."
        )

    fitted_rho = float(mean[rho_index])

    fitted_betas = mean[1:rho_index]

    # --------------------------------------------------------
    # 2. Build the spatial multiplier matrix
    # --------------------------------------------------------

    W = np.asarray(
        w.sparse.toarray(),
        dtype=float,
    )

    identity = np.eye(n)

    multiplier_matrix = np.linalg.solve(
        identity - fitted_rho * W,
        identity,
    )

    if not np.isfinite(multiplier_matrix).all():
        raise ValueError(
            "Non-finite spatial multiplier matrix."
        )

    # Matrix-based average direct and total multipliers.
    direct_multiplier_matrix = (
        np.trace(multiplier_matrix) / n
    )

    total_multiplier_matrix = (
        multiplier_matrix.sum(axis=1).mean()
    )

    indirect_multiplier_matrix = (
        total_multiplier_matrix
        - direct_multiplier_matrix
    )

    # --------------------------------------------------------
    # 3. Independently verify row-standardized identity
    # --------------------------------------------------------

    total_multiplier_identity = (
        1.0 / (1.0 - fitted_rho)
    )

    total_identity_error = abs(
        total_multiplier_matrix
        - total_multiplier_identity
    )

    # Eigenvalue-based trace calculation.
    eigenvalues = np.linalg.eigvals(W)

    direct_multiplier_eigen = np.real(
        np.sum(
            1.0 / (1.0 - fitted_rho * eigenvalues)
        ) / n
    )

    direct_eigen_error = abs(
        direct_multiplier_matrix
        - direct_multiplier_eigen
    )

    verification_tolerance = 1e-7

    matrix_validation_pass = (
        total_identity_error <= verification_tolerance
        and direct_eigen_error <= verification_tolerance
    )

    validation_record = {
        "matrix_validation_pass": bool(
            matrix_validation_pass
        ),
        "fitted_rho": fitted_rho,
        "matrix_direct_multiplier": float(
            direct_multiplier_matrix
        ),
        "eigen_direct_multiplier": float(
            direct_multiplier_eigen
        ),
        "direct_multiplier_absolute_error": float(
            direct_eigen_error
        ),
        "matrix_total_multiplier": float(
            total_multiplier_matrix
        ),
        "identity_total_multiplier": float(
            total_multiplier_identity
        ),
        "total_multiplier_absolute_error": float(
            total_identity_error
        ),
        "matrix_indirect_multiplier": float(
            indirect_multiplier_matrix
        ),
    }

    print("\nSpatial multiplier validation:")
    print(validation_record)

    if not matrix_validation_pass:
        raise ValueError(
            "Independent spatial multiplier validation failed."
        )

    # --------------------------------------------------------
    # 4. Calculate point estimates at fitted parameters
    # --------------------------------------------------------

    point_estimates = {}

    for index, predictor in enumerate(predictor_names):

        beta = float(fitted_betas[index])

        direct = (
            beta * direct_multiplier_matrix
        )

        total = (
            beta * total_multiplier_matrix
        )

        indirect = total - direct

        point_estimates[predictor] = {
            "direct": float(direct),
            "indirect": float(indirect),
            "total": float(total),
        }

    # --------------------------------------------------------
    # 5. Prepare multivariate parameter simulation
    # --------------------------------------------------------

    eigenvalues_cov, eigenvectors_cov = np.linalg.eigh(
        covariance
    )

    if eigenvalues_cov.min() < -1e-8:
        raise ValueError(
            "Covariance matrix is not positive semidefinite."
        )

    eigenvalues_cov = np.maximum(
        eigenvalues_cov,
        0,
    )

    covariance_root = (
        eigenvectors_cov
        @ np.diag(np.sqrt(eigenvalues_cov))
    )

    rng = np.random.default_rng(RANDOM_SEED)

    standard_normal = rng.standard_normal(
        (SIMULATION_DRAWS, len(mean))
    )

    draws = (
        mean
        + standard_normal @ covariance_root.T
    )

    rho_draws = draws[:, rho_index]

    valid = (
        np.isfinite(draws).all(axis=1)
        & (rho_draws > -0.999)
        & (rho_draws < 0.999)
    )

    draws = draws[valid]
    rho_draws = rho_draws[valid]

    valid_count = len(draws)

    if valid_count < SIMULATION_DRAWS * 0.5:
        raise ValueError(
            "Insufficient valid parameter simulation draws."
        )

    # --------------------------------------------------------
    # 6. Calculate simulated spatial multipliers
    # --------------------------------------------------------

    # For row-standardized W, average row sum of S is
    # exactly 1 / (1-rho).

    total_multiplier_draws = (
        1.0 / (1.0 - rho_draws)
    )

    # Trace(S)/n using the eigenvalues of W.
    direct_multiplier_draws = np.empty(
        valid_count,
        dtype=float,
    )

    for start in range(0, valid_count, 500):

        stop = min(
            start + 500,
            valid_count,
        )

        rho_chunk = rho_draws[start:stop]

        values = 1.0 / (
            1.0
            - rho_chunk[:, None]
            * eigenvalues[None, :]
        )

        direct_multiplier_draws[start:stop] = (
            np.real(values.sum(axis=1)) / n
        )

    indirect_multiplier_draws = (
        total_multiplier_draws
        - direct_multiplier_draws
    )

    # --------------------------------------------------------
    # 7. Calculate confidence intervals
    # --------------------------------------------------------

    lower_q = (
        1 - CONFIDENCE_LEVEL
    ) / 2

    upper_q = 1 - lower_q

    impact_rows = []

    for index, predictor in enumerate(predictor_names):

        beta_draws = draws[:, index + 1]

        direct_draws = (
            beta_draws
            * direct_multiplier_draws
        )

        total_draws = (
            beta_draws
            * total_multiplier_draws
        )

        indirect_draws = (
            total_draws
            - direct_draws
        )

        simulated = {
            "direct": direct_draws,
            "indirect": indirect_draws,
            "total": total_draws,
        }

        for impact_type, values in simulated.items():

            lower, upper = np.quantile(
                values,
                [lower_q, upper_q],
            )

            impact_rows.append({
                **specification,

                "estimation_method": method,
                "predictor": predictor,
                "impact_type": impact_type,

                # Actual fitted-parameter point estimate.
                "estimate": point_estimates[
                    predictor
                ][impact_type],

                # Simulation distribution diagnostics.
                "simulation_median": float(
                    np.median(values)
                ),

                "ci_lower": float(lower),
                "ci_upper": float(upper),

                "confidence_level": CONFIDENCE_LEVEL,
                "simulation_draws_requested": SIMULATION_DRAWS,
                "simulation_draws_valid": valid_count,

                "interval_excludes_zero": bool(
                    lower > 0 or upper < 0
                ),

                **validation_record,
            })

    return impact_rows, covariance_check



# ============================================================
# FIT FULL AND ORD MODELS
# ============================================================

def fit_specification(
    data,
    w,
    weight_name,
    predictors,
    specification,
):

    X = data[predictors].to_numpy(dtype=float)

    y = data[
        DEPENDENT
    ].to_numpy(dtype=float).reshape(-1, 1)

    if not np.isfinite(X).all():
        raise ValueError("Non-finite predictors.")

    if not np.isfinite(y).all():
        raise ValueError("Non-finite dependent variable.")

    if np.linalg.matrix_rank(
        np.column_stack([np.ones(len(X)), X])
    ) != X.shape[1] + 1:
        raise ValueError("Rank-deficient model design.")

    fitted = {}
    comparison_rows = []
    covariance_rows = []
    impact_rows = []

    for method in ["full", "ord"]:

        print(
            f"\nFitting {specification} | "
            f"{weight_name} | method={method}"
        )

        captured = []

        try:

            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")

                model = ML_Lag(
                    y,
                    X,
                    w=w,
                    method=method,
                    epsilon=1e-7,
                    spat_diag=True,
                    vm=True,
                    name_y=DEPENDENT,
                    name_x=predictors,
                    name_w=weight_name,
                )

                captured = [
                    str(item.message)
                    for item in caught
                ]

            beta = np.asarray(
                model.betas,
                dtype=float,
            ).flatten()

            if not np.isfinite(beta).all():
                raise ValueError(
                    "Non-finite fitted parameters."
                )

            rho = float(
                np.asarray(model.rho).flatten()[0]
            )

            if not (-1 < rho < 1):
                raise ValueError(
                    f"Spatial rho outside expected interval: {rho}"
                )

            covariance_check = inspect_covariance(model)

            fitted[method] = model

            comparison_rows.append({
                **specification,
                "weights": weight_name,
                "method": method,
                "rho": rho,
                "pseudo_r2": float(model.pr2),
                "aic": float(model.aic),
                "log_likelihood": float(model.logll),
                "warning_count": len(captured),
                "warning_messages": " | ".join(
                    sorted(set(captured))
                ),
                "fit_success": True,
            })

            covariance_rows.append({
                **specification,
                "weights": weight_name,
                "method": method,
                **covariance_check,
            })

            print("rho:", rho)
            print("pseudo-R²:", float(model.pr2))
            print("AIC:", float(model.aic))
            print("Warnings:", len(captured))
            print(
                "Covariance usable:",
                covariance_check[
                    "covariance_usable_for_simulation"
                ],
            )

        except Exception as exc:

            comparison_rows.append({
                **specification,
                "weights": weight_name,
                "method": method,
                "fit_success": False,
                "failure_reason": str(exc),
                "warning_count": len(captured),
                "warning_messages": " | ".join(
                    sorted(set(captured))
                ),
            })

            print("FIT FAILED:", exc)

    # --------------------------------------------------------
    # Compare full and ord estimates.
    # --------------------------------------------------------

    if "full" in fitted and "ord" in fitted:

        full = fitted["full"]
        ord_model = fitted["ord"]

        full_beta = np.asarray(full.betas).flatten()
        ord_beta = np.asarray(ord_model.betas).flatten()

        difference = np.abs(
            full_beta - ord_beta
        )

        comparison_rows.append({
            **specification,
            "weights": weight_name,
            "method": "full_vs_ord",
            "max_parameter_difference": float(
                difference.max()
            ),
            "rho_difference": abs(
                float(full.rho) - float(ord_model.rho)
            ),
            "aic_difference": abs(
                float(full.aic) - float(ord_model.aic)
            ),
            "method_equivalent": bool(
                difference.max() <= METHOD_COMPARISON_TOL
            ),
            "fit_success": True,
        })

    # --------------------------------------------------------
    # Use ORD for inference.
    # --------------------------------------------------------

    if "ord" in fitted:

        model = fitted["ord"]

        impact_specification = {
            **specification,
            "weights": weight_name,
        }

        impacts, covariance_check = simulate_impacts(
            model=model,
            w=w,
            predictor_names=predictors,
            specification=impact_specification,
            method="ord",
        )

        impact_rows.extend(impacts)

        if not covariance_check[
            "covariance_usable_for_simulation"
        ]:
            print(
                "Confidence intervals unavailable for this model."
            )

    return (
        comparison_rows,
        covariance_rows,
        impact_rows,
    )


# ============================================================
# MAIN PIPELINE
# ============================================================


def main():

    print("=" * 75)
    print("NMMC SAR NUMERICAL STABILITY AND INFERENCE VALIDATION")
    print("=" * 75)

    # --------------------------------------------------------
    # 1. LOAD DATA
    # --------------------------------------------------------

    boundaries, lulc, ecological = load_source_data()

    normalized = prepare_normalized_data(
        boundaries,
        lulc,
        ecological,
    )

    model_comparisons = []
    covariance_audits = []
    impacts_all = []

    # --------------------------------------------------------
    # 2. NORMALIZED BASELINE SAR
    # --------------------------------------------------------

    print("\n" + "=" * 75)
    print("NORMALIZED BASELINE SAR")
    print("=" * 75)

    normalized_predictors = [
        "builtup_change_pct_points",
        "vegetation_change_pct_points",
        "wetlands_change_pct_points",
    ]

    normalized_weights = build_weights(normalized)

    for weight_name, w in normalized_weights.items():

        specification = {
            "model_type": "NORMALIZED_SAR",
            "replacement_factor_pixels": np.nan,
        }

        results = fit_specification(
            data=normalized,
            w=w,
            weight_name=weight_name,
            predictors=normalized_predictors,
            specification=specification,
        )

        model_comparisons.extend(results[0])
        covariance_audits.extend(results[1])
        impacts_all.extend(results[2])

    # --------------------------------------------------------
    # 3. ILR SAR MODELS
    # --------------------------------------------------------

    for factor in REPLACEMENT_FACTORS:

        print("\n" + "=" * 75)
        print(f"ILR ZERO REPLACEMENT: {factor} PIXEL(S)")
        print("=" * 75)

        ilr_data = prepare_ilr_data(
            boundaries,
            lulc,
            ecological,
            factor,
        )

        predictors = [
            f"delta_{name}"
            for name in ILR_BALANCES
        ]

        ilr_weights = build_weights(ilr_data)

        for weight_name, w in ilr_weights.items():

            specification = {
                "model_type": "ILR_SAR",
                "replacement_factor_pixels": factor,
            }

            results = fit_specification(
                data=ilr_data,
                w=w,
                weight_name=weight_name,
                predictors=predictors,
                specification=specification,
            )

            model_comparisons.extend(results[0])
            covariance_audits.extend(results[1])
            impacts_all.extend(results[2])

    # --------------------------------------------------------
    # 4. CREATE OUTPUT DATAFRAMES
    # --------------------------------------------------------

    comparison_df = pd.DataFrame(model_comparisons)
    covariance_df = pd.DataFrame(covariance_audits)
    impacts_df = pd.DataFrame(impacts_all)

    # --------------------------------------------------------
    # 5. EXPORT CSV REPORTS
    # --------------------------------------------------------

    files = {
        "NMMC_SAR_Method_Comparison.csv": comparison_df,
        "NMMC_SAR_Covariance_Validation.csv": covariance_df,
        "NMMC_SAR_Impact_Confidence_Intervals.csv": impacts_df,
    }

    print("\n" + "=" * 75)
    print("EXPORTING REPORTS")
    print("=" * 75)

    for filename, frame in files.items():

        path = OUTPUT_DIR / filename

        frame.to_csv(
            path,
            index=False,
        )

        print("Saved:", path)

    # --------------------------------------------------------
    # 6. METHOD COMPARISON SUMMARY
    # --------------------------------------------------------

    print("\n" + "=" * 75)
    print("METHOD COMPARISON SUMMARY")
    print("=" * 75)

    summary_columns = [
        "model_type",
        "replacement_factor_pixels",
        "weights",
        "method",
        "rho",
        "pseudo_r2",
        "aic",
        "warning_count",
        "fit_success",
        "method_equivalent",
    ]

    available = [
        col for col in summary_columns
        if col in comparison_df.columns
    ]

    print(
        comparison_df[available]
        .to_string(index=False)
    )

    # --------------------------------------------------------
    # 7. COVARIANCE VALIDATION SUMMARY
    # --------------------------------------------------------

    print("\n" + "=" * 75)
    print("COVARIANCE VALIDATION SUMMARY")
    print("=" * 75)

    covariance_columns = [
        "model_type",
        "replacement_factor_pixels",
        "weights",
        "method",
        "covariance_finite",
        "covariance_symmetric",
        "covariance_min_eigenvalue",
        "covariance_condition_number",
        "covariance_positive_semidefinite",
        "covariance_usable_for_simulation",
        "failure_reason",
    ]

    available = [
        col for col in covariance_columns
        if col in covariance_df.columns
    ]

    print(
        covariance_df[available]
        .to_string(index=False)
    )

    # --------------------------------------------------------
    # 8. INDEPENDENT SPATIAL MULTIPLIER VALIDATION
    # --------------------------------------------------------

    print("\n" + "=" * 75)
    print("SPATIAL MULTIPLIER VALIDATION SUMMARY")
    print("=" * 75)

    if len(impacts_df):

        validation_columns = [
            "model_type",
            "replacement_factor_pixels",
            "weights",
            "matrix_validation_pass",
            "fitted_rho",
            "matrix_direct_multiplier",
            "eigen_direct_multiplier",
            "direct_multiplier_absolute_error",
            "matrix_total_multiplier",
            "identity_total_multiplier",
            "total_multiplier_absolute_error",
        ]

        available = [
            col for col in validation_columns
            if col in impacts_df.columns
        ]

        # One row per fitted model; validation values repeat
        # across its predictors and impact types.
        validation_df = (
            impacts_df[available]
            .drop_duplicates(
                subset=[
                    "model_type",
                    "replacement_factor_pixels",
                    "weights",
                ]
            )
        )

        print(
            validation_df
            .round(10)
            .to_string(index=False)
        )

        validation_pass_count = int(
            validation_df[
                "matrix_validation_pass"
            ].fillna(False).sum()
        )

        validation_model_count = len(validation_df)

        print(
            f"\nSpatial multiplier validation: "
            f"{validation_pass_count}/{validation_model_count} PASS"
        )

    else:

        print(
            "Spatial multiplier validation unavailable "
            "because no impact records were generated."
        )

    # --------------------------------------------------------
    # 9. IMPACT CONFIDENCE INTERVAL SUMMARY
    # --------------------------------------------------------

    print("\n" + "=" * 75)
    print("IMPACT CONFIDENCE INTERVAL SUMMARY")
    print("=" * 75)

    if len(impacts_df):

        impact_columns = [
            "model_type",
            "replacement_factor_pixels",
            "weights",
            "predictor",
            "impact_type",
            "estimate",
            "simulation_median",
            "ci_lower",
            "ci_upper",
            "interval_excludes_zero",
            "simulation_draws_valid",
        ]

        available = [
            col for col in impact_columns
            if col in impacts_df.columns
        ]

        print(
            impacts_df[available]
            .round(8)
            .to_string(index=False)
        )

        # Count confidence intervals excluding zero.
        significant_count = int(
            impacts_df[
                "interval_excludes_zero"
            ].fillna(False).sum()
        )

        total_intervals = len(impacts_df)

        print(
            f"\nIntervals excluding zero: "
            f"{significant_count}/{total_intervals}"
        )

        # Check all simulation draw counts.
        if "simulation_draws_valid" in impacts_df.columns:

            draw_counts = (
                impacts_df[
                    "simulation_draws_valid"
                ].value_counts().sort_index()
            )

            print("\nSimulation draw count distribution:")
            print(draw_counts.to_string())

    else:

        print(
            "No confidence intervals generated. "
            "Covariance validation did not pass."
        )

    # --------------------------------------------------------
    # 10. FINAL VALIDATION SUMMARY
    # --------------------------------------------------------

    print("\n" + "=" * 75)
    print("FINAL VALIDATION SUMMARY")
    print("=" * 75)

    # Actual model fits: exclude full_vs_ord comparison rows.
    fit_rows = comparison_df[
        comparison_df["method"].isin(["full", "ord"])
    ]

    successful_fits = int(
        fit_rows["fit_success"].fillna(False).sum()
    )

    total_fits = len(fit_rows)

    usable_covariances = int(
        covariance_df[
            "covariance_usable_for_simulation"
        ].fillna(False).sum()
    )

    total_covariances = len(covariance_df)

    print(
        f"Successful model fits: "
        f"{successful_fits}/{total_fits}"
    )

    print(
        f"Usable covariance matrices: "
        f"{usable_covariances}/{total_covariances}"
    )

    print(
        f"Impact intervals generated: "
        f"{len(impacts_df)}"
    )

    if len(impacts_df):

        matrix_pass = bool(
            impacts_df[
                "matrix_validation_pass"
            ].fillna(False).all()
        )

        print(
            "All spatial multiplier checks passed:",
            matrix_pass,
        )

        if "simulation_draws_valid" in impacts_df.columns:

            all_draws_valid = bool(
                (
                    impacts_df["simulation_draws_valid"]
                    == SIMULATION_DRAWS
                ).all()
            )

            print(
                "All requested simulation draws retained:",
                all_draws_valid,
            )

    print("\n" + "=" * 75)
    print("VALIDATION PIPELINE COMPLETED")
    print("=" * 75)


if __name__ == "__main__":
    main()
