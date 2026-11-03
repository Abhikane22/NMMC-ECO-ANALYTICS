
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
YEAR_COLUMN = "year"

DEPENDENT = "lst_change_c"

CLASSES = {
    "water": "Water_area_km2",
    "vegetation": "Vegetation_area_km2",
    "builtup": "Built-up_area_km2",
    "barren": "Barren_area_km2",
    "wetlands": "Coastal Wetlands_area_km2",
}

VALID_AREA = "valid_raster_area_km2"

# 30 m × 30 m = 900 m² = 0.0009 km².
PIXEL_AREA_KM2 = 0.0009

# Sensitivity magnitudes, expressed as equivalent full pixels.
REPLACEMENT_FACTORS = [0.5, 1.0, 2.0]

EXPECTED_WARDS = 111
RANDOM_SEED = 42
PERMUTATIONS = 999


# ============================================================
# ILR BASIS
# ============================================================

CLASS_NAMES = list(CLASSES.keys())

BALANCE_NAMES = [
    "ILR1_water_vs_rest",
    "ILR2_vegetation_vs_rest",
    "ILR3_builtup_vs_rest",
    "ILR4_barren_vs_wetlands",
]

# Rows are orthonormal balance vectors.
# Column order: water, vegetation, builtup, barren, wetlands.

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


# ============================================================
# DATA LOADING
# ============================================================

def load_inputs():

    print("\nLoading boundaries, LULC and ecological data...")

    boundaries = gpd.read_file(BOUNDARY_FILE)

    if BOUNDARY_ID not in boundaries.columns:
        raise ValueError(
            f"Missing boundary identifier: {BOUNDARY_ID}"
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

    lulc = pd.read_csv(LULC_FILE)

    required_lulc = [
        WARD_ID,
        YEAR_COLUMN,
        VALID_AREA,
        *CLASSES.values(),
    ]

    missing = [
        col for col in required_lulc
        if col not in lulc.columns
    ]

    if missing:
        raise ValueError(
            f"Missing LULC columns: {missing}"
        )

    lulc[WARD_ID] = pd.to_numeric(
        lulc[WARD_ID],
        errors="raise",
    ).astype(int)

    lulc[YEAR_COLUMN] = pd.to_numeric(
        lulc[YEAR_COLUMN],
        errors="raise",
    ).astype(int)

    lulc = lulc[
        lulc[YEAR_COLUMN].isin([1994, 2024])
    ].copy()

    if lulc.duplicated([WARD_ID, YEAR_COLUMN]).any():
        raise ValueError("Duplicate LULC ward-year records.")

    numeric_columns = [
        VALID_AREA,
        *CLASSES.values(),
    ]

    for col in numeric_columns:
        lulc[col] = pd.to_numeric(
            lulc[col],
            errors="coerce",
        )

    if not np.isfinite(
        lulc[numeric_columns].to_numpy(dtype=float)
    ).all():
        raise ValueError(
            "Missing or non-finite LULC area values."
        )

    if (lulc[numeric_columns] < 0).any().any():
        raise ValueError("Negative LULC area found.")

    for year in [1994, 2024]:
        if year not in lulc[YEAR_COLUMN].unique():
            raise ValueError(f"Missing LULC year: {year}")

    # Verify five-class closure against valid area.
    class_sum = lulc[list(CLASSES.values())].sum(axis=1)

    closure_error = (
        class_sum - lulc[VALID_AREA]
    ).abs().max()

    print("Maximum five-class area closure error:", closure_error)

    if closure_error > 1e-8:
        raise ValueError(
            "LULC five-class area closure failed."
        )

    ecological = pd.read_csv(ECOLOGICAL_FILE)

    if WARD_ID not in ecological.columns:
        raise ValueError(
            f"Ecological data missing {WARD_ID}"
        )

    if DEPENDENT not in ecological.columns:
        raise ValueError(
            f"Ecological data missing {DEPENDENT}"
        )

    ecological[WARD_ID] = pd.to_numeric(
        ecological[WARD_ID],
        errors="raise",
    ).astype(int)

    ecological[DEPENDENT] = pd.to_numeric(
        ecological[DEPENDENT],
        errors="coerce",
    )

    if ecological[WARD_ID].duplicated().any():
        raise ValueError(
            "Duplicate ward records in ecological data."
        )

    if ecological[DEPENDENT].isna().any():
        raise ValueError("Missing LST change observations.")

    # Require identical ward sets.
    boundary_ids = set(boundaries[WARD_ID])
    ecological_ids = set(ecological[WARD_ID])

    for year in [1994, 2024]:
        year_ids = set(
            lulc.loc[
                lulc[YEAR_COLUMN] == year,
                WARD_ID,
            ]
        )

        if year_ids != boundary_ids:
            raise ValueError(
                f"Ward mismatch between boundaries and LULC {year}."
            )

    if ecological_ids != boundary_ids:
        raise ValueError(
            "Ward mismatch between boundaries and ecological data."
        )

    print("Boundary wards:", len(boundaries))
    print("LULC records:", len(lulc))
    print("Ecological wards:", len(ecological))

    return boundaries, lulc, ecological


# ============================================================
# MULTIPLICATIVE ZERO REPLACEMENT
# ============================================================

def replace_zeros_multiplicative(
    areas,
    valid_area,
    pixel_factor,
):
    """
    Convert class areas to proportions.

    Replace observed zeros using a resolution-informed
    detection threshold, then proportionally rescale
    nonzero components to preserve closure.

    Replacement is capped:
      1. Below half the smallest observed positive component.
      2. So total replacement mass does not exceed 25%.

    This is a sensitivity treatment, not proof that
    observed zeros are sampling zeros.
    """

    proportions = np.asarray(
        areas,
        dtype=float,
    ) / float(valid_area)

    if not np.isfinite(proportions).all():
        raise ValueError("Non-finite composition.")

    if (proportions < 0).any():
        raise ValueError("Negative composition.")

    total = proportions.sum()

    if not np.isclose(total, 1.0, atol=1e-8):
        raise ValueError(
            f"Composition does not sum to 1: {total}"
        )

    zero_mask = proportions == 0
    zero_count = int(zero_mask.sum())

    if zero_count == 0:
        return proportions.copy(), 0.0

    positive = proportions[~zero_mask]

    if len(positive) == 0:
        raise ValueError(
            "Composition contains no positive classes."
        )

    # Equivalent pixel area as a fraction of valid area.
    raw_delta = (
        pixel_factor * PIXEL_AREA_KM2 / valid_area
    )

    # Do not assign more than half the smallest positive part.
    positive_cap = positive.min() * 0.5

    # Limit total mass allocated to zeros.
    total_mass_cap = 0.25 / zero_count

    delta = min(
        raw_delta,
        positive_cap,
        total_mass_cap,
    )

    if not np.isfinite(delta) or delta <= 0:
        raise ValueError("Invalid zero replacement value.")

    replaced = proportions.copy()

    replaced[zero_mask] = delta

    # Multiplicative rescaling of nonzero components.
    original_positive_sum = positive.sum()

    remaining_mass = 1.0 - zero_count * delta

    replaced[~zero_mask] = (
        positive
        * remaining_mass
        / original_positive_sum
    )

    if (replaced <= 0).any():
        raise ValueError(
            "Zero replacement produced non-positive values."
        )

    if not np.isclose(
        replaced.sum(),
        1.0,
        atol=1e-10,
    ):
        raise ValueError(
            "Zero-replaced composition failed closure."
        )

    return replaced, delta


# ============================================================
# CONSTRUCT ILR COMPOSITIONS
# ============================================================

def construct_ilr_data(
    lulc,
    pixel_factor,
):

    results = []
    replacement_records = []

    for _, row in lulc.iterrows():

        ward_id = int(row[WARD_ID])
        year = int(row[YEAR_COLUMN])

        areas = np.array([
            row[CLASSES[class_name]]
            for class_name in CLASS_NAMES
        ], dtype=float)

        valid_area = float(row[VALID_AREA])

        composition, delta = replace_zeros_multiplicative(
            areas=areas,
            valid_area=valid_area,
            pixel_factor=pixel_factor,
        )

        balances = ILR_BASIS @ np.log(composition)

        result = {
            WARD_ID: ward_id,
            YEAR_COLUMN: year,
        }

        for name, value in zip(
            BALANCE_NAMES,
            balances,
        ):
            result[name] = float(value)

        results.append(result)

        replacement_records.append({
            WARD_ID: ward_id,
            YEAR_COLUMN: year,
            "replacement_factor_pixels": pixel_factor,
            "zero_classes": int((areas == 0).sum()),
            "replacement_delta": delta,
            "replacement_mass": (
                int((areas == 0).sum()) * delta
            ),
        })

    ilr_df = pd.DataFrame(results)

    replacement_df = pd.DataFrame(
        replacement_records
    )

    ilr_1994 = (
        ilr_df[ilr_df[YEAR_COLUMN] == 1994]
        .set_index(WARD_ID)
    )

    ilr_2024 = (
        ilr_df[ilr_df[YEAR_COLUMN] == 2024]
        .set_index(WARD_ID)
    )

    if set(ilr_1994.index) != set(ilr_2024.index):
        raise ValueError(
            "Ward mismatch between ILR endpoints."
        )

    change = pd.DataFrame(
        index=pd.Index(
            sorted(ilr_1994.index),
            name=WARD_ID,
        )
    )

    for balance in BALANCE_NAMES:
        change[f"delta_{balance}"] = (
            ilr_2024.loc[change.index, balance].to_numpy()
            -
            ilr_1994.loc[change.index, balance].to_numpy()
        )

    return change, replacement_df


# ============================================================
# SPATIAL WEIGHTS
# ============================================================

def build_weights(data):

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

        print(
            f"{name}: n={w.n}, "
            f"mean neighbours={w.mean_neighbors:.3f}"
        )

    return weights


# ============================================================
# FIT OLS BASELINE AND SAR
# ============================================================

def fit_model(
    data,
    w,
    weight_name,
    pixel_factor,
):

    predictor_columns = [
        f"delta_{name}"
        for name in BALANCE_NAMES
    ]

    X = data[
        predictor_columns
    ].to_numpy(dtype=float)

    y = data[
        DEPENDENT
    ].to_numpy(dtype=float).reshape(-1, 1)

    if not np.isfinite(X).all() or not np.isfinite(y).all():
        raise ValueError("Non-finite model values.")

    design = np.column_stack([
        np.ones(len(y)),
        X,
    ])

    if np.linalg.matrix_rank(design) != design.shape[1]:
        raise ValueError(
            "ILR design matrix is rank deficient."
        )

    # OLS baseline.
    beta_ols = np.linalg.lstsq(
        design,
        y.flatten(),
        rcond=None,
    )[0]

    ols_residuals = y.flatten() - design @ beta_ols

    ss_res = np.sum(ols_residuals ** 2)
    ss_total = np.sum((y.flatten() - y.mean()) ** 2)

    ols_r2 = (
        1 - ss_res / ss_total
        if ss_total > 0
        else np.nan
    )

    np.random.seed(RANDOM_SEED)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        ols_moran = Moran(
            ols_residuals,
            w,
            permutations=PERMUTATIONS,
        )

    # SAR maximum likelihood.
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")

        model = ML_Lag(
            y,
            X,
            w=w,
            method="ord",
            epsilon=1e-7,
            spat_diag=True,
            vm=True,
            name_y=DEPENDENT,
            name_x=BALANCE_NAMES,
            name_w=weight_name,
        )

    rho = float(
        np.asarray(model.rho).flatten()[0]
    )

    residuals = np.asarray(model.u).flatten()

    np.random.seed(RANDOM_SEED)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        sar_moran = Moran(
            residuals,
            w,
            permutations=PERMUTATIONS,
        )

    # SAR coefficients.
    betas = np.asarray(model.betas).flatten()
    z_stats = list(model.z_stat)

    if len(betas) < 6:
        raise ValueError(
            f"Unexpected SAR coefficient count: {len(betas)}"
        )

    # Expected order:
    # intercept, four ILR coefficients, spatial lag coefficient.
    coefficient_rows = []

    for index, balance in enumerate(BALANCE_NAMES):

        coefficient = float(betas[index + 1])

        z_value = np.nan
        p_value = np.nan

        if index + 1 < len(z_stats):
            z_value = float(z_stats[index + 1][0])
            p_value = float(z_stats[index + 1][1])

        coefficient_rows.append({
            "replacement_factor_pixels": pixel_factor,
            "weights": weight_name,
            "balance": balance,
            "coefficient": coefficient,
            "z_statistic": z_value,
            "p_value": p_value,
        })

    # Exact SAR impacts.
    W = w.sparse.toarray()

    multiplier = np.linalg.solve(
        np.eye(w.n) - rho * W,
        np.eye(w.n),
    )

    average_diagonal = (
        np.trace(multiplier) / w.n
    )

    average_row_sum = (
        multiplier.sum(axis=1).mean()
    )

    impact_rows = []

    for index, balance in enumerate(BALANCE_NAMES):

        coefficient = float(betas[index + 1])

        direct = coefficient * average_diagonal
        total = coefficient * average_row_sum
        indirect = total - direct

        impact_rows.append({
            "replacement_factor_pixels": pixel_factor,
            "weights": weight_name,
            "balance": balance,
            "coefficient": coefficient,
            "direct_impact": direct,
            "indirect_impact": indirect,
            "total_impact": total,
        })

    diagnostic = {
        "replacement_factor_pixels": pixel_factor,
        "weights": weight_name,
        "n_wards": w.n,
        "ols_r2": ols_r2,
        "ols_residual_moran_i": ols_moran.I,
        "ols_residual_moran_p": ols_moran.p_sim,
        "sar_pseudo_r2": float(model.pr2),
        "sar_aic": float(model.aic),
        "sar_bic": float(model.schwarz),
        "sar_log_likelihood": float(model.logll),
        "rho": rho,
        "sar_residual_moran_i": sar_moran.I,
        "sar_residual_moran_p": sar_moran.p_sim,
        "numerical_warning_count": len(caught),
        "numerical_warnings": " | ".join(
            str(item.message) for item in caught
        ),
    }

    print(
        f"\nReplacement={pixel_factor} pixel(s) | "
        f"{weight_name}"
    )
    print("OLS R²:", round(ols_r2, 6))
    print("SAR pseudo-R²:", round(float(model.pr2), 6))
    print("SAR AIC:", round(float(model.aic), 6))
    print("rho:", round(rho, 6))
    print(
        "SAR residual Moran's I:",
        round(sar_moran.I, 6),
    )
    print(
        "SAR residual Moran p:",
        sar_moran.p_sim,
    )
    print("Numerical warnings:", len(caught))

    return diagnostic, coefficient_rows, impact_rows


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("NMMC ILR COMPOSITIONAL SPATIAL REGRESSION")
    print("=" * 70)

    boundaries, lulc, ecological = load_inputs()

    diagnostics_all = []
    coefficients_all = []
    impacts_all = []
    replacement_all = []

    for pixel_factor in REPLACEMENT_FACTORS:

        print("\n" + "=" * 70)
        print(
            f"ZERO REPLACEMENT: {pixel_factor} "
            "EQUIVALENT 30M PIXEL(S)"
        )
        print("=" * 70)

        ilr_change, replacement_log = construct_ilr_data(
            lulc,
            pixel_factor,
        )

        replacement_all.append(replacement_log)

        data = boundaries.merge(
            ilr_change.reset_index(),
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

        data = data.sort_values(
            WARD_ID
        ).reset_index(drop=True)

        if len(data) != EXPECTED_WARDS:
            raise ValueError(
                f"Expected {EXPECTED_WARDS} wards; got {len(data)}"
            )

        # Save model inputs without geometry.
        input_columns = [
            WARD_ID,
            DEPENDENT,
            *[
                f"delta_{name}"
                for name in BALANCE_NAMES
            ],
        ]

        data[input_columns].to_csv(
            OUTPUT_DIR
            / (
                "NMMC_ILR_Model_Input_"
                f"{pixel_factor:g}_pixel.csv"
            ),
            index=False,
        )

        weights = build_weights(data)

        for weight_name, w in weights.items():

            diagnostic, coefficients, impacts = fit_model(
                data=data,
                w=w,
                weight_name=weight_name,
                pixel_factor=pixel_factor,
            )

            diagnostics_all.append(diagnostic)
            coefficients_all.extend(coefficients)
            impacts_all.extend(impacts)

    # --------------------------------------------------------
    # Export results
    # --------------------------------------------------------

    diagnostics_df = pd.DataFrame(diagnostics_all)
    coefficients_df = pd.DataFrame(coefficients_all)
    impacts_df = pd.DataFrame(impacts_all)

    replacement_df = pd.concat(
        replacement_all,
        ignore_index=True,
    )

    outputs = {
        "NMMC_ILR_SAR_Diagnostics.csv": diagnostics_df,
        "NMMC_ILR_SAR_Coefficients.csv": coefficients_df,
        "NMMC_ILR_SAR_Impacts.csv": impacts_df,
        "NMMC_ILR_Zero_Replacement_Audit.csv": replacement_df,
    }

    for filename, frame in outputs.items():

        path = OUTPUT_DIR / filename

        frame.to_csv(
            path,
            index=False,
        )

        print("\nSaved:", path)

    # --------------------------------------------------------
    # Final comparison
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("ILR SAR MODEL SENSITIVITY SUMMARY")
    print("=" * 70)

    print(
        diagnostics_df[
            [
                "replacement_factor_pixels",
                "weights",
                "ols_r2",
                "sar_pseudo_r2",
                "sar_aic",
                "rho",
                "sar_residual_moran_i",
                "sar_residual_moran_p",
                "numerical_warning_count",
            ]
        ].round(6).to_string(index=False)
    )

    print("\nILR coefficient sensitivity:")

    print(
        coefficients_df[
            [
                "replacement_factor_pixels",
                "weights",
                "balance",
                "coefficient",
                "p_value",
            ]
        ].round(6).to_string(index=False)
    )

    print("\n" + "=" * 70)
    print("ILR ANALYSIS COMPLETED")
    print("=" * 70)


if __name__ == "__main__":
    main()
