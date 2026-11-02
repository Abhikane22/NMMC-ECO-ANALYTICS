
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt

from libpysal.weights import Queen
from esda import Moran, G_Local
from scipy.stats import false_discovery_control

from config import PROJECT_ROOT, REPORTS_DIR


# ---------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------

BOUNDARY_FILE = (
    PROJECT_ROOT
    / "data/boundaries/NMMC_Ward_Boundaries_2024_CLEAN.geojson"
)

LULC_FILE = (
    REPORTS_DIR
    / "NMMC_Ward_Level_LULC_Statistics_PRELIMINARY_ALL_YEARS.csv"
)

ECOLOGICAL_FILE = (
    REPORTS_DIR
    / "NMMC_Ward_Level_Ecological_Statistics.csv"
)

OUTPUT_DIR = PROJECT_ROOT / "outputs"
CHARTS_DIR = OUTPUT_DIR / "charts"
MAPS_DIR = OUTPUT_DIR / "maps"

WARD_ID_FIELD = "ward_lgd_c"

PERMUTATIONS = 999
RANDOM_SEED = 2026
FDR_ALPHA = 0.05

YEARS = [1994, 2004, 2014, 2024]


# ---------------------------------------------------------
# LOAD AND VALIDATE INPUT DATA
# ---------------------------------------------------------

def load_inputs():

    for path in [
        BOUNDARY_FILE,
        LULC_FILE,
        ECOLOGICAL_FILE,
    ]:
        if not path.exists():
            raise FileNotFoundError(path)

    wards = gpd.read_file(BOUNDARY_FILE)

    if len(wards) != 111:
        raise ValueError(
            f"Expected 111 wards, found {len(wards)}"
        )

    if wards.crs is None:
        raise ValueError("Ward boundary CRS is missing.")

    if WARD_ID_FIELD not in wards.columns:
        raise ValueError(
            f"Missing ward ID field: {WARD_ID_FIELD}"
        )

    wards["ward_id"] = (
        wards[WARD_ID_FIELD].astype(str)
    )

    if wards["ward_id"].duplicated().any():
        raise ValueError("Duplicate ward IDs in boundaries.")

    lulc = pd.read_csv(LULC_FILE)
    ecological = pd.read_csv(ECOLOGICAL_FILE)

    lulc["ward_id"] = lulc["ward_id"].astype(str)
    ecological["ward_id"] = ecological["ward_id"].astype(str)

    # Confirm four-year preliminary dataset.
    if set(lulc["year"].unique()) != set(YEARS):
        raise ValueError(
            "Preliminary LULC dataset does not contain all four years."
        )

    if lulc.duplicated(["ward_id", "year"]).any():
        raise ValueError("Duplicate preliminary ward-year records.")

    if ecological.duplicated(["ward_id", "year"]).any():
        raise ValueError("Duplicate ecological ward-year records.")

    if set(lulc["ward_id"]) != set(wards["ward_id"]):
        raise ValueError("LULC ward IDs do not match boundaries.")

    if set(ecological["ward_id"]) != set(wards["ward_id"]):
        raise ValueError("Ecological ward IDs do not match boundaries.")

    return wards, lulc, ecological


# ---------------------------------------------------------
# CREATE TEMPORAL CHANGE INDICATORS
# ---------------------------------------------------------

def create_change_indicators(wards, lulc, ecological):

    indicators = wards[
        ["ward_id", "geometry"]
    ].copy()

    # LULC indicators: preliminary classifications only.
    lulc_variables = {
        "Built-up_area_km2": "builtup_expansion_km2",
        "Vegetation_area_km2": "vegetation_loss_km2",
        "Coastal Wetlands_area_km2": "wetland_loss_km2",
    }

    for source_column, output_column in lulc_variables.items():

        pivot = lulc.pivot(
            index="ward_id",
            columns="year",
            values=source_column,
        )

        if not {1994, 2024}.issubset(pivot.columns):
            raise ValueError(
                f"Missing years for {source_column}"
            )

        if source_column == "Built-up_area_km2":
            change = pivot[2024] - pivot[1994]

        else:
            change = pivot[1994] - pivot[2024]

        indicators[output_column] = (
            indicators["ward_id"].map(change)
        )

    # Ecological indicators.
    ecological_variables = {
        "lst_c_mean": "lst_increase_c",
        "ndvi_mean": "ndvi_decline",
    }

    for source_column, output_column in ecological_variables.items():

        pivot = ecological.pivot(
            index="ward_id",
            columns="year",
            values=source_column,
        )

        if not {1994, 2024}.issubset(pivot.columns):
            raise ValueError(
                f"Missing ecological years for {source_column}"
            )

        if source_column == "lst_c_mean":
            change = pivot[2024] - pivot[1994]

        else:
            change = pivot[1994] - pivot[2024]

        indicators[output_column] = (
            indicators["ward_id"].map(change)
        )

    indicator_columns = [
        "builtup_expansion_km2",
        "vegetation_loss_km2",
        "wetland_loss_km2",
        "lst_increase_c",
        "ndvi_decline",
    ]

    if indicators[indicator_columns].isna().any().any():
        raise ValueError("Missing values in change indicators.")

    if not np.isfinite(
        indicators[indicator_columns].to_numpy(dtype=float)
    ).all():
        raise ValueError("Non-finite indicator values detected.")

    return indicators, indicator_columns


# ---------------------------------------------------------
# SPATIAL WEIGHTS
# ---------------------------------------------------------

def create_spatial_weights(wards):

    weights = Queen.from_dataframe(
        wards,
        ids=wards["ward_id"].tolist(),
    )

    print("\nSPATIAL WEIGHTS")
    print("Components:", weights.n_components)
    print("Islands:", weights.islands)
    print(
        "Average neighbours:",
        round(
            np.mean(list(weights.cardinalities.values())),
            3,
        ),
    )

    if weights.n_components != 1:
        raise ValueError(
            "Spatial weights contain disconnected components."
        )

    if weights.islands:
        raise ValueError(
            f"Isolated wards detected: {weights.islands}"
        )

    return weights


# ---------------------------------------------------------
# GLOBAL MORAN'S I
# ---------------------------------------------------------

def calculate_global_moran(values, weights, indicator):

    # Make permutation results reproducible.
    np.random.seed(RANDOM_SEED)

    result = Moran(
        values,
        weights,
        transformation="r",
        permutations=PERMUTATIONS,
    )

    return {
        "indicator": indicator,
        "moran_i": float(result.I),
        "expected_i": float(result.EI),
        "z_score": float(result.z_sim),
        "permutation_p_value": float(result.p_sim),
        "permutations": PERMUTATIONS,
        "weight_method": "Queen contiguity, row-standardized",
    }


# ---------------------------------------------------------
# LOCAL GETIS-ORD Gi*
# ---------------------------------------------------------

def calculate_local_gi_star(values, weights, indicator):

    result = G_Local(
        values,
        weights,
        transform="B",
        permutations=PERMUTATIONS,
        star=True,
        keep_simulations=False,
        n_jobs=-1,
        seed=RANDOM_SEED,
    )

    p_values = np.asarray(result.p_sim, dtype=float)

    # Correct for 111 simultaneous local tests.
    adjusted_p = false_discovery_control(
        p_values,
        method="bh",
    )

    z_scores = np.asarray(result.Zs, dtype=float)

    hotspot_class = []

    for z, q in zip(z_scores, adjusted_p):

        if q < FDR_ALPHA and z > 0:
            hotspot_class.append("Hotspot")

        elif q < FDR_ALPHA and z < 0:
            hotspot_class.append("Coldspot")

        else:
            hotspot_class.append("Not significant")

    return pd.DataFrame({
        "ward_id": weights.id_order,
        "indicator": indicator,
        "gi_star_z_score": z_scores,
        "permutation_p_value": p_values,
        "fdr_adjusted_p_value": adjusted_p,
        "hotspot_class": hotspot_class,
    })


# ---------------------------------------------------------
# MAP GENERATION
# ---------------------------------------------------------

def plot_hotspot_map(gdf, indicator):

    colors = {
        "Hotspot": "#D7191C",
        "Coldspot": "#2C7BB6",
        "Not significant": "#D9D9D9",
    }

    fig, ax = plt.subplots(figsize=(10, 10))

    for category, color in colors.items():

        subset = gdf[
            gdf["hotspot_class"] == category
        ]

        if not subset.empty:
            subset.plot(
                ax=ax,
                color=color,
                edgecolor="black",
                linewidth=0.25,
                label=category,
            )

    ax.set_title(
        f"Getis-Ord Gi* Hotspot Analysis\n{indicator}",
        fontsize=13,
    )

    ax.set_axis_off()
    ax.legend(title="Classification", loc="lower right")

    fig.tight_layout()

    output = MAPS_DIR / f"hotspot_{indicator}.png"

    fig.savefig(
        output,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)

    print("Map saved:", output)


# ---------------------------------------------------------
# MAIN PIPELINE
# ---------------------------------------------------------

def main():

    print("\nNMMC SPATIAL HOTSPOT ANALYSIS")
    print("=" * 65)

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    MAPS_DIR.mkdir(parents=True, exist_ok=True)
    CHARTS_DIR.mkdir(parents=True, exist_ok=True)

    wards, lulc, ecological = load_inputs()

    print("\nInput validation: PASS")
    print("Wards:", len(wards))

    indicators, indicator_columns = (
        create_change_indicators(
            wards,
            lulc,
            ecological,
        )
    )

    weights = create_spatial_weights(wards)

    global_results = []
    local_results = []

    for indicator in indicator_columns:

        print(f"\nAnalysing: {indicator}")

        values = indicators[indicator].to_numpy(dtype=float)

        if np.std(values) <= 1e-12:
            raise ValueError(
                f"Indicator has no spatial variation: {indicator}"
            )

        # Global spatial autocorrelation.
        global_result = calculate_global_moran(
            values,
            weights,
            indicator,
        )

        global_results.append(global_result)

        # Local spatial clusters.
        local = calculate_local_gi_star(
            values,
            weights,
            indicator,
        )

        local_results.append(local)

        print(
            "Moran's I:",
            round(global_result["moran_i"], 5),
        )

        print(
            "Global permutation p-value:",
            round(global_result["permutation_p_value"], 5),
        )

        print(
            "Hotspots:",
            int((local["hotspot_class"] == "Hotspot").sum()),
        )

        print(
            "Coldspots:",
            int((local["hotspot_class"] == "Coldspot").sum()),
        )

        print(
            "Not significant:",
            int((local["hotspot_class"] == "Not significant").sum()),
        )

    # Combine outputs.
    global_df = pd.DataFrame(global_results)

    local_df = pd.concat(
        local_results,
        ignore_index=True,
    )

    local_df = local_df.merge(
        indicators.drop(columns="geometry"),
        on="ward_id",
        how="left",
        validate="many_to_one",
    )

    # Save statistical reports.
    global_path = (
        REPORTS_DIR / "NMMC_Global_Morans_I_Results.csv"
    )

    local_path = (
        REPORTS_DIR / "NMMC_Local_Gi_Star_Hotspot_Results.csv"
    )

    global_df.to_csv(global_path, index=False)
    local_df.to_csv(local_path, index=False)

    # Generate five spatial maps.
    for indicator in indicator_columns:

        indicator_results = local_df[
            local_df["indicator"] == indicator
        ]

        map_gdf = wards.merge(
            indicator_results,
            on="ward_id",
            how="left",
            validate="one_to_one",
        )

        plot_hotspot_map(
            map_gdf,
            indicator,
        )

    # Summary.
    summary = (
        local_df.groupby(
            ["indicator", "hotspot_class"]
        )
        .size()
        .unstack(fill_value=0)
    )

    summary_path = (
        REPORTS_DIR / "NMMC_Hotspot_Summary.csv"
    )

    summary.to_csv(summary_path)

    print("\n" + "=" * 65)
    print("SPATIAL HOTSPOT ANALYSIS COMPLETED")
    print("=" * 65)

    print("\nGlobal Moran's I results:")
    print(
        global_df[
            [
                "indicator",
                "moran_i",
                "z_score",
                "permutation_p_value",
            ]
        ].round(5).to_string(index=False)
    )

    print("\nLocal hotspot summary:")
    print(summary.to_string())

    print("\nReports:")
    print(global_path)
    print(local_path)
    print(summary_path)

    print("\nMaps:", MAPS_DIR)

    return global_df, local_df


if __name__ == "__main__":
    main()
