
import numpy as np
import pandas as pd
import geopandas as gpd

from pathlib import Path
from scipy.stats import false_discovery_control, spearmanr

from libpysal.weights import Queen, Rook
from esda import Moran, G_Local

from config import PROJECT_ROOT, REPORTS_DIR

from spatial_hotspot_analysis import (
    load_inputs,
    create_change_indicators,
)


PERMUTATIONS = 999
RANDOM_SEED = 2026
ALPHA = 0.05

OUTPUT_DIR = REPORTS_DIR


def classify_local_results(z_scores, p_values):

    adjusted = false_discovery_control(
        p_values,
        method="bh",
    )

    labels = []

    for z, q in zip(z_scores, adjusted):

        if q < ALPHA and z > 0:
            labels.append("Hotspot")

        elif q < ALPHA and z < 0:
            labels.append("Coldspot")

        else:
            labels.append("Not significant")

    return adjusted, labels


def run_global_moran(values, weights):

    np.random.seed(RANDOM_SEED)

    result = Moran(
        values,
        weights,
        transformation="r",
        permutations=PERMUTATIONS,
    )

    return {
        "moran_i": float(result.I),
        "expected_i": float(result.EI),
        "z_score": float(result.z_sim),
        "p_value": float(result.p_sim),
    }


def run_local_gi(values, weights):

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

    z = np.asarray(result.Zs)
    p = np.asarray(result.p_sim)

    adjusted, labels = classify_local_results(z, p)

    return {
        "z_scores": z,
        "p_values": p,
        "adjusted_p": adjusted,
        "labels": labels,
    }


def main():

    print("\nNMMC SPATIAL WEIGHTS SENSITIVITY ANALYSIS")
    print("=" * 65)

    wards, lulc, ecological = load_inputs()

    indicators, indicator_columns = create_change_indicators(
        wards,
        lulc,
        ecological,
    )

    # Both weight matrices use exactly the same ward order.
    queen = Queen.from_dataframe(
        wards,
        ids=wards["ward_id"].tolist(),
    )

    rook = Rook.from_dataframe(
        wards,
        ids=wards["ward_id"].tolist(),
    )

    if queen.id_order != rook.id_order:
        raise ValueError("Queen and Rook ward ordering differs.")

    if queen.islands or rook.islands:
        raise ValueError("Spatial islands detected.")

    queen.transform = "R"
    rook.transform = "R"

    weight_sets = {
        "Queen": queen,
        "Rook": rook,
    }

    global_records = []
    local_records = []

    for indicator in indicator_columns:

        print(f"\nIndicator: {indicator}")

        values = indicators[indicator].to_numpy(dtype=float)

        if not np.isfinite(values).all():
            raise ValueError(f"Invalid values in {indicator}")

        local_by_method = {}

        for method, weights in weight_sets.items():

            print(f"  Running {method}...")

            global_result = run_global_moran(
                values,
                weights,
            )

            global_records.append({
                "indicator": indicator,
                "weights": method,
                **global_result,
            })

            local = run_local_gi(
                values,
                weights,
            )

            local_by_method[method] = local

            for i, ward_id in enumerate(wards["ward_id"]):

                local_records.append({
                    "ward_id": ward_id,
                    "indicator": indicator,
                    "weights": method,
                    "gi_star_z_score": local["z_scores"][i],
                    "permutation_p_value": local["p_values"][i],
                    "fdr_adjusted_p_value": local["adjusted_p"][i],
                    "hotspot_class": local["labels"][i],
                })

        # Compare Queen and Rook classifications.
        q_labels = np.asarray(
            local_by_method["Queen"]["labels"]
        )

        r_labels = np.asarray(
            local_by_method["Rook"]["labels"]
        )

        q_z = local_by_method["Queen"]["z_scores"]
        r_z = local_by_method["Rook"]["z_scores"]

        agreement = np.mean(q_labels == r_labels) * 100

        spearman = spearmanr(q_z, r_z)

        print(
            f"  Classification agreement: {agreement:.2f}%"
        )

        print(
            f"  Gi* Z-score correlation: {spearman.statistic:.4f}"
        )

    global_df = pd.DataFrame(global_records)
    local_df = pd.DataFrame(local_records)

    # -----------------------------------------------------
    # GLOBAL COMPARISON
    # -----------------------------------------------------

    global_comparison = global_df.pivot(
        index="indicator",
        columns="weights",
        values=["moran_i", "p_value"],
    )

    global_comparison.columns = [
        f"{metric}_{method}"
        for metric, method in global_comparison.columns
    ]

    global_comparison = global_comparison.reset_index()

    # -----------------------------------------------------
    # LOCAL CLASSIFICATION COMPARISON
    # -----------------------------------------------------

    classification_comparison = []

    for indicator in indicator_columns:

        subset = local_df[
            local_df["indicator"] == indicator
        ]

        q = subset[
            subset["weights"] == "Queen"
        ].sort_values("ward_id")

        r = subset[
            subset["weights"] == "Rook"
        ].sort_values("ward_id")

        if q["ward_id"].tolist() != r["ward_id"].tolist():
            raise ValueError("Ward ordering mismatch.")

        q_labels = q["hotspot_class"].to_numpy()
        r_labels = r["hotspot_class"].to_numpy()

        classification_comparison.append({
            "indicator": indicator,
            "classification_agreement_pct": (
                np.mean(q_labels == r_labels) * 100
            ),
            "queen_hotspots": int(
                (q_labels == "Hotspot").sum()
            ),
            "rook_hotspots": int(
                (r_labels == "Hotspot").sum()
            ),
            "queen_coldspots": int(
                (q_labels == "Coldspot").sum()
            ),
            "rook_coldspots": int(
                (r_labels == "Coldspot").sum()
            ),
            "queen_rook_z_score_correlation": spearmanr(
                q["gi_star_z_score"],
                r["gi_star_z_score"],
            ).statistic,
        })

    classification_df = pd.DataFrame(
        classification_comparison
    )

    # -----------------------------------------------------
    # SAVE
    # -----------------------------------------------------

    global_df.to_csv(
        OUTPUT_DIR / "NMMC_Sensitivity_Global_Morans.csv",
        index=False,
    )

    local_df.to_csv(
        OUTPUT_DIR / "NMMC_Sensitivity_Local_Gi_Star.csv",
        index=False,
    )

    global_comparison.to_csv(
        OUTPUT_DIR / "NMMC_Sensitivity_Global_Comparison.csv",
        index=False,
    )

    classification_df.to_csv(
        OUTPUT_DIR / "NMMC_Sensitivity_Classification_Comparison.csv",
        index=False,
    )

    print("\n" + "=" * 65)
    print("SENSITIVITY ANALYSIS COMPLETED")
    print("=" * 65)

    print("\nGlobal Moran's I comparison:")
    print(global_comparison.round(5).to_string(index=False))

    print("\nLocal classification comparison:")
    print(classification_df.round(3).to_string(index=False))

    print("\nReports saved in:", OUTPUT_DIR)


if __name__ == "__main__":
    main()
