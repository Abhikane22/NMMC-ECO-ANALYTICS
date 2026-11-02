
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
import matplotlib.pyplot as plt

from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.patches import Patch

from src.config import (
    LULC_DIR,
    REPORTS_DIR,
    MAPS_DIR,
    CHARTS_DIR,
    LULC_CLASSES,
    CLASS_COLORS,
)


# ---------------------------------------------------------
# 1. CONFIGURATION
# ---------------------------------------------------------

YEAR = 2024

LULC_RASTER = (
    LULC_DIR / "NMMC_2024_LULC_SCIENTIFIC_FINAL.tif"
)

GEE_AREA_CSV = (
    LULC_DIR / "NMMC_2024_LULC_AREA_FINAL_2.csv"
)

EXPECTED_CLASSES = set(LULC_CLASSES.keys())


# ---------------------------------------------------------
# 2. LOAD AND VALIDATE LULC RASTER
# ---------------------------------------------------------

def load_lulc_raster():

    if not LULC_RASTER.exists():
        raise FileNotFoundError(
            f"LULC raster not found: {LULC_RASTER}"
        )

    with rasterio.open(LULC_RASTER) as src:

        if src.count != 1:
            raise ValueError(
                "Scientific LULC raster must have one band."
            )

        data = src.read(1, masked=True)

        metadata = {
            "crs": str(src.crs),
            "resolution": src.res,
            "transform": src.transform,
            "width": src.width,
            "height": src.height,
            "nodata": src.nodata,
        }

    valid_values = np.asarray(
        data.compressed()
    )

    actual_classes = set(
        np.unique(valid_values).tolist()
    )

    unexpected = actual_classes - EXPECTED_CLASSES

    if unexpected:
        raise ValueError(
            f"Unexpected LULC classes: {unexpected}"
        )

    if not actual_classes:
        raise ValueError("LULC raster has no valid pixels.")

    print("\nLULC raster validated successfully.")
    print(f"CRS: {metadata['crs']}")
    print(f"Resolution: {metadata['resolution']}")
    print(f"Dimensions: {metadata['width']} x {metadata['height']}")
    print(f"NoData: {metadata['nodata']}")
    print(f"Classes present: {sorted(actual_classes)}")

    return data, metadata


# ---------------------------------------------------------
# 3. CALCULATE RASTER CLASS AREAS
# ---------------------------------------------------------

def calculate_raster_class_areas(data, metadata):
    """
    Calculate class areas from the exported raster grid.

    For projected metre-based CRS, pixel area is derived
    from the affine transformation determinant.
    """

    with rasterio.open(LULC_RASTER) as src:
        if not src.crs or not src.crs.is_projected:
            raise ValueError(
                "Projected CRS required for pixel-area calculation."
            )

        units = src.crs.linear_units_factor

        if units is None:
            raise ValueError("Cannot determine CRS linear units.")

        metres_per_unit = units[1]

    transform = metadata["transform"]

    pixel_area_m2 = abs(
        transform.a * transform.e
        - transform.b * transform.d
    ) * metres_per_unit ** 2

    valid_pixels = data.compressed()

    total_valid = len(valid_pixels)

    records = []

    for class_code, class_name in LULC_CLASSES.items():

        count = int(
            np.count_nonzero(valid_pixels == class_code)
        )

        area_km2 = count * pixel_area_m2 / 1_000_000

        percentage = (
            count / total_valid * 100
            if total_valid > 0
            else 0
        )

        records.append({
            "year": YEAR,
            "class_code": class_code,
            "class_name": class_name,
            "pixel_count": count,
            "area_km2_raster": area_km2,
            "percentage_raster": percentage,
        })

    return pd.DataFrame(records)


# ---------------------------------------------------------
# 4. COMPARE AGAINST GEE AREA CSV
# ---------------------------------------------------------

def compare_gee_areas(raster_areas):

    if not GEE_AREA_CSV.exists():
        print(
            "\nGEE area CSV not found. "
            "Skipping comparison."
        )
        return None

    gee = pd.read_csv(GEE_AREA_CSV)

    required = {
        "class_code",
        "area_km2",
        "percentage",
    }

    missing = required - set(gee.columns)

    if missing:
        raise ValueError(
            f"GEE CSV missing columns: {missing}"
        )

    gee["class_code"] = gee["class_code"].astype(int)

    comparison = raster_areas.merge(
        gee[
            ["class_code", "area_km2", "percentage"]
        ],
        on="class_code",
        how="left",
        validate="one_to_one",
    )

    comparison = comparison.rename(columns={
        "area_km2": "area_km2_gee",
        "percentage": "percentage_gee",
    })

    comparison["area_difference_km2"] = (
        comparison["area_km2_raster"]
        - comparison["area_km2_gee"]
    )

    comparison["percentage_difference_points"] = (
        comparison["percentage_raster"]
        - comparison["percentage_gee"]
    )

    return comparison


# ---------------------------------------------------------
# 5. GENERATE SCIENTIFIC LULC MAP
# ---------------------------------------------------------

def plot_lulc_map(data, metadata):

    MAPS_DIR.mkdir(parents=True, exist_ok=True)

    colors = [
        CLASS_COLORS[code]
        for code in sorted(LULC_CLASSES)
    ]

    cmap = ListedColormap(colors)
    cmap.set_bad(color="white", alpha=0)

    norm = BoundaryNorm(
        np.arange(-0.5, len(colors) + 0.5, 1),
        cmap.N,
    )

    fig, ax = plt.subplots(figsize=(8, 12))

    ax.imshow(
        data,
        cmap=cmap,
        norm=norm,
        interpolation="nearest",
    )

    ax.set_title(
        "NMMC Land Use / Land Cover — 2024",
        fontsize=15,
        fontweight="bold",
    )

    ax.set_xlabel("Raster Column")
    ax.set_ylabel("Raster Row")

    legend = [
        Patch(
            facecolor=CLASS_COLORS[code],
            label=LULC_CLASSES[code],
        )
        for code in sorted(LULC_CLASSES)
    ]

    ax.legend(
        handles=legend,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.12),
        ncol=2,
        frameon=False,
    )

    output_path = MAPS_DIR / "NMMC_LULC_2024.png"

    fig.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)

    print(f"Saved map: {output_path}")


# ---------------------------------------------------------
# 6. CLASS DISTRIBUTION CHART
# ---------------------------------------------------------

def plot_class_distribution(areas):

    CHARTS_DIR.mkdir(parents=True, exist_ok=True)

    colors = [
        CLASS_COLORS[code]
        for code in areas["class_code"]
    ]

    fig, ax = plt.subplots(figsize=(10, 6))

    bars = ax.bar(
        areas["class_name"],
        areas["area_km2_raster"],
        color=colors,
    )

    ax.bar_label(
        bars,
        fmt="%.2f",
        padding=3,
    )

    ax.set_title(
        "NMMC LULC Class Distribution — 2024",
        fontsize=14,
        fontweight="bold",
    )

    ax.set_ylabel("Area from Exported Raster (km²)")
    ax.tick_params(axis="x", rotation=20)
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()

    output_path = (
        CHARTS_DIR / "NMMC_LULC_2024_class_distribution.png"
    )

    fig.savefig(output_path, dpi=300)

    plt.close(fig)

    print(f"Saved chart: {output_path}")


# ---------------------------------------------------------
# 7. MAIN PIPELINE
# ---------------------------------------------------------

def run_lulc_analysis():

    print("\n" + "=" * 65)
    print("NMMC LULC ANALYSIS — 2024")
    print("=" * 65)

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    data, metadata = load_lulc_raster()

    print("\n[1/4] Calculating raster class areas...")
    areas = calculate_raster_class_areas(
        data,
        metadata,
    )

    print(areas.to_string(index=False))

    area_path = (
        REPORTS_DIR / "NMMC_LULC_2024_raster_areas.csv"
    )

    areas.to_csv(area_path, index=False)

    print("\n[2/4] Comparing GEE area statistics...")

    comparison = compare_gee_areas(areas)

    if comparison is not None:

        print(comparison.round(4).to_string(index=False))

        comparison_path = (
            REPORTS_DIR
            / "NMMC_LULC_2024_area_comparison.csv"
        )

        comparison.to_csv(
            comparison_path,
            index=False,
        )

        print(f"Saved comparison: {comparison_path}")

    print("\n[3/4] Generating classified LULC map...")
    plot_lulc_map(data, metadata)

    print("\n[4/4] Generating class distribution chart...")
    plot_class_distribution(areas)

    print("\nLULC analysis completed successfully.")

    return areas, comparison


if __name__ == "__main__":
    run_lulc_analysis()
