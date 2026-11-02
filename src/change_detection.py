
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
import matplotlib.pyplot as plt

from pyproj import Transformer, Geod
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.patches import Patch

from src.config import (
    YEARS,
    LULC_DIR,
    LULC_CLASSES,
    CLASS_COLORS,
    REPORTS_DIR,
    MAPS_DIR,
    CHARTS_DIR,
)


# ---------------------------------------------------------
# 1. CONFIGURATION
# ---------------------------------------------------------

LULC_FILES = {
    1994: LULC_DIR / "NMMC_LULC_1994_PRELIMINARY.tif",
    2004: LULC_DIR / "NMMC_LULC_2004_PRELIMINARY.tif",
    2014: LULC_DIR / "NMMC_LULC_2014_PRELIMINARY.tif",
    2024: LULC_DIR / "NMMC_2024_LULC_SCIENTIFIC_FINAL.tif",
}

PERIODS = [
    (1994, 2004),
    (2004, 2014),
    (2014, 2024),
    (1994, 2024),
]

CLASS_CODES = sorted(LULC_CLASSES.keys())

NODATA_VALUE = 255

REPORTS_DIR.mkdir(parents=True, exist_ok=True)
MAPS_DIR.mkdir(parents=True, exist_ok=True)
CHARTS_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------
# 2. LOAD AND VALIDATE ALL CLASSIFICATION RASTERS
# ---------------------------------------------------------

def load_lulc_rasters():
    """
    Load all four classified GeoTIFFs.

    Verify that every raster has the same grid,
    projection, dimensions and transform.
    """

    rasters = {}
    reference_metadata = None

    for year in YEARS:

        path = LULC_FILES[year]

        if not path.exists():
            raise FileNotFoundError(
                f"Missing LULC raster for {year}: {path}"
            )

        with rasterio.open(path) as src:

            if src.count != 1:
                raise ValueError(
                    f"{year}: Expected a single-band raster."
                )

            if src.crs is None or not src.crs.is_projected:
                raise ValueError(
                    f"{year}: A projected CRS is required."
                )

            data = src.read(1, masked=True)

            metadata = {
                "crs": src.crs,
                "transform": src.transform,
                "width": src.width,
                "height": src.height,
                "nodata": src.nodata,
                "bounds": src.bounds,
                "profile": src.profile.copy(),
            }

        valid = data.compressed()

        if valid.size == 0:
            raise ValueError(
                f"{year}: No valid classified pixels."
            )

        actual_classes = set(
            np.unique(valid).astype(int).tolist()
        )

        unexpected = actual_classes - set(CLASS_CODES)

        if unexpected:
            raise ValueError(
                f"{year}: Unexpected class codes: {unexpected}"
            )

        if reference_metadata is None:

            reference_metadata = metadata

        else:

            checks = [
                ("CRS", metadata["crs"], reference_metadata["crs"]),
                (
                    "Transform",
                    metadata["transform"],
                    reference_metadata["transform"],
                ),
                (
                    "Width",
                    metadata["width"],
                    reference_metadata["width"],
                ),
                (
                    "Height",
                    metadata["height"],
                    reference_metadata["height"],
                ),
                (
                    "Bounds",
                    metadata["bounds"],
                    reference_metadata["bounds"],
                ),
            ]

            for name, current, reference in checks:

                if current != reference:
                    raise ValueError(
                        f"Raster alignment error in {year}: "
                        f"{name} differs."
                    )

        rasters[year] = data

        print(
            f"{year}: PASS | "
            f"{metadata['width']} x {metadata['height']} | "
            f"{metadata['crs']} | "
            f"{len(valid)} valid pixels"
        )

    print("\nAll four LULC rasters are aligned.")

    return rasters, reference_metadata


# ---------------------------------------------------------
# 3. CALCULATE GEODESIC PIXEL AREAS
# ---------------------------------------------------------

def calculate_geodesic_pixel_areas(metadata):
    """
    Calculate the ground area of every raster pixel.

    Pixel corners are transformed from the raster CRS
    to geographic coordinates. Geodesic polygon areas
    are calculated using the WGS84 ellipsoid.

    This accounts for variation in ground area across
    the projected raster grid.
    """

    transform = metadata["transform"]
    crs = metadata["crs"]

    width = metadata["width"]
    height = metadata["height"]

    if crs.linear_units != "metre":
        raise ValueError(
            "This implementation expects a metre-based projected CRS."
        )

    transformer = Transformer.from_crs(
        crs,
        "EPSG:4326",
        always_xy=True,
    )

    geod = Geod(ellps="WGS84")

    areas = np.zeros(
        (height, width),
        dtype=np.float64,
    )

    # Calculate each pixel's geodesic quadrilateral area.
    for row in range(height):

        for col in range(width):

            x0, y0 = transform * (col, row)
            x1, y1 = transform * (col + 1, row)
            x2, y2 = transform * (col + 1, row + 1)
            x3, y3 = transform * (col, row + 1)

            lon, lat = transformer.transform(
                [x0, x1, x2, x3],
                [y0, y1, y2, y3],
            )

            area_m2, _ = geod.polygon_area_perimeter(
                lon,
                lat,
            )

            areas[row, col] = abs(area_m2)

    print(
        "\nGeodesic pixel-area calculation completed."
    )

    print(
        f"Pixel area range: "
        f"{areas.min():.3f} - {areas.max():.3f} m²"
    )

    return areas


# ---------------------------------------------------------
# 4. ANNUAL LULC AREA STATISTICS
# ---------------------------------------------------------

def calculate_annual_lulc_areas(rasters, pixel_areas):
    """
    Calculate area and percentage for each LULC class
    in each study year.
    """

    records = []

    for year in YEARS:

        raster = rasters[year]

        values = np.asarray(raster.data)

        valid_mask = ~np.ma.getmaskarray(raster)

        valid_mask &= np.isin(values, CLASS_CODES)

        total_area = pixel_areas[valid_mask].sum()
        total_area_km2 = total_area / 1_000_000
        for code in CLASS_CODES:

            class_mask = valid_mask & (values == code)

            pixel_count = int(class_mask.sum())

            area_km2 = (
                pixel_areas[class_mask].sum() / 1_000_000
            )

            percentage = (
                    area_km2 / total_area_km2 * 100
                    if total_area_km2 > 0
                    else 0
                )

            records.append({
                "year": year,
                "class_code": code,
                "class_name": LULC_CLASSES[code],
                "pixel_count": pixel_count,
                "area_km2": area_km2,
                "percentage": percentage,
            })

    return pd.DataFrame(records)


# ---------------------------------------------------------
# 5. CREATE TRANSITION MATRIX
# ---------------------------------------------------------

def calculate_transition_matrix(
    start_year,
    end_year,
    rasters,
    pixel_areas,
):
    """
    Calculate transitions from one class to another.

    Transition code = old class * 10 + new class.

    Only pixels valid in both years are included.
    """

    start = rasters[start_year]
    end = rasters[end_year]

    old_values = np.asarray(start.data)
    new_values = np.asarray(end.data)

    old_mask = np.ma.getmaskarray(start)
    new_mask = np.ma.getmaskarray(end)

    common_mask = (
        ~old_mask
        & ~new_mask
        & np.isin(old_values, CLASS_CODES)
        & np.isin(new_values, CLASS_CODES)
    )

    old = old_values[common_mask].astype(np.int16)
    new = new_values[common_mask].astype(np.int16)

    valid_areas = pixel_areas[common_mask]

    transition_codes = old * 10 + new

    records = []

    matrix_area = np.zeros(
        (len(CLASS_CODES), len(CLASS_CODES)),
        dtype=np.float64,
    )

    matrix_pixels = np.zeros(
        (len(CLASS_CODES), len(CLASS_CODES)),
        dtype=np.int64,
    )

    for from_code in CLASS_CODES:

        for to_code in CLASS_CODES:

            selected = (
                (old == from_code)
                & (new == to_code)
            )

            count = int(selected.sum())

            area_km2 = (
                valid_areas[selected].sum() / 1_000_000
            )

            matrix_pixels[from_code, to_code] = count

            matrix_area[from_code, to_code] = area_km2

            records.append({
                "start_year": start_year,
                "end_year": end_year,
                "transition_code": from_code * 10 + to_code,
                "from_class_code": from_code,
                "from_class": LULC_CLASSES[from_code],
                "to_class_code": to_code,
                "to_class": LULC_CLASSES[to_code],
                "pixel_count": count,
                "area_km2": area_km2,
                "percentage_of_common_area": (
                    area_km2
                    / (valid_areas.sum() / 1_000_000)
                    * 100
                    if valid_areas.sum() > 0
                    else 0
                ),
            })

    return (
        pd.DataFrame(records),
        matrix_area,
        matrix_pixels,
        common_mask,
        transition_codes,
    )


# ---------------------------------------------------------
# 6. CLASS GAINS, LOSSES AND NET CHANGE
# ---------------------------------------------------------

def calculate_class_changes(
    transition_area,
    start_year,
    end_year,
):
    """
    Calculate class-wise gains, losses and net changes.

    Gain = area entering a class from other classes.
    Loss = area leaving a class for other classes.
    Net change = gain - loss.
    """

    records = []

    for code in CLASS_CODES:

        row = transition_area[code, :]
        column = transition_area[:, code]

        unchanged = transition_area[code, code]

        loss = row.sum() - unchanged
        gain = column.sum() - unchanged

        net_change = gain - loss

        records.append({
            "start_year": start_year,
            "end_year": end_year,
            "class_code": code,
            "class_name": LULC_CLASSES[code],
            "gain_km2": gain,
            "loss_km2": loss,
            "net_change_km2": net_change,
        })

    return pd.DataFrame(records)


# ---------------------------------------------------------
# 7. EXPORT TRANSITION RASTER
# ---------------------------------------------------------

def export_transition_raster(
    start_year,
    end_year,
    rasters,
    metadata,
):
    """
    Save a categorical transition GeoTIFF.

    Example:
        12 = Vegetation -> Built-up
        21 = Built-up -> Vegetation

    255 = NoData
    """

    start = rasters[start_year]
    end = rasters[end_year]

    old_values = np.asarray(start.data)
    new_values = np.asarray(end.data)

    common_mask = (
        ~np.ma.getmaskarray(start)
        & ~np.ma.getmaskarray(end)
        & np.isin(old_values, CLASS_CODES)
        & np.isin(new_values, CLASS_CODES)
    )

    transition = np.full(
        old_values.shape,
        NODATA_VALUE,
        dtype=np.uint8,
    )

    transition[common_mask] = (
        old_values[common_mask] * 10
        + new_values[common_mask]
    ).astype(np.uint8)

    profile = metadata["profile"].copy()

    profile.update({
        "driver": "GTiff",
        "count": 1,
        "dtype": "uint8",
        "nodata": NODATA_VALUE,
        "compress": "lzw",
    })

    output_path = (
        MAPS_DIR
        / f"NMMC_TRANSITION_{start_year}_{end_year}.tif"
    )

    with rasterio.open(
        output_path,
        "w",
        **profile,
    ) as dst:

        dst.write(transition, 1)

    print(f"Saved transition raster: {output_path.name}")

    return transition


# ---------------------------------------------------------
# 8. VISUALIZE TRANSITION MAP
# ---------------------------------------------------------

def plot_transition_map(
    transition,
    start_year,
    end_year,
):
    """
    Generate a categorical transition map.

    Each present transition receives its own colour.
    """

    valid = transition != NODATA_VALUE

    present_codes = sorted(
        np.unique(transition[valid]).tolist()
    )

    if not present_codes:
        print(
            f"No valid transitions for "
            f"{start_year}-{end_year}"
        )
        return

    colors = plt.get_cmap("tab20")(
        np.linspace(0, 1, len(present_codes))
    )

    cmap = ListedColormap(colors)

    bounds = (
        np.arange(len(present_codes) + 1) - 0.5
    )

    norm = BoundaryNorm(
        bounds,
        cmap.N,
    )

    lookup = np.full(
        transition.shape,
        -1,
        dtype=np.int16,
    )

    for index, code in enumerate(present_codes):

        lookup[transition == code] = index

    lookup = np.ma.masked_where(
        lookup < 0,
        lookup,
    )

    fig, ax = plt.subplots(figsize=(9, 12))

    ax.imshow(
        lookup,
        cmap=cmap,
        norm=norm,
        interpolation="nearest",
    )

    ax.set_title(
        f"LULC Transitions: {start_year}–{end_year}",
        fontsize=15,
        fontweight="bold",
    )

    ax.set_xlabel("Raster Column")
    ax.set_ylabel("Raster Row")

    legend = []

    for index, code in enumerate(present_codes):

        from_code = code // 10
        to_code = code % 10

        label = (
            f"{LULC_CLASSES[from_code]} → "
            f"{LULC_CLASSES[to_code]}"
        )

        legend.append(
            Patch(
                facecolor=colors[index],
                label=label,
            )
        )

    ax.legend(
        handles=legend,
        loc="upper left",
        bbox_to_anchor=(1.02, 1),
        fontsize=8,
        frameon=False,
    )

    output_path = (
        MAPS_DIR
        / f"NMMC_TRANSITION_{start_year}_{end_year}.png"
    )

    fig.savefig(
        output_path,
        dpi=250,
        bbox_inches="tight",
    )

    plt.close(fig)

    print(f"Saved transition map: {output_path.name}")


# ---------------------------------------------------------
# 9. TRANSITION MATRIX HEATMAP
# ---------------------------------------------------------

def plot_transition_matrix(
    matrix_area,
    start_year,
    end_year,
):
    """Plot a transition matrix using geodesic area."""

    labels = [
        LULC_CLASSES[code]
        for code in CLASS_CODES
    ]

    fig, ax = plt.subplots(figsize=(9, 7))

    image = ax.imshow(
        matrix_area,
        cmap="YlGnBu",
    )

    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))

    ax.set_xticklabels(
        labels,
        rotation=35,
        ha="right",
    )

    ax.set_yticklabels(labels)

    ax.set_xlabel("Destination Class")
    ax.set_ylabel("Original Class")

    ax.set_title(
        f"Land-Cover Transition Matrix "
        f"({start_year}–{end_year})",
        fontsize=14,
        fontweight="bold",
    )

    for row in range(len(labels)):

        for col in range(len(labels)):

            value = matrix_area[row, col]

            ax.text(
                col,
                row,
                f"{value:.2f}",
                ha="center",
                va="center",
                fontsize=9,
            )

    fig.colorbar(
        image,
        ax=ax,
        label="Transition Area (km²)",
    )

    output_path = (
        CHARTS_DIR
        / f"transition_matrix_{start_year}_{end_year}.png"
    )

    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)

    print(f"Saved matrix chart: {output_path.name}")


# ---------------------------------------------------------
# 10. ANNUAL AREA DISTRIBUTION CHART
# ---------------------------------------------------------

def plot_annual_lulc_areas(annual_areas):

    fig, ax = plt.subplots(figsize=(11, 6))

    for code in CLASS_CODES:

        subset = annual_areas[
            annual_areas["class_code"] == code
        ].sort_values("year")

        ax.plot(
            subset["year"],
            subset["area_km2"],
            marker="o",
            linewidth=2,
            color=CLASS_COLORS[code],
            label=LULC_CLASSES[code],
        )

    ax.set_title(
        "NMMC Land-Cover Area Changes (1994–2024)",
        fontsize=14,
        fontweight="bold",
    )

    ax.set_xlabel("Year")
    ax.set_ylabel("Geodesic Area (km²)")
    ax.set_xticks(YEARS)

    ax.grid(True, linestyle="--", alpha=0.35)
    ax.legend()

    output_path = (
        CHARTS_DIR / "NMMC_LULC_annual_area_trends.png"
    )

    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)

    print(f"Saved annual area chart: {output_path.name}")


# ---------------------------------------------------------
# 11. MAIN CHANGE-DETECTION PIPELINE
# ---------------------------------------------------------

def run_change_detection():

    print("\n" + "=" * 70)
    print("NMMC LULC CHANGE DETECTION")
    print("=" * 70)

    # Load and validate data.
    print("\n[1/6] Loading and validating LULC rasters...")

    rasters, metadata = load_lulc_rasters()

    # Calculate geodesic pixel areas.
    print("\n[2/6] Calculating geodesic pixel areas...")

    pixel_areas = calculate_geodesic_pixel_areas(
        metadata
    )

    # Annual class areas.
    print("\n[3/6] Calculating annual LULC area statistics...")

    annual_areas = calculate_annual_lulc_areas(
        rasters,
        pixel_areas,
    )

    annual_path = (
        REPORTS_DIR / "NMMC_LULC_annual_area_statistics.csv"
    )

    annual_areas.to_csv(
        annual_path,
        index=False,
    )

    print(annual_areas.round(4).to_string(index=False))

    # Transition analysis.
    print("\n[4/6] Calculating transition matrices...")

    all_transitions = []
    all_class_changes = []

    for start_year, end_year in PERIODS:

        print(
            f"\nAnalysing {start_year}–{end_year}..."
        )

        (
            transition_table,
            matrix_area,
            matrix_pixels,
            common_mask,
            transition_codes,
        ) = calculate_transition_matrix(
            start_year,
            end_year,
            rasters,
            pixel_areas,
        )

        all_transitions.append(transition_table)

        class_changes = calculate_class_changes(
            matrix_area,
            start_year,
            end_year,
        )

        all_class_changes.append(class_changes)

        transition_path = (
            REPORTS_DIR
            / f"transition_matrix_{start_year}_{end_year}.csv"
        )

        matrix_df = pd.DataFrame(
            matrix_area,
            index=[
                LULC_CLASSES[c]
                for c in CLASS_CODES
            ],
            columns=[
                LULC_CLASSES[c]
                for c in CLASS_CODES
            ],
        )

        matrix_df.to_csv(transition_path)

        transition_table.to_csv(
            REPORTS_DIR
            / f"transition_details_{start_year}_{end_year}.csv",
            index=False,
        )

        print("\nTransition matrix (km²):")
        print(matrix_df.round(4).to_string())

        print("\nClass gains and losses:")
        print(class_changes.round(4).to_string(index=False))

        # Export categorical transition raster.
        transition = export_transition_raster(
            start_year,
            end_year,
            rasters,
            metadata,
        )

        # Visual outputs.
        plot_transition_map(
            transition,
            start_year,
            end_year,
        )

        plot_transition_matrix(
            matrix_area,
            start_year,
            end_year,
        )

    # Save combined transition details.
    print("\n[5/6] Saving combined reports...")

    combined_transitions = pd.concat(
        all_transitions,
        ignore_index=True,
    )

    combined_changes = pd.concat(
        all_class_changes,
        ignore_index=True,
    )

    combined_transitions.to_csv(
        REPORTS_DIR / "NMMC_LULC_all_transition_details.csv",
        index=False,
    )

    combined_changes.to_csv(
        REPORTS_DIR / "NMMC_LULC_class_gains_losses.csv",
        index=False,
    )

    plot_annual_lulc_areas(annual_areas)

    # Long-term summary.
    print("\n[6/6] Generating long-term summary...")

    long_term = combined_changes[
        (combined_changes["start_year"] == 1994)
        & (combined_changes["end_year"] == 2024)
    ]

    print("\n1994–2024 net class changes:")
    print(
        long_term[
            [
                "class_name",
                "gain_km2",
                "loss_km2",
                "net_change_km2",
            ]
        ].round(4).to_string(index=False)
    )

    print("\n" + "=" * 70)
    print("CHANGE DETECTION COMPLETED")
    print("=" * 70)

    print(f"\nReports: {REPORTS_DIR}")
    print(f"Maps: {MAPS_DIR}")
    print(f"Charts: {CHARTS_DIR}")

    return {
        "annual_areas": annual_areas,
        "transition_details": combined_transitions,
        "class_changes": combined_changes,
    }


# ---------------------------------------------------------
# 12. EXECUTION
# ---------------------------------------------------------

if __name__ == "__main__":
    run_change_detection()
