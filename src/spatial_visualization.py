
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import rasterio

from src.config import YEARS, CHARTS_DIR
from src.raster_processor import get_raster_path


# Indicators to map and their corresponding raster types.
INDICATORS = {
    "NDVI": {
        "raster_type": "optical",
        "cmap": "RdYlGn",
        "label": "Normalized Difference Vegetation Index",
        "fixed_limits": (-1, 1),
    },
    "MNDWI": {
        "raster_type": "optical",
        "cmap": "BrBG",
        "label": "Modified Normalized Difference Water Index",
        "fixed_limits": (-1, 1),
    },
    "LST_C": {
        "raster_type": "thermal",
        "cmap": "inferno",
        "label": "Land Surface Temperature (°C)",
        "fixed_limits": None,
    },
    "UTFVI": {
        "raster_type": "thermal",
        "cmap": "coolwarm",
        "label": "Urban Thermal Field Variation Index",
        "fixed_limits": None,
    },
}


def load_indicator_raster(year, indicator):
    """Load one indicator band and its geographic metadata."""

    raster_type = INDICATORS[indicator]["raster_type"]
    raster_path = get_raster_path(year, raster_type)

    if not raster_path.exists():
        raise FileNotFoundError(
            f"Raster not found: {raster_path}"
        )

    with rasterio.open(raster_path) as src:
        band_names = list(src.descriptions)

        if indicator not in band_names:
            raise ValueError(
                f"{indicator} not found in {raster_path.name}. "
                f"Available bands: {band_names}"
            )

        band_index = band_names.index(indicator) + 1
        data = src.read(band_index, masked=True)

        metadata = {
            "bounds": src.bounds,
            "crs": src.crs,
            "resolution": src.res,
            "width": src.width,
            "height": src.height,
        }

    return data, metadata


def calculate_display_limits(indicator, all_data):
    """Calculate common colour limits for all four years."""

    fixed_limits = INDICATORS[indicator]["fixed_limits"]

    if fixed_limits is not None:
        return fixed_limits

    valid_arrays = []

    for data in all_data.values():
        values = data.compressed()
        values = values[np.isfinite(values)]

        if values.size:
            valid_arrays.append(values)

    if not valid_arrays:
        raise ValueError(
            f"No valid pixels found for {indicator}"
        )

    combined = np.concatenate(valid_arrays)

    if indicator == "UTFVI":
        limit = float(np.percentile(np.abs(combined), 98))
        limit = max(limit, 1e-6)
        return -limit, limit

    lower, upper = np.percentile(combined, [2, 98])

    return float(lower), float(upper)


def create_spatial_map(
    year,
    indicator,
    data,
    metadata,
    vmin,
    vmax
):
    """Create and save one publication-quality spatial map."""

    CHARTS_DIR.mkdir(parents=True, exist_ok=True)

    settings = INDICATORS[indicator]
    bounds = metadata["bounds"]

    fig, ax = plt.subplots(figsize=(9, 7))

    cmap = plt.get_cmap(settings["cmap"]).copy()
    cmap.set_bad("white")

    image = ax.imshow(
        data,
        extent=(
            bounds.left,
            bounds.right,
            bounds.bottom,
            bounds.top
        ),
        origin="upper",
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        interpolation="nearest"
    )

    colorbar = fig.colorbar(
        image,
        ax=ax,
        shrink=0.82,
        pad=0.03
    )

    colorbar.set_label(settings["label"])

    ax.set_title(
        f"NMMC {indicator} Spatial Distribution ({year})",
        fontsize=14,
        fontweight="bold",
        pad=14
    )

    ax.set_xlabel("Easting (m)")
    ax.set_ylabel("Northing (m)")

    ax.set_aspect("equal")
    ax.ticklabel_format(
        style="plain",
        axis="both",
        useOffset=False
    )

    ax.text(
        0.02,
        0.02,
        f"CRS: {metadata['crs']} | "
        f"Resolution: {metadata['resolution'][0]} m",
        transform=ax.transAxes,
        fontsize=8,
        verticalalignment="bottom",
        bbox={
            "facecolor": "white",
            "alpha": 0.8,
            "edgecolor": "none"
        }
    )

    fig.tight_layout()

    filename = f"{indicator}_{year}_spatial_map.png"

    output_path = CHARTS_DIR / filename

    fig.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close(fig)

    print(f"Saved: {output_path}")


def generate_spatial_maps():
    print("\n" + "=" * 65)
    print("NMMC SPATIAL INDICATOR MAPPING")
    print("=" * 65)

    for indicator in INDICATORS:

        print(f"\nProcessing indicator: {indicator}")

        all_data = {}
        all_metadata = {}

        # Load all years before plotting so the colour scale
        # remains consistent across the four maps.
        for year in YEARS:
            data, metadata = load_indicator_raster(
                year,
                indicator
            )

            all_data[year] = data
            all_metadata[year] = metadata

        vmin, vmax = calculate_display_limits(
            indicator,
            all_data
        )

        print(
            f"Common display range: "
            f"{vmin:.4f} to {vmax:.4f}"
        )

        for year in YEARS:
            create_spatial_map(
                year,
                indicator,
                all_data[year],
                all_metadata[year],
                vmin,
                vmax
            )

    print("\n" + "=" * 65)
    print("SPATIAL MAPPING COMPLETE")
    print(f"Maps saved in: {CHARTS_DIR}")
    print("=" * 65)


if __name__ == "__main__":
    generate_spatial_maps()
