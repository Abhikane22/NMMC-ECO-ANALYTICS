
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import YEARS, REPORTS_DIR
from src.raster_processor import (
    get_band,
    calculate_band_statistics,
)


# ---------------------------------------------------------
# 1. CONFIGURATION
# ---------------------------------------------------------

OPTICAL_INDICATORS = [
    "NDVI",
    "NDBI",
    "NDMI",
    "NDWI",
    "MNDWI",
]

THERMAL_INDICATORS = [
    "LST_C",
    "UTFVI",
]

ALL_INDICATORS = OPTICAL_INDICATORS + THERMAL_INDICATORS

REPORTS_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------
# 2. LOAD AN INDICATOR
# ---------------------------------------------------------

def load_indicator(indicator, year):
    """
    Load one ecological indicator for a given year.

    Returns a masked NumPy array.
    """

    if indicator in OPTICAL_INDICATORS:
        raster_type = "optical"

    elif indicator in THERMAL_INDICATORS:
        raster_type = "thermal"

    else:
        raise ValueError(f"Unknown ecological indicator: {indicator}")

    return get_band(
        year=year,
        band_name=indicator,
        raster_type=raster_type,
    )


# ---------------------------------------------------------
# 3. ANNUAL STATISTICS
# ---------------------------------------------------------

def calculate_annual_statistics():
    """
    Calculate annual statistics for all ecological indicators.

    Statistics:
        count, mean, standard deviation,
        minimum, maximum, median.
    """

    records = []

    for year in YEARS:

        print(f"\nProcessing year: {year}")

        for indicator in ALL_INDICATORS:

            band = load_indicator(indicator, year)

            stats = calculate_band_statistics(band)

            record = {
                        "year": year,
                        "indicator": indicator,
                        "valid_pixels": stats["valid_pixels"],
                        "mean": stats["mean"],
                        "std": stats["std_dev"],
                        "min": stats["min"],
                        "max": stats["max"],
                        "median": stats["median"],
                    }

            records.append(record)

            print(
                f"{indicator}: "
                f"mean={stats['mean']:.4f}, "
                f"std={stats['std_dev']:.4f}"
            )

    return pd.DataFrame(records)


# ---------------------------------------------------------
# 4. TEMPORAL CHANGE: 1994 TO 2024
# ---------------------------------------------------------

def calculate_temporal_change(
    indicator,
    start_year=1994,
    end_year=2024,
):
    """
    Calculate pixel-wise change between two years.

    Only pixels valid in both years are compared.
    The rasters must have matching dimensions.
    """

    start_data = load_indicator(indicator, start_year)
    end_data = load_indicator(indicator, end_year)

    if start_data.shape != end_data.shape:
        raise ValueError(
            f"Raster dimensions differ for {indicator}: "
            f"{start_data.shape} vs {end_data.shape}"
        )

    start_values = np.asarray(start_data.data, dtype=np.float64)
    end_values = np.asarray(end_data.data, dtype=np.float64)

    start_mask = np.ma.getmaskarray(start_data)
    end_mask = np.ma.getmaskarray(end_data)

    common_mask = (
        start_mask
        | end_mask
        | ~np.isfinite(start_values)
        | ~np.isfinite(end_values)
    )

    valid_start = start_values[~common_mask]
    valid_end = end_values[~common_mask]

    if valid_start.size == 0:
        raise ValueError(
            f"No common valid pixels for {indicator}"
        )

    change = valid_end - valid_start

    start_mean = float(np.mean(valid_start))
    end_mean = float(np.mean(valid_end))

    absolute_mean_change = end_mean - start_mean

    # Percentage change is not very meaningful for
    # normalized indices or means close to zero.
    percentage_change = np.nan

    if abs(start_mean) > 0.05:
        percentage_change = (
            absolute_mean_change / abs(start_mean)
        ) * 100

    return {
        "indicator": indicator,
        "start_year": start_year,
        "end_year": end_year,
        "start_mean_common_pixels": start_mean,
        "end_mean_common_pixels": end_mean,
        "mean_change": absolute_mean_change,
        "percentage_change": percentage_change,
        "pixel_change_min": float(np.min(change)),
        "pixel_change_max": float(np.max(change)),
        "pixel_change_std": float(np.std(change)),
        "common_valid_pixels": int(valid_start.size),
    }


def calculate_all_temporal_changes():
    """
    Calculate 1994-2024 changes for all indicators.
    """

    records = []

    for indicator in ALL_INDICATORS:

        print(f"Calculating temporal change: {indicator}")

        records.append(
            calculate_temporal_change(indicator)
        )

    return pd.DataFrame(records)


# ---------------------------------------------------------
# 5. DESCRIPTIVE TEMPORAL TRENDS
# ---------------------------------------------------------

def calculate_temporal_trends(annual_statistics):
    """
    Estimate a simple linear slope across the four study years.

    This is descriptive analysis only. It does not establish
    statistical significance or causation.
    """

    records = []

    for indicator in ALL_INDICATORS:

        subset = annual_statistics[
            annual_statistics["indicator"] == indicator
        ].dropna(subset=["mean"])

        if len(subset) < 2:
            continue

        x = subset["year"].to_numpy(dtype=float)
        y = subset["mean"].to_numpy(dtype=float)

        slope, intercept = np.polyfit(x, y, 1)

        predicted = slope * x + intercept

        ss_res = np.sum((y - predicted) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)

        r_squared = (
            1 - ss_res / ss_tot
            if ss_tot > 0
            else np.nan
        )

        records.append({
            "indicator": indicator,
            "slope_per_year": float(slope),
            "slope_per_decade": float(slope * 10),
            "r_squared_descriptive": float(r_squared),
            "number_of_years": len(subset),
        })

    return pd.DataFrame(records)


# ---------------------------------------------------------
# 6. MAIN PIPELINE
# ---------------------------------------------------------

def run_ecological_analysis():

    print("\n" + "=" * 65)
    print("NMMC ECOLOGICAL ANALYSIS")
    print("=" * 65)

    # Annual statistics
    print("\n[1/3] Calculating annual statistics...")

    annual_stats = calculate_annual_statistics()

    annual_path = (
        REPORTS_DIR / "ecological_annual_raster_statistics.csv"
    )

    annual_stats.to_csv(annual_path, index=False)

    # Temporal changes
    print("\n[2/3] Calculating temporal changes...")

    temporal_changes = calculate_all_temporal_changes()

    change_path = (
        REPORTS_DIR / "ecological_temporal_changes_1994_2024.csv"
    )

    temporal_changes.to_csv(change_path, index=False)

    # Temporal trends
    print("\n[3/3] Calculating descriptive trends...")

    trends = calculate_temporal_trends(annual_stats)

    trends_path = (
        REPORTS_DIR / "ecological_temporal_trends.csv"
    )

    trends.to_csv(trends_path, index=False)

    # Display results
    print("\n" + "=" * 65)
    print("ANNUAL STATISTICS")
    print("=" * 65)

    print(annual_stats.round(4).to_string(index=False))

    print("\n" + "=" * 65)
    print("TEMPORAL CHANGES: 1994-2024")
    print("=" * 65)

    print(temporal_changes.round(4).to_string(index=False))

    print("\n" + "=" * 65)
    print("DESCRIPTIVE TEMPORAL TRENDS")
    print("=" * 65)

    print(trends.round(4).to_string(index=False))

    print("\nReports saved successfully:")
    print(annual_path)
    print(change_path)
    print(trends_path)

    return {
        "annual_statistics": annual_stats,
        "temporal_changes": temporal_changes,
        "temporal_trends": trends,
    }


# ---------------------------------------------------------
# 7. EXECUTION
# ---------------------------------------------------------

if __name__ == "__main__":
    run_ecological_analysis()
