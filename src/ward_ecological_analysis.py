
import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
import shapely

from pathlib import Path
from rasterio.features import geometry_window
from shapely.geometry import mapping

from config import PROJECT_ROOT, REPORTS_DIR, YEARS


BOUNDARY_FILE = (
    PROJECT_ROOT / "data" / "boundaries" /
    "NMMC_Ward_Boundaries_2024_CLEAN.geojson"
)

OPTICAL_DIR = PROJECT_ROOT / "data" / "ecological" / "optical_indices"
THERMAL_DIR = PROJECT_ROOT / "data" / "ecological" / "thermal_indicators"

WARD_ID_FIELD = "ward_lgd_c"

OPTICAL_BANDS = ["NDVI", "NDBI", "NDMI", "NDWI", "MNDWI"]
THERMAL_BANDS = ["LST_C", "UTFVI"]


def calculate_pixel_intersections(geometry, raster):
    """Return pixel intersection areas and the corresponding raster window."""

    window = geometry_window(raster, [mapping(geometry)])

    window_transform = raster.window_transform(window)

    rows, cols = np.indices(
        (int(window.height), int(window.width))
    )

    x1 = window_transform.c + cols * window_transform.a
    x2 = x1 + window_transform.a

    y1 = window_transform.f + rows * window_transform.e
    y2 = y1 + window_transform.e

    pixels = shapely.box(
        np.minimum(x1, x2),
        np.minimum(y1, y2),
        np.maximum(x1, x2),
        np.maximum(y1, y2),
    )

    areas = shapely.area(
        shapely.intersection(pixels, geometry)
    )

    return window, areas


def calculate_band_statistics(
    band_data,
    intersection_areas,
    ward_area_m2,
):
    """Calculate area-weighted statistics for one raster band."""

    values = np.asarray(band_data.filled(np.nan), dtype=np.float64)

    valid = (
        np.isfinite(values)
        & np.isfinite(intersection_areas)
        & (intersection_areas > 0)
    )

    if not valid.any():
        return {
            "mean": np.nan,
            "min": np.nan,
            "max": np.nan,
            "valid_area_km2": 0.0,
            "coverage_pct": 0.0,
        }

    valid_values = values[valid]
    valid_areas = intersection_areas[valid]

    valid_area = valid_areas.sum()

    weighted_mean = np.average(
        valid_values,
        weights=valid_areas,
    )

    return {
        "mean": float(weighted_mean),
        "min": float(valid_values.min()),
        "max": float(valid_values.max()),
        "valid_area_km2": float(valid_area / 1_000_000),
        "coverage_pct": float(
            valid_area / ward_area_m2 * 100
        ),
    }


def process_indicator_group(
    wards,
    year,
    raster_path,
    expected_bands,
    group_name,
):
    """Calculate indicator statistics for all wards in one raster."""

    results = []

    with rasterio.open(raster_path) as raster:

        if raster.crs is None or raster.crs.to_epsg() != 32643:
            raise ValueError(f"Unexpected CRS: {raster.crs}")

        year_wards = wards.to_crs(raster.crs)

        if raster.count != len(expected_bands):
            raise ValueError(
                f"Unexpected band count in {raster_path}"
            )

        for _, ward in year_wards.iterrows():

            ward_id = ward[WARD_ID_FIELD]
            geometry = ward.geometry
            ward_area_m2 = geometry.area

            if geometry is None or geometry.is_empty:
                raise ValueError(f"Invalid ward geometry: {ward_id}")

            window, intersection_areas = (
                calculate_pixel_intersections(geometry, raster)
            )

            row = {
                "year": year,
                "ward_id": ward_id,
                "indicator_group": group_name,
            }

            for band_index, band_name in enumerate(
                expected_bands, start=1
            ):

                band_data = raster.read(
                    band_index,
                    window=window,
                    masked=True,
                )

                stats = calculate_band_statistics(
                    band_data,
                    intersection_areas,
                    ward_area_m2,
                )

                prefix = band_name.lower()

                row[f"{prefix}_mean"] = stats["mean"]
                row[f"{prefix}_min"] = stats["min"]
                row[f"{prefix}_max"] = stats["max"]
                row[f"{prefix}_valid_area_km2"] = (
                    stats["valid_area_km2"]
                )
                row[f"{prefix}_coverage_pct"] = (
                    stats["coverage_pct"]
                )

            results.append(row)

    return results


def main():

    print("\nNMMC WARD-LEVEL ECOLOGICAL ANALYSIS")
    print("=" * 60)

    wards = gpd.read_file(BOUNDARY_FILE)

    if len(wards) != 111:
        raise ValueError(
            f"Expected 111 wards, found {len(wards)}"
        )

    if wards[WARD_ID_FIELD].isna().any():
        raise ValueError("Missing ward identifiers.")

    if wards[WARD_ID_FIELD].duplicated().any():
        raise ValueError("Duplicate ward identifiers.")

    all_results = []

    for year in YEARS:

        print(f"\nProcessing year: {year}")

        optical_path = (
            OPTICAL_DIR / f"NMMC_Optical_Indices_{year}.tif"
        )

        thermal_path = (
            THERMAL_DIR / f"NMMC_Thermal_Indicators_{year}.tif"
        )

        if not optical_path.exists():
            raise FileNotFoundError(optical_path)

        if not thermal_path.exists():
            raise FileNotFoundError(thermal_path)

        optical_results = process_indicator_group(
            wards,
            year,
            optical_path,
            OPTICAL_BANDS,
            "OPTICAL",
        )

        thermal_results = process_indicator_group(
            wards,
            year,
            thermal_path,
            THERMAL_BANDS,
            "THERMAL",
        )

        # Merge optical and thermal results by ward and year.
        optical_df = pd.DataFrame(optical_results)
        thermal_df = pd.DataFrame(thermal_results)

        merged = optical_df.drop(
            columns=["indicator_group"]
        ).merge(
            thermal_df.drop(columns=["indicator_group"]),
            on=["year", "ward_id"],
            how="inner",
            validate="one_to_one",
        )

        if len(merged) != 111:
            raise ValueError(
                f"Expected 111 merged records for {year}"
            )

        all_results.append(merged)

        print(
            f"Completed {year}: "
            f"{len(merged)} wards"
        )

    final_df = pd.concat(
        all_results,
        ignore_index=True,
    )

    if len(final_df) != 444:
        raise ValueError(
            f"Expected 444 records, got {len(final_df)}"
        )

    if final_df.duplicated(
        subset=["year", "ward_id"]
    ).any():
        raise ValueError("Duplicate ward-year records.")

    # Check that valid coverage does not exceed ward area.
    coverage_columns = [
        f"{name.lower()}_coverage_pct"
        for name in OPTICAL_BANDS + THERMAL_BANDS
    ]

    for column in coverage_columns:
        if (final_df[column] > 100.01).any():
            raise ValueError(
                f"Coverage exceeds ward area: {column}"
            )

    REPORTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = (
        REPORTS_DIR /
        "NMMC_Ward_Level_Ecological_Statistics.csv"
    )

    final_df.to_csv(
        output_path,
        index=False,
    )

    print("\n" + "=" * 60)
    print("ECOLOGICAL ANALYSIS COMPLETED")
    print("=" * 60)

    print("Total records:", len(final_df))
    print("Unique wards:", final_df["ward_id"].nunique())
    print("Years:", sorted(final_df["year"].unique()))

    print("\nMean coverage by year:")

    print(
        final_df.groupby("year")[coverage_columns]
        .mean()
        .round(3)
        .to_string()
    )

    print("\nCity-wide mean indicator values by year:")

    indicator_columns = [
        f"{name.lower()}_mean"
        for name in OPTICAL_BANDS + THERMAL_BANDS
    ]

    print(
        final_df.groupby("year")[indicator_columns]
        .mean()
        .round(4)
        .to_string()
    )

    print("\nOutput:", output_path)


if __name__ == "__main__":
    main()
