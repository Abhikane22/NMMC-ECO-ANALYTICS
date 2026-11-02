
import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
import shapely

from rasterio.features import geometry_window
from shapely.geometry import mapping
from pathlib import Path

from config import (
    PROJECT_ROOT,
    LULC_DIR,
    REPORTS_DIR,
    YEARS,
    LULC_CLASSES,
)


BOUNDARY_FILE = (
    PROJECT_ROOT
    / "data"
    / "boundaries"
    / "NMMC_Ward_Boundaries_2024_CLEAN.geojson"
)

WARD_ID_FIELD = "ward_lgd_c"


def calculate_ward_lulc(geometry, raster):
    """
    Calculate class areas using exact pixel-ward intersection areas.

    Returns class areas in km², total valid raster coverage,
    and the ward's total area in km².
    """

    transform = raster.transform

    if not np.isclose(transform.b, 0) or not np.isclose(transform.d, 0):
        raise ValueError("Rotated raster grids are not supported.")

    # Find the smallest raster window covering the ward.
    window = geometry_window(
        raster,
        [mapping(geometry)]
    )

    data = raster.read(1, window=window)
    window_transform = raster.window_transform(window)

    rows, cols = np.indices(data.shape)

    # Construct the actual projected footprint of every pixel.
    x1 = window_transform.c + cols * window_transform.a
    x2 = x1 + window_transform.a
    y1 = window_transform.f + rows * window_transform.e
    y2 = y1 + window_transform.e

    pixel_polygons = shapely.box(
        np.minimum(x1, x2),
        np.minimum(y1, y2),
        np.maximum(x1, x2),
        np.maximum(y1, y2),
    )

    # Exact intersection area between each pixel and the ward.
    intersection_areas = shapely.area(
        shapely.intersection(pixel_polygons, geometry)
    )

    # Only count pixels with a valid LULC class.
    valid_mask = (
        (data >= 0)
        & (data <= 4)
        & (intersection_areas > 0)
    )

    valid_classes = data[valid_mask].astype(np.int32)
    valid_areas = intersection_areas[valid_mask]

    class_areas = np.bincount(
        valid_classes,
        weights=valid_areas,
        minlength=len(LULC_CLASSES),
    )

    ward_area_km2 = geometry.area / 1_000_000
    valid_area_km2 = valid_areas.sum() / 1_000_000

    return {
        "ward_area_km2": ward_area_km2,
        "valid_area_km2": valid_area_km2,
        "valid_coverage_pct": (
            valid_area_km2 / ward_area_km2 * 100
            if ward_area_km2 > 0 else 0
        ),
        "class_areas_km2": class_areas / 1_000_000,
    }


def main():
    print("\nNMMC WARD-LEVEL LULC ANALYSIS")
    print("=" * 55)

    if not BOUNDARY_FILE.exists():
        raise FileNotFoundError(BOUNDARY_FILE)

    wards = gpd.read_file(BOUNDARY_FILE)

    if WARD_ID_FIELD not in wards.columns:
        raise ValueError(
            f"Missing ward ID field: {WARD_ID_FIELD}. "
            f"Available fields: {list(wards.columns)}"
        )

    if wards[WARD_ID_FIELD].isna().any():
        raise ValueError("Some wards have missing ward IDs.")

    if wards[WARD_ID_FIELD].duplicated().any():
        raise ValueError("Ward IDs are not unique.")

    if len(wards) != 111:
        raise ValueError(
            f"Expected 111 wards, found {len(wards)}."
        )

    results = []

    for year in YEARS:
        raster_path = (
            LULC_DIR / f"NMMC_{year}_LULC_SCIENTIFIC_FINAL.tif"
        )

        # Historical rasters have PRELIMINARY filenames.
        if not raster_path.exists():
            raster_path = (
                LULC_DIR / f"NMMC_LULC_{year}_PRELIMINARY.tif"
            )

        if not raster_path.exists():
            raise FileNotFoundError(
                f"No LULC raster found for {year}: {raster_path}"
            )

        print(f"\nProcessing {year}: {raster_path.name}")

        with rasterio.open(raster_path) as raster:

            if raster.crs is None:
                raise ValueError(f"Missing CRS in {raster_path}")

            if raster.crs.to_epsg() != 32643:
                raise ValueError(
                    f"Expected EPSG:32643, got {raster.crs}"
                )

            year_wards = wards.to_crs(raster.crs)

            for _, ward in year_wards.iterrows():
                ward_id = ward[WARD_ID_FIELD]
                geometry = ward.geometry

                if geometry is None or geometry.is_empty:
                    raise ValueError(
                        f"Empty geometry for ward {ward_id}"
                    )

                if not geometry.is_valid:
                    raise ValueError(
                        f"Invalid geometry for ward {ward_id}"
                    )

                stats = calculate_ward_lulc(
                    geometry,
                    raster,
                )

                row = {
                    "year": year,
                    "ward_id": ward_id,
                    "ward_area_km2": stats["ward_area_km2"],
                    "valid_raster_area_km2": stats["valid_area_km2"],
                    "valid_coverage_pct": stats["valid_coverage_pct"],
                }

                for class_id, class_name in LULC_CLASSES.items():
                    class_area = stats["class_areas_km2"][class_id]

                    row[f"{class_name}_area_km2"] = class_area

                    row[f"{class_name}_pct_of_ward"] = (
                        class_area / stats["ward_area_km2"] * 100
                        if stats["ward_area_km2"] > 0 else 0
                    )

                    row[f"{class_name}_pct_of_valid_area"] = (
                        class_area / stats["valid_area_km2"] * 100
                        if stats["valid_area_km2"] > 0 else 0
                    )

                results.append(row)

        print(f"Completed {year}: {len(year_wards)} wards")

    df = pd.DataFrame(results)

    # Validate that class areas add up to valid raster area.
    class_area_columns = [
        f"{name}_area_km2"
        for name in LULC_CLASSES.values()
    ]

    df["class_area_sum_km2"] = df[class_area_columns].sum(axis=1)

    df["area_difference_km2"] = (
        df["valid_raster_area_km2"] - df["class_area_sum_km2"]
    )

    if not np.allclose(
        df["area_difference_km2"],
        0,
        atol=1e-8,
    ):
        raise ValueError(
            "Class area totals do not match valid raster areas."
        )

    if (df["valid_coverage_pct"] > 100.01).any():
        raise ValueError(
            "Raster coverage exceeds ward area. "
            "Check overlapping or inconsistent geometries."
        )

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    output_path = (
        REPORTS_DIR / "NMMC_Ward_Level_LULC_Statistics.csv"
    )

    df.to_csv(output_path, index=False)

    print("\n" + "=" * 55)
    print("WARD-LEVEL ANALYSIS COMPLETED")
    print("=" * 55)
    print("Total records:", len(df))
    print("Unique wards:", df["ward_id"].nunique())
    print("Years:", sorted(df["year"].unique()))
    print("Area accounting validation: PASS")
    print("Output:", output_path)

    print("\nMean valid raster coverage by year:")
    print(
        df.groupby("year")["valid_coverage_pct"]
        .mean()
        .round(3)
        .to_string()
    )

    print("\nSample results:")
    print(
        df[
            [
                "year",
                "ward_id",
                "ward_area_km2",
                "valid_raster_area_km2",
                "valid_coverage_pct",
            ]
        ].head(10).to_string(index=False)
    )


if __name__ == "__main__":
    main()
