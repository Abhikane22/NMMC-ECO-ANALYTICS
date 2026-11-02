
import numpy as np
import geopandas as gpd
import rasterio
import pandas as pd
import shapely

from rasterio.features import geometry_window
from shapely.geometry import mapping

from config import PROJECT_ROOT, REPORTS_DIR, LULC_CLASSES


BOUNDARY_FILE = (
    PROJECT_ROOT
    / "data/boundaries/NMMC_Ward_Boundaries_2024_CLEAN.geojson"
)

RASTER_FILE = (
    PROJECT_ROOT
    / "data/lulc/NMMC_LULC_2024_PRELIMINARY.tif"
)

WARD_ID_FIELD = "ward_lgd_c"


def calculate_ward_lulc(geometry, raster):

    window = geometry_window(
        raster,
        [mapping(geometry)]
    )

    data = raster.read(1, window=window)
    transform = raster.window_transform(window)

    rows, cols = np.indices(data.shape)

    x1 = transform.c + cols * transform.a
    x2 = x1 + transform.a
    y1 = transform.f + rows * transform.e
    y2 = y1 + transform.e

    pixels = shapely.box(
        np.minimum(x1, x2),
        np.minimum(y1, y2),
        np.maximum(x1, x2),
        np.maximum(y1, y2),
    )

    intersections = shapely.area(
        shapely.intersection(pixels, geometry)
    )

    valid = (
        np.isin(data, list(LULC_CLASSES.keys()))
        & (intersections > 0)
    )

    class_areas = np.bincount(
        data[valid].astype(int),
        weights=intersections[valid],
        minlength=len(LULC_CLASSES),
    )

    valid_area = intersections[valid].sum()
    ward_area = geometry.area

    return {
        "ward_area_km2": ward_area / 1e6,
        "valid_raster_area_km2": valid_area / 1e6,
        "valid_coverage_pct": (
            valid_area / ward_area * 100 if ward_area else 0
        ),
        "class_areas_km2": class_areas / 1e6,
    }


def main():

    wards = gpd.read_file(BOUNDARY_FILE)

    if len(wards) != 111:
        raise ValueError(f"Expected 111 wards, found {len(wards)}")

    if wards[WARD_ID_FIELD].isna().any():
        raise ValueError("Missing ward IDs")

    if wards[WARD_ID_FIELD].duplicated().any():
        raise ValueError("Duplicate ward IDs")

    results = []

    with rasterio.open(RASTER_FILE) as raster:

        if raster.crs.to_epsg() != 32643:
            raise ValueError("Expected EPSG:32643")

        wards = wards.to_crs(raster.crs)

        for _, ward in wards.iterrows():

            geometry = ward.geometry

            if geometry is None or geometry.is_empty or not geometry.is_valid:
                raise ValueError(
                    f"Invalid geometry: {ward[WARD_ID_FIELD]}"
                )

            stats = calculate_ward_lulc(geometry, raster)

            row = {
                "year": 2024,
                "ward_id": ward[WARD_ID_FIELD],
                "ward_area_km2": stats["ward_area_km2"],
                "valid_raster_area_km2": stats["valid_raster_area_km2"],
                "valid_coverage_pct": stats["valid_coverage_pct"],
            }

            for class_id, class_name in LULC_CLASSES.items():

                area = stats["class_areas_km2"][class_id]

                row[f"{class_name}_area_km2"] = area

                row[f"{class_name}_pct_of_ward"] = (
                    area / stats["ward_area_km2"] * 100
                    if stats["ward_area_km2"] else 0
                )

                row[f"{class_name}_pct_of_valid_area"] = (
                    area / stats["valid_raster_area_km2"] * 100
                    if stats["valid_raster_area_km2"] else 0
                )

            results.append(row)

    df = pd.DataFrame(results)

    class_columns = [
        f"{name}_area_km2"
        for name in LULC_CLASSES.values()
    ]

    df["class_area_sum_km2"] = df[class_columns].sum(axis=1)

    df["area_difference_km2"] = (
        df["valid_raster_area_km2"] - df["class_area_sum_km2"]
    )

    if not np.allclose(df["area_difference_km2"], 0, atol=1e-8):
        raise ValueError("Class area accounting failed")

    if (df["valid_coverage_pct"] > 100.01).any():
        raise ValueError("Coverage exceeds ward area")

    output = (
        REPORTS_DIR
        / "NMMC_Ward_Level_LULC_2024_PRELIMINARY.csv"
    )

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(output, index=False)

    print("\n2024 PRELIMINARY WARD LULC COMPLETED")
    print("Records:", len(df))
    print("Unique wards:", df["ward_id"].nunique())
    print("Area accounting: PASS")
    print("Output:", output)

    print("\nClass areas (km²):")
    print(df[class_columns].sum().round(4).to_string())


if __name__ == "__main__":
    main()
