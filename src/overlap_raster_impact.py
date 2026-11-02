
import geopandas as gpd
import rasterio
import shapely
import numpy as np

from pathlib import Path
from rasterio.features import geometry_window
from shapely.geometry import mapping

ROOT = Path(__file__).resolve().parent.parent

BOUNDARY = (
    ROOT / "data" / "boundaries" /
    "NMMC_Ward_Boundaries_2024_CLEAN.geojson"
)

RASTER = (
    ROOT / "data" / "lulc" /
    "NMMC_2024_LULC_SCIENTIFIC_FINAL.tif"
)

wards = gpd.read_file(BOUNDARY).to_crs("EPSG:32643")

# Calculate the total area covered more than once.
# This is different from simply summing all pairwise overlaps
# when three or more polygons intersect at the same location.
overlap_regions = []

for i in range(len(wards)):
    for j in range(i + 1, len(wards)):
        intersection = wards.geometry.iloc[i].intersection(
            wards.geometry.iloc[j]
        )

        if not intersection.is_empty and intersection.area > 0:
            overlap_regions.append(intersection)

if not overlap_regions:
    print("No overlapping regions.")
    raise SystemExit

overlap_union = shapely.union_all(overlap_regions)

with rasterio.open(RASTER) as src:
    window = geometry_window(
        src,
        [mapping(overlap_union)]
    )

    data = src.read(1, window=window)
    transform = src.window_transform(window)

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

    intersection_areas = shapely.area(
        shapely.intersection(pixels, overlap_union)
    )

    valid = (
        (data >= 0)
        & (data <= 4)
        & (intersection_areas > 0)
    )

    print("\nOVERLAP RASTER IMPACT — 2024")
    print("-" * 45)
    print("Overlap geometry area (m²):", overlap_union.area)
    print("Valid intersecting pixels:", int(valid.sum()))
    print(
        "Classified raster area inside overlaps (m²):",
        intersection_areas[valid].sum()
    )

    for class_id, class_name in {
        0: "Water",
        1: "Vegetation",
        2: "Built-up",
        3: "Barren",
        4: "Coastal Wetlands",
    }.items():
        area = intersection_areas[
            valid & (data == class_id)
        ].sum()

        print(f"{class_name}: {area:.6f} m²")
