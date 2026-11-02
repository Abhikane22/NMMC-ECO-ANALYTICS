
import json
from pathlib import Path
from shapely.geometry import shape
from shapely.ops import transform
from pyproj import Transformer

ROOT = Path(__file__).resolve().parent.parent
BOUNDARY_DIR = ROOT / "data" / "boundaries"

original_path = BOUNDARY_DIR / "NMMC_Ward_Boundaries_2024.geojson"
clean_path = BOUNDARY_DIR / "NMMC_Ward_Boundaries_2024_CLEAN.geojson"

with open(original_path) as f:
    original = json.load(f)

with open(clean_path) as f:
    cleaned = json.load(f)

assert len(original["features"]) == len(cleaned["features"])

# Project geographic coordinates to UTM Zone 43N for area comparison
project = Transformer.from_crs(
    "EPSG:4326", "EPSG:32643", always_xy=True
).transform

original_features = {
    str(f["properties"]["ward_lgd_c"]): f
    for f in original["features"]
}

clean_features = {
    str(f["properties"]["ward_lgd_c"]): f
    for f in cleaned["features"]
}

assert original_features.keys() == clean_features.keys()

area_differences = []

for ward_id, original_feature in original_features.items():
    clean_feature = clean_features[ward_id]

    assert original_feature["properties"] == clean_feature["properties"]

    original_geom = shape(original_feature["geometry"])
    clean_geom = shape(clean_feature["geometry"])

    original_polygons = [
        part for part in (
            original_geom.geoms
            if original_geom.geom_type == "GeometryCollection"
            else [original_geom]
        )
        if part.geom_type in ("Polygon", "MultiPolygon")
    ]

    original_polygon = original_polygons[0] if len(original_polygons) == 1 else None

    if original_polygon is None:
        raise ValueError(f"Unexpected polygon structure in ward {ward_id}")

    original_area = transform(project, original_polygon).area
    clean_area = transform(project, clean_geom).area

    difference = abs(original_area - clean_area)
    area_differences.append(difference)

    assert difference < 0.01, f"Area changed for ward {ward_id}"

print("\nBOUNDARY PRESERVATION REPORT")
print("-" * 40)
print("Original wards:", len(original_features))
print("Cleaned wards:", len(clean_features))
print("Ward attributes preserved: PASS")
print("All polygon areas preserved: PASS")
print("Maximum area difference (m²):", max(area_differences))
print("Validation: SUCCESS")
