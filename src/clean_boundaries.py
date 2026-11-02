
import json
from pathlib import Path
from collections import Counter

from shapely.geometry import shape, mapping, Polygon, MultiPolygon, GeometryCollection

PROJECT_ROOT = Path(__file__).resolve().parent.parent

INPUT_FILE = (
    PROJECT_ROOT / "data" / "boundaries" /
    "NMMC_Ward_Boundaries_2024.geojson"
)

OUTPUT_FILE = (
    PROJECT_ROOT / "data" / "boundaries" /
    "NMMC_Ward_Boundaries_2024_CLEAN.geojson"
)


def extract_polygons(geometry):
    """Extract polygon components recursively."""
    if isinstance(geometry, (Polygon, MultiPolygon)):
        return [geometry] if isinstance(geometry, Polygon) else list(geometry.geoms)

    if isinstance(geometry, GeometryCollection):
        polygons = []
        for part in geometry.geoms:
            polygons.extend(extract_polygons(part))
        return polygons

    return []


def main():
    with open(INPUT_FILE, "r") as file:
        geojson = json.load(file)

    cleaned_features = []
    original_types = Counter()
    cleaned_types = Counter()

    for feature in geojson["features"]:
        original_geometry = shape(feature["geometry"])
        original_types[original_geometry.geom_type] += 1

        polygons = extract_polygons(original_geometry)

        if not polygons:
            raise ValueError(
                f"No polygon found for ward "
                f"{feature['properties'].get('ward_lgd_c')}"
            )

        if len(polygons) == 1:
            cleaned_geometry = polygons[0]
        else:
            cleaned_geometry = MultiPolygon(polygons)

        if cleaned_geometry.is_empty or not cleaned_geometry.is_valid:
            raise ValueError(
                f"Invalid cleaned geometry for ward "
                f"{feature['properties'].get('ward_lgd_c')}"
            )

        cleaned_feature = {
            "type": "Feature",
            "properties": feature["properties"],
            "geometry": mapping(cleaned_geometry)
        }

        cleaned_features.append(cleaned_feature)
        cleaned_types[cleaned_geometry.geom_type] += 1

    cleaned_geojson = {
        "type": "FeatureCollection",
        "features": cleaned_features
    }

    with open(OUTPUT_FILE, "w") as file:
        json.dump(cleaned_geojson, file)

    print("\nBOUNDARY CLEANING REPORT")
    print("-" * 40)
    print("Original features:", len(geojson["features"]))
    print("Cleaned features:", len(cleaned_features))
    print("Original geometry types:", dict(original_types))
    print("Cleaned geometry types:", dict(cleaned_types))
    print("All geometries valid:", all(
        shape(f["geometry"]).is_valid for f in cleaned_features
    ))
    print("All geometries non-empty:", all(
        not shape(f["geometry"]).is_empty for f in cleaned_features
    ))
    print("Output:", OUTPUT_FILE)


if __name__ == "__main__":
    main()
