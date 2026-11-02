
import geopandas as gpd
from pathlib import Path
from shapely.strtree import STRtree

ROOT = Path(__file__).resolve().parent.parent

path = (
    ROOT / "data" / "boundaries" /
    "NMMC_Ward_Boundaries_2024_CLEAN.geojson"
)

wards = gpd.read_file(path).to_crs("EPSG:32643")

geometries = wards.geometry.to_numpy()
ward_ids = wards["ward_lgd_c"].to_numpy()

tree = STRtree(geometries)

overlaps = []

for i, geom in enumerate(geometries):
    # Query only spatially intersecting candidate geometries.
    candidates = tree.query(geom)

    for j in candidates:
        j = int(j)

        # Avoid self-intersections and duplicate pairs.
        if j <= i:
            continue

        intersection = geom.intersection(geometries[j])

        # Only polygonal overlap counts; shared boundaries do not.
        overlap_area = intersection.area

        if overlap_area > 0.01:
            overlaps.append({
                "ward_1": ward_ids[i],
                "ward_2": ward_ids[j],
                "overlap_area_m2": overlap_area,
                "overlap_area_km2": overlap_area / 1e6,
            })

print("\nWARD-TO-WARD OVERLAP DETAILS")
print("-" * 55)

if not overlaps:
    print("No meaningful polygon overlaps detected.")
else:
    for item in sorted(
        overlaps,
        key=lambda x: x["overlap_area_m2"],
        reverse=True
    ):
        print(item)

    print("\nTotal overlapping pairs:", len(overlaps))
    print(
        "Total pairwise overlap (m²):",
        sum(x["overlap_area_m2"] for x in overlaps)
    )
