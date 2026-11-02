
import geopandas as gpd
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

path = (
    ROOT / "data" / "boundaries" /
    "NMMC_Ward_Boundaries_2024_CLEAN.geojson"
)

wards = gpd.read_file(path).to_crs("EPSG:32643")

# Total area when individual ward areas are added.
sum_area = wards.geometry.area.sum()

# Area covered by the union of all ward polygons.
union_geometry = wards.geometry.union_all()
union_area = union_geometry.area

overlap_area = sum_area - union_area

print("\nWARD OVERLAP AUDIT")
print("-" * 45)
print(f"Number of wards: {len(wards)}")
print(f"Sum of ward areas: {sum_area / 1e6:.6f} km²")
print(f"Union area: {union_area / 1e6:.6f} km²")
print(f"Overlapping area: {overlap_area / 1e6:.9f} km²")
print(
    f"Overlap percentage: "
    f"{overlap_area / sum_area * 100:.9f}%"
)

print(
    "Result:",
    "PASS" if abs(overlap_area) < 1 else "INVESTIGATE"
)
