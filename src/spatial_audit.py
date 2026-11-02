
import json
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio

from src.config import (
    LULC_DIR,
    ECOLOGICAL_DIR,
    YEARS,
    LULC_CLASSES,
    OUTPUT_DIR,
)

REPORT_DIR = OUTPUT_DIR / "reports"
REPORT_DIR.mkdir(parents=True, exist_ok=True)

LULC_FILES = {
    1994: LULC_DIR / "NMMC_LULC_1994_PRELIMINARY.tif",
    2004: LULC_DIR / "NMMC_LULC_2004_PRELIMINARY.tif",
    2014: LULC_DIR / "NMMC_LULC_2014_PRELIMINARY.tif",
    2024: LULC_DIR / "NMMC_2024_LULC_SCIENTIFIC_FINAL.tif",
}

OPTICAL_DIR = ECOLOGICAL_DIR / "optical_indices"
THERMAL_DIR = ECOLOGICAL_DIR / "thermal_indicators"

RASTERS = []

for year in YEARS:
    RASTERS.extend([
        {
            "year": year,
            "type": "LULC",
            "path": LULC_FILES[year],
        },
        {
            "year": year,
            "type": "OPTICAL",
            "path": OPTICAL_DIR / f"NMMC_Optical_Indices_{year}.tif",
        },
        {
            "year": year,
            "type": "THERMAL",
            "path": THERMAL_DIR / f"NMMC_Thermal_Indicators_{year}.tif",
        },
    ])


def audit_raster(item):
    path = item["path"]

    if not path.exists():
        return {
            "year": item["year"],
            "type": item["type"],
            "file": str(path),
            "status": "MISSING",
        }

    records = []

    with rasterio.open(path) as src:

        for band_index in range(1, src.count + 1):

            data = src.read(band_index, masked=True)
            valid = data.compressed()

            valid_count = int(valid.size)
            total_count = int(src.width * src.height)
            nodata_count = total_count - valid_count

            record = {
                "year": item["year"],
                "type": item["type"],
                "file": path.name,
                "band": band_index,
                "band_name": src.descriptions[band_index - 1],
                "width": src.width,
                "height": src.height,
                "total_pixels": total_count,
                "valid_pixels": valid_count,
                "nodata_pixels": nodata_count,
                "valid_percentage": (
                    valid_count / total_count * 100
                    if total_count else None
                ),
                "minimum": None,
                "maximum": None,
                "mean": None,
                "std": None,
                "nan_count": 0,
                "infinite_count": 0,
                "status": "PASS",
            }

            if valid_count:

                record["minimum"] = float(np.min(valid))
                record["maximum"] = float(np.max(valid))
                record["mean"] = float(np.mean(valid))
                record["std"] = float(np.std(valid))

                record["nan_count"] = int(np.isnan(valid).sum())
                record["infinite_count"] = int(
                    np.isinf(valid).sum()
                )

                if record["nan_count"] or record["infinite_count"]:
                    record["status"] = "WARNING"

            if valid_count == 0:
                record["status"] = "FAIL"

            if item["type"] == "LULC":

                observed = set(np.unique(valid).astype(int))
                expected = set(LULC_CLASSES.keys())

                record["observed_classes"] = sorted(observed)
                record["unexpected_classes"] = sorted(
                    observed - expected
                )

                if observed - expected:
                    record["status"] = "WARNING"

            records.append(record)

    return records


def check_alignment():

    groups = {
        "LULC": [],
        "OPTICAL": [],
        "THERMAL": [],
    }

    for item in RASTERS:

        if not item["path"].exists():
            continue

        with rasterio.open(item["path"]) as src:
            groups[item["type"]].append({
                "year": item["year"],
                "crs": str(src.crs),
                "transform": tuple(src.transform),
                "width": src.width,
                "height": src.height,
                "bounds": tuple(src.bounds),
            })

    results = {}

    for raster_type, entries in groups.items():

        if not entries:
            results[raster_type] = {
                "status": "FAIL",
                "reason": "No rasters found",
            }
            continue

        reference = entries[0]

        aligned = all(
            entry["crs"] == reference["crs"]
            and entry["transform"] == reference["transform"]
            and entry["width"] == reference["width"]
            and entry["height"] == reference["height"]
            and entry["bounds"] == reference["bounds"]
            for entry in entries
        )

        results[raster_type] = {
            "status": "PASS" if aligned else "WARNING",
            "years_checked": [e["year"] for e in entries],
            "reference_crs": reference["crs"],
            "reference_dimensions": [
                reference["width"],
                reference["height"],
            ],
        }

    return results


def main():

    print("\nNMMC SPATIAL DATA QUALITY AUDIT")
    print("=" * 70)

    all_records = []

    for item in RASTERS:

        result = audit_raster(item)

        if isinstance(result, dict):
            all_records.append(result)
        else:
            all_records.extend(result)

    df = pd.DataFrame(all_records)

    csv_path = REPORT_DIR / "NMMC_spatial_data_quality_audit.csv"
    df.to_csv(csv_path, index=False)

    alignment = check_alignment()

    json_path = REPORT_DIR / "NMMC_spatial_alignment_audit.json"

    with open(json_path, "w") as f:
        json.dump(alignment, f, indent=4)

    print("\nRASTER BAND QUALITY")
    print("-" * 70)

    print(
        df[
            [
                "year",
                "type",
                "band_name",
                "valid_percentage",
                "minimum",
                "maximum",
                "status",
            ]
        ].to_string(index=False)
    )

    print("\nSPATIAL ALIGNMENT")
    print("-" * 70)

    for raster_type, result in alignment.items():
        print(f"{raster_type}: {result}")

    print("\nFILES GENERATED")
    print(csv_path)
    print(json_path)

    print("\nAUDIT COMPLETED")


if __name__ == "__main__":
    main()
