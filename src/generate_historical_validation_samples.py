
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio

from pyproj import Transformer


# ============================================================
# CONFIGURATION
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

LULC_DIR = ROOT / "data" / "lulc"

OUTPUT_DIR = (
    ROOT / "data" / "validation"
    / "historical_reference_samples"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

YEARS = [1994, 2004, 2014]

RASTER_FILES = {
    1994: LULC_DIR / "NMMC_LULC_1994_PRELIMINARY.tif",
    2004: LULC_DIR / "NMMC_LULC_2004_PRELIMINARY.tif",
    2014: LULC_DIR / "NMMC_LULC_2014_PRELIMINARY.tif",
}

CLASS_NAMES = {
    0: "Water",
    1: "Vegetation",
    2: "Built-up",
    3: "Barren",
    4: "Coastal Wetlands",
}

SAMPLES_PER_CLASS = 50
RANDOM_SEED = 42


# ============================================================
# SAMPLE GENERATION
# ============================================================

def generate_samples(year, raster_path, rng):

    print(f"\nProcessing {year}: {raster_path.name}")

    if not raster_path.exists():
        raise FileNotFoundError(raster_path)

    with rasterio.open(raster_path) as src:

        raster = src.read(1, masked=True)

        if src.crs is None:
            raise ValueError(f"{year}: Raster has no CRS.")

        if src.crs.to_epsg() != 32643:
            raise ValueError(
                f"{year}: Expected EPSG:32643, found {src.crs}"
            )

        transform = src.transform

        height, width = raster.shape

        valid_mask = ~np.ma.getmaskarray(raster)

        values = np.asarray(raster.data)

        transformer = Transformer.from_crs(
            src.crs,
            "EPSG:4326",
            always_xy=True
        )

        records = []

        for class_code, class_name in CLASS_NAMES.items():

            rows, cols = np.where(
                valid_mask & (values == class_code)
            )

            available = len(rows)

            if available < SAMPLES_PER_CLASS:
                raise ValueError(
                    f"{year}: Class {class_name} has only "
                    f"{available} pixels."
                )

            selected = rng.choice(
                available,
                size=SAMPLES_PER_CLASS,
                replace=False
            )

            selected_rows = rows[selected]
            selected_cols = cols[selected]

            print(
                f"  {class_name}: "
                f"{available:,} available pixels, "
                f"{SAMPLES_PER_CLASS} selected"
            )

            for row, col in zip(
                selected_rows,
                selected_cols
            ):

                x, y = rasterio.transform.xy(
                    transform,
                    int(row),
                    int(col),
                    offset="center"
                )

                longitude, latitude = transformer.transform(
                    x, y
                )

                records.append({
                    "point_id": (
                        f"NMMC_{year}_{class_code}_"
                        f"{len(records) + 1:03d}"
                    ),
                    "year": year,
                    "mapped_class_code": class_code,
                    "mapped_class_name": class_name,
                    "row": int(row),
                    "column": int(col),
                    "x_utm": x,
                    "y_utm": y,
                    "longitude": longitude,
                    "latitude": latitude,
                    "reference_class_code": None,
                    "reference_class_name": None,
                    "reference_source": None,
                    "interpretation_confidence": None,
                    "notes": None,
                })

    return pd.DataFrame(records)


# ============================================================
# MAIN
# ============================================================

def main():

    rng = np.random.default_rng(RANDOM_SEED)

    all_samples = []

    for year in YEARS:

        samples = generate_samples(
            year,
            RASTER_FILES[year],
            rng
        )

        year_file = (
            OUTPUT_DIR
            / f"NMMC_Historical_Reference_Samples_{year}.csv"
        )

        samples.to_csv(year_file, index=False)

        all_samples.append(samples)

        print(f"Saved: {year_file}")

    combined = pd.concat(
        all_samples,
        ignore_index=True
    )

    combined_file = (
        OUTPUT_DIR
        / "NMMC_Historical_Reference_Samples_ALL.csv"
    )

    combined.to_csv(
        combined_file,
        index=False
    )

    summary = (
        combined.groupby(
            ["year", "mapped_class_name"]
        )
        .size()
        .reset_index(name="sample_count")
    )

    summary_file = (
        OUTPUT_DIR
        / "NMMC_Historical_Reference_Sample_Summary.csv"
    )

    summary.to_csv(
        summary_file,
        index=False
    )

    print("\n" + "=" * 65)
    print("HISTORICAL REFERENCE SAMPLE GENERATION COMPLETE")
    print("=" * 65)

    print("Total samples:", len(combined))
    print("Years:", YEARS)
    print("Samples per class:", SAMPLES_PER_CLASS)

    print("\nSample summary:")
    print(summary.to_string(index=False))

    print("\nCombined file:")
    print(combined_file)


if __name__ == "__main__":
    main()
