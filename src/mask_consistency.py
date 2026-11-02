
import numpy as np
import rasterio

from src.config import LULC_DIR, ECOLOGICAL_DIR, YEARS

LULC_FILES = {
    1994: LULC_DIR / "NMMC_LULC_1994_PRELIMINARY.tif",
    2004: LULC_DIR / "NMMC_LULC_2004_PRELIMINARY.tif",
    2014: LULC_DIR / "NMMC_LULC_2014_PRELIMINARY.tif",
    2024: LULC_DIR / "NMMC_2024_LULC_SCIENTIFIC_FINAL.tif",
}

OPTICAL_DIR = ECOLOGICAL_DIR / "optical_indices"


def get_valid_mask(path, band=1):
    with rasterio.open(path) as src:
        return src.read_masks(band) > 0


def compare_masks(name, masks):

    print(f"\n{name}")
    print("-" * 65)

    reference_year = list(masks.keys())[0]
    reference = masks[reference_year]

    all_identical = True

    for year, mask in masks.items():

        if mask.shape != reference.shape:
            print(f"{year}: DIMENSION MISMATCH")
            all_identical = False
            continue

        identical = np.array_equal(reference, mask)

        different = int(np.count_nonzero(reference != mask))

        print(
            f"{year}: "
            f"{'IDENTICAL' if identical else 'DIFFERENT'} | "
            f"Different pixels: {different}"
        )

        if not identical:
            all_identical = False

    print(
        "RESULT:",
        "PASS - all masks identical"
        if all_identical
        else "WARNING - masks differ"
    )

    return all_identical


def main():

    print("\nNMMC VALID PIXEL MASK CONSISTENCY")
    print("=" * 70)

    lulc_masks = {}
    optical_masks = {}

    for year in YEARS:

        lulc_masks[year] = get_valid_mask(
            LULC_FILES[year]
        )

        optical_path = (
            OPTICAL_DIR /
            f"NMMC_Optical_Indices_{year}.tif"
        )

        optical_masks[year] = get_valid_mask(
            optical_path
        )

    compare_masks("LULC YEAR-TO-YEAR", lulc_masks)

    compare_masks("OPTICAL YEAR-TO-YEAR", optical_masks)

    compare_masks(
        "LULC VS OPTICAL - SAME YEAR",
        {
            year: lulc_masks[year] ^ optical_masks[year]
            for year in YEARS
        }
    )

    print("\nOPTICAL BAND MASK CONSISTENCY")
    print("-" * 65)

    for year in YEARS:

        path = (
            OPTICAL_DIR /
            f"NMMC_Optical_Indices_{year}.tif"
        )

        masks = [
            get_valid_mask(path, band)
            for band in range(1, 6)
        ]

        identical = all(
            np.array_equal(masks[0], mask)
            for mask in masks[1:]
        )

        print(
            f"{year}: "
            f"{'PASS' if identical else 'WARNING'}"
        )

    print("\nMASK AUDIT COMPLETED")


if __name__ == "__main__":
    main()
