
import rasterio
import numpy as np

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

ECO_DIR = ROOT / "data" / "ecological"

YEARS = [1994, 2004, 2014, 2024]

RASTER_GROUPS = {
    "OPTICAL": (
        ECO_DIR / "optical_indices",
        "NMMC_Optical_Indices_{}.tif",
        ["NDVI", "NDBI", "NDMI", "NDWI", "MNDWI"],
        5,
        30,
    ),
    "THERMAL": (
        ECO_DIR / "thermal_indicators",
        "NMMC_Thermal_Indicators_{}.tif",
        ["LST_C", "UTFVI"],
        2,
        120,
    ),
}


def audit_raster(group, year, path, expected_bands, expected_resolution):
    print(f"\n{group} — {year}")
    print("-" * 55)

    if not path.exists():
        raise FileNotFoundError(path)

    with rasterio.open(path) as src:

        print("File:", path.name)
        print("CRS:", src.crs)
        print("Dimensions:", src.width, "x", src.height)
        print("Band count:", src.count)
        print("Resolution:", src.res)
        print("Data type:", src.dtypes)
        print("NoData:", src.nodata)
        print("Descriptions:", src.descriptions)

        assert src.count == expected_bands
        assert src.crs.to_epsg() == 32643
        assert np.isclose(abs(src.res[0]), expected_resolution)
        assert np.isclose(abs(src.res[1]), expected_resolution)
        assert src.nodata == -9999

        for band_index, expected_name in enumerate(
            RASTER_GROUPS[group][2], start=1
        ):
            band = src.read(band_index, masked=True)
            values = band.compressed()

            if values.size == 0:
                raise ValueError(
                    f"No valid observations: {path.name}, "
                    f"band {expected_name}"
                )

            print(
                f"{expected_name}: "
                f"valid={values.size}, "
                f"min={values.min():.6f}, "
                f"max={values.max():.6f}, "
                f"mean={values.mean():.6f}"
            )

            if not np.isfinite(values).all():
                raise ValueError(
                    f"Non-finite values in {expected_name}"
                )

            if group == "OPTICAL":
                if values.min() < -1.01 or values.max() > 1.01:
                    raise ValueError(
                        f"Unexpected index range: {expected_name}"
                    )

            if expected_name == "UTFVI":
                if values.min() < -1 or values.max() > 1:
                    raise ValueError("Unexpected UTFVI range")

        print("Metadata validation: PASS")


def main():
    print("\nNMMC ECOLOGICAL RASTER AUDIT")
    print("=" * 60)

    for group, config in RASTER_GROUPS.items():
        directory, filename, _, _, _ = config

        for year in YEARS:
            path = directory / filename.format(year)

            audit_raster(
                group,
                year,
                path,
                config[3],
                config[4],
            )

    print("\n" + "=" * 60)
    print("ALL ECOLOGICAL RASTER AUDITS PASSED")


if __name__ == "__main__":
    main()
