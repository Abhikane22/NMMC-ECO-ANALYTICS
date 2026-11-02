
from pathlib import Path

import numpy as np
import rasterio

from src.config import (
    PROJECT_ROOT,
    ECOLOGICAL_DIR,
    YEARS,
)


# ============================================================
# RASTER DIRECTORY CONFIGURATION
# ============================================================

OPTICAL_DIR = ECOLOGICAL_DIR / "optical_indices"
THERMAL_DIR = ECOLOGICAL_DIR / "thermal_indicators"

OPTICAL_BANDS = [
    "NDVI",
    "NDBI",
    "NDMI",
    "NDWI",
    "MNDWI",
]

THERMAL_BANDS = [
    "LST_C",
    "UTFVI",
]

EXPECTED_NODATA = -9999.0


# ============================================================
# RASTER PATH RESOLUTION
# ============================================================

def get_raster_path(year: int, raster_type: str) -> Path:
    """
    Return the path of an ecological raster for a given year.

    raster_type:
        optical or thermal
    """

    if year not in YEARS:
        raise ValueError(
            f"Unsupported year: {year}. Expected one of {YEARS}"
        )

    if raster_type == "optical":
        directory = OPTICAL_DIR
        filename = f"NMMC_Optical_Indices_{year}.tif"

    elif raster_type == "thermal":
        directory = THERMAL_DIR
        filename = f"NMMC_Thermal_Indicators_{year}.tif"

    else:
        raise ValueError(
            "raster_type must be either 'optical' or 'thermal'"
        )

    path = directory / filename

    if not path.exists():
        raise FileNotFoundError(
            f"Raster not found: {path}"
        )

    return path


# ============================================================
# LOAD RASTER DATA
# ============================================================

def load_raster(year: int, raster_type: str) -> dict:
    """
    Load a raster and return its bands and metadata.

    Masked arrays preserve NoData information.
    """

    path = get_raster_path(year, raster_type)

    with rasterio.open(path) as src:

        data = src.read(masked=True)

        metadata = {
            "year": year,
            "type": raster_type,
            "file": path.name,
            "path": str(path),

            "width": src.width,
            "height": src.height,
            "band_count": src.count,

            "crs": str(src.crs),
            "transform": src.transform,
            "bounds": src.bounds,
            "resolution": src.res,

            "nodata": src.nodata,
            "dtype": src.dtypes,

            "descriptions": list(src.descriptions),
            "valid_pixels_per_band": [
                int(np.ma.count(data[index]))
                for index in range(src.count)
            ],
        }

    band_names = (
        OPTICAL_BANDS
        if raster_type == "optical"
        else THERMAL_BANDS
    )

    if metadata["descriptions"] != band_names:
        raise ValueError(
            f"Unexpected band descriptions in {path.name}: "
            f"{metadata['descriptions']}"
        )

    bands = {
        name: data[index]
        for index, name in enumerate(band_names)
    }

    return {
        "data": data,
        "bands": bands,
        "metadata": metadata,
    }


def load_optical(year: int) -> dict:
    """Load the five optical indices for a given year."""

    return load_raster(year, "optical")


def load_thermal(year: int) -> dict:
    """Load LST and UTFVI for a given year."""

    return load_raster(year, "thermal")


# ============================================================
# BAND EXTRACTION
# ============================================================

def get_band(
    year: int,
    band_name: str,
    raster_type: str = "optical",
) -> np.ma.MaskedArray:
    """
    Load and return a specific raster band.

    Examples:
        get_band(2024, "NDVI")
        get_band(2024, "LST_C", "thermal")
    """

    raster = load_raster(year, raster_type)

    if band_name not in raster["bands"]:
        available = list(raster["bands"].keys())

        raise ValueError(
            f"Band '{band_name}' not available. "
            f"Available bands: {available}"
        )

    return raster["bands"][band_name]


# ============================================================
# RASTER INFORMATION
# ============================================================

def raster_info(year: int, raster_type: str) -> dict:
    """Return metadata for a raster without returning its pixels."""

    path = get_raster_path(year, raster_type)

    with rasterio.open(path) as src:

        return {
            "file": path.name,
            "year": year,
            "type": raster_type,
            "dimensions": (src.height, src.width),
            "bands": src.count,
            "crs": str(src.crs),
            "resolution": src.res,
            "nodata": src.nodata,
            "dtype": src.dtypes,
            "bounds": tuple(src.bounds),
            "transform": tuple(src.transform),
            "descriptions": list(src.descriptions),
        }


# ============================================================
# VALID PIXEL STATISTICS
# ============================================================

def calculate_band_statistics(
    band: np.ma.MaskedArray,
) -> dict:
    """
    Calculate statistics using valid pixels only.

    NoData values are excluded through the masked array.
    """

    values = band.compressed()

    values = values[np.isfinite(values)]

    if values.size == 0:
        raise ValueError("Band contains no valid finite pixels.")

    return {
        "valid_pixels": int(values.size),
        "min": float(values.min()),
        "max": float(values.max()),
        "mean": float(values.mean()),
        "std_dev": float(values.std()),
        "median": float(np.median(values)),
    }


def get_band_statistics(
    year: int,
    band_name: str,
    raster_type: str = "optical",
) -> dict:
    """Load a band and calculate its statistics."""

    band = get_band(year, band_name, raster_type)

    return calculate_band_statistics(band)


# ============================================================
# CROSS-YEAR RASTER ALIGNMENT
# ============================================================

def check_raster_alignment(
    raster_type: str,
    years: list = None,
) -> dict:
    """
    Check whether rasters from multiple years share
    the same dimensions, CRS, transform and resolution.
    """

    if years is None:
        years = YEARS

    metadata = [
        raster_info(year, raster_type)
        for year in years
    ]

    reference = metadata[0]

    differences = []

    for item in metadata[1:]:

        if item["dimensions"] != reference["dimensions"]:
            differences.append(
                f"{item['year']}: dimensions differ"
            )

        if item["crs"] != reference["crs"]:
            differences.append(
                f"{item['year']}: CRS differs"
            )

        if item["resolution"] != reference["resolution"]:
            differences.append(
                f"{item['year']}: resolution differs"
            )

        if not np.allclose(
            tuple(item["bounds"]),
            tuple(reference["bounds"]),
        ):
            differences.append(
                f"{item['year']}: spatial bounds differ"
            )

        if not np.allclose(
            tuple(item["transform"]),
            tuple(reference["transform"]),
        ):
            differences.append(
                f"{item['year']}: affine transform differs"
            )

    return {
        "raster_type": raster_type,
        "years_checked": years,
        "aligned": len(differences) == 0,
        "differences": differences,
    }


# ============================================================
# LOAD ALL YEARS
# ============================================================

def load_all_years(raster_type: str) -> dict:
    """
    Load the specified raster type for all study years.
    """

    return {
        year: load_raster(year, raster_type)
        for year in YEARS
    }
