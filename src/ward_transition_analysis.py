
import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
import shapely

from rasterio.features import geometry_window
from shapely.geometry import mapping

from config import (
    PROJECT_ROOT,
    LULC_DIR,
    REPORTS_DIR,
    YEARS,
    LULC_CLASSES,
)


# ---------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------

BOUNDARY_FILE = (
    PROJECT_ROOT
    / "data"
    / "boundaries"
    / "NMMC_Ward_Boundaries_2024_CLEAN.geojson"
)

WARD_ID_FIELD = "ward_lgd_c"

PERIODS = [
    (1994, 2004),
    (2004, 2014),
    (2014, 2024),
    (1994, 2024),
]

CLASS_CODES = sorted(LULC_CLASSES.keys())
NUM_CLASSES = len(CLASS_CODES)

NODATA_VALUE = 255


def get_lulc_path(year):
    """Always use preliminary classifications for temporal analysis."""

    path = (
        LULC_DIR
        / f"NMMC_LULC_{year}_PRELIMINARY.tif"
    )

    if not path.exists():
        raise FileNotFoundError(
            f"Preliminary LULC raster missing for {year}: {path}"
        )

    return path


# ---------------------------------------------------------
# RASTER GRID VALIDATION
# ---------------------------------------------------------

def validate_raster_pair(start_raster, end_raster):
    """Ensure both rasters share an identical spatial grid."""

    if start_raster.crs is None or end_raster.crs is None:
        raise ValueError("One or both rasters have missing CRS.")

    if start_raster.crs != end_raster.crs:
        raise ValueError("Raster CRS mismatch.")

    if start_raster.crs.to_epsg() != 32643:
        raise ValueError(
            f"Expected EPSG:32643, got {start_raster.crs}"
        )

    if (
        start_raster.width != end_raster.width
        or start_raster.height != end_raster.height
    ):
        raise ValueError("Raster dimensions do not match.")

    if start_raster.transform != end_raster.transform:
        raise ValueError("Raster transforms do not match.")

    if start_raster.count != 1 or end_raster.count != 1:
        raise ValueError("Expected single-band LULC rasters.")


# ---------------------------------------------------------
# WARD-LEVEL TRANSITION CALCULATION
# ---------------------------------------------------------

def calculate_ward_transitions(
    geometry,
    start_raster,
    end_raster,
):
    """
    Calculate transition areas using exact pixel-ward
    intersection areas.

    Transition code:
        old_class * NUM_CLASSES + new_class

    Pixel counts represent pixels with positive intersection
    area inside the individual ward.

    Areas are calculated in projected EPSG:32643 units (m²)
    and converted to km².
    """

    transform = start_raster.transform

    if not np.isclose(transform.b, 0) or not np.isclose(
        transform.d, 0
    ):
        raise ValueError("Rotated raster grids are unsupported.")

    # Find the raster window covering the ward.
    window = geometry_window(
        start_raster,
        [mapping(geometry)],
    )

    start_data = start_raster.read(1, window=window)
    end_data = end_raster.read(1, window=window)

    if start_data.shape != end_data.shape:
        raise ValueError("Window arrays have different shapes.")

    window_transform = start_raster.window_transform(window)

    rows, cols = np.indices(start_data.shape)

    # Construct projected pixel footprints.
    x1 = window_transform.c + cols * window_transform.a
    x2 = x1 + window_transform.a

    y1 = window_transform.f + rows * window_transform.e
    y2 = y1 + window_transform.e

    pixel_polygons = shapely.box(
        np.minimum(x1, x2),
        np.minimum(y1, y2),
        np.maximum(x1, x2),
        np.maximum(y1, y2),
    )

    # Calculate exact pixel-ward intersection areas.
    intersection_areas = shapely.area(
        shapely.intersection(pixel_polygons, geometry)
    )

    # Both years must contain valid LULC classes.
    valid_mask = (
        np.isin(start_data, CLASS_CODES)
        & np.isin(end_data, CLASS_CODES)
        & (intersection_areas > 0)
    )

    old_classes = start_data[valid_mask].astype(np.int32)
    new_classes = end_data[valid_mask].astype(np.int32)

    valid_areas = intersection_areas[valid_mask]

    # Encode each transition as a unique integer.
    transition_codes = (
        old_classes * NUM_CLASSES + new_classes
    )

    # Calculate intersection area for all possible transitions.
    transition_areas = np.bincount(
        transition_codes,
        weights=valid_areas,
        minlength=NUM_CLASSES * NUM_CLASSES,
    )

    # Count pixels for each transition.
    transition_pixels = np.bincount(
        transition_codes,
        minlength=NUM_CLASSES * NUM_CLASSES,
    )

    transition_areas = transition_areas.reshape(
        NUM_CLASSES,
        NUM_CLASSES,
    )

    transition_pixels = transition_pixels.reshape(
        NUM_CLASSES,
        NUM_CLASSES,
    )

    ward_area_km2 = geometry.area / 1_000_000

    valid_area_km2 = valid_areas.sum() / 1_000_000

    return {
        "ward_area_km2": ward_area_km2,
        "valid_area_km2": valid_area_km2,
        "valid_coverage_pct": (
            valid_area_km2 / ward_area_km2 * 100
            if ward_area_km2 > 0 else 0
        ),
        "transition_areas_km2": (
            transition_areas / 1_000_000
        ),
        "transition_pixels": transition_pixels,
        "valid_pixel_count": int(valid_mask.sum()),
    }


# ---------------------------------------------------------
# TRANSITION DETAIL REPORT
# ---------------------------------------------------------

def build_transition_records(
    ward_id,
    start_year,
    end_year,
    stats,
):
    """Create 25 transition records for one ward and period."""

    records = []

    matrix = stats["transition_areas_km2"]
    pixel_matrix = stats["transition_pixels"]

    valid_area = stats["valid_area_km2"]
    ward_area = stats["ward_area_km2"]

    for i, old_class in enumerate(CLASS_CODES):

        for j, new_class in enumerate(CLASS_CODES):

            area_km2 = matrix[i, j]

            records.append({
                "ward_id": ward_id,
                "start_year": start_year,
                "end_year": end_year,
                "from_class_code": old_class,
                "from_class": LULC_CLASSES[old_class],
                "to_class_code": new_class,
                "to_class": LULC_CLASSES[new_class],
                "pixel_count": int(pixel_matrix[i, j]),
                "transition_area_km2": area_km2,
                "transition_pct_of_ward": (
                    area_km2 / ward_area * 100
                    if ward_area > 0 else 0
                ),
                "transition_pct_of_valid_area": (
                    area_km2 / valid_area * 100
                    if valid_area > 0 else 0
                ),
                "ward_area_km2": ward_area,
                "common_valid_area_km2": valid_area,
                "valid_coverage_pct": stats["valid_coverage_pct"],
            })

    return records


# ---------------------------------------------------------
# CLASS GAINS AND LOSSES
# ---------------------------------------------------------

def build_class_change_records(
    ward_id,
    start_year,
    end_year,
    stats,
):
    """Calculate gains, losses and net change per LULC class."""

    matrix = stats["transition_areas_km2"]

    records = []

    for i, class_code in enumerate(CLASS_CODES):

        # Exclude diagonal (unchanged) transitions.
        gain_km2 = (
            matrix[:, i].sum() - matrix[i, i]
        )

        loss_km2 = (
            matrix[i, :].sum() - matrix[i, i]
        )

        net_change_km2 = gain_km2 - loss_km2

        records.append({
            "ward_id": ward_id,
            "start_year": start_year,
            "end_year": end_year,
            "class_code": class_code,
            "class_name": LULC_CLASSES[class_code],
            "gain_km2": gain_km2,
            "loss_km2": loss_km2,
            "net_change_km2": net_change_km2,
            "ward_area_km2": stats["ward_area_km2"],
            "common_valid_area_km2": stats["valid_area_km2"],
        })

    return records


# ---------------------------------------------------------
# MAIN PIPELINE
# ---------------------------------------------------------

def main():

    print("\nNMMC WARD-LEVEL TRANSITION ANALYSIS")
    print("=" * 65)

    if not BOUNDARY_FILE.exists():
        raise FileNotFoundError(BOUNDARY_FILE)

    wards = gpd.read_file(BOUNDARY_FILE)

    # Validate ward identifiers and geometries.
    if WARD_ID_FIELD not in wards.columns:
        raise ValueError(
            f"Missing ward ID field: {WARD_ID_FIELD}"
        )

    if wards[WARD_ID_FIELD].isna().any():
        raise ValueError("Missing ward IDs detected.")

    if wards[WARD_ID_FIELD].duplicated().any():
        raise ValueError("Duplicate ward IDs detected.")

    if len(wards) != 111:
        raise ValueError(
            f"Expected 111 wards, found {len(wards)}."
        )

    if wards.crs is None:
        raise ValueError("Ward boundary CRS is missing.")

    all_transition_records = []
    all_change_records = []

    for start_year, end_year in PERIODS:

        print(
            f"\nProcessing {start_year}–{end_year}..."
        )

        start_path = get_lulc_path(start_year)
        end_path = get_lulc_path(end_year)

        print("Start raster:", start_path.name)
        print("End raster:", end_path.name)

        with rasterio.open(start_path) as start_raster, \
             rasterio.open(end_path) as end_raster:

            validate_raster_pair(
                start_raster,
                end_raster,
            )

            year_wards = wards.to_crs(start_raster.crs)

            period_transition_records = []
            period_change_records = []

            for _, ward in year_wards.iterrows():

                ward_id = ward[WARD_ID_FIELD]
                geometry = ward.geometry

                if geometry is None or geometry.is_empty:
                    raise ValueError(
                        f"Empty geometry for ward {ward_id}"
                    )

                if not geometry.is_valid:
                    raise ValueError(
                        f"Invalid geometry for ward {ward_id}"
                    )

                stats = calculate_ward_transitions(
                    geometry,
                    start_raster,
                    end_raster,
                )

                # Validate transition area accounting.
                transition_area_sum = (
                    stats["transition_areas_km2"].sum()
                )

                if not np.isclose(
                    transition_area_sum,
                    stats["valid_area_km2"],
                    atol=1e-8,
                ):
                    raise ValueError(
                        f"Transition area mismatch for ward "
                        f"{ward_id}, {start_year}-{end_year}."
                    )

                # Validate pixel count accounting.
                pixel_count_sum = (
                    stats["transition_pixels"].sum()
                )

                if pixel_count_sum != stats["valid_pixel_count"]:
                    raise ValueError(
                        f"Transition pixel count mismatch for "
                        f"ward {ward_id}, {start_year}-{end_year}."
                    )

                if stats["valid_coverage_pct"] > 100.01:
                    raise ValueError(
                        f"Valid coverage exceeds ward area "
                        f"for ward {ward_id}."
                    )

                period_transition_records.extend(
                    build_transition_records(
                        ward_id,
                        start_year,
                        end_year,
                        stats,
                    )
                )

                period_change_records.extend(
                    build_class_change_records(
                        ward_id,
                        start_year,
                        end_year,
                        stats,
                    )
                )

            all_transition_records.extend(
                period_transition_records
            )

            all_change_records.extend(
                period_change_records
            )

        print(
            f"Completed {start_year}–{end_year}: "
            f"{len(year_wards)} wards, "
            f"{len(period_transition_records)} "
            f"transition records."
        )

    # -----------------------------------------------------
    # SAVE REPORTS
    # -----------------------------------------------------

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    transition_df = pd.DataFrame(
        all_transition_records
    )

    changes_df = pd.DataFrame(
        all_change_records
    )

    transition_path = (
            REPORTS_DIR
            / "NMMC_Ward_Level_Transition_Statistics_PRELIMINARY.csv"
        )

    changes_path = (
            REPORTS_DIR
            / "NMMC_Ward_Level_Class_Gains_Losses_PRELIMINARY.csv"
        )

    transition_df.to_csv(
        transition_path,
        index=False,
    )

    changes_df.to_csv(
        changes_path,
        index=False,
    )

    # -----------------------------------------------------
    # FINAL VALIDATION
    # -----------------------------------------------------

    expected_transition_rows = (
        len(wards) * len(PERIODS) * NUM_CLASSES ** 2
    )

    expected_change_rows = (
        len(wards) * len(PERIODS) * NUM_CLASSES
    )

    if len(transition_df) != expected_transition_rows:
        raise ValueError(
            f"Expected {expected_transition_rows} transition "
            f"records, found {len(transition_df)}."
        )

    if len(changes_df) != expected_change_rows:
        raise ValueError(
            f"Expected {expected_change_rows} class-change "
            f"records, found {len(changes_df)}."
        )

    if transition_df.duplicated(
        [
            "ward_id",
            "start_year",
            "end_year",
            "from_class_code",
            "to_class_code",
        ]
    ).any():
        raise ValueError("Duplicate transition records found.")

    if changes_df.duplicated(
        [
            "ward_id",
            "start_year",
            "end_year",
            "class_code",
        ]
    ).any():
        raise ValueError("Duplicate class-change records found.")

    print("\n" + "=" * 65)
    print("WARD-LEVEL TRANSITION ANALYSIS COMPLETED")
    print("=" * 65)

    print("Unique wards:", wards[WARD_ID_FIELD].nunique())
    print("Comparison periods:", len(PERIODS))
    print("Possible transitions per ward-period:", NUM_CLASSES ** 2)
    print("Transition records:", len(transition_df))
    print("Class-change records:", len(changes_df))

    print("\nValidation:")
    print("Transition area accounting: PASS")
    print("Transition pixel accounting: PASS")
    print("Coverage validation: PASS")
    print("Duplicate record validation: PASS")
    print("Expected row count validation: PASS")

    print("\nReports:")
    print(transition_path)
    print(changes_path)

    print("\nTransition area totals by period (km²):")
    print(
        transition_df.groupby(
            ["start_year", "end_year"]
        )["transition_area_km2"]
        .sum()
        .round(4)
        .to_string()
    )

    return transition_df, changes_df


if __name__ == "__main__":
    main()
