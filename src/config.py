
from pathlib import Path

# Project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Data directories
DATA_DIR = PROJECT_ROOT / "data"
LULC_DIR = DATA_DIR / "lulc"
ECOLOGICAL_DIR = DATA_DIR / "ecological"
STATISTICS_DIR = DATA_DIR / "statistics"

# Output directories
OUTPUT_DIR = PROJECT_ROOT / "outputs"
MAPS_DIR = OUTPUT_DIR / "maps"
CHARTS_DIR = OUTPUT_DIR / "charts"
REPORTS_DIR = OUTPUT_DIR / "reports"

# Study configuration
STUDY_AREA = "Navi Mumbai Municipal Corporation"

YEARS = [1994, 2004, 2014, 2024]

LULC_CLASSES = {
    0: "Water",
    1: "Vegetation",
    2: "Built-up",
    3: "Barren",
    4: "Coastal Wetlands"
}

CLASS_COLORS = {
    0: "#419BDF",
    1: "#397D49",
    2: "#E31A1C",
    3: "#C2B280",
    4: "#00A884"
}
