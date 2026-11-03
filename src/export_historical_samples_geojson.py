
import json
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

SAMPLE_DIR = (
    ROOT / "data" / "validation"
    / "historical_reference_samples"
)

files = [
    SAMPLE_DIR / "NMMC_Historical_Reference_Samples_ALL.csv",
    SAMPLE_DIR / "NMMC_Historical_Reference_Samples_1994.csv",
    SAMPLE_DIR / "NMMC_Historical_Reference_Samples_2004.csv",
    SAMPLE_DIR / "NMMC_Historical_Reference_Samples_2014.csv",
]

for csv_path in files:

    df = pd.read_csv(csv_path)

    features = []

    for _, row in df.iterrows():

        properties = {}

        for column, value in row.items():
            if pd.isna(value):
                properties[column] = None
            elif hasattr(value, "item"):
                properties[column] = value.item()
            else:
                properties[column] = value

        features.append({
            "type": "Feature",
            "geometry": {
                "type": "Point",
                "coordinates": [
                    float(row["longitude"]),
                    float(row["latitude"])
                ]
            },
            "properties": properties
        })

    geojson = {
        "type": "FeatureCollection",
        "features": features
    }

    output_path = csv_path.with_suffix(".geojson")

    with open(output_path, "w") as f:
        json.dump(geojson, f, indent=2)

    print(f"Exported: {output_path.name} ({len(features)} points)")
