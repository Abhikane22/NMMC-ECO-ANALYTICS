# NMMC-ECO-ANALYTICS

**Multitemporal Land Use/Land Cover (LULC) and Ecological Impact Analysis for Navi Mumbai Municipal Corporation (NMMC)**

A geospatial research and decision-support project examining land-cover patterns, temporal transitions, ward-level ecological indicators, spatial hotspots and spatially dependent relationships across NMMC.

> **Research status:** The spatial regression validation pipeline (Phase 5 core) has completed its numerical validation. Historical temporal LULC products are preliminary. The 2024 scientific-final classification is maintained as a separate product and must not be confused with the preliminary temporal-analysis track.

## Study overview

- **Study area:** Navi Mumbai Municipal Corporation (NMMC), Maharashtra, India.
- **Spatial units:** 111 municipal wards.
- **Analysis years:** 1994, 2004, 2014 and 2024.
- **Primary raster sources:** Landsat Collection 2 Level-2 composites and derived indices.
- **Analysis CRS:** EPSG:32643 for ward-level area calculations.
- **LULC classes:** Water, Vegetation, Built-up, Barren and Coastal wetlands.

### Project objectives

1. Map and compare LULC patterns across four observation years.
2. Quantify class transitions and ward-level gains and losses.
3. Derive ward-level ecological indicators from optical and thermal raster products.
4. Identify spatial clustering and local hotspots in long-term change indicators.
5. Examine ward-level ecological associations using correlation and spatial regression.
6. Provide a foundation for an urban ecological decision-support dashboard and reproducible research outputs.

## Important data and interpretation policy

### Two separate LULC tracks

The project maintains two analytical tracks:

**Track A — preliminary temporal analysis**
- `data/lulc/NMMC_LULC_1994_PRELIMINARY.tif`
- `data/lulc/NMMC_LULC_2004_PRELIMINARY.tif`
- `data/lulc/NMMC_LULC_2014_PRELIMINARY.tif`
- `data/lulc/NMMC_LULC_2024_PRELIMINARY.tif`

Use this consistent preliminary series for temporal transition and 1994–2024 change analyses.

**Track B — 2024 scientific-final classification**
- `data/lulc/NMMC_2024_LULC_SCIENTIFIC_FINAL.tif`

This is a separate 2024 classification/accuracy-assessment product. Do not silently substitute it into the preliminary temporal series or describe the historical preliminary maps as independently validated final maps.

### Ecological rasters

- Optical indices: `data/ecological/optical_indices/`
- Thermal indicators: `data/ecological/thermal_indicators/`

The optical products are 30 m and the thermal products are 120 m. Thermal and optical indicators therefore have different spatial resolutions. The Landsat seasonal composites use January–May windows and span different Landsat sensors. Interpret inter-year differences with these limitations in mind.

### Boundaries

- Original ward boundaries: `data/boundaries/NMMC_Ward_Boundaries_2024.geojson`
- Cleaned ward boundaries: `data/boundaries/NMMC_Ward_Boundaries_2024_CLEAN.geojson`

The cleaned file contains 111 valid polygon features and preserves the ward attributes. The source ward geometries have a very small measured overlap; this is documented in the project validation work. Ward-level intersection summaries may consequently double-count a negligible shared area. Do not silently alter source geometries to remove it.

### Scientific caveats

- Historical LULC maps in the temporal series are preliminary.
- The reported 2024 classifier holdout accuracy is not an independent validation of the 1994, 2004 or 2014 maps.
- The 2024 scientific-final classification is a separate track.
- The five LULC class proportions form a composition: their shares sum to one, and their changes sum to zero. Do not interpret all five class changes as independent predictors in a model with an intercept.
- ILR balances represent relative contrasts among groups of classes, not isolated single-class effects.
- ILR zero replacement is a sensitivity procedure; it does not establish whether zeros are structural or sampling zeros.
- SAR impact intervals are simulation-based and conditional on the fitted model, covariance estimate, spatial weights and zero-replacement procedure. They do not propagate all classification, measurement, zero-type or model-selection uncertainty.
- Spatial associations are observational and do not establish causation.
- NDBI and NDMI in the current optical products are exact negatives under the implemented band-order convention. Do not treat them as independent corroborating indicators without reviewing the index definitions.

## Repository structure

```text
NMMC-ECO-ANALYTICS/
├── dashboard/
│   └── app.py
├── data/
│   ├── boundaries/
│   ├── ecological/
│   │   ├── optical_indices/
│   │   └── thermal_indicators/
│   ├── lulc/
│   └── statistics/
├── gee_scripts/
│   └── 03_landsat_composites_ecological_analysis.js
├── outputs/
│   ├── charts/
│   ├── maps/
│   └── reports/
├── src/
│   ├── compositional_sar_sensitivity.py
│   ├── ecological_impact_analysis.py
│   ├── ilr_compositional_sar.py
│   ├── lulc_composition_audit.py
│   ├── sar_inference_validation.py
│   ├── spatial_ecological_regression.py
│   ├── spatial_ecological_sensitivity.py
│   ├── spatial_hotspot_analysis.py
│   ├── spatial_sensitivity_analysis.py
│   ├── ward_lulc_2024_preliminary.py
│   └── ward_transition_analysis.py
├── .gitignore
├── README.md
└── requirements.txt
```

Additional source modules and generated reports may be present. The tree above highlights the principal project components rather than claiming to be an exhaustive listing.

## Environment setup

The dependency versions below were captured from the working project environment on 2 November 2026. The local environment used Python 3.14.3 on macOS Apple Silicon.

Create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Check the Python version:

```bash
python --version
```

The requirements file pins the versions observed in the development environment. Reinstallation on another operating system or Python version may require resolving platform-specific binary dependencies for geospatial libraries such as Rasterio, GDAL/PROJ and GeoPandas.

## Data preparation

The project uses raster and boundary inputs stored under `data/`. Preserve the distinction between original and cleaned boundaries and between preliminary and scientific-final LULC products.

The Landsat processing workflow is documented in:

```text
gee_scripts/03_landsat_composites_ecological_analysis.js
```

The script is intended for the Google Earth Engine Code Editor and uses the NMMC ward asset configured in the project workflow. Confirm that the Earth Engine asset is accessible to your account before attempting to reproduce exports.

Expected raster families include:
- Four annual LULC rasters for the preliminary temporal series.
- One separate 2024 scientific-final LULC raster.
- Four annual optical-index rasters.
- Four annual thermal-indicator rasters.

Do not overwrite source rasters during audits or post-processing. Write derived products to designated output locations and retain their provenance.

## Analysis workflow

The analysis is organized into phases. Run scripts from the repository root so relative paths resolve as expected. The following are the principal analysis modules currently included in `src/`.

### Ward-level statistics and validation

Relevant modules include the ward LULC, ecological-statistics and validation scripts in `src/`. These calculate ward-level class areas and ecological summaries using raster–ward intersections and produce CSV reports under `outputs/reports/`.

### Temporal change detection

The temporal workflow compares the preliminary LULC series for:
- 1994–2004
- 2004–2014
- 2014–2024
- 1994–2024

It produces annual area statistics, transition details, class gains/losses and ward-level transition outputs. Preserve the preliminary qualifier in filenames and interpretation.

### Spatial hotspot analysis

Principal modules:

```bash
python src/spatial_hotspot_analysis.py
python src/spatial_sensitivity_analysis.py
```

The hotspot analysis uses Queen contiguity and permutation-based global/local spatial statistics. The sensitivity analysis compares Queen and Rook contiguity. The project used 999 permutations for the reported hotspot analyses.

### Ecological impact analysis

```bash
python src/ecological_impact_analysis.py
```

This module examines ward-level changes and correlations. Correlations are exploratory and should not be described as causal effects. Spatial dependence and compositional closure must be considered when interpreting p-values and associations.

### Spatial regression and compositional analysis

The principal Phase 5 modules are:

```bash
python src/lulc_composition_audit.py
python src/spatial_ecological_sensitivity.py
python src/compositional_sar_sensitivity.py
python src/ilr_compositional_sar.py
python src/sar_inference_validation.py
```

These scripts implement compositional closure checks, SAR model comparisons, reference-class sensitivity, ILR balance models, zero-replacement sensitivity, covariance checks and simulation-based impact intervals.

**Execution note:** Run each module only after its required upstream CSVs and rasters have been generated. Check the input paths in each script before running it in a fresh checkout. The exact end-to-end execution order and any optional/legacy modules should be confirmed against the current source code before treating the repository as a one-command pipeline.

## Phase 5 validation snapshot

The final validation run reported:

| Validation item | Result |
|---|---:|
| Successful model fits | 16 / 16 |
| Usable covariance matrices | 16 / 16 |
| Impact interval records | 90 |
| Spatial multiplier checks | All passed (8 / 8 specifications) |
| Simulation draws retained | 10,000 per impact record |

The detailed report is:

```text
outputs/reports/NMMC_Phase5_Spatial_Regression_Validation_Report.md
```

The reported confidence intervals are conditional simulation intervals, not a full uncertainty analysis. Direct, indirect and total impacts, as well as Queen/Rook and zero-replacement variants, are dependent results and must not be counted as independent discoveries.

## Dashboard

The Streamlit dashboard entry point is:

```text
dashboard/app.py
```

Start it from the repository root:

```bash
streamlit run dashboard/app.py
```

The dashboard source references several report filenames in `outputs/reports/`. Before describing the dashboard as fully operational in a fresh clone, verify that every configured input file exists and matches the current report-generation pipeline. Some configured filenames may reflect an earlier output schema. The dashboard should be treated as an integration workstream until those dependencies are reconciled and the application is tested end to end.

## Outputs

Generated analytical products are stored in:

- `outputs/reports/` — CSV tables, diagnostics and validation reports.
- `outputs/maps/` — spatial map products.
- `outputs/charts/` — charts and figures.

CSV reports should be regenerated from the source scripts where possible. Keep the input data, script version, package versions, random seeds and execution date with any formal research release.

## Reproducibility checklist

Before a formal rerun or research-paper submission:

- [ ] Confirm all required source rasters and boundary files are present.
- [ ] Confirm the preliminary and scientific-final LULC tracks are not mixed.
- [ ] Confirm the ward ID field and CRS used by each workflow.
- [ ] Re-run data and composition closure audits.
- [ ] Record package versions and random seeds.
- [ ] Retain model diagnostics, covariance validation and impact simulation outputs.
- [ ] Inspect warnings rather than suppressing them without explanation.
- [ ] Verify dashboard input paths and run the dashboard end to end.
- [ ] Record all methodological changes from the reported analysis.
- [ ] State observational limitations and avoid causal claims.

## Current project status

| Phase | Status |
|---|---|
| Phase 1 — Ward-level statistics and data quality | Completed |
| Phase 2 — Temporal change detection | Completed |
| Phase 3 — Spatial hotspot analysis | Completed |
| Phase 4 — Ecological impact correlations | Completed; exploratory |
| Phase 5 — Spatial regression and inference | Core validation completed; interpretation and limitations documented |
| Phase 6 — Decision-support dashboard | In progress / integration pending |
| Phase 7 — Research paper and reproducibility | Pending |

## Acknowledgements and data provenance

The project uses Landsat Collection 2 Level-2 imagery and derived geospatial products processed through Google Earth Engine, together with NMMC ward boundary data used in the study workflow. Include the appropriate dataset citations, asset ownership/permission details and data-use terms in any public research release.

## Citation

A formal citation for this project should be added once the authorship, institutional affiliation, version and public release record have been finalized.

---

**Disclaimer:** This repository is a research and analytical prototype. Its outputs should be reviewed with domain experts and relevant authorities before being used for planning, regulatory or investment decisions.
