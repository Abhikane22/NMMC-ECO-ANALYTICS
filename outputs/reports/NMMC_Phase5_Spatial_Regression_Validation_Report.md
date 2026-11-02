# NMMC Phase 5 — Spatial Regression and Inference Validation Report

**Project:** NMMC Multitemporal LULC and Ecological Impact Analysis  
**Study area:** Navi Mumbai Municipal Corporation (NMMC), Maharashtra, India  
**Analysis period:** 1994–2024  
**Spatial unit:** 111 NMMC wards  
**Report status:** Core numerical validation completed; scientific interpretation and limitations documented  
**Prepared:** 2 November 2026

---

## 1. Executive summary

Phase 5 evaluates the relationship between ward-level land-use/land-cover (LULC) composition changes and changes in land surface temperature (LST), accounting for spatial dependence between neighbouring wards.

The analysis progressed from ordinary least squares (OLS) diagnostics to spatial autoregressive (SAR) models, compositional sensitivity analyses, and simulation-based impact inference. Queen and Rook contiguity weights were used to test sensitivity to the spatial-neighbour definition.

The final validation pipeline completed successfully:

| Validation item | Result |
|---|---:|
| Successful model fits | 16 / 16 |
| Usable covariance matrices | 16 / 16 |
| Impact confidence-interval records | 90 |
| Spatial multiplier checks | All passed (8 / 8 specifications) |
| Requested simulation draws retained | All retained (10,000 per impact record) |

The main compositional SAR sensitivity results are stable across the tested zero-replacement factors (0.5, 1, and 2 Landsat pixel areas) and Queen/Rook weights. In the ILR models, the vegetation-versus-rest balance and barren-versus-wetlands balance show consistently non-zero associations with ward-level LST change; the water-versus-rest and built-up-versus-rest balances do not show confidence intervals excluding zero in the tested specifications.

These are observational, ward-level associations. They do not establish that a particular land-cover change caused a temperature change. The historical LULC maps used in the temporal analysis are preliminary classifications, and the inference does not include full classification, zero-replacement, or model-selection uncertainty.

## 2. Scope and analytical data

### 2.1 Study design

The study uses 111 NMMC wards and compares 1994 with 2024. Ward-level changes are defined consistently as:

\[
\Delta X = X_{2024} - X_{1994}
\]

The response variable for the principal spatial regression analyses is:

- `lst_change_c`: ward-level change in mean LST, in degrees Celsius.

The main predictor families are:

1. **Absolute class-area changes** in km².
2. **Normalized class-area changes** in percentage points of ward area.
3. **Compositional ILR balances** derived from the five-class LULC composition.

### 2.2 Five LULC classes

The five classes used in the compositional analysis are:

- Water
- Vegetation
- Built-up
- Barren
- Coastal wetlands

The five-class closure audit passed. For each ward, the class-area changes sum to approximately zero, subject only to floating-point precision:

\[
\Delta W + \Delta V + \Delta B + \Delta R + \Delta C = 0
\]

The maximum absolute ward-level closure error was approximately \(3.0 \times 10^{-15}\) km². This confirms that the five class changes are compositionally constrained and cannot all be entered as independent predictors alongside an intercept.

### 2.3 Data-track qualification

The temporal analysis uses the preliminary LULC maps for 1994, 2004, 2014 and 2024. The 2024 scientific-final classification is maintained as a separate classification/accuracy-assessment track. The SAR models in this report use the preliminary temporal-analysis track.

The 2024 classifier's reported holdout accuracy is not an independent validation of the 1994, 2004 or 2014 maps. Findings from the 1994–2024 regression should therefore be interpreted in light of the preliminary status of the historical classifications.

## 3. Why spatial regression was required

OLS residuals exhibited statistically significant positive spatial autocorrelation in the tested specifications. This indicates that neighbouring wards' model errors were not spatially independent under OLS.

The SAR specification incorporates a spatially lagged response term:

\[
y = \rho Wy + X\beta + \varepsilon
\]

where:

- \(y\) is the ward-level LST-change response;
- \(W\) is the spatial-weights matrix;
- \(\rho\) is the spatial autoregressive parameter;
- \(X\) contains the explanatory variables;
- \(\beta\) contains regression coefficients; and
- \(\varepsilon\) is the remaining error term.

Queen contiguity treats wards that touch at an edge or vertex as neighbours. Rook contiguity uses shared edges. Both weights structures were row-standardized for the spatial analyses.

The Queen and Rook weights each formed one connected component across 111 wards, with no islands. Queen had a mean of approximately 5.35 neighbours per ward; Rook had approximately 4.99.

## 4. Absolute and normalized SAR comparison

Two non-compositional predictor parameterizations were compared.

- **Absolute model:** changes in class area in km².
- **Normalized model:** changes in class share, expressed in percentage points of ward area.

| Specification | OLS R² | OLS residual Moran's I (p) | SAR pseudo-R² | AIC | SAR rho | SAR residual Moran's I (p) |
|---|---:|---:|---:|---:|---:|---:|
| Absolute — Queen | 0.0889 | 0.5141 (0.001) | 0.6042 | 397.883 | 0.7660 | -0.0077 (0.492) |
| Absolute — Rook | 0.0889 | 0.5180 (0.001) | 0.6078 | 398.073 | 0.7631 | -0.0123 (0.498) |
| Normalized — Queen | 0.4057 | 0.3800 (0.001) | 0.6798 | 363.680 | 0.6642 | -0.0233 (0.428) |
| Normalized — Rook | 0.4057 | 0.3748 (0.001) | 0.6805 | 364.019 | 0.6606 | -0.0305 (0.364) |

The normalized models have lower AIC than the absolute-area models in this comparison. OLS residual Moran's I is significant in all four cases, while SAR residual Moran's I is not significant at the 0.05 level in these diagnostics.

The normalized SAR coefficients were statistically significant for all three normalized class-change predictors in the fitted specifications. However, these normalized predictors do not remove the compositional constraint. The normalized SAR results are therefore treated as a comparison/sensitivity analysis, not as the preferred final compositional interpretation.

**Important interpretation warning:** the negative built-up coefficient in the normalized SAR model must not be interpreted as evidence that built-up expansion reduces temperature. It is a conditional coefficient in a model with spatial feedback and interdependent land-cover shares. The compositional model below is the more appropriate framework for interpreting relative changes in the five-class composition.

## 5. Five-reference-class compositional sensitivity

A five-class closure constraint means that only four independent class-change coordinates can be represented with an intercept. A reference-class parameterization was therefore tested by omitting each of the five classes in turn.

Ten models were fitted: five reference classes under Queen weights and five under Rook weights.

The fitted results were equivalent across reference-class choices to numerical precision:

- Queen SAR pseudo-R²: approximately 0.711330.
- Rook SAR pseudo-R²: approximately 0.713028.
- Queen rho: approximately 0.586356.
- Rook rho: approximately 0.584441.
- Maximum prediction differences between reference-class parameterizations were below \(8 \times 10^{-12}\).
- Maximum AIC difference was approximately \(5.7 \times 10^{-14}\).

The reference class changes the coefficient parameterization and its interpretation, but not the fitted model in this closed five-part composition. The five parameterizations are not five independent replications or five independent hypothesis tests.

## 6. ILR compositional SAR model

### 6.1 Rationale

The five class proportions are compositional data: they are non-negative parts constrained to sum to one. Ordinary regression on all five shares would create exact linear dependence. The analysis therefore uses an orthonormal isometric log-ratio (ILR) representation.

The four sequential balances are:

1. `ILR1_water_vs_rest`
2. `ILR2_vegetation_vs_rest`
3. `ILR3_builtup_vs_rest`
4. `ILR4_barren_vs_wetlands`

The balance coordinates are relative log-ratio contrasts. They should not be described as isolated effects of one land-cover class. For example, `ILR2_vegetation_vs_rest` expresses vegetation relative to the geometric balance of the remaining classes under the specified sequential basis.

The basis orthonormality check passed.

### 6.2 Zero-value treatment

Some ward-year compositions contain zero-valued classes. Log-ratio transformations cannot be computed directly with zeros. The analysis used a resolution-informed multiplicative zero-replacement sensitivity procedure.

- One 30 m pixel corresponds to 0.0009 km².
- Replacement factors tested: 0.5, 1 and 2 pixel areas.
- Replacement was capped relative to the smallest positive component.
- The replacement mass was limited, and non-zero components were rescaled multiplicatively so each composition continued to sum to one.

This is a sensitivity procedure, not proof that the observed zeros are sampling zeros. Some zeros may represent genuine absence or resolution-related absence. The results must be interpreted with this uncertainty in mind.

### 6.3 ILR model diagnostics

Six ILR-SAR models were evaluated: three zero-replacement factors crossed with Queen and Rook weights.

| Replacement | Weights | OLS R² | OLS residual Moran's I (p) | SAR pseudo-R² | AIC | rho | SAR residual Moran's I (p) |
|---:|---|---:|---:|---:|---:|---:|---:|
| 0.5 pixel | Queen | 0.5336 | 0.2152 (0.001) | 0.6797 | 360.280 | 0.5468 | -0.0231 (0.411) |
| 0.5 pixel | Rook | 0.5336 | 0.2121 (0.001) | 0.6789 | 360.889 | 0.5433 | -0.0261 (0.382) |
| 1 pixel | Queen | 0.5135 | 0.2365 (0.001) | 0.6737 | 362.948 | 0.5627 | -0.0197 (0.436) |
| 1 pixel | Rook | 0.5135 | 0.2334 (0.001) | 0.6732 | 363.498 | 0.5594 | -0.0231 (0.396) |
| 2 pixels | Queen | 0.4918 | 0.2594 (0.001) | 0.6672 | 365.750 | 0.5783 | -0.0170 (0.444) |
| 2 pixels | Rook | 0.4918 | 0.2551 (0.001) | 0.6670 | 366.263 | 0.5750 | -0.0213 (0.405) |

OLS residual spatial autocorrelation is significant in all six specifications. SAR residual Moran's I is not significant at the 0.05 level in any of the six specifications.

### 6.4 ILR coefficient sensitivity

The coefficient pattern is consistent across the tested replacement factors and spatial weights:

- `ILR1_water_vs_rest`: not statistically significant in any of the six specifications.
- `ILR2_vegetation_vs_rest`: negative and statistically significant in all six specifications.
- `ILR3_builtup_vs_rest`: not statistically significant in any of the six specifications.
- `ILR4_barren_vs_wetlands`: positive and statistically significant in all six specifications.

Representative coefficients for the 1-pixel Queen model:

| ILR balance | Coefficient | p-value |
|---|---:|---:|
| Water versus rest | 0.078394 | 0.3832 |
| Vegetation versus rest | -0.312969 | 0.00538 |
| Built-up versus rest | -0.121084 | 0.2753 |
| Barren versus wetlands | 0.370488 | 3.70 × 10⁻⁷ |

A Benjamini–Hochberg adjustment was also applied across the 24 coefficient tests from the six sensitivity models. The vegetation and barren-versus-wetlands balances remained significant across the tested specifications. Because the six models are dependent sensitivity variants, this pooled adjustment is best viewed as a conservative descriptive check, not as 24 independent scientific hypotheses.

## 7. Direct, indirect and total impacts

### 7.1 Impact interpretation

In a SAR model, a predictor change can affect the response in the same ward and propagate through the spatial system. Therefore, the regression coefficient alone is not the full marginal effect.

The impact analysis reports:

- **Direct impact:** average own-ward effect, including feedback through the spatial system.
- **Indirect impact:** average spillover effect on other wards.
- **Total impact:** direct plus indirect impact.

Confidence intervals were generated using 10,000 multivariate simulation draws from each fitted model's parameter covariance matrix. The reported point estimate is the impact evaluated at the fitted parameter estimates; the simulation median is retained separately in the output.

### 7.2 Representative primary ILR specification

The 1-pixel replacement, Queen-weights ILR-SAR model is shown here as a representative specification. It is not claimed to be uniquely correct; the other replacement and weights variants are used to assess robustness.

| ILR balance | Impact | Estimate | 95% simulation interval |
|---|---|---:|---:|
| Water versus rest | Direct | 0.0852 | [-0.1092, 0.2807] |
| Water versus rest | Indirect | 0.0941 | [-0.1217, 0.3872] |
| Water versus rest | Total | 0.1793 | [-0.2322, 0.6509] |
| Vegetation versus rest | Direct | -0.3402 | [-0.5772, -0.1115] |
| Vegetation versus rest | Indirect | -0.3755 | [-0.8183, -0.1152] |
| Vegetation versus rest | Total | -0.7157 | [-1.3321, -0.2329] |
| Built-up versus rest | Direct | -0.1316 | [-0.3642, 0.1038] |
| Built-up versus rest | Indirect | -0.1453 | [-0.4616, 0.1284] |
| Built-up versus rest | Total | -0.2769 | [-0.7987, 0.2294] |
| Barren versus wetlands | Direct | 0.4028 | [0.2508, 0.5497] |
| Barren versus wetlands | Indirect | 0.4445 | [0.2496, 0.7876] |
| Barren versus wetlands | Total | 0.8473 | [0.5514, 1.2618] |

The vegetation-versus-rest and barren-versus-wetlands balances have direct, indirect and total intervals excluding zero in this representative model. Water-versus-rest and built-up-versus-rest intervals include zero.

The balance units are ILR-coordinate units; these estimates are not degrees Celsius per km² or per percentage point. They describe modelled marginal response to a unit change in the corresponding log-ratio balance, with the other model terms held as specified.

### 7.3 Robustness across sensitivity specifications

The same qualitative impact-interval pattern was observed across all six ILR specifications:

- All direct, indirect and total intervals for `ILR2_vegetation_vs_rest` excluded zero and were negative.
- All direct, indirect and total intervals for `ILR4_barren_vs_wetlands` excluded zero and were positive.
- All intervals for `ILR1_water_vs_rest` included zero.
- All intervals for `ILR3_builtup_vs_rest` included zero.

This consistency across Queen/Rook weights and 0.5/1/2-pixel zero-replacement factors supports robustness to these particular modelling choices. It does not establish robustness to all possible model specifications or data uncertainties.

### 7.4 Normalized SAR impacts: comparison only

The normalized SAR Queen model produced negative direct, indirect and total impact intervals for all three normalized class-change predictors. For example, the built-up percentage-point total impact estimate was -0.0343, with a 95% interval of [-0.0689, -0.0014].

These normalized results are not the preferred basis for class-specific ecological claims because the five class shares are compositionally constrained. In particular, they should not be used to claim that built-up expansion cools wards.

## 8. Numerical and covariance validation

### 8.1 Model fit and covariance checks

The final validation run reported:

- 16 successful model fits out of 16.
- 16 usable covariance matrices out of 16.
- 90 impact interval records.
- 10,000 valid simulation draws retained for every impact record.

Covariance matrices were checked for finite values, symmetry, positive-semidefinite behaviour and finite standard errors. All 16 covariance records were marked usable for simulation.

The covariance condition numbers differed substantially by model family:

- ILR-SAR: approximately 326–360.
- Normalized SAR: approximately 25,415–26,370.

The higher condition numbers in the normalized models indicate greater numerical sensitivity and should be disclosed. They do not, by themselves, invalidate the fitted models.

### 8.2 Full versus Ord estimation

Full and Ord estimation methods were compared across the eight model specifications. The comparison rows reported equivalent estimates to numerical precision. For example, maximum parameter differences were on the order of \(10^{-12}\), with negligible differences in rho and AIC.

The Ord method produced fewer numerical warnings than the full method in the comparison run. It is therefore a reasonable computational choice for the reported run, while the warning history and equivalence check should remain part of the reproducibility record.

Warnings included determinant-related numerical warnings during some likelihood calculations and pandas/NumPy deprecation warnings. All model fits returned finite results and passed the covariance and impact validation checks. These warnings should not be silently discarded; package versions and warning output should be retained with the analysis environment.

### 8.3 Independent spatial multiplier validation

The impact implementation was checked independently using matrix-based and eigenvalue-based calculations.

Across all eight specifications:

- Matrix direct multiplier and eigenvalue direct multiplier agreed to floating-point precision.
- Matrix total multiplier agreed with \(1/(1-\rho)\) to floating-point precision.
- All spatial multiplier checks passed.

The maximum reported discrepancies were approximately \(10^{-15}\), consistent with floating-point rounding.

## 9. Inference policy

The following policy should be used in the research paper and dashboard.

1. **Primary compositional interpretation:** use the ILR-SAR analysis, not a five-share regression with all five class changes entered alongside an intercept.
2. **Sensitivity analysis:** report Queen and Rook weights and all three zero-replacement factors. Do not count these model variants as independent replications.
3. **Hypothesis family:** pre-specify the four ILR balance effects as the primary coefficient hypothesis family for the selected primary model. Treat the other weight and replacement variants as robustness analyses. If formal multiplicity-adjusted inference is presented, state the exact family and correction method.
4. **Impact intervals:** describe the 95% intervals as simulation-based and conditional on the fitted model, covariance estimate, spatial weights and chosen zero-replacement procedure.
5. **No causal language:** use “associated with”, “conditional association” or “modelled relationship”; do not use “caused”, “led to” or “resulted in” based on these cross-sectional ward-level models.
6. **No over-counting of evidence:** direct, indirect and total impacts are mathematically related. Queen/Rook and zero-replacement variants are dependent. The 90 intervals are not 90 independent hypotheses.
7. **Compositional interpretation:** describe ILR balances as relative contrasts among groups of classes, not as isolated effects of a single class.
8. **Uncertainty:** explicitly acknowledge that the intervals do not propagate LULC classification uncertainty, zero-type uncertainty, raster measurement uncertainty, or all model-selection uncertainty.

## 10. Limitations

### 10.1 Preliminary historical LULC classifications

The 1994, 2004 and 2014 maps, and the 2024 map used in the preliminary temporal track, are preliminary. The separate 2024 scientific-final classification does not independently validate the historical maps. Misclassification can affect estimated class changes and therefore the regression predictors.

### 10.2 Temporal and sensor comparability

The analysis uses seasonal Landsat composites from different sensors across the study period. Although the same seasonal window and processing framework were applied, sensor characteristics, atmospheric conditions, acquisition availability and residual cloud effects may contribute to observed differences.

### 10.3 Thermal resolution

Thermal indicators were exported at 120 m, while optical indices and LULC were at 30 m. Ward-level thermal summaries therefore represent a coarser spatial measurement. Small wards and heterogeneous urban areas may be particularly sensitive to this resolution difference.

### 10.4 Zero replacement

The ILR analysis requires positive compositions. The tested multiplicative replacement values are resolution-informed sensitivity choices. The underlying zero mechanism is not known and may include structural and resolution-related zeros.

### 10.5 Spatial model specification

Queen and Rook contiguity are two plausible neighbourhood definitions, not an exhaustive set of spatial-weights structures. A non-significant residual Moran's I indicates that residual spatial autocorrelation was not detected by the selected diagnostic under the tested weights; it does not prove that all spatial dependence or omitted-variable bias has been removed.

### 10.6 Observational and ecological inference

The analysis is cross-sectional at the ward level, based on changes between two time points. It does not establish temporal ordering at finer intervals or causal mechanisms. Ward-level associations should not be transferred directly to individual residents, properties or parcels.

### 10.7 Other measurement and modelling limitations

The study has documented small overlaps among ward polygons. Their total area is negligible relative to the study area, but ward-intersection summaries may double-count tiny shared regions. In addition, NDBI and NDMI in the current index products are exact negatives due to the implemented band-order convention; they must not be treated as independent corroborating indicators without revisiting the index definitions.

## 11. Reproducibility: outputs and audit files

The following generated outputs should be retained with the project:

| Output | Purpose |
|---|---|
| `NMMC_SAR_Impact_Confidence_Intervals.csv` | Direct, indirect and total impact estimates and simulation intervals |
| `NMMC_SAR_Covariance_Validation.csv` | Covariance matrix and standard-error validation |
| `NMMC_SAR_Method_Comparison.csv` | Full versus Ord estimation comparison |
| `NMMC_ILR_SAR_Diagnostics.csv` | ILR-SAR model diagnostics |
| `NMMC_ILR_SAR_Coefficients.csv` | ILR balance coefficients |
| `NMMC_ILR_SAR_Impacts.csv` | ILR impact estimates |
| `NMMC_ILR_Zero_Replacement_Audit.csv` | Ward-year zero-replacement audit |
| `NMMC_Compositional_SAR_Reference_Diagnostics.csv` | Five-reference-class equivalence diagnostics |
| `NMMC_Compositional_SAR_Reference_Equivalence.csv` | Prediction and fit equivalence across reference classes |
| `NMMC_SAR_Sensitivity_Diagnostics.csv` | Absolute and normalized SAR diagnostics |
| `NMMC_SAR_Sensitivity_Coefficients.csv` | Absolute and normalized SAR coefficients |
| `NMMC_Sensitivity_Global_Comparison.csv` | Queen/Rook global spatial sensitivity |
| `NMMC_Sensitivity_Classification_Comparison.csv` | Queen/Rook local hotspot classification comparison |
| `NMMC_LULC_Compositional_Closure_Audit.csv` | Five-class composition closure validation |

The exact filenames may include version suffixes in a working export directory. The final repository should use one canonical version of each output and record the script version, input files, software versions, random seeds and run date.

## 12. Conclusion and Phase 5 status

The Phase 5 computational validation is complete. The SAR models passed the final fit, covariance, simulation and spatial multiplier checks. The compositional ILR results showed a consistent pattern across the tested Queen/Rook weights and zero-replacement factors: the vegetation-versus-rest and barren-versus-wetlands balances had impact intervals excluding zero, while the water-versus-rest and built-up-versus-rest balances did not.

These findings are suitable for cautious reporting as ward-level spatial associations. They should not be presented as causal effects or as fully uncertainty-propagated estimates. The preliminary status of the historical LULC maps and the limits of the zero-replacement and conditional simulation procedures must remain visible in the paper and dashboard.

**Phase 5 status:** Core spatial regression, compositional sensitivity, covariance validation, impact interval generation and independent multiplier validation completed. The report records the interpretation policy and limitations required before using these findings in the dashboard and research paper.
