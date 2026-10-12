# FAERS analysis of GLP-1 receptor agonists and pancreatitis reporting

Analysis code for the manuscript by Balidemaj F and Burleigh S on pancreatitis reporting with GLP-1 receptor agonists in FAERS (submitted to BMC Gastroenterology: "Temporal shifts in pancreatitis reporting with GLP-1 receptor agonists: active-comparator disproportionality analyses and machine-learning report prioritization in FAERS").

The code reads the public FDA Adverse Event Reporting System (FAERS) quarterly ASCII files, deduplicates them at case level, and runs the disproportionality, robustness, time-to-onset and machine-learning analyses reported in the manuscript.

## Data

FAERS data are not redistributed here. Download the 44 quarterly ASCII zip files from 2015 Q1 to 2025 Q4 (`faers_ascii_2015q1.zip` ... `faers_ascii_2025q4.zip`) from the FDA FAERS Quarterly Data Extract Files page:
https://fis.fda.gov/extensions/FPD-QDE-FAERS/FPD-QDE-FAERS.html

Place the zip files (no need to unzip) in one folder. By default the scripts look in `~/Desktop/ASCII2015-2025`; to use another folder, set the environment variable `FAERS_BASE_DIR` or edit `BASE_DIR` in `config.py`. Results are written to `results/` (CSV tables) and `results/figures/` (600-dpi PNG); intermediate files go to `work/`.

## Licence

MIT (see `LICENSE`).

## Software

- Python 3.13.9 with the packages in `requirements.txt`
- R 4.5.3 with openEBGM 0.10.0 (for `03_run_openebgm.R`; install with `remotes::install_version("openEBGM", version = "0.10.0")`)
- About 8 GB of memory and 10 GB of free disk space

## Running the analysis

Run the scripts in order from this folder (for example `python 01_extract_faers_ascii.py`).

| Step | Script | Purpose |
|---|---|---|
| 1 | `01_extract_faers_ascii.py` | Reads all quarters, removes FDA-deleted cases, keeps the latest version of each case, and builds the report-level dataset (30–90 min; resumes if interrupted) |
| 1b | `01b_ebgm_counts.py` | EBGM input: report counts for every active ingredient × MedDRA preferred term pair, and for the GLP-1 RAs and DPP-4 inhibitors × outcome term sets (30–60 min; resumes if interrupted) |
| 2 | `02_disproportionality.py` | PRR, ROR and IC; suspect-only and DPP-4 inhibitor analyses; yearly counts |
| 3 | `03_run_openebgm.R` (in R) | EBGM with EB05 and EB95 (gamma-Poisson shrinker, openEBGM): prior estimated from all ingredient × preferred term pairs, posterior for each agent × outcome term set |
| 4 | `04_tables_figures_dpa.py` | Disproportionality tables and forest plot |
| 5 | `05_machine_learning.py` | Primary machine-learning models (intake predictors only): random split, temporal validation, top-k, calibration, recalibration, bootstrap, interpretability, secondary outcomes |
| 5b | `05_machine_learning.py --sensitivity` | Sensitivity model adding seriousness criteria and report type (outputs prefixed `SENS_`) |
| 5c | `05b_ml_initial_version.py` | Sensitivity analysis: primary models with predictors from the initial (first-received) version of each case instead of the latest version (one pass over the DRUG tables) |
| 6 | `06_export_glp1_subset.py` | Optional: exports GLP-1 RA and DPP-4 inhibitor reports to a compressed CSV |
| 7 | `07_extract_robustness_data.py` | Another pass over the zip files: co-reported pancreatitis causes, indications (by class and by GLP-1 RA), therapy and event dates |
| 8 | `08_robustness_tto_cases.py` | Robustness analyses, reporting composition by year, case characteristics, time to onset (Weibull) |
| 9 | `09_brand_adjusted_controls.py` | Pass for product names; product-based indication, adjusted active-comparator RORs (logistic regression), stepwise adjustment, recorded-indication comparison, gallbladder/biliary positive control |
| 9b | `09b_sglt2_comparator.py` | Second active comparator: SGLT2 inhibitors (crude, year-only, year and reporter type, and fully adjusted RORs; yearly pancreatitis proportions; one pass over the DRUG tables) |
| 9c | `09c_year_lawyer_sensitivity.py` | Year-specific active-comparator RORs with a test for variation between years, and exclusion of lawyer-submitted reports |
| 9d | `09d_year_only_subgroups.py` | Year-only adjusted RORs (primary adjustment) for the DPP-4 inhibitor subgroups: health-professional reports, diabetes products, both combined, recorded diabetes indication; re-estimates the fully adjusted model as a check (reads the CSV from step 6) |
| 10 | `10_ml_extensions.py` | Simple baselines, paired bootstrap of AUROC differences, annual retraining (expanding and sliding windows) vs frozen model, subgroup performance, population shift, review workload |
| 11 | `11_figure_comparator.py` | Fig. 2: pancreatitis proportions by year (GLP-1 RA, DPP-4 and SGLT2 inhibitors) and GLP-1 RA class RORs, crude, year-adjusted (primary) and fully adjusted |

Steps 7 to 11 depend on earlier steps: run 07 before 08 and 09, 09 before 09b, 09b before 09c and 11, 06 and 09 before 09d, 09d before 11, and 05 before 05b. Each script prints the next step when it finishes. Intermediate files for each quarter are cached in `work/`; delete the relevant cache files (for example `work/extra_*.pkl` for step 7) to rebuild them after changing a definition.

The machine-learning scripts (05, 10, 11) can alternatively read the CSV written by step 6 instead of `work/reports.pkl` by setting the environment variable `FAERS_ML_INPUT` to its path.

For a quick test, set `FAERS_TEST_QUARTERS` (for example `2022q2,2023q3`) before step 1; delete `work/` and `results/` before the full run.

## Main manuscript outputs

Output file names reflect the order in which the analyses were developed and do not always match the final figure and table numbers.

| Manuscript | Script | Output |
|---|---|---|
| Table 1 | 02, 03, 04 | `Table1_disproportionality_pancreatitis.csv` |
| Table 2 | 08, 09, 09b, 09d | `Robustness_ladder.csv`, `Adjusted_comparator.csv`, `Adjustment_steps.csv`, `SGLT2_comparator.csv`, `Year_only_subgroups_dpp4.csv` |
| Table 3 | 08 | `Case_characteristics.csv`, `Time_to_onset.csv`, `Descriptive_numbers.csv` |
| Table 4 | 05, 10 | `Table2_model_performance.csv`, `ML_baselines.csv`, `ML_auroc_differences.csv`, `SENS_Table2_model_performance.csv` |
| Table 5 | 10 | `ML_workload.csv` |
| Fig. 1 | 04 | `figures/Fig1_forest_ROR_pancreatitis.png` |
| Fig. 2 | 09, 09b, 09d, 11 | `figures/Fig_comparator_positive_control.png`, `SGLT2_comparator.csv`, `SGLT2_by_year.csv` |
| Fig. 3 | 08 | `figures/Fig_reporting_composition.png` |
| Fig. 4 | 08 | `figures/Fig_time_to_onset.png` |
| Fig. 5 | 05 | `figures/Fig3_ROC_PR_random_temporal.png` |
| Fig. 6 | 10 | `figures/Fig_ml_retraining.png` |

Supplementary tables and figures are produced by the same scripts (see Additional file 1 of the manuscript), for example: data flow `data_flow.csv` (01); EBGM `ebgm_results.csv` (03); secondary outcomes `TableS_disproportionality_secondary_outcomes.csv` (04); matched pancreatitis-cause and indication terms `check_riskfactor_terms.csv`, `check_indication_terms.csv` (07); reporting composition `Reporting_composition_by_year.csv` (08); product-based indication `Brand_indication.csv`, `Brand_indication_coverage.csv` and control outcomes `Positive_control.csv` (09); annual retraining `ML_annual_retraining.csv`, subgroups `ML_subgroups.csv` and population shift `ML_population_shift.csv` (10); pancreatitis proportions by year `Comparator_by_year.csv` (11); bootstrap, calibration, recalibration, top-k, coefficients and permutation importance `TableS*.csv`, `Table3_topk.csv`, `Table4_calibration_recalibration.csv` (05; `SENS_` prefix for the sensitivity model).

## Notes

- Gallbladder/biliary events are analysed as a positive control (Additional file 1). Step 5 also writes temporal machine-learning results for gallbladder/biliary events (`TableS_secondary_outcomes_*.csv`), which are exploratory and not reported in the manuscript.

- A fixed random seed (42) is used throughout.
- Results describe reporting patterns in a spontaneous reporting database and are not estimates of incidence or causal risk.
