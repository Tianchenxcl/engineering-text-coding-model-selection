# Reproducibility and version notes

We assembled this repository from the final 240-task experiment workflow, the archived statistical analyses and the final manuscript figures. The analysis units and validation splits are described in the main README and data dictionary.

## Relationship to the original implementation

The training and statistical scripts retain the original model settings, sampling rules and numerical calculations. We adapted filesystem locations for the repository layout and separated private manual adjudications from the preprocessing code. The original 101 tie decisions must be supplied as a private input when rerunning corpus preparation; no substitute labels are provided.

The `SOURCE_MANIFEST.csv` file records the source-to-release mapping and checksums. Internal variable names are retained so that the released records remain compatible with the analysis scripts. `SHA256SUMS.csv` supports verification of the published files.

The label-similarity script was restored from the notebook calculation documented in our project records. It retains the 11 label names and definitions, normalized sentence embeddings and cosine-similarity calculation. We added command-line options for the model location and output directory. The label order agrees with the saved 11 × 11 matrix, and all 55 values in the saved ranked-pair table agree with that matrix. The original embedding model revision and complete environment were not retained. We have not rerun the restored embedding script against the archived matrix.

The repository also includes utilities for verification, staged statistical reproduction and conversion of the private intermediate data to training workbooks. The openpyxl workbook exporter preserves the original row/column schema but does not reproduce the formatting of the original exporter. The original `.mjs` exporters are retained for reference.

## Checks completed for this release

| Check | Result |
|---|---|
| Python syntax | All released scripts compile |
| Main experiment scripts | The executable calculation structure of all 16 original scripts is unchanged after removal of standalone manuscript-text output functions |
| Prediction data | 600 unique model–task records and 40 label combinations |
| Original experiment records | 240 supervised tasks, 480 supervised evaluations and 360 LLM evaluations |
| Record-stratified Kappa prediction RMSE | 0.08413100480560769 for gradient boosting |
| Grouped Kappa prediction RMSE | 0.11041358299504746 for ridge regression |
| Grouped fold assignments | All records from a label combination remain in one outer fold |
| Step 1 public-data checks | 54 passed; private sentence-level alignment explicitly skipped |
| Steps 2, 7 and 8 | All 46 comparable CSV outputs agree with the archived results within numerical tolerance |
| Fractional-logit analysis | Runs successfully from the released supervised records |
| Saved ridge predictor | Produces results for the five example task profiles |
| Final figure reproduction | All six regenerated PNG figures are pixel-identical to the final figures |

We checked that the publication files do not contain the private sentence data or API credentials. Real contract examples, individual annotations and conversation records are excluded.

## Scope of verification

These checks used the existing local dependency installations. We did not repeat the original text-model training, make new LLM API calls, refit the full nested-CV performance-prediction workflow, refit the models in Steps 3–5, regenerate the embeddings or test installation in a fresh environment as part of these release checks.

Public task-level records support statistical reanalysis and figure reproduction. They do not permit independent reconstruction of the sentence-level evaluation scores without the private corpus and annotations. The original cloud training dependency specification uses version ranges and is not a complete frozen environment.
