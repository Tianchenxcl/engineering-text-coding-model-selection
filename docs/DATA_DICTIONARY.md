# Data dictionary and analysis units

## Complexity dimensions

| Manuscript term | Principal machine-readable fields | Interpretation |
|---|---|---|
| Label number | `label_number`, `label_number_complexity`, `label_number_log2` | Number of permitted labels; the regression/prediction scripts define the applicable encoding |
| Class imbalance | `class_imbalance`, `realized_class_imbalance_index`, `class_imbalance_complexity` | One minus normalized class entropy for the supervised training sample |
| Training sample size | `training_sample_size`, `training_size_complexity`, `training_size_log2_filled` | Number of training sentences available for a supervised task; log2 transformations are defined in code |
| Label semantic similarity | `semantic_similarity`, `avg_pairwise_semantic_similarity`, `semantic_similarity_complexity` | Mean pairwise label similarity from the supplied matrix |
| Human coding disagreement rate | `annotation_disagreement`, `sampled_annotation_disagreement_rate`, `annotation_disagreement_rate`, `annotation_disagreement_complexity` | Human disagreement measure for the relevant task; supervised sampled and LLM combination aggregates retain their original definitions |

Machine-readable field names are preserved for compatibility. The words “annotation disagreement” and “human coding disagreement” refer to the manuscript dimension, but weighted/sampled and unweighted combination aggregates must not be treated as interchangeable. Missing LLM training-size/class-imbalance values mean “not applicable”, not zero. Predictor-specific filled variables are constructed by code.

## Identifiers and units

- `combination_id`: a selected set of labels, shared by related tasks. This is the grouping unit in held-out-combination validation.
- `task_no` / `original_task_id`: one controlled supervised data configuration. SVM and DeBERTa are paired within a task.
- `llm_task_id`: an LLM label-combination task.
- `prompt_strategy`: `zero_shot`, `few_shot` or `codebook_guided`.
- `repeat_id`: one of three LLM calls for a combination-strategy cell.
- `record_id` in the 600-row prediction data: a model-task configuration ID, not a sentence identifier.
- `outer_fold`: saved out-of-fold prediction assignment.
- `actual`, `prediction`, `residual`, `absolute_error`: observed task-level performance, its out-of-fold prediction, and their differences.

## Dataset counts

| Dataset | Rows | Unit |
|---|---:|---|
| `data/task_records/supervised_performance_long.csv` | 480 | 240 tasks × 2 supervised models |
| `data/task_records/llm_performance_long.csv` | 360 | 40 combinations × 3 strategies × 3 calls |
| `data/task_records/all_performance_long.csv` | 840 | All individual supervised and LLM evaluations |
| `data/task_records/common24_cross_model_long.csv` | 264 | 48 supervised evaluations + 216 LLM calls on 24 matched combinations |
| `results/step1_data_quality/llm_combination_prompt_means.csv` | 120 | LLM outcomes averaged across three calls |
| `results/step1_data_quality/common24_primary_analysis_ready.csv` | 72 | 24 combinations × 3 primary model configurations |
| `results/step6_performance_prediction/diagnostics/prediction_unified_analysis_data.csv` | 600 | 480 supervised records + 120 LLM means |

Each full out-of-fold prediction file has 7,200 rows: 600 configurations × 3 metrics × 4 predictors. These are not 7,200 independent experimental tasks. Kappa and MCC have theoretical range [-1, 1], while F1 has range [0, 1]. Unconstrained regression predictions can exceed metric bounds; they have not been silently clipped in this release.

Additional variable processing details are available in `results/step1_data_quality/analysis_data_dictionary.csv` and `standardization_parameters.csv`. The full prompt implementation is `code/run_deepseek_llm.py:build_messages`; placeholder exports reproduce its structure without publishing the real demonstrations. The placeholder export lists one illustrative placeholder per label and is not a new experimental demonstration bank.
