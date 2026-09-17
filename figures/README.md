# Manuscript figures

We provide the six final figures as individual files. Each figure has PNG, PDF and SVG versions. The underlying plotted data are in [`source_data/`](source_data/).

| Figure | Content | Files | Analysis source |
|---|---|---|---|
| 1 | Reliability distributions across candidate configurations | [PNG](Figure1_reliability_distributions.png) · [PDF](Figure1_reliability_distributions.pdf) · [SVG](Figure1_reliability_distributions.svg) | Step 2 |
| 2 | Supervised-model responses to task complexity | [PNG](Figure2_supervised_complexity_response.png) · [PDF](Figure2_supervised_complexity_response.pdf) · [SVG](Figure2_supervised_complexity_response.svg) | Step 3 |
| 3 | LLM prompting responses to task complexity | [PNG](Figure3_llm_prompt_complexity_response.png) · [PDF](Figure3_llm_prompt_complexity_response.pdf) · [SVG](Figure3_llm_prompt_complexity_response.svg) | Step 4 |
| 4 | Matched model comparison on Common24 | [PNG](Figure4_common24_model_comparison.png) · [PDF](Figure4_common24_model_comparison.pdf) · [SVG](Figure4_common24_model_comparison.svg) | Step 5 |
| 5 | Task-performance prediction | [PNG](Figure5_model_performance_prediction.png) · [PDF](Figure5_model_performance_prediction.pdf) · [SVG](Figure5_model_performance_prediction.svg) | Both Step 6 validation designs |
| 6 | Calibration under label-combination-held-out validation | [PNG](Figure6_grouped_holdout_calibration.png) · [PDF](Figure6_grouped_holdout_calibration.pdf) · [SVG](Figure6_grouped_holdout_calibration.svg) | Grouped Step 6 validation |

From the repository root, run:

```sh
python reproduce.py --steps figures
```

The script reads archived statistical results and writes regenerated PNG, PDF, SVG and TIFF files to `reproduced/figures/`. The existing figure files remain unchanged.
