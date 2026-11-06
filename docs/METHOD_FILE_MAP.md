# Method-to-file map

| Research component | Code | Released evidence |
|---|---|---|
| Label-definition similarity | `preprocessing/compute_label_similarity_recovered.py` | `metadata/label_semantic_similarity_matrix.xlsx`, ranked pair table; recovered source has not been rerun |
| Deduplication and disagreement aggregation | `preprocessing/process_data_0713.py` | Aggregate `preprocessing/processed/label_statistics.csv`; original sentences and adjudications withheld |
| Label-combination and task selection | `preprocessing/select_label_sets_0713.py`, `generate_task_datasets_0713.py` | `manifests/source_manifests/`, aggregate candidate tables |
| SVM and DeBERTa experiments | `code/train_svm.py`, `train_deberta_v3.py` | `data/task_records/supervised_performance_long.csv` |
| Three LLM prompting strategies | `code/run_deepseek_llm.py` | `data/task_records/llm_performance_long.csv`, `materials/prompt_templates_with_placeholders.json` |
| Analysis assembly / quality checks | `code/assemble_analysis_tables.py`, `prepare_step1_analysis.py` | `results/step1_data_quality/` |
| Reliability distributions / Figure 1 | `code/describe_step2_reliability.py` | `results/step2_descriptive_reliability/` |
| Effects of task characteristics on supervised models / Figure 2 | `code/fit_step3_supervised_complexity_models.py` | `results/step3_supervised_complexity/` |
| LLM prompting and task characteristics / Figure 3 | `code/fit_step4_llm_prompt_models.py` | `results/step4_llm_prompt_analysis/` |
| Matched Common24 comparison / Figure 4 | `code/fit_step5_common24_cross_model.py` | `results/step5_common24_cross_model/` |
| Performance prediction / Figures 5–6 | Both `code/fit_step6_*.py` scripts | Both `results/step6_*/` directories, including OOF records and saved models |
| Diagnostic scenarios | `code/build_step7_diagnostic_guidance.py` | `results/step7_diagnostic_guidance/` |
| Robustness and scope checks | `code/run_step8_robustness_checks.py`, `run_fractional_logit_robustness.py` | `results/step8_robustness/`, `results/bounded_kappa_robustness/` |
| Final journal figure styling | `code/figures/redraw_optimized_figures.py` | `figures/` and `figures/source_data/` |

The original task designs and recorded model/API identifiers are retained. Historical filenames and variable names containing `complexity` are retained for reproducibility. `DATA_DICTIONARY.md` maps the fields to the five task characteristics used in the current manuscript.
