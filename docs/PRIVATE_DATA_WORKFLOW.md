# Private-data workflow

The public repository starts statistical reproduction from task-level records. The following original preprocessing/training workflow is supplied for inspection and for use by authorized holders of the private inputs; it cannot run end to end from the public release alone.

## Required private assets

- The original paired-annotation workbook at repository root, using the filename expected by `process_data_0713.py`.
- The legacy fixed-test workbook `data/test.xlsx` and demonstration workbook `data/deepseek_v4_11_label_examples.xlsx`.
- `private_inputs/tie_adjudications.json`: the original 101 deterministic tie decisions. Schema: a JSON object mapping a tie-sequence integer string to `[integer_label_id, rationale]`. The release does not invent replacement decisions.
- Alternatively, the already generated `supervised_data/trainN.xlsx`, `supervised_data/testN.xlsx` (N = 1…240), and `llm_data/LLM*.json` can be supplied from the private study archive.

The similarity matrix at `metadata/label_semantic_similarity_matrix.xlsx` and aggregate manifests are public. The original corpus and annotation records, demonstrations, source-row links and per-sentence predictions are not.

## Preparation sequence

```sh
python preprocessing/process_data_0713.py
python preprocessing/select_label_sets_0713.py
python preprocessing/generate_task_datasets_0713.py
python preprocessing/prepare_hy_tmp_package_0713.py
python preprocessing/export_training_workbooks.py
python preprocessing/validate_hy_tmp_package_0713.py
```

The openpyxl exporter is a release convenience utility for the same intermediate row/column schema; it does not sample, relabel or score observations. It does not reproduce the original workbook styling. The two original `.mjs` exporters are preserved as provenance and depend on `@oai/artifact-tool`; they are not required for the public statistical workflow. `validate_generated_datasets_0713.py` refers to the intermediate formatted-workbook layout produced by the original exporter.

Preprocessing can regenerate private text-bearing files. They are ignored by `.gitignore` and must remain outside the public release. Keep the supplied public release separate from any private experimental working tree.

## Training and evaluation entry points

Install the original training dependency specification in `requirements-training.txt` into a separate suitable environment. This specification uses version ranges and is not a full frozen cloud environment.

```sh
python code/train_svm.py --help
python code/train_deberta_v3.py --help
python code/run_deepseek_llm.py --help
python code/assemble_analysis_tables.py --help
```

The training scripts default to this repository's `supervised_data/` and `manifests/task_file_mapping.xlsx`. Select a locally available DeBERTa model through the documented CLI; the historical `/hy-tmp/` default is an original cloud-environment fallback, not a bundled model. The LLM runner defaults to `llm_data/`; it uses API-key environment variables and will make paid requests only when explicitly run with private inputs and credentials. Its outputs may contain original text and raw API content, so do not upload those output directories.

No original text-model retraining or API call was performed to prepare this release.
