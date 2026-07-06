# Reproducibility Status

This file records what is complete in the public package and what still requires a data/checkpoint rerun.

## Complete

- Repository status no longer says private or "public after acceptance."
- `LICENSE`, `CITATION.cff`, `environment.yml`, pinned `requirements.txt`, and `data/README.md` are present.
- Legacy development notebooks were moved to `legacy_notebooks/`.
- Canonical notebooks were added under `notebooks/`.
- `src/data_preprocessing.py` implements fold-safe preprocessing with train-fold-only scaler fitting.
- `src/model_variation_pairwise.py` implements true pairwise RQ2 comparisons.
- `src/explain_shap_lime.py` separates main 500-instance SHAP/LIME checkpoints from LIME stochasticity and grid-search checkpoints.
- `src/train_neural.py` uses CPU-safe CUDA detection.
- `MANUSCRIPT_PATCH_NOTES.md` contains exact manuscript replacements for code availability, data availability, LIME protocol, RQ2 wording, and preprocessing wording.
- Result CSVs were extracted from the manuscript tables rather than fabricated.

## Needs rerun before archival release

- `metadata/edgeiiot_feature_names.csv` and `metadata/cicids2017_feature_names.csv` need exact feature exports from the prepared matrices or checkpoints.
- `results/table_08_rq2_model_variation_pairwise.csv` contains manuscript-reported means and `n_pairs = 1225`; standard-deviation columns are blank because the manuscript table does not report them. Regenerate from saved SHAP arrays with `src/model_variation_pairwise.py`.
- If fold-safe preprocessing changes any metric after a fresh run, update the corresponding `results/*.csv` files and manuscript tables.
- Run `pytest` after installing `requirements.txt` or `environment.yml`. The local Codex bundled runtime used during this edit did not include the scientific dependencies needed to execute the tests.

## Regeneration commands

```bash
pip install -r requirements.txt
pytest
```

```bash
python -m src.export_metadata --csv data/ML-EdgeIIoT-dataset.csv --target Attack_type --out metadata/edgeiiot_feature_names.csv
python -m src.export_metadata --csv data/cicids2017_prepared.csv --target Label --out metadata/cicids2017_feature_names.csv
```

```bash
python -m src.model_variation_pairwise --dataset Edge-IIoT --model RF --npz data/edgeiiot_checkpoints/rf_shap_instances.npz
```
