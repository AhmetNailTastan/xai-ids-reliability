# Reproducibility Status

This file records what is complete in the public package and what still requires a data/checkpoint rerun.

## Complete

- Repository status no longer says private or "public after acceptance."
- `LICENSE`, `CITATION.cff`, `environment.yml`, pinned `requirements.txt`, and `data/README.md` are present.
- Legacy development notebooks were moved to `legacy_notebooks/`.
- Canonical notebooks were added under `notebooks/`.
- `src/data_preprocessing.py` implements fold-safe preprocessing with train-fold-only scaler fitting.
- `src/model_variation_pairwise.py` implements true pairwise RQ2 comparisons.
- `src/export_checkpoint_artifacts.py` exports feature metadata and can regenerate RQ2 summaries from local checkpoints.
- `src/explain_shap_lime.py` separates main 500-instance SHAP/LIME checkpoints from LIME stochasticity and grid-search checkpoints.
- `src/train_neural.py` uses CPU-safe CUDA detection.
- `MANUSCRIPT_PATCH_NOTES.md` contains exact manuscript replacements for code availability, data availability, LIME protocol, RQ2 wording, and preprocessing wording.
- Result CSVs were extracted from the manuscript tables rather than fabricated.
- `metadata/edgeiiot_feature_names.csv` and `metadata/cicids2017_feature_names.csv` were exported from prepared checkpoint matrices.
- DNN/LSTM rows in `results/table_08_rq2_model_variation_pairwise.csv` now include true pairwise standard deviations from saved SHAP arrays.

## Needs rerun before archival release

- Classical-model rows in `results/table_08_rq2_model_variation_pairwise.csv` still contain manuscript-reported means and blank standard-deviation fields. They require a long-running SHAP recomputation from the saved `modeller_seed*.pkl` and `modeller_cicids_seed*.pkl` checkpoints.
- If fold-safe preprocessing changes any metric after a fresh run, update the corresponding `results/*.csv` files and manuscript tables.

## Regeneration commands

```bash
pip install -r requirements.txt
pytest
```

```bash
python -m src.export_checkpoint_artifacts --skip-classical-rq2 --skip-neural-rq2
```

```bash
python -m src.export_checkpoint_artifacts
```

The full checkpoint command recomputes classical-model SHAP matrices for 50 trained instances per algorithm and may take a long time on CPU. If precomputed NPZ files with one SHAP matrix per trained model instance are available, `src/model_variation_pairwise.py` can summarize them directly.
