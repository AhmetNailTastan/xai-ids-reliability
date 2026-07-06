# Metadata

Class mappings are extracted from the manuscript's processed class-distribution table.

Feature-name files are deliberately marked `needs_rerun` because the exact prepared feature matrices are not stored in Git. Do not infer or hand-type feature names from memory. Regenerate them from raw/prepared data with:

```bash
python -m src.export_metadata --csv data/ML-EdgeIIoT-dataset.csv --target Attack_type --out metadata/edgeiiot_feature_names.csv
python -m src.export_metadata --csv data/cicids2017_prepared.csv --target Label --out metadata/cicids2017_feature_names.csv
```
