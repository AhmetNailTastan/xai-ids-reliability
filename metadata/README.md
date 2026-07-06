# Metadata

Class mappings are extracted from the manuscript's processed class-distribution table.

Feature-name files were exported from the prepared Edge-IIoT and CIC-IDS-2017 checkpoint matrices. Do not infer or hand-type feature names from memory. They can be regenerated from the local checkpoints with:

```bash
python -m src.export_checkpoint_artifacts --skip-classical-rq2 --skip-neural-rq2
```

If prepared checkpoint files are unavailable, use `src/export_metadata.py` with a prepared CSV and the correct target column.
