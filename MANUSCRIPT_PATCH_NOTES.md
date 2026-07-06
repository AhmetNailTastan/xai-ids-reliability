# Manuscript Patch Notes

Use these exact replacements when preparing the next manuscript revision.

## Code Availability replacement

```text
Code Availability: The experimental code, configuration files, result-generation notebooks, and CSV files corresponding to the reported tables are publicly available at https://github.com/AhmetNailTastan/xai-ids-reliability. Large raw datasets are not redistributed; download instructions and expected directory structure are provided in the repository. Serialized checkpoints can be regenerated from the provided code and are available upon reasonable request when file size prevents repository hosting.
```

## Data availability replacement

```text
The Edge-IIoTset and CIC-IDS-2017 datasets analysed in this study are publicly available: Edge-IIoTset via IEEE DataPort and CIC-IDS-2017 via the Canadian Institute for Cybersecurity (https://www.unb.ca/cic/datasets/ids-2017.html). The experimental code, configuration files, table CSVs, and figure-generation workflow are publicly available at https://github.com/AhmetNailTastan/xai-ids-reliability. Raw datasets are not redistributed; expected local paths are documented in the repository.
```

## LIME protocol replacement

```text
For the main paired SHAP-LIME comparisons, LIME was run with a fixed random seed to ensure reproducible paired comparisons. Intrinsic LIME stochasticity was evaluated separately by repeating explanations 10 times per instance under varying random seeds.
```

## RQ2 wording

```text
For model-variation consistency, all 50 trained instances of each algorithm were compared pairwise, yielding 1,225 model pairs per algorithm. Pairwise SHAP rankings were evaluated on a shared reference sample using Spearman correlation, Jaccard@5, and Jaccard@10.
```

## Preprocessing wording

```text
All scaling operations were fitted exclusively on the training partition of each fold and then applied to the corresponding test partition. No scaler was fitted on the full dataset before cross-validation.
```

## Result-table note

```text
The public repository stores manuscript table values as CSV files. Where the manuscript reports only means, standard-deviation fields in the CSV are marked for rerun rather than inferred.
```
