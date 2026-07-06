# XAI Reliability Evaluation for Intrusion Detection Systems

This repository contains the reproducibility package for the manuscript:

**"Evaluating the Reliability of Post-Hoc Explanations in Intrusion Detection: A Multi-Dimensional Assessment Framework for SHAP and LIME"**

Repository status: **Public research code accompanying the submitted manuscript.**
Citation metadata and archival DOI will be updated after publication.

## Canonical reproduction path

1. Prepare datasets according to `data/README.md`.
2. Install the environment using `environment.yml` or `requirements.txt`.
3. Run the canonical scripts in `src/` or the notebooks in `notebooks/`.
4. Use `notebooks/04_generate_paper_tables_figures.ipynb` to regenerate manuscript tables and figures from `results/*.csv`.

The original development notebooks are preserved in `legacy_notebooks/` for transparency. They are not the canonical reproduction path.

See `REPRODUCIBILITY_STATUS.md` for the current regeneration status and the items that require raw data/checkpoints before final archival release.

## Repository structure

```text
xai-ids-reliability/
├── README.md
├── LICENSE
├── CITATION.cff
├── requirements.txt
├── environment.yml
├── data/
│   └── README.md
├── metadata/
│   ├── edgeiiot_class_mapping.csv
│   ├── cicids2017_class_mapping.csv
│   ├── edgeiiot_feature_names.csv
│   └── cicids2017_feature_names.csv
├── results/
│   ├── table_01_class_distribution.csv
│   ├── table_05_classification_performance.csv
│   ├── table_06_rq1_perturbation_stability.csv
│   ├── table_07_lime_stochasticity.csv
│   ├── table_08_rq2_model_variation_pairwise.csv
│   ├── table_09_rq3_f1_xai_correlation.csv
│   ├── table_10_rq4_fidelity.csv
│   ├── table_12_statistical_tests.csv
│   ├── table_13_xai_score.csv
│   └── table_14_adversarial.csv
├── figures/
│   └── paper_figures/
├── notebooks/
├── legacy_notebooks/
├── src/
└── tests/
```

## Result files

The CSV files in `results/` correspond directly to the manuscript tables. The current manuscript numbers the statistical, XAI-Score, and adversarial tables as Tables 12, 13, and 14 respectively. Compatibility aliases are also included for older task notes that referred to them as Tables 11-13.

| Manuscript item | Repository file |
|---|---|
| Table 1 | `results/table_01_class_distribution.csv` |
| Table 5 | `results/table_05_classification_performance.csv` |
| Table 6 | `results/table_06_rq1_perturbation_stability.csv` |
| Table 7 | `results/table_07_lime_stochasticity.csv` |
| Table 8 | `results/table_08_rq2_model_variation_pairwise.csv` |
| Table 9 | `results/table_09_rq3_f1_xai_correlation.csv` |
| Tables 10-11 | `results/table_10_rq4_fidelity.csv` |
| Table 12 | `results/table_12_statistical_tests.csv` |
| Table 13 | `results/table_13_xai_score.csv` |
| Table 14 | `results/table_14_adversarial.csv` |

Table 8 is intended to use true pairwise comparisons: 50 trained instances per algorithm yield 1,225 model pairs per dataset/model combination. The current CSV marks row-level provenance: DNN/LSTM rows include standard deviations recomputed from saved pairwise SHAP arrays, while classical-model rows keep manuscript-reported means and are marked for a long-running checkpoint rerun.

`metadata/*_class_mapping.csv` is populated from the manuscript class-distribution table. `metadata/*_feature_names.csv` is exported from the prepared Edge-IIoT and CIC-IDS-2017 checkpoint matrices, not inferred by hand.

## Protocol clarifications

### Preprocessing

Canonical preprocessing in `src/data_preprocessing.py` is fold-safe: scalers and categorical encoders are fitted only on each training fold and then applied to the corresponding test fold. Legacy notebooks may contain earlier whole-matrix preprocessing cells and should not be cited as the final protocol.

### LIME

Main SHAP/LIME comparisons use 500 test instances per model. The main paired SHAP-LIME comparison uses a fixed seed for reproducibility, while intrinsic LIME stochasticity is evaluated separately using 30 instances x 10 repeats with varying seeds. LIME hyperparameter sensitivity is diagnostic and is not used for the main RQ1 table.

Checkpoint names are separated by experiment type:

```text
xai_edgeiiot_main500_{model}.pkl
xai_cicids2017_main500_{model}.pkl
lime_stochastic_edgeiiot_30x10_{model}.pkl
lime_stochastic_cicids2017_30x10_{model}.pkl
lime_grid_edgeiiot_{model}.pkl
lime_grid_cicids2017_{model}.pkl
```

### CPU and GPU execution

CPU execution is possible but slow for DNN/LSTM experiments. The canonical neural helper in `src/train_neural.py` checks CUDA availability before requesting a CUDA device name.

## Large files

Raw datasets and serialized checkpoints are not tracked by Git. Public dataset download instructions are provided in `data/README.md`. Large checkpoints can be regenerated from the provided code and may be shared upon reasonable request if file size prevents repository hosting.

## Setup

Python 3.12.3 is recommended.

```bash
conda env create -f environment.yml
conda activate xai-ids-reliability
```

or:

```bash
pip install -r requirements.txt
```

## Validation

```bash
pytest
```

`notebooks/04_generate_paper_tables_figures.ipynb` does not require raw datasets or model checkpoints; it reads `results/*.csv` and writes reviewer-facing artifacts to `figures/paper_figures/`.

## Citation

Please use `CITATION.cff`. Article DOI and final citation metadata will be added after publication.
