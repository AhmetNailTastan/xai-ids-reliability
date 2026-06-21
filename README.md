# XAI Reliability Evaluation for Intrusion Detection Systems

Experimental code for evaluating the reliability of SHAP and LIME explanations
across multiple IDS datasets and model families.

> **Repository status:** Private — will be made public upon paper acceptance.
> Citation information will be added here after publication.

---

## Repository structure

```
xai-ids-reliability/
├── notebooks/
│   ├── 01_EdgeIIoT_ML_XAI.ipynb          Classical ML models on Edge-IIoT
│   ├── 02_CICIDS2017_ML_XAI.ipynb        Classical ML models on CIC-IDS-2017
│   ├── 03_DNN_LSTM_XAI.ipynb             DNN and LSTM models on both datasets
│   └── 04_Cross_Dataset_Comparison.ipynb Cross-dataset comparison figures
├── figures/                               Generated result figures
├── data/                                  Datasets and checkpoints (not tracked)
│   └── README.md                          Data setup instructions
├── requirements.txt
└── .gitignore
```

---

## Setup

**Python 3.12.3** is recommended (exact version used in the experiments).

```bash
pip install -r requirements.txt
```

GPU (CUDA) is required for Notebook 03 (DNN/LSTM). The code falls back to CPU
if CUDA is unavailable, but training will be significantly slower.

---

## Datasets

See [`data/README.md`](data/README.md) for download links and the expected
directory layout.

- **Edge-IIoT:** IEEE DataPort
- **CIC-IDS-2017:** https://www.unb.ca/cic/datasets/ids-2017.html

---

## Running the notebooks

Open each notebook in Jupyter Lab or Jupyter Notebook.
Run them in order, as later notebooks depend on checkpoints saved by earlier ones.

1. `01_EdgeIIoT_ML_XAI.ipynb` — trains 7 classical models on Edge-IIoT and runs SHAP/LIME
2. `02_CICIDS2017_ML_XAI.ipynb` — trains 7 classical models on CIC-IDS-2017 and runs SHAP/LIME
3. `03_DNN_LSTM_XAI.ipynb` — trains DNN and LSTM on both datasets and runs SHAP/LIME
4. `04_Cross_Dataset_Comparison.ipynb` — aggregates results and generates comparison figures

**Before running any notebook**, update the `BASE_DIR` variable in the
**Configuration** cell (second cell of each notebook) to point to your local
data directory.

---

## Hardware used

- CPU: AMD Ryzen 9 5900X
- RAM: 32 GB DDR4
- GPU: NVIDIA RTX 3080 (10 GB)

Results may differ slightly on different hardware, particularly the runtime
component of the XAI-Score.

---

## Citation

*(Will be updated upon publication)*
