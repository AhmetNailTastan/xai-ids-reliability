# Data Directory

This directory is **not tracked by git** (see `.gitignore`).  
Place your dataset files and checkpoint sub-directories here before running the notebooks.

---

## Required datasets

### Edge-IIoT
- **Source:** IEEE DataPort — search for "Edge-IIoT-set"
- **File needed:** `ML-EdgeIIoT-dataset.csv`
- Place it at: `data/ML-EdgeIIoT-dataset.csv`

### CIC-IDS-2017
- **Source:** Canadian Institute for Cybersecurity
  https://www.unb.ca/cic/datasets/ids-2017.html
- **Files needed:** the Friday, Monday, Tuesday, Thursday, Wednesday CSV files
  (e.g. `Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv`, etc.)
- Place them at: `data/cicids2017_raw/`

---

## Expected checkpoint directory structure

After running the notebooks, intermediate results (pickled DataFrames and arrays)
will be saved here automatically:

```
data/
├── ML-EdgeIIoT-dataset.csv
├── cicids2017_raw/
│   ├── Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv
│   ├── Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv
│   ├── Friday-WorkingHours-Morning.pcap_ISCX.csv
│   ├── Monday-WorkingHours.pcap_ISCX.csv
│   ├── Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv
│   ├── Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv
│   ├── Tuesday-WorkingHours.pcap_ISCX.csv
│   └── Wednesday-workingHours.pcap_ISCX.csv
├── edgeiiot_checkpoints/    <- created by Notebook 01
├── cicids2017_checkpoints/  <- created by Notebook 02
└── dnn_lstm_checkpoints/    <- created by Notebook 03
```

> **Note:** Serialized checkpoints and trained model files are available upon
> formal request to the corresponding author.
