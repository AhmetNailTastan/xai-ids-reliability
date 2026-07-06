# Data setup

Raw datasets and model checkpoints are intentionally not tracked by Git.

Prepare the local data directory as follows:

```text
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
├── edgeiiot_checkpoints/
├── cicids2017_checkpoints/
└── dnn_lstm_checkpoints/
```

Expected canonical paths:

```python
BASE_DIR = Path("data")
EDGE_RAW_PATH = BASE_DIR / "ML-EdgeIIoT-dataset.csv"
RAW_CIC_DIR = BASE_DIR / "cicids2017_raw"
VERI_KLASORU = str(RAW_CIC_DIR)
```

Dataset sources:

- Edge-IIoTset: IEEE DataPort.
- CIC-IDS-2017: https://www.unb.ca/cic/datasets/ids-2017.html

Do not commit raw CSVs, serialized checkpoints, or generated pickle files. Use `metadata/` for lightweight class and feature metadata and `results/` for manuscript table CSVs.
