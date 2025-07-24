# Training Data

CSV files containing captured CSI samples and ground truth labels used by the
machine learning scripts.

## File overview

- `02-07-2025-training-data-set.csv` - example dataset recorded on 02 July 2025.
- `11-07-2025-training-data-set.csv` - dataset recorded on 11 July 2025.

Each row contains the timestamp, MAC address, sequence number, estimated AoA and
raw IQ samples for the master and worker receivers.

## Usage

Use `data_collection_scripts/csi_pipeline.py` to generate new datasets. Specify
an output CSV file inside this folder:

```bash
python scripts/csi_pipeline.py --port-master /dev/ttyUSB0 --port-worker /dev/ttyUSB1 \
  --output training_data/new-session.csv
```

The resulting file can then be fed into the scripts under `ML_Scripts/` for
training or evaluation.
