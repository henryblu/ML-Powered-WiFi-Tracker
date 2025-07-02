# CSI Pipeline

This project processes Channel State Information (CSI) from two ESP32-S3 devices to estimate Wi-Fi Angle-of-Arrival (AoA).
It reads raw lines from two serial ports, parses frames, pairs matching packets, performs AoA estimation and logs results to a rotating CSV file.

## Setup

Install the runtime dependencies using `dev-requirements.txt`:

```bash
python -m pip install -r dev-requirements.txt
```

## Usage

Run the pipeline from the repository root:

```bash
python scripts/csi_pipeline.py \
  --port-master /dev/ttyUSB0 --port-worker /dev/ttyUSB1 \
  --output aoa.csv
```

or on windows:

```bash
python scripts/csi_pipeline.py  --port-master COM3 --port-worker COM4  --output aoa.csv
```

The serial readers use standard ``pyserial`` threads for compatibility with
Windows and POSIX. No `PYTHONPATH` tweak is required—the script adjusts the
path automatically.

Environment variables prefixed with ``CSI_PIPELINE_`` are read first, then
overridden by CLI options.

## Development

Run the test suite with `pytest`:

```bash
pytest -q
```

The package adheres to `black` formatting and `ruff` linting.
