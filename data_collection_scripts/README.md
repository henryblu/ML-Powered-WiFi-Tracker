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

The modules in this package can also be used directly. For instance,
`csi_reader.py` opens a serial port and yields raw CSI lines that may be parsed
with `csi_parser.py`:

```python
from data_collection_scripts.csi_reader import CSISerialReader
from data_collection_scripts.csi_parser import parse_csi_line

reader = CSISerialReader('/dev/ttyUSB0')
async for line in reader:
    packet = parse_csi_line(line)
    print(packet)
```

## Development

Run the test suite with `pytest`:

```bash
pytest -q
```

The package adheres to `black` formatting and `ruff` linting.

## nn_exploration

.py
Streamlined script used to test different neural network architectures with a single run.
Edit the `dataset` path near the top of `ML_Scripts/nn_exploration.py` to point
to one of the CSV files in `training_data/` and run:

```bash
python ML_Scripts/nn_exploration.py
```

.ipynb
Notebook version of the script for interactive debugging and iteration.

## File overview

- `__init__.py` - package marker so the scripts can be imported during testing.
- `csi_reader.py` - asynchronous serial reader that yields raw CSI lines from an
  ESP32 device.
- `csi_parser.py` - converts the raw text produced by the firmware into
  structured `CSIPacket` objects.
- `csi_matcher.py` - pairs CSI packets from master and worker receivers based on
  sequence numbers.
- `csi_estimator.py` - computes the Angle-of-Arrival from matched packet pairs.
- `csi_estimator_new.py` - alternative implementation of the estimator using
  direct parsing of the raw CSI strings.
- `csi_logger.py` - CSV logger with automatic file rotation for long
  experiments.
- `csi_pipeline.py` - command line entry point that wires together reader,
  parser, matcher and estimator to produce AoA measurements.
- `csi_stats.py` - optional coroutine that prints running statistics about the
  matcher queues.
