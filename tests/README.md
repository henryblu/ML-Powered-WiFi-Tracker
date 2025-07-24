# Test Suite

The `tests` folder contains unit tests for both the Python data collection
pipeline and the embedded components.

## File overview

- `conftest.py` - adjusts `sys.path` so the project root is importable during
  testing.
- `test_math.py` - verifies helper functions in `csi_estimator` such as
  `wavelength` and `weighted_phase_mean`.
- `test_serial_reader.py` - tests the asynchronous serial reader using a
  loopback transport.
- `test_parser.py` - exercises the CSI text parser.
- `test_e2e_async.py` - end-to-end test that feeds packets through matcher and
  estimator coroutines.
- `hardware/` - small ESP-IDF project containing a unity test for the time
  component.

## Usage

Run all Python tests with:

```bash
pytest -q
```

The hardware tests can be built and flashed using ESP-IDF:

```bash
cd tests/hardware
idf.py -p "$ESPPORT" flash monitor
```
