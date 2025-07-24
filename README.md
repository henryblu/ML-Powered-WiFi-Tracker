# ML-Powered WiFi Tracker

This project collects Wi-Fi Channel State Information (CSI) with two ESP32-S3 boards and processes it to estimate the angle of arrival of a signal. The repository contains the firmware for the microcontrollers, a Python pipeline for logging and estimating AoA, and several machine learning experiments.

## 1. Hardware setup

1. Connect **two** ESP32-S3 boards to your computer. If your board exposes two USB-C ports use the **left** port for flashing and monitoring.
2. Determine the serial ports for each device (e.g. `/dev/ttyUSB0` and `/dev/ttyUSB1` on Linux or `COM3` and `COM4` on Windows).
3. Each board must be assigned a unique ID and a role:
   - Run `idf.py menuconfig`.
   - Navigate to **ESP32 CSI Tool Config → Device ID** and set a different string on each board.
   - Under **Device role** choose one board as **Master** and the other as **Worker**.
4. Save the configuration and exit `menuconfig`.

## 2. Building and flashing the firmware

1. Install the ESP‑IDF v5.x toolchain and export the environment variables:

   ```bash
   export IDF_PATH=/path/to/esp-idf
   . "$IDF_PATH/export.sh"
   ```
2. Install the Python utilities (preferably inside a virtual environment):

   ```bash
   python -m pip install -r dev-requirements.txt
   ```
3. Set the serial port for the board you want to flash:

   ```bash
   export ESPPORT=/dev/ttyUSB0  # or COMx on Windows
   ```
4. Build and flash the firmware:

   ```bash
   idf.py build
   idf.py -p "$ESPPORT" flash monitor
   ```
5. Repeat the last two steps for the second board (adjusting `ESPPORT`).

## 3. Running the CSI pipeline

After flashing both devices you can collect and process CSI streams in real time.

1. Install the runtime dependencies:

   ```bash
   python -m pip install -r requirements.txt
   ```
2. Run the pipeline specifying the master and worker ports and an output CSV file:

   ```bash
   python data_collection_scripts/csi_pipeline.py \
       --port-master /dev/ttyUSB0 --port-worker /dev/ttyUSB1 \
       --output aoa_log.csv
   ```
3. The script continuously logs matched packets and estimated AoA values to the chosen file. Press `Ctrl+C` to stop.

## 4. Using the machine learning code

The folder `ML_Scripts/` contains a variety of experiments that operate on the CSV files produced by the pipeline. Update the dataset path near the top of a script and run it with Python. For example:

```bash
python ML_Scripts/NN_RawCSI.py
```

Notebooks such as `simple_nn.ipynb` can be opened in Jupyter for interactive exploration. Example datasets are provided under `training_data/`.

## 5. Report
The report summarizing the project findings is available in `/report/WIFI_Source_Tracker.pdf`. It includes a detailed account on the motivation, hardware setup, data collection process, and machine learning results.

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
