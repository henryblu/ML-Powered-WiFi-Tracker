# Passive CSI collection

Listens passively for packets on WiFi channel 6 by default. This channel can be changed in `main/main.cc` or by editing `sdkconfig`.
To use run `idf.py flash monitor` from a terminal.

This project collects CSI data by fetching CSI information from packets transmitted from the nearby WiFi router 

This repository targets the **ESP32-S3** microcontroller.
The `.vscode/settings.json` file configures the ESP-IDF extension for Visual Studio Code. It sets the `IDF_TARGET` to `esp32s3` so the project builds specifically for the ESP32-S3 chip.

Update the serial port in this file to match your development board. On Windows this is the `idf.portWin` field, while Linux/macOS use the equivalent setting (`idf.port`).

## Required tools

- ESP-IDF 5.x installed on your system.
- Python 3 environment for running `idf.py`.
- A compatible ESP32 board connected via USB.

## Setup

Set the `IDF_PATH` environment variable to point to your ESP-IDF installation and load the environment:

```bash
export IDF_PATH=/path/to/esp-idf
. "$IDF_PATH/export.sh"
```

Choose the serial/USB port where your board is connected:

```bash
export ESPPORT=/dev/ttyUSB0  # or COMx on Windows
```

## Build and flash

Compile the project and flash the firmware to your board:

```bash
idf.py build
idf.py -p "$ESPPORT" flash monitor
```

## Changing the WiFi channel

The default channel is configured through the `WIFI_CHANNEL` value. You can change it either in `main/main.cc` or by modifying the `CONFIG_WIFI_CHANNEL` setting in `sdkconfig`.
