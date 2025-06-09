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

If your ESP32 board has two USB-C ports, connect to the **left** port for
flashing and monitoring. The right port is typically for USB-OTG and does not
provide the USB-to-serial connection used by `idf.py`. Using the wrong port
can result in null responces when monitoring the device or trying to pull
the csi information. 

## Build and flash

Compile the project and flash the firmware to your board:

```bash
idf.py build
idf.py -p "$ESPPORT" flash monitor
```

## Changing the WiFi channel

The default channel is configured through the `WIFI_CHANNEL` value. You can change it either in `main/main.cc` or by modifying the `CONFIG_WIFI_CHANNEL` setting in `sdkconfig`.

## ESP32-CSI-Tool components

This project reuses helper components from the [ESP32-CSI-Tool](https://github.com/StevenMHernandez/ESP32-CSI-Tool) repository. The required header files are included in the `_components` directory. If you need to update or re-fetch these files, clone the CSI tool repository and copy its `_components` folder into the root of this project:

```bash
git clone https://github.com/StevenMHernandez/ESP32-CSI-Tool.git
cp -r ESP32-CSI-Tool/_components .
```

Alternatively you can track the dependency using a git submodule:

```bash
git submodule add https://github.com/StevenMHernandez/ESP32-CSI-Tool.git external/ESP32-CSI-Tool
```

The headers provide the implementations for `nvs_component`, `sd_component`, `csi_component`, `time_component`, and `input_component` used by `main/main.cc`.
