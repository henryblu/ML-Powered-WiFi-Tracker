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

## Pre-build steps

Before compiling the firmware or running the Python utilities, install the
required packages (preferably inside a virtual environment):

```bash
python -m pip install -r dev-requirements.txt
```

## Build and flash

Compile the project and flash the firmware to your board:

```bash
idf.py build
idf.py -p "$ESPPORT" flash monitor
```

## Changing the WiFi channel

The default channel is configured through the `WIFI_CHANNEL` value. You can change it either in `main/main.cc` or by modifying the `CONFIG_WIFI_CHANNEL` setting in `sdkconfig`.

## Device ID

Each board should be built with a unique identifier using the `CONFIG_DEVICE_ID` option. Run `idf.py menuconfig` and set a different string for every device so CSI data can be traced back to the correct board.

## Device role

Time synchronisation relies on two boards. One acts as the **master** and
broadcasts its clock using ESP-NOW. The other is the **worker** and adjusts its
local time based on those messages so calls to `get_synced_time()` line up with
the master's timestamps.

Choose the role for each board in `idf.py menuconfig`:

1. Run `idf.py menuconfig`.
2. Navigate to **ESP32 CSI Tool Config → Device role**.
3. Select **Master** or **Worker**.

Build and flash the firmware after setting the role on each board:

```bash
idf.py -p "$ESPPORT" flash
```

Reconfigure and re-flash if you later switch a board between master and worker.
The default role is **Worker**.

## Capturing additional frame types

`main/main.cc` configures the ESP32 in promiscuous mode. By default the firmware only listened for data frames, which limited how often CSI callbacks were triggered. The code now enables `WIFI_PROMIS_FILTER_MASK_ALL` so CSI is reported for management and control frames as well. This increases the CSI sampling rate when there is little data traffic on the monitored channel.

## Capturing the full CSI payload

By default, only the legacy long training field (LLTF) is stored for each packet. To include the high throughput long training field (HT-LTF) and obtain more data per packet, set `CONFIG_SHOULD_COLLECT_ONLY_LLTF` to `n` using `idf.py menuconfig` or by editing `sdkconfig` directly.
i.e. disabling it. 
However, it is recommended to Enable `y` if:
1. You want stable, consistent CSI data without missing samples.
2. You don’t need MIMO or 40MHz channel analysis.
3. You want lower bandwidth usage (important for serial/SD logging).
   
Note that printing the full CSI over serial can reduce the effective sampling rate if the serial link becomes saturated.

## CSI output fields

Each logged line begins with metadata followed by the CSI payload. The header printed by `_print_csi_csv_header()` lists the fields in order. A `seq_ctrl` column has been added just before the CSI data to store the IEEE 802.11 sequence control value from the received frame.

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

## Modifying the format of CSI 
Head to the csi_component.h file where _wifi_csi_cb() function caters the pattern of the CSI data. Following modifications can be made according to the application and need: 
1. Reducing/Adding fields
2. Choosing CSI Type [There are 3 options available: CSI_RAW, CSI_AMPLITUDE and CSI_PHASE]. The default type is CSI_RAW

## Collecting CSI from multiple devices

Use `collect_multi_csi.py` to read CSI data from several ESP32 boards at once. Each device is specified as an `ID:PORT` pair. Lines from all devices are saved to one CSV file and are prefixed with the given ID.

Example:

```bash
python3 collect_multi_csi.py --device STA1:/dev/ttyUSB0 --device STA2:/dev/ttyUSB1 --output csi_log.csv
```


### CSV format v2

CSI lines are written in comma-separated form. Version 2 replaces the
previous `local_timestamp` and `real_timestamp` fields with a single
`timestamp` column and adds `phy_timestamp` which exposes the Wi-Fi
hardware TSF value. The `timestamp` field reports epoch microseconds once
the system clock is set. Before that, masters use monotonic microseconds
since boot and workers use the value from `get_synced_time()`.

The `real_time_set` column now flags whether `timestamp` contains epoch
time (`1`) or a monotonic/synchronized value (`0`). Downstream tools that
expected the old `real_timestamp` column should switch to `timestamp` and
may need to update column indices accordingly.

### simple_nn

Simple neural network that uses CSI, either from one or 2 ESPs, as features, and the measurement location as the label. "simple_nn_multi.ipynb" is for data collected from 2 ESPs, use "collect_multi_csi_nn.py" to collect the data for the neural network. "collect_multi_csi_nn.py" is a derivative from "collect_multi_csi_raw.py" and works similarly. However, collect the CSI data from different locations into seperate CSV files, and modify the "target_mac" parameter to the MAC address that you want to collect CSI from.



