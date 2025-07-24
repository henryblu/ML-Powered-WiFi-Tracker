# Firmware Source

This directory contains the ESP-IDF project used to collect CSI packets.
The build system treats this folder as the `main` component.

## File overview

- `main.cc` - application entry point configuring Wi-Fi, starting CSI capture and
  handling configuration flags.
- `CMakeLists.txt` - registers the source file and include directory with the
  ESP-IDF CMake build.
- `component.mk` - legacy makefile used when building with the old build system.
- `Kconfig.projbuild` - project configuration options selectable via
  `menuconfig`.

## Usage

1. Ensure the ESP-IDF tools are in your `PATH` by sourcing `export.sh` from your
   ESP-IDF installation.
2. Connect your ESP32-S3 board and set the serial port:

   ```bash
   export ESPPORT=/dev/ttyUSB0  # or COMx on Windows
   ```

3. Build and flash the firmware:

   ```bash
   idf.py build
   idf.py -p "$ESPPORT" flash monitor
   ```

Use `idf.py menuconfig` to adjust options like the Wi‑Fi channel and device ID
before building.
