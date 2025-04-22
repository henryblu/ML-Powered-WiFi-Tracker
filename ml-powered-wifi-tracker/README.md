## 🛠 Project Configuration (ESP32‑S3 DevKitC‑1 N16R8)

This project is built using [PlatformIO](https://platformio.org/) in Visual Studio Code, targeting the **Diymore ESP32‑S3 DevKitC‑1 N16R8** board (16 MB flash / 8 MB PSRAM).

### Folder Structure

```
.
├── .pio/                 # PlatformIO build output
├── .vscode/             # VS Code settings
├── boards/              # Custom board definition for ESP32-S3 N16R8
├── include/             # Header files
├── lib/                 # External libraries
├── src/                 # Main source code (e.g. main.cpp)
├── test/                # PlatformIO test files
└── platformio.ini       # Project configuration
```

### Board Setup

This board (N16R8) was not listed by default in PlatformIO. A custom board manifest was added manually:

1. **Created a custom `boards/` directory**:
   ```
   mkdir boards
   ```

2. **Downloaded the official `esp32-s3-devkitc-1-n16r8v.json`** board file:
   - From: [esp32-s3-devkitc-1-n16r8v.json](https://raw.githubusercontent.com/platformio/platform-espressif32/develop/boards/esp32-s3-devkitc-1-n16r8v.json)
   - Saved as: `boards/esp32-s3-devkitc-1-n16r8v.json`

3. **Updated `platformio.ini`** to use this board:

   ```ini
   [env:esp32-s3-devkitc-1-n16r8v]
   platform = espressif32
   board = esp32-s3-devkitc-1-n16r8v
   framework = arduino
   monitor_speed = 115200
   ```

### Uploading to the Board

First try both ports on the esp32r32-s3 devkitc-1 n16r8v board. 

If this dosn't work, the ESP32‑S3 sometimes requires manual intervention to enter bootloader mode:

1. Connect the board via USB.
2. **Hold the `BOOT` button** on the board.
3. In VS Code, run **Upload** via the PlatformIO toolbar or:
   ```
   pio run --target upload
   ```
4. Release `BOOT` once uploading begins.

If this dosn't work, try the follo