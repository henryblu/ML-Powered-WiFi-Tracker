# Passive CSI collection

Listens passively for packets on WiFi channel 6. 
This channel can be changed in `main/main.cc` depending on the channel of the device you wish to passively listen for.
To use run `idf.py flash monitor` from a terminal.

This project collects CSI data by fetching CSI information from packets transmitted from the nearby WiFi router 

This repository targets the **ESP32-S3** microcontroller.
The `.vscode/settings.json` file configures the ESP-IDF extension for Visual Studio Code. It sets the `IDF_TARGET` to `esp32s3` so the project builds specifically for the ESP32-S3 chip.

Update the serial port in this file to match your development board. On Windows this is the `idf.portWin` field, while Linux/macOS use the equivalent setting (`idf.port`).

