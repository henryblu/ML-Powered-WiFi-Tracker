# ESP32 Timestamp Synchronization Examples

This folder contains simple ESP-IDF examples that demonstrate how two ESP32 modules can
exchange timestamps using ESP-NOW and apply an offset to correct their local clock.

- `sync_master.c` – periodically broadcasts the current time in microseconds.
- `sync_slave.c`  – receives the master's timestamp, computes the difference
  between the received value and its local `esp_timer_get_time()` and stores
  it as an offset. The function `get_synced_time()` returns the corrected time.

These examples assume that both devices are already configured with a common
crystal source or hardware clock. They show how to perform a simple software
correction for any remaining skew.

To build either example, replace the contents of `main/main.cc` with the desired
file or create a new project using the provided source code.
