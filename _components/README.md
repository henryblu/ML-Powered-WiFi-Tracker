## Global Components for each ESP32 Sub-project

The files in this directory allow us to create reusable modules that can be
shared across all ESP-IDF projects in this repository.

### File overview

- `component.mk` - declares the include directory so these headers can be used by
  the firmware build system.
- `csi_component.h` - helper functions for configuring the Wi-Fi interface and
  capturing CSI frames from the ESP32.
- `input_component.h` - simple serial input handler used to accept commands such
  as timestamp updates from the host.
- `nvs_component.h` - initializes the non‑volatile storage (NVS) subsystem.
- `sd_component.h` - mounts the SD card and exposes helpers for writing
  collected data to disk.
- `sockets_component.h` - lightweight socket wrappers used by the examples.
- `sync_component.h` - implements ESP‑NOW based time synchronisation between a
  master and worker device.
- `time_component.h` - utilities for parsing and formatting timestamps that are
  shared across projects.

## Usage

Add this directory to `EXTRA_COMPONENT_DIRS` when invoking `idf.py` or copy the
headers into your project's `components` folder. Once included, your firmware
can call the helpers provided here. For example, the CSI component exposes
`csi_init()` which configures Wi‑Fi and registers a callback for CSI frames.
