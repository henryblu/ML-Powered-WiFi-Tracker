# Passive CSI collection

Listens passively for packets on channel 3 (same channel as both active_ap and active_sta). This channel can be changed in `main/main.c` depending on the channel of the device you wish to passively listen for.

To use run `idf.py flash monitor` from a terminal.
