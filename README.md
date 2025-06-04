# Passive CSI collection

Listens passively for packets on WiFi channel 6. 
This channel can be changed in `main/main.cc` depending on the channel of the device you wish to passively listen for.
To use run `idf.py flash monitor` from a terminal.

This project collects CSI data by fetching CSI information from packets transmitted from the nearby WiFi router 
