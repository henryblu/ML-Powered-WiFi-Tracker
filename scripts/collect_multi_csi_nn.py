#Use this file to display collected CSI from multiple devices in a proper format 
import argparse
import csv
import serial
import threading
from typing import List

target_mac = "3A:FB:99:B4:E0:FC" # replace with your target device mac

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect CSI data from multiple serial devices")
    parser.add_argument(
        "--device",
        action="append",
        required=True,
        help="Specify device as ID:PORT",
    )
    parser.add_argument(
        "--baud",
        type=int,
        default=921600,
        help="Serial baud rate (default: 921600)",
    )
    parser.add_argument(
        "--output",
        default="multi_csi.csv",
        help="Output CSV file",
    )
    return parser.parse_args()

def open_serial(port: str, baud: int) -> serial.Serial:
    return serial.Serial(port, baud, timeout=1)

# Indices: ROLE: 1, MAC: 2, RSSI: 3, timestamp: 18, Seq. num: 25, CSI: 26, PHY_TIME: 19
indices = [1,2,3,18,19,25,26]
def parse_csi_line(line: str) -> List[str]:
    """Parse CSI data line into individual components"""
    if line.startswith("CSI_DATA"):
        parts = (line.split(","))
        if len(parts) >= 4:  # Adjust based on your actual CSI format
            MAC = parts[2]
            if MAC == target_mac:
                print("MAC:",parts[2]) 
                return [parts[i] for i in indices]  # Skip the "CSI_DATA" prefix
    return []

def reader(device_id: str, ser: serial.Serial, writer: csv.writer, lock: threading.Lock, stop_evt: threading.Event):
    while not stop_evt.is_set():
        try:
            line = ser.readline().decode("utf-8", errors="ignore").strip()
            if line:
                csi_components = parse_csi_line(line)
                if csi_components:
                    with lock:
                        writer.writerow([device_id] + csi_components)
        except serial.SerialException:
            break

def main() -> None:
    args = parse_args()
    devices: List[tuple[str, str]] = []
    for dev in args.device:
        if ":" not in dev:
            raise SystemExit(f"Invalid --device format: {dev}. Expected ID:PORT")
        device_id, port = dev.split(":", 1)
        devices.append((device_id, port))
    serials = []
    stop_evt = threading.Event()
    lock = threading.Lock()
    threads = []

    try:
        # Create directory if it doesn't exist
        #os.makedirs(os.path.dirname(args.output), exist_ok=True)
        # output_dir = os.path.dirname(args.output) or "."  # Use current dir if no directory specified
        # os.makedirs(output_dir, exist_ok=True)
        location = 'hallway'
        filename = '../data/home_' + location + '_multi.csv'
        with open(filename, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            # Write header row (adjust columns based on your CSI format)
            headers_all = ([
                "device_id",
                "role", "mac", "rssi", "rate", "sig_mode", "mcs",
                "bandwidth", "smoothing", "not_sounding", "aggregation",
                "stbc", "fec_coding", "sgi", "noise_floor", "ampdu_cnt",
                "channel", "secondary_channel", "timestamp", "phy_timestamp",
                "ant", "sig_len", "rx_state", "real_time_set", "len",
                "seq_ctrl", "csi_data"
            ])
            selected_i = [0,1,2,3,18,19,25,26]
            headers_selected = [headers_all[i] for i in selected_i]
            writer.writerow(
                headers_selected
            )
            
            for device_id, port in devices:
                ser = open_serial(port, args.baud)
                serials.append(ser)
                t = threading.Thread(
                    target=reader, args=(device_id, ser, writer, lock, stop_evt), daemon=True
                )
                t.start()
                threads.append(t)

            while True:
                for t in threads:
                    t.join(0.1)
    except KeyboardInterrupt:
        print("\nStopping...")
        stop_evt.set()
    except PermissionError as e:
        print(f"\nPermission denied: {e}")
        print("Try specifying a different output path or check write permissions")
        stop_evt.set()
    finally:
        for ser in serials:
            if ser.is_open:
                ser.close()

if __name__ == "__main__":
    main()
