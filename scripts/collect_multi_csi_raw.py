import argparse
import csv
import serial
import threading
from typing import List


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
        default="../data/multi_csi.csv",
        help="Output CSV file",
    )
    return parser.parse_args()


def open_serial(port: str, baud: int) -> serial.Serial:
    return serial.Serial(port, baud, timeout=1)


def reader(device_id: str, ser: serial.Serial, writer: csv.writer, lock: threading.Lock, stop_evt: threading.Event):
    while not stop_evt.is_set():
        try:
            line = ser.readline().decode("utf-8", errors="ignore").strip()
            if line:
                with lock:
                    writer.writerow([device_id, line])
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
        with open(args.output, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["device_id", "raw_csi"])
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
    finally:
        for ser in serials:
            if ser.is_open:
                ser.close()


if __name__ == "__main__":
    main()
