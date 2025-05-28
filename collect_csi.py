import serial
import csv

# === Update your serial port and baud rate here ===
SERIAL_PORT = 'COM5'     # Replace with your actual port
BAUD_RATE = 921600       # This is standard for ESP32-CSI
CSV_FILE = 'Findings_1.csv'

try:
    ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
    print(f"Connected to {SERIAL_PORT} at {BAUD_RATE} baud.")
    
    with open(CSV_FILE, 'w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(['Raw CSI Data'])  # CSV Header

        while True:
            try:
                line = ser.readline().decode('utf-8', errors='ignore').strip()
                print(line)  
                if "CSI_DATA" in line:
                    print(line)  # Optional: view in console
                    writer.writerow([line])
            except KeyboardInterrupt:
                print("\nStopped by user.")
                break

except serial.SerialException as e:
    print(f"Serial error: {e}")
except Exception as e:
    print(f"Error: {e}")
finally:
    if 'ser' in locals() and ser.is_open:
        ser.close()
        print("Serial connection closed.")
