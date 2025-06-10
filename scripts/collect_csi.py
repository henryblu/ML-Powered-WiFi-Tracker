import serial
import csv

# === Update your serial port and baud rate here ===
SERIAL_PORT = 'COM3'     # Replace with your actual port
BAUD_RATE = 921600       # This is standard for ESP32-CSI
CSV_FILE = '../data/Findings_1.csv'

# CSI Data headers according to the format and sequence 
HEADERS = [
    "type", "role", "mac", "rssi", "rate", "sig_mode", "mcs", "bandwidth",
    "smoothing", "not_sounding", "aggregation", "stbc", "fec_coding", "sgi",
    "noise_floor", "ampdu_cnt", "channel", "secondary_channel", "local_timestamp",
    "ant", "sig_len", "rx_state", "real_time_set", "real_timestamp", "len", "CSI_DATA"
]

try:
    ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
    print(f"Connected to {SERIAL_PORT} at {BAUD_RATE} baud.")
    
    with open(CSV_FILE, 'w', newline='', encoding='utf-8') as csv_file:
        writer = csv.writer(csv_file)
        # Write headers
        writer.writerow(HEADERS)
        last_line = None
        while True:
            try:
                line = ser.readline().decode('utf-8', errors='ignore').strip()
                if "CSI_DATA" in line:
                    if line != last_line:
                        print(line)
                        last_line = line
                        # Find the position where CSI_DATA starts
                        csi_start = line.find('[')
                        if csi_start == -1:
                            continue  # Skip if no CSI data found
                            
                        # Split the part before CSI_DATA into fields
                        prefix = line[:csi_start]
                        fields = prefix.split(',')[:25]  # Get up to 25 fields
                        
                        # The len value is the last field before CSI_DATA
                        len_value = fields[-1].strip() if len(fields) >= 25 else ''
                        
                        # The CSI data is everything from '[' onwards
                        csi_data = line[csi_start:]
                        
                        # If we have exactly 25 fields including len in prefix
                        if len(fields) >= 24:
                            # First 24 fields are fields[0] to fields[23]
                            # 25th field is len_value
                            row_data = fields[:24] + [len_value] + [csi_data]
                        else:
                            # Handle case where we don't have all fields
                            row_data = fields + [''] * (25 - len(fields)) + [csi_data]
                        
                        # Write to CSV
                        writer.writerow(row_data)
                        
            except KeyboardInterrupt:
                print("\nStopped by user.")
                break
            except Exception as e:
                print(f"Error processing line: {e}")
                continue

except serial.SerialException as e:
    print(f"Serial error: {e}")
except Exception as e:
    print(f"Error: {e}")
finally:
    if 'ser' in locals() and ser.is_open:
        ser.close()
        print("Serial connection closed.")
