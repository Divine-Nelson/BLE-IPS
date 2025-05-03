import os
import csv
import pandas as pd
from collections import defaultdict

# Paths to RSSI data
RAW_PATH = "Ref_files"
MEDIAN_PATH = "filtered_median"
KALMAN_PATH = "filtered_kalman"
METADATA_FILE = r"CSV\New_RF.csv"

# Load metadata
metadata = pd.read_csv(METADATA_FILE)

def build_fingerprint_db(rssi_dir, output_file):
    fingerprint_rows = []
    all_macs = set()

    for _, row in metadata.iterrows():
        rp_id = row['ID']
        x = row['X']
        y = row['Y']
        file_name = row['File']

        # Construct path to file depending on directory
        if rssi_dir == RAW_PATH:
            full_path = os.path.join(rssi_dir, f"{file_name}.txt")
            delimiter = '\t'
        elif rssi_dir == MEDIAN_PATH:
            full_path = os.path.join(rssi_dir, f"{file_name}_medianfilter.txt")
            delimiter = ','  # median-filtered uses commas
        elif rssi_dir == KALMAN_PATH:
            full_path = os.path.join(rssi_dir, f"{file_name}_filtered.txt")
            delimiter = ','  # kalman-filtered uses commas
        else:
            continue

        if not os.path.isfile(full_path):
            print(f"Missing file: {full_path}, skipping...")
            continue

        mac_rssi = defaultdict(list)

        with open(full_path, newline='') as f:
            reader = csv.DictReader(f, delimiter=delimiter)
            for line in reader:
                mac = line.get('Device Address')
                rssi = None
                if 'Filtered_RSSI' in line:
                    rssi = line.get('Filtered_RSSI')
                elif 'RSSI' in line:
                    rssi = line.get('RSSI')

                if mac and rssi:
                    try:
                        mac_rssi[mac].append(float(rssi))
                    except ValueError:
                        continue

        if not mac_rssi:
            print(f"No valid RSSI data in {full_path}")
            continue

        averaged_rssi = {mac: sum(vals)/len(vals) for mac, vals in mac_rssi.items()}
        all_macs.update(averaged_rssi.keys())
        averaged_rssi.update({'RP_ID': rp_id, 'X': x, 'Y': y})
        fingerprint_rows.append(averaged_rssi)

    all_macs_sorted = sorted(all_macs)
    fieldnames = ['RP_ID', 'X', 'Y'] + all_macs_sorted

    with open(output_file, 'w', newline='') as out:
        writer = csv.DictWriter(out, fieldnames=fieldnames)
        writer.writeheader()
        for row in fingerprint_rows:
            writer.writerow({key: row.get(key, -100) for key in fieldnames})

    return output_file

# Run for all three types
raw_fp = build_fingerprint_db(RAW_PATH, "fingerprints_raw.csv")
median_fp = build_fingerprint_db(MEDIAN_PATH, "fingerprints_median.csv")
kalman_fp = build_fingerprint_db(KALMAN_PATH, "fingerprints_kalman.csv")

print(raw_fp, median_fp, kalman_fp)
