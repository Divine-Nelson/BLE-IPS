import csv
import matplotlib.pyplot as plt
import glob

# Find all filtered CSV files
CSV_FILES = glob.glob("filtered_kalman_test/*.txt")

for file_path in CSV_FILES:
    rssi_values = []

    with open(file_path, 'r') as file:
        reader = csv.DictReader(file)
        for row in reader:
            try:
                rssi_values.append(float(row["Filtered_RSSI"]))
            except Exception as e:
                print(f"Skipping row in {file_path} due to error: {e}")

    # Generate x-axis as index (1, 2, 3, ...)
    timestamps = list(range(len(rssi_values)))

    # Plotting
    plt.figure(figsize=(10, 5))
    plt.plot(timestamps, rssi_values, marker='o', linestyle='-', color='blue', label='Filtered RSSI')
    plt.title(f"Filtered RSSI - {file_path.split('/')[-1]}")
    plt.xlabel("Sample Index")
    plt.ylabel("RSSI (dBm)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()
