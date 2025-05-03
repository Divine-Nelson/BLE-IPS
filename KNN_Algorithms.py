import os
import csv
import cv2
import pandas as pd
import numpy as np
from collections import defaultdict
from math import sqrt

TEST_METADATA_FILE = "CSV/Test_RFs.csv"
K = 3
image_path = "./image/New_RPs.png"

def load_fingerprint_db(path):
    return pd.read_csv(path)

def load_test_data(file_path):
    mac_rssi = defaultdict(list)

    # Heuristic for delimiter
    delimiter = '\t' if 'Test_files' in file_path or file_path.endswith("RS1.txt") else ','

    with open(file_path, newline='') as f:
        reader = csv.DictReader(f, delimiter=delimiter)
        #print(f"\nReading {file_path} with delimiter: {'TAB' if delimiter == '\t' else 'COMMA'}")
        for i, line in enumerate(reader):
            #print(f"Line {i}: keys={list(line.keys())}")  # Print column names
            try:
                mac = line.get('Device Address')
                rssi = float(line.get('Filtered_RSSI') or line.get('RSSI'))
                mac_rssi[mac].append(rssi)
            except (ValueError, TypeError):
                print(f"  Skipped line due to error.")
                continue

    if not mac_rssi:
        print(f"No valid RSSI data in {file_path}")
    averaged = {mac: np.mean(rssis) for mac, rssis in mac_rssi.items()}
    return averaged




def knn_predict(test_sample, fingerprint_df, k=3):
    distances = []
    for _, row in fingerprint_df.iterrows():
        ref_vec = []
        test_vec = []
        for mac in fingerprint_df.columns[3:]:  # skip RP_ID, X, Y
            ref_vec.append(row[mac])
            test_vec.append(test_sample.get(mac, -100))
        dist = np.linalg.norm(np.array(ref_vec) - np.array(test_vec))
        distances.append((dist, row['X'], row['Y']))
    if not distances:
        return None, None
    distances.sort(key=lambda x: x[0])
    top_k = distances[:k]
    avg_x = np.mean([x[1] for x in top_k])
    avg_y = np.mean([x[2] for x in top_k])
    return avg_x, avg_y

def evaluate(fingerprint_db, test_meta_df, label, test_folder, suffix=""):
    errors = []
    for _, row in test_meta_df.iterrows():
        test_file = os.path.join(test_folder, f"{row['File']}{suffix}.txt")
        if not os.path.exists(test_file):
            print(f"Missing test file: {test_file}")
            continue
        test_rssi = load_test_data(test_file)
        if not test_rssi:
            continue
        pred_x, pred_y = knn_predict(test_rssi, fingerprint_db, K)
        if pred_x is None:
            continue
        true_x, true_y = row['X'], row['Y']
        error = sqrt((pred_x - true_x) ** 2 + (pred_y - true_y) ** 2)
        errors.append(error)
    print(f"\n== {label.upper()} Results ==")
    if errors:
        print(f"Mean Error: {np.mean(errors):.2f} units")
        print(f"Std Dev Error: {np.std(errors):.2f} units")
    else:
        print("No valid test predictions made.")
    return errors

def visualize_predictions(fingerprint_dbs, test_metadata, test_folders, background_img_path=image_path, k=3):
    img = cv2.imread(background_img_path)
    if img is None:
        print("Failed to load background image.")
        return

    vis_img = img.copy()
    colors = {
        "Raw": (0, 0, 255),
        "Median": (255, 0, 0),
        "Kalman": (0, 165, 255),
        "GT": (0, 255, 0)
    }

    for _, row in test_metadata.iterrows():
        true_x, true_y = int(row['X']), int(row['Y'])
        cv2.circle(vis_img, (true_x, true_y), 6, colors["GT"], -1)
        cv2.putText(vis_img, "GT", (true_x+5, true_y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, colors["GT"], 1)

        for label, db in fingerprint_dbs.items():
            test_folder = test_folders[label]
            test_file = os.path.join(test_folder, f"{row['File']}.txt")
            if not os.path.exists(test_file):
                continue
            test_rssi = load_test_data(test_file)
            if not test_rssi:
                continue
            pred_x, pred_y = knn_predict(test_rssi, db, k)
            if pred_x is None:
                continue
            pred_x, pred_y = int(pred_x), int(pred_y)
            cv2.circle(vis_img, (pred_x, pred_y), 6, colors[label], -1)
            cv2.line(vis_img, (true_x, true_y), (pred_x, pred_y), colors[label], 1)
            cv2.putText(vis_img, label[0], (pred_x+5, pred_y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, colors[label], 1)

    cv2.imwrite("prediction_visualization.png", vis_img)
    cv2.imshow("Predictions", vis_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def log_results_to_csv(results, output_file="knn_error_results.csv"):
    """
    results: list of dicts with keys: 'k', 'method', 'mean_error', 'std_error'
    output_file: path to save the CSV file
    """
    file_exists = os.path.isfile(output_file)

    with open(output_file, mode='a', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=["k", "method", "mean_error", "std_error"])

        if not file_exists:
            writer.writeheader()

        for result in results:
            writer.writerow(result)

def main():
    test_metadata = pd.read_csv(TEST_METADATA_FILE)

    raw_fp = load_fingerprint_db("fingerprints_raw.csv")
    median_fp = load_fingerprint_db("fingerprints_median.csv")
    kalman_fp = load_fingerprint_db("fingerprints_kalman.csv")

    raw_errors = evaluate(raw_fp, test_metadata, "Raw", "Test_files")
    median_errors = evaluate(median_fp, test_metadata, "Median", "filtered_median_test", "_medianfilter")
    kalman_errors = evaluate(kalman_fp, test_metadata, "Kalman", "filtered_kalman_test", "_filtered")

    results_summary = [
        {"k": K, "method": "Raw", "mean_error": np.mean(raw_errors), "std_error": np.std(raw_errors)},
        {"k": K, "method": "Median", "mean_error": np.mean(median_errors), "std_error": np.std(median_errors)},
        {"k": K, "method": "Kalman", "mean_error": np.mean(kalman_errors), "std_error": np.std(kalman_errors)},
    ]
    log_results_to_csv(results_summary)

    visualize_predictions(
        {
            "Raw": raw_fp,
            "Median": median_fp,
            "Kalman": kalman_fp
        },
        test_metadata,
        {
            "Raw": "Test_files",
            "Median": "filtered_median_test",
            "Kalman": "filtered_kalman_test"
        }
    )

if __name__ == "__main__":
    main()
