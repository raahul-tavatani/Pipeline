import os
import json
import math
import csv
import numpy as np
from datetime import datetime

BASE_PATH = r"C:\Pipeline\saved_data"

# Add timestamp to output file name
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
CSV_OUTPUT = os.path.join(BASE_PATH, f"evaluation_summary_{timestamp}.csv")

def load_json(path):
    with open(path, 'r') as f:
        return json.load(f)

def compute_metrics(gt_boxes, pred_boxes, threshold=2.0):
    matched_gt = set()
    matched_pred = set()
    TP = 0

    for i, pred in enumerate(pred_boxes):
        pred_center = np.array([pred[0], -pred[1], pred[2]])

        for j, gt in enumerate(gt_boxes):
            center = gt['center_lidar']
            offset = gt.get('bounding_box_offset', {'x': 0, 'y': 0, 'z': 0})

            gt_center = np.array([
                center["x"] - offset["x"],
                center["y"] - offset["y"],
                center["z"] + offset["z"]
            ])

            dist = np.linalg.norm(pred_center - gt_center)

            if dist <= threshold and j not in matched_gt:
                TP += 1
                matched_gt.add(j)
                matched_pred.add(i)
                break

    FP = len(pred_boxes) - len(matched_pred)
    FN = len(gt_boxes) - len(matched_gt)

    precision = TP / (TP + FP + 1e-6)
    recall = TP / (TP + FN + 1e-6)
    f1 = 2 * (precision * recall) / (precision + recall + 1e-6)

    return TP, FP, FN, round(precision, 3), round(recall, 3), round(f1, 3)

def evaluate_all():
    all_results = []

    for test_folder in os.listdir(BASE_PATH):
        folder_path = os.path.join(BASE_PATH, test_folder)
        json_path = os.path.join(folder_path, "json")
        pred_path = os.path.join(folder_path, "pred_json")

        if not (os.path.isdir(json_path) and os.path.isdir(pred_path)):
            continue

        print(f"\nEvaluating folder: {test_folder}")
        gt_files = sorted(os.listdir(json_path))
        pred_files = sorted(os.listdir(pred_path))

        folder_metrics = {
            'TP': 0,
            'FP': 0,
            'FN': 0,
        }

        for filename in gt_files:
            if filename not in pred_files:
                print(f"Skipping (missing prediction): {filename}")
                continue

            gt_file = os.path.join(json_path, filename)
            pred_file = os.path.join(pred_path, filename)

            gt_data = load_json(gt_file)
            pred_data = load_json(pred_file)
            pred_boxes = pred_data.get("boxes", [])

            TP, FP, FN, precision, recall, f1 = compute_metrics(gt_data, pred_boxes)

            folder_metrics['TP'] += TP
            folder_metrics['FP'] += FP
            folder_metrics['FN'] += FN

            all_results.append({
                "Folder": test_folder,
                "File": filename,
                "TP": TP,
                "FP": FP,
                "FN": FN,
                "Precision": precision,
                "Recall": recall,
                "F1-Score": f1
            })

        # Add per-folder summary row
        total_TP = folder_metrics['TP']
        total_FP = folder_metrics['FP']
        total_FN = folder_metrics['FN']
        if total_TP + total_FP + total_FN > 0:
            precision = total_TP / (total_TP + total_FP + 1e-6)
            recall = total_TP / (total_TP + total_FN + 1e-6)
            f1 = 2 * (precision * recall) / (precision + recall + 1e-6)
        else:
            precision = recall = f1 = 0.0

        all_results.append({
            "Folder": test_folder,
            "File": "[TOTAL]",
            "TP": total_TP,
            "FP": total_FP,
            "FN": total_FN,
            "Precision": round(precision, 3),
            "Recall": round(recall, 3),
            "F1-Score": round(f1, 3)
        })

    # Save to CSV with timestamp
    with open(CSV_OUTPUT, mode='w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=["Folder", "File", "TP", "FP", "FN", "Precision", "Recall", "F1-Score"])
        writer.writeheader()
        writer.writerows(all_results)

    print(f"\n✅ Evaluation complete. Results saved to: {CSV_OUTPUT}")

if __name__ == "__main__":
    evaluate_all()
