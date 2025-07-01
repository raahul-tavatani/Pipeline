import subprocess
import os
import csv

# Constants
PORT = 4000
SCRIPT_DIR = os.path.dirname(__file__)
BASE_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
CSV_PATH = os.path.join(BASE_DIR, "scripts", "curvy_trajectory.csv")
GENERATOR_SCRIPT = os.path.join(BASE_DIR, "scripts", "curvy_trajectory_csv_generator.py")
PIPELINE_SCRIPT = os.path.join(BASE_DIR, "Single_pipeline.py")

# Ensure CSV exists
if not os.path.exists(CSV_PATH):
    print(f"CSV not found at {CSV_PATH}. Generating...")
    subprocess.run(["python", GENERATOR_SCRIPT])

if not os.path.exists(CSV_PATH):
    raise FileNotFoundError(f"CSV still not found after attempting generation: {CSV_PATH}")

# Load CSV
with open(CSV_PATH, newline='') as csvfile:
    reader = csv.reader(csvfile)
    next(reader)  # Skip header
    rows = list(reader)

# Run tests
for i, row in enumerate(rows):
    x_val = float(row[0])
    y_val = float(row[1])
    yaw_val = float(row[2]) if len(row) > 2 else 0

    tag = f"trajectory_test_{i}"
    cmd = [
        "python", PIPELINE_SCRIPT,
        "--port", str(PORT),
        "--x_dist_1", str(x_val),
        "--y_dist_1", str(y_val),
        "--yaw_1", str(yaw_val),
        "--save_tag", tag,
        "--test_type", "Trajectory_tests"
    ]
    print(f"🚗 Running trajectory test {i}: x={x_val}, y={y_val}, yaw={yaw_val}")
    subprocess.run(cmd)
