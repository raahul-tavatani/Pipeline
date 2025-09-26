import subprocess
import os
import numpy as np



BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
PIPELINE_SCRIPT = os.path.join(BASE_DIR, "Single_pipeline.py")

PORT = 3000
X_POS = 30
GAPS = np.arange(2, 3.6, 0.1)  # Includes 5.0


for gap in GAPS:
    rounded_gap = round(gap, 1)  # Round to 1 decimal place
    tag = f"azimutal_separability_{rounded_gap}m"
    y1 = -rounded_gap / 2
    y2 = rounded_gap / 2

    save_tag = os.path.join("Azimutal_tests", tag) 
    cmd = [
        "python", PIPELINE_SCRIPT,
        "--port", str(PORT),
        "--x_dist_1", str(X_POS),
        "--y_dist_1", str(y1),
        "--x_dist_2", str(X_POS),
        "--y_dist_2", str(y2),
        "--save_tag", tag,
        "--test_type", "Azimutal_tests"
    ]

    print(f"Running azimutal test with gap = {rounded_gap}m with save tag: {tag}")
    subprocess.run(cmd)
