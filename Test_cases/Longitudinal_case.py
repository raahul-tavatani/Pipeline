import subprocess
import os

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
PIPELINE_SCRIPT = os.path.join(BASE_DIR, "Single_pipeline.py")

PORT = 2000
X_START = 10
X_END = 100
STEP = 10
GAP = 3.3  # lateral gap between vehicles


for x in range(X_START, X_END + 1, STEP):
    tag = f"radial_test_{x}m"
    save_tag = os.path.join("Radial_tests", tag)  
    cmd = [
        "python", PIPELINE_SCRIPT,
        "--port", str(PORT),
        "--x_dist_1", str(x),
        "--y_dist_1", str(-GAP / 2),
        "--x_dist_2", str(x),
        "--y_dist_2", str(GAP / 2),
        "--save_tag", tag,
        "--test_type", "Radial_tests"  
    ]
    print(f" Running radial test at x = {x}m with save tag: {tag}")
    subprocess.run(cmd)
