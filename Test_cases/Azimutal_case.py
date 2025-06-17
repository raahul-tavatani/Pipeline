import subprocess
import os

PORT = 3000
X_POS = 60
GAPS = list(range(2, 51, 5))  # 2m to 50m in 5m steps

# SAVE_ROOT = r"C:\Pipeline\saved_data\Azimutal_tests"
# os.makedirs(SAVE_ROOT, exist_ok=True)

for gap in GAPS:
    tag = f"azimutal_separability_{gap}m"
    y1 = -gap / 2
    y2 = gap / 2

    save_tag = os.path.join("Azimutal_tests", tag)  # For consistency in your saving scheme
    cmd = [
        "python", r"C:\Pipeline\Single_pipeline.py",
        "--port", str(PORT),
        "--x_dist_1", str(X_POS),
        "--y_dist_1", str(y1),
        "--x_dist_2", str(X_POS),
        "--y_dist_2", str(y2),
        "--save_tag", tag,
        "--test_type", "Azimutal_tests"
    ]

    print(f"Running azimutal test with gap = {gap}m with save tag: {tag}")
    subprocess.run(cmd)
