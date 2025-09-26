import os
import subprocess
from pathlib import Path

#  Configuration
BASE_DIR = Path(r"C:\Pipeline\saved_data")
DEMO_SCRIPT = Path(r"C:\Pipeline\OpenPCDet-Win11-Compatible\\tools\\demo.py")
CFG_FILE = "C:\Pipeline\\Scripts\\cfgs\\kitti_models\\pv_rcnn.yaml"
CKPT_PATH = r"C:\Pipeline\models\pv_rcnn_8369.pth"  

def run_demo_on_bin(bin_file: Path, cfg_file: str, ckpt_path: str, pred_json_dir: Path):
    pred_json_dir.mkdir(parents=True, exist_ok=True)
    output_name = bin_file.stem + ".json"
    output_path = pred_json_dir / output_name

    PYTHON_EXE = r"C:\Pipeline\py_env_openpcdet\Scripts\python.exe"


    cmd = [
        PYTHON_EXE, str(DEMO_SCRIPT),
        #"python", str(DEMO_SCRIPT),
        "--cfg_file", cfg_file,
        "--ckpt", ckpt_path,
        "--data_path", str(bin_file),
        "--ext", ".bin"
    ]

    print(f"🚀 Running: {bin_file.name}")
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Failed to process {bin_file.name}: {e}")
        return

    # Move the prediction to pred_json/ and rename
    generated_json = Path("prediction_000.json")
    if generated_json.exists():
        generated_json.rename(output_path)
        print(f" Saved: {output_path}")
    else:
        print(f" Prediction not found for: {bin_file.name}")

def process_test_folder(test_folder: Path):
    bin_dir = test_folder / "bin"
    pred_json_dir = test_folder / "pred_json"

    if not bin_dir.exists():
        print(f" Skipping {test_folder.name}: No bin folder.")
        return

    bin_files = list(bin_dir.glob("*.bin"))
    if not bin_files:
        print(f" No .bin files found in {bin_dir}")
        return

    for bin_file in bin_files:
        run_demo_on_bin(bin_file, CFG_FILE, CKPT_PATH, pred_json_dir)

def main():
    print(f"🔍 Scanning test folders in: {BASE_DIR}")
    for test_folder in BASE_DIR.iterdir():
        if test_folder.is_dir():
            process_test_folder(test_folder)

if __name__ == "__main__":
    main()
