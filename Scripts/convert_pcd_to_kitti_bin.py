# convert_pcd_to_kitti_bin.py
import open3d as o3d
import numpy as np
import os

def convert_pcd_to_bin_open3d(pcd_path, bin_path):
    try:
        print(f" Reading: {pcd_path}")
        pcd = o3d.io.read_point_cloud(pcd_path)
        points = np.asarray(pcd.points)

        if points.shape[0] == 0:
            print(f" No points in {pcd_path}, skipping.")
            return

        # Flip Y-axis if needed for KITTI format
        points[:, 1] = -points[:, 1]

        # Dummy intensity (0.5 for all points)
        intensity = np.full((points.shape[0], 1), 0.5, dtype=np.float32)

        # Combine into [x, y, z, intensity]
        data = np.hstack((points.astype(np.float32), intensity))

        # Ensure output directory exists
        os.makedirs(os.path.dirname(bin_path), exist_ok=True)

        data.tofile(bin_path)
        print(f" Saved .bin → {bin_path}")

    except Exception as e:
        print(f" Error processing {pcd_path}: {e}")

def convert_all_pcds(base_dir):
    print(f"🔍 Recursively scanning for PCD files in: {base_dir}")

    for root, dirs, files in os.walk(base_dir):
        # We only convert files inside any "pcd" folder
        if os.path.basename(root).lower() == "pcd":
            # Compute corresponding "bin" folder path
            bin_folder = os.path.join(os.path.dirname(root), "bin")
            os.makedirs(bin_folder, exist_ok=True)

            for file in files:
                if file.lower().endswith(".pcd"):
                    pcd_path = os.path.join(root, file)
                    bin_name = os.path.splitext(file)[0] + ".bin"
                    bin_path = os.path.join(bin_folder, bin_name)
                    convert_pcd_to_bin_open3d(pcd_path, bin_path)

if __name__ == "__main__":
    BASE_DIR = r"C:\Pipeline\saved_data"
    convert_all_pcds(BASE_DIR)
    print(" All PCD files converted to BIN format.")
