import open3d as o3d
import numpy as np
import os

def convert_pcd_to_bin(pcd_path, bin_path):
    # Load the PCD file
    pcd = o3d.io.read_point_cloud(pcd_path)
    points = np.asarray(pcd.points)

    if points.shape[1] != 3:
        raise ValueError("Expected XYZ coordinates only. Got unexpected point shape.")

    # Flip Y axis to match KITTI format
    points[:, 1] *= -1

    # Add dummy intensity
    intensity = np.full((points.shape[0], 1), 0.5, dtype=np.float32)
    points_intensity = np.hstack((points, intensity))

    # Ensure output directory exists
    os.makedirs(os.path.dirname(bin_path), exist_ok=True)

    # Save to .bin format
    points_intensity.astype(np.float32).tofile(bin_path)
    print(f"[✓] Saved converted KITTI .bin file to: {bin_path}")


# Absolute paths
pcd_path = r"C:\Pipeline\saved_data\single_frame.pcd"
bin_path = r"C:\Pipeline\saved_data\single_frame.bin"

convert_pcd_to_bin(pcd_path, bin_path)