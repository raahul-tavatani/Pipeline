from pypcd.pypcd import PointCloud
import numpy as np
import os

def convert_pcd_to_bin_pypcd(pcd_path, bin_path):
    print(f"📂 Reading PCD file: {pcd_path}")
    pc = PointCloud.from_path(pcd_path)

    # Required fields
    x = pc.pc_data['x'].astype(np.float32)
    y = pc.pc_data['y'].astype(np.float32)
    z = pc.pc_data['z'].astype(np.float32)

    # Optional: Flip Y axis if needed (KITTI format might require it)
    y = -y

    # Use intensity if present, otherwise use dummy value
    if 'intensity' in pc.fields:
        intensity = pc.pc_data['intensity'].astype(np.float32)
        # Normalize intensity if needed: e.g., /256 if values are 0-255
        if intensity.max() > 1:
            intensity /= 256.0
    else:
        intensity = np.full_like(x, fill_value=0.5, dtype=np.float32)

    # Combine into Nx4 format (x, y, z, i)
    points = np.stack((x, y, z, intensity), axis=-1)

    # Ensure output directory exists
    os.makedirs(os.path.dirname(bin_path), exist_ok=True)

    # Save to .bin file
    points.astype(np.float32).tofile(bin_path)
    print(f"[✓] Saved .bin to: {bin_path}")


# Example usage
if __name__ == "__main__":
    pcd_path = r"C:\Pipeline\saved_data\single_frame.pcd"
    bin_path = r"C:\Pipeline\saved_data\single_frame.bin"
    convert_pcd_to_bin_pypcd(pcd_path, bin_path)
