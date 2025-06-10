import open3d as o3d
import numpy as np
import os

def load_pcd(file_path):
    """Load a PCD file using Open3D."""
    pcd = o3d.io.read_point_cloud(file_path)
    return pcd

file_path = r"C:\Users\tavatanr\OneDrive - Bertrandt AG\Desktop\PCD visualisation\lidar_pointcloud.pcd"
pcd = load_pcd(file_path)
print(pcd)  # Print the point cloud object to check if it's loaded correctly

o3d.visualization.draw_geometries([pcd])
