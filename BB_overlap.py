import json
import numpy as np
import open3d as o3d
import os
import math


def load_point_cloud(pcd_path):
    print(f" Loading point cloud: {pcd_path}")
    return o3d.io.read_point_cloud(pcd_path)


def load_json(path):
    print(f" Loading JSON: {path}")
    with open(path, 'r') as f:
        return json.load(f)


def create_bbox(center, extent, yaw_deg, color):
    bbox = o3d.geometry.OrientedBoundingBox()
    bbox.center = center
    bbox.extent = extent

    R = bbox.get_rotation_matrix_from_xyz((0, 0, math.radians(yaw_deg)))
    bbox.R = R
    bbox.color = color
    return bbox


def visualize(pcd, pred_boxes, gt_boxes):
    geometries = [pcd]

    #  Red: Predicted boxes
    for box in pred_boxes:
        center = box[:3]
        center_flipped = [center[0], -center[1], center[2]]
        extent = box[3:6]
        yaw_rad = box[6]
        yaw_deg = math.degrees(yaw_rad)
        bbox = create_bbox(center_flipped, extent, -yaw_deg, color=(1, 0, 0))
        geometries.append(bbox)

    #  Green: Ground truth boxes
    for obj in gt_boxes:
        center = obj['center_lidar']
        extent = obj['extent']
        yaw = obj['rotation']['yaw']  

        offset = obj.get('bounding_box_offset', {'x': 0, 'y': 0, 'z': 0})

        center_vec = [
        center["x"] - offset['x'],
        center["y"] - offset['y'],
        center["z"] + offset['z']
        ]
        extent_vec = [
            extent["x"] * 2,
            extent["y"] * 2,
            extent["z"] * 2
        ]

        bbox = create_bbox(center_vec, extent_vec, yaw, color=(0, 1, 0))
        geometries.append(bbox)

    print(" Rendering point cloud with predicted red and ground truth green boxes...")
    o3d.visualization.draw_geometries(geometries)


def main():
    #base_path = r"C:\Pipeline\saved_data\Trajectory_tests"
    pcd_path = os.path.join(r"C:\Pipeline\saved_data\Trajectory_tests\pcd", "trajectory_test_10.pcd")
    pred_json_path = os.path.join(r"C:\Pipeline\saved_data\Trajectory_tests\pred_json", "trajectory_test_10.json")
    gt_json_path = os.path.join(r"C:\Pipeline\saved_data\Trajectory_tests\json", "trajectory_test_10.json")

    for path in [pcd_path, pred_json_path, gt_json_path]:
        if not os.path.exists(path):
            print(f" File not found: {path}")
            return

    pcd = load_point_cloud(pcd_path)
    predictions = load_json(pred_json_path)
    ground_truth = load_json(gt_json_path)

    pred_boxes = predictions.get("boxes", [])  # List of [x, y, z, dx, dy, dz, yaw_rad]
    visualize(pcd, pred_boxes, ground_truth)


if __name__ == "__main__":
    main()
