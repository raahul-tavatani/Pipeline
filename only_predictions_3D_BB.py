import json
import math
import os
import open3d as o3d


def load_point_cloud(pcd_path):
    print(f"📂 Loading point cloud: {pcd_path}")
    return o3d.io.read_point_cloud(pcd_path)


def load_predictions(json_path):
    print(f"📂 Loading predictions: {json_path}")
    with open(json_path, 'r') as f:
        data = json.load(f)
    return data.get("boxes", [])  # List of [x, y, z, dx, dy, dz, yaw_rad]


def create_bbox(center, extent, yaw_rad, color=(1, 0, 0)):
    bbox = o3d.geometry.OrientedBoundingBox()
    bbox.center = center
    bbox.extent = extent
    bbox.R = bbox.get_rotation_matrix_from_xyz((0, 0, yaw_rad))
    bbox.color = color
    return bbox


def visualize(pcd, pred_boxes):
    geometries = [pcd]

    for box in pred_boxes:
        center = box[:3]
        extent = box[3:6]
        yaw_rad = box[6]
        bbox = create_bbox(center, extent, yaw_rad)
        geometries.append(bbox)

    print("🎨 Rendering point cloud with predicted boxes (🔴)...")
    o3d.visualization.draw_geometries(geometries)


def main():
    base_path = r"C:\Pipeline\saved_data"
    pcd_path = os.path.join(base_path, "single_frame.pcd")
    pred_json_path = os.path.join(base_path, "prediction_000.json")

    if not os.path.exists(pcd_path):
        print(f"❌ Point cloud not found: {pcd_path}")
        return
    if not os.path.exists(pred_json_path):
        print(f"❌ Prediction file not found: {pred_json_path}")
        return

    pcd = load_point_cloud(pcd_path)
    pred_boxes = load_predictions(pred_json_path)
    visualize(pcd, pred_boxes)


if __name__ == "__main__":
    main()
