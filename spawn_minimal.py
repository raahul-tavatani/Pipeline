import os, json, numpy as np, open3d as o3d

BASE_DIR = r"C:\Pipeline\saved_data\Radial_tests"#"C:\Pipeline\saved_data\Azimutal_tests"
FILE_TAG = "radial_test_50m"

pcd_path = os.path.join(BASE_DIR, "pcd", f"{FILE_TAG}.pcd")
gt_json_path = os.path.join(BASE_DIR, "json", f"{FILE_TAG}.json")
pred_json_path = os.path.join(BASE_DIR, "pred_json", f"{FILE_TAG}.json")

def rot_z(deg):
    a = np.deg2rad(deg)
    c, s = np.cos(a), np.sin(a)
    return np.array([[c,-s,0],[s,c,0],[0,0,1]], dtype=np.float64)

def obb_lineset(center_xyz, extent_full, yaw_deg, rgb):
    R = rot_z(yaw_deg)
    obb = o3d.geometry.OrientedBoundingBox(center=center_xyz, R=R, extent=extent_full)
    ls  = o3d.geometry.LineSet.create_from_oriented_bounding_box(obb)
    cols = np.tile(np.array(rgb, dtype=np.float64), (len(ls.lines), 1))
    ls.colors = o3d.utility.Vector3dVector(cols)
    return ls

# ── Load PCD ───────────────────────────────────────
pcd = o3d.io.read_point_cloud(pcd_path)
pts = np.asarray(pcd.points)
if pts.size == 0:
    raise RuntimeError(f"No points in {pcd_path}")

print(f"PCD loaded: {pts.shape[0]} pts; XYZ min={pts.min(0)}, max={pts.max(0)}")

# ── Ground Truth (CARLA) ───────────────────────────
with open(gt_json_path) as f:
    gt_data = json.load(f)

gt_lsets = []
for box in gt_data:
    ext_full = 2.0 * np.array([box["extent"]["x"], box["extent"]["y"], box["extent"]["z"]], float)
    center_lidar = np.array([box["center_lidar"]["x"], box["center_lidar"]["y"], box["center_lidar"]["z"]], float)
    off = np.array([box["bounding_box_offset"]["x"], box["bounding_box_offset"]["y"], box["bounding_box_offset"]["z"]], float)
    yaw = float(box["rotation"]["yaw"])  # deg
    center = center_lidar + rot_z(yaw) @ off
    gt_lsets.append(obb_lineset(center, ext_full, yaw, (0,1,0)))  # green

# ── Predictions (KITTI → CARLA) ────────────────────
with open(pred_json_path) as f:
    pred_data = json.load(f)

pred_lsets = []
for arr in pred_data["boxes"]:
    center = np.array(arr[0:3], float)
    center[1] *= -1.0                      # flip y
    ext_full = np.array(arr[3:6], float)   # already full sizes
    yaw_deg  = -np.rad2deg(arr[6])         # flip heading sign due to y-flip
    pred_lsets.append(obb_lineset(center, ext_full, yaw_deg, (1,0,0)))  # red

# ── Diagnostics (check proximity) ──────────────────
def brief(v): return np.round(v, 3).tolist()
if gt_lsets:
    # get first GT center by reconstructing from its lineset bbox
    gt_center = np.asarray(gt_lsets[0].get_axis_aligned_bounding_box().get_center())
    print("Sample GT center near:", brief(gt_center))
if pred_lsets:
    pr_center = np.asarray(pred_lsets[0].get_axis_aligned_bounding_box().get_center())
    print("Sample Pred center near:", brief(pr_center))

# ── Visualize ──────────────────────────────────────
geoms = [pcd, *gt_lsets, *pred_lsets, o3d.geometry.TriangleMesh.create_coordinate_frame(size=2.0)]
vis = o3d.visualization.Visualizer()
vis.create_window("PCD + GT (green) + Pred (red)", 1280, 720)
for g in geoms:
    vis.add_geometry(g)

# Fit camera to **all** geometry
all_bbox = o3d.geometry.AxisAlignedBoundingBox.create_from_points(
    o3d.utility.Vector3dVector(
        np.vstack([pts] + [
            np.asarray(ls.get_axis_aligned_bounding_box().get_box_points()) for ls in (gt_lsets + pred_lsets)
        ])
    )
)
ctr = vis.get_view_control()
ctr.set_lookat(all_bbox.get_center())
ctr.set_up([0,0,1])         # Z-up
ctr.set_front([ -1, 0, -0.2])
ctr.set_zoom(0.4)

opt = vis.get_render_option()
opt.point_size = 2.0        # make points thicker
opt.line_width = 2.0        # if supported

vis.run()
vis.destroy_window()
