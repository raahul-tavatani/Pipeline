# tools/visualize_rs1_gap_clipped.py
# -*- coding: utf-8 -*-
"""
Visualize an RS-1 gap folder with:
  • PCD points (one frame or merged)
  • target_vicinity.json (OBB, drawn with thick cylindrical edges)
  • control_volume.json (AABB, thick edges)
  • cluster_map.json tubes (clipped to vicinity using same ray/OBB math as your evaluator)
  • optional coloring from grid_tubes_map.csv (is_separable)

Usage (Windows CMD, from repo root):
  python -m tools.visualize_rs1_gap_clipped ^
    --gap .\outputs\rs1\scenario_001\d0_10m\gap_0.50m ^
    --frame all --voxel 0.02 ^
    --tube-rho-scale 1.5 --edge-radius 0.02 ^
    --color-from-grid

Tip: increase --tube-radius-min or --tube-rho-scale if tubes are too skinny to see.
"""

import os, re, json, math, argparse, csv
from typing import Dict, List, Tuple, Optional
import numpy as np
import open3d as o3d

# ---------------- I/O ----------------

def _load_json(path: str) -> Optional[dict]:
    if not os.path.exists(path): return None
    with open(path, "r") as f: return json.load(f)

def _read_pcds(gap_dir: str, frame: str = "0") -> Optional[o3d.geometry.PointCloud]:
    files = []
    if frame == "all":
        files = [f for f in os.listdir(gap_dir) if f.lower().endswith(".pcd") and f.startswith("frame_")]
        files.sort()
    else:
        files = [f"frame_{int(frame):03}.pcd"]
    pcs = []
    for fn in files:
        p = os.path.join(gap_dir, fn)
        if not os.path.exists(p):
            print(f"[WARN] missing PCD: {p}")
            continue
        pc = o3d.io.read_point_cloud(p)
        pcs.append(pc)
    if not pcs: return None
    if len(pcs) == 1: return pcs[0]
    out = o3d.geometry.PointCloud()
    for pc in pcs: out += pc
    return out

# ---------------- math helpers ----------------

def _dir_from_az_el_deg(az_deg: float, el_deg: float) -> np.ndarray:
    az = math.radians(az_deg); el = math.radians(el_deg)
    c = math.cos(el)
    return np.array([c*math.cos(az), c*math.sin(az), math.sin(el)], dtype=float)

def _ray_obb_intersection(obb: o3d.geometry.OrientedBoundingBox,
                          ray_o: np.ndarray,
                          ray_d: np.ndarray) -> Optional[Tuple[float, float]]:
    """Intersect ray O + t D with OBB. Returns (t_enter, t_exit) if hit in front; else None."""
    C = np.asarray(obb.center, dtype=float)
    R = np.asarray(obb.R if hasattr(obb, "R") else obb.rotation, dtype=float)
    ext = np.asarray(obb.extent if hasattr(obb, "extent") else obb.get_extent(), dtype=float) * 0.5
    ro = R.T @ (ray_o - C); rd = R.T @ ray_d
    tmin, tmax = -1e30, 1e30; eps = 1e-12
    for i in range(3):
        if abs(rd[i]) < eps:
            if ro[i] < -ext[i] or ro[i] > ext[i]: return None
        else:
            t1 = (-ext[i] - ro[i]) / rd[i]
            t2 = ( ext[i] - ro[i]) / rd[i]
            if t1 > t2: t1, t2 = t2, t1
            if t1 > tmin: tmin = t1
            if t2 < tmax: tmax = t2
            if tmax < tmin: return None
    if tmax < max(tmin, 0.0): return None
    return float(max(tmin, 0.0)), float(tmax)

# ---------------- thick edge drawing ----------------

def _cyl_between(p0: np.ndarray, p1: np.ndarray, radius: float, color=(0.8,0.2,0.2)) -> o3d.geometry.TriangleMesh:
    axis = p1 - p0
    L = float(np.linalg.norm(axis))
    if L <= 1e-9:
        s = o3d.geometry.TriangleMesh.create_sphere(radius=radius)
        s.paint_uniform_color(color)
        s.translate(p0); s.compute_vertex_normals()
        return s
    cyl = o3d.geometry.TriangleMesh.create_cylinder(radius=max(radius,1e-5), height=L, resolution=24)
    cyl.compute_vertex_normals(); cyl.paint_uniform_color(color)
    z = np.array([0.0,0.0,1.0]); a = axis / L
    v = np.cross(z, a); s = np.linalg.norm(v); c = np.dot(z, a)
    if s < 1e-9:
        R = np.eye(3) if c > 0 else np.array([[1,0,0],[0,-1,0],[0,0,-1]])
    else:
        vx = np.array([[0,-v[2],v[1]],[v[2],0,-v[0]],[-v[1],v[0],0]], dtype=float)
        R = np.eye(3) + vx + vx @ vx * ((1 - c) / (s**2 + 1e-12))
    T = np.eye(4); T[:3,:3] = R; T[:3,3] = (p0+p1)*0.5
    cyl.transform(T); return cyl

def _box_edges_mesh_from_bbox(bbox, radius: float, color=(0.8,0.2,0.2)):
    corners = np.asarray(bbox.get_box_points())
    # Open3D corner indexing order for boxes:
    # 0..3 bottom face, 4..7 top face (but not guaranteed which exact order).
    # We'll reconstruct edges using the bbox.get_oriented_bounding_box() convention.
    # Build by nearest-neighbor face connections: this stable set works for AxisAligned and Oriented.
    # Edges: square bottom (0-1,1-3,3-2,2-0), square top (4-5,5-7,7-6,6-4), verticals (0-4,1-5,2-6,3-7)
    edges = [(0,1),(1,3),(3,2),(2,0),(4,5),(5,7),(7,6),(6,4),(0,4),(1,5),(2,6),(3,7)]
    meshes = []
    for i,j in edges:
        meshes.append(_cyl_between(corners[i], corners[j], radius=radius, color=color))
    # Merge
    out = o3d.geometry.TriangleMesh()
    for m in meshes: out += m
    out.compute_vertex_normals()
    return out

# ---------------- scene pieces ----------------

def _vicinity_obb_mesh(gap_dir: str, edge_radius: float, color=(0.0,0.7,0.2)) -> Optional[o3d.geometry.TriangleMesh]:
    p = os.path.join(gap_dir, "target_vicinity.json")
    v = _load_json(p)
    if not v or "corner_points_xyz" not in v:
        print(f"[WARN] no target_vicinity.json in {gap_dir}")
        return None
    pts = np.array(list(v["corner_points_xyz"].values()), dtype=float).reshape(-1,3)
    obb = o3d.geometry.OrientedBoundingBox.create_from_points(o3d.utility.Vector3dVector(pts))
    return _box_edges_mesh_from_bbox(obb, radius=edge_radius, color=color), obb

def _control_volume_aabb_mesh(gap_dir: str, edge_radius: float, color=(0.8,0.2,0.2)) -> Optional[o3d.geometry.TriangleMesh]:
    p = os.path.join(gap_dir, "control_volume.json")
    v = _load_json(p)
    if not v or "corner_points_xyz" not in v:
        print(f"[WARN] no control_volume.json in {gap_dir}")
        return None
    corners = np.array(list(v["corner_points_xyz"].values()), dtype=float).reshape(-1,3)
    aabb = o3d.geometry.AxisAlignedBoundingBox(corners.min(axis=0), corners.max(axis=0))
    return _box_edges_mesh_from_bbox(aabb, radius=edge_radius, color=color)

def _load_cluster_map(path: str) -> List[dict]:
    d = _load_json(path) or {}
    tubes = []
    for t in d.get("tubes", []):
        az = float(t["azimuth_deg"]); el = float(t["elevation_deg"])
        rho = float(t.get("rho_deg", 0.08))
        s0  = float(t.get("s_start_m", 0.0))
        s1  = float(t.get("s_end_m", s0))
        tubes.append({
            "tube_id": int(t.get("tube_id", len(tubes))),
            "az": az, "el": el, "rho_deg": rho, "rho_rad": math.radians(rho),
            "dir": _dir_from_az_el_deg(az, el),
            "s_start": s0, "s_end": s1
        })
    return tubes

def _grid_colors_from_csv(gap_dir: str) -> Dict[int, int]:
    """Map cell_id -> is_separable from grid_tubes_map.csv if present."""
    out = {}
    p = os.path.join(gap_dir, "grid_tubes_map.csv")
    if not os.path.exists(p): return out
    with open(p, "r", newline="") as f:
        rdr = csv.DictReader(f)
        for row in rdr:
            try:
                cid = int(row["cell_id"]); sep = int(row.get("is_separable","0"))
                out[cid] = sep
            except Exception:
                continue
    return out

def _tube_meshes(gap_dir: str,
                 obb_for_clip: o3d.geometry.OrientedBoundingBox,
                 tube_rho_scale: float,
                 tube_radius_min: float,
                 tube_max: Optional[int],
                 color_from_grid: bool) -> List[o3d.geometry.Geometry]:
    cm_path = os.path.join(gap_dir, "cluster_map.json")
    tubes = _load_cluster_map(cm_path)
    if not tubes:
        print(f"[WARN] no cluster_map.json or empty tubes in {gap_dir}")
        return []

    sep_map = _grid_colors_from_csv(gap_dir) if color_from_grid else {}

    geoms: List[o3d.geometry.Geometry] = []
    ray_o = np.zeros(3, dtype=float)
    count = 0
    for t in tubes:
        # clip tube to vicinity OBB (like evaluator)
        hit = _ray_obb_intersection(obb_for_clip, ray_o, t["dir"])
        if hit is None:
            continue
        s_enter, s_exit = hit
        s0 = max(s_enter, t["s_start"])
        s1 = min(s_exit,  t["s_end"])
        if s1 <= s0 + 1e-9:
            continue

        p0 = t["dir"] * s0
        p1 = t["dir"] * s1
        r0 = max(s0 * math.tan(t["rho_rad"]) * tube_rho_scale, tube_radius_min)

        col = (0.20, 0.60, 1.00)  # default blue-ish
        if t["tube_id"] in sep_map:
            col = (0.10, 0.80, 0.20) if sep_map[t["tube_id"]] == 1 else (0.55, 0.55, 0.55)

        cyl = _cyl_between(p0, p1, radius=r0, color=col)
        geoms.append(cyl)
        # draw end rings for context
        try:
            axis = (p1 - p0); axis_n = axis / (np.linalg.norm(axis) + 1e-12)
            geoms.append(_ring_wire(p0, axis_n, r0, color=col))
            r1 = max(s1 * math.tan(t["rho_rad"]) * tube_rho_scale, tube_radius_min)
            geoms.append(_ring_wire(p1, axis_n, r1, color=col))
        except Exception:
            pass

        count += 1
        if tube_max is not None and count >= tube_max:
            break
    return geoms

def _ring_wire(center: np.ndarray, normal: np.ndarray, radius: float, color=(0.3,0.75,1.0), segments: int = 48):
    n = normal / (np.linalg.norm(normal) + 1e-12)
    a = np.array([1.0,0.0,0.0]) if abs(n[0]) < 0.9 else np.array([0.0,1.0,0.0])
    u = np.cross(n, a); u /= (np.linalg.norm(u)+1e-12)
    v = np.cross(n, u)
    pts = []
    for k in range(segments):
        th = 2*math.pi*k/segments
        pts.append(center + radius*(math.cos(th)*u + math.sin(th)*v))
    ls = o3d.geometry.LineSet()
    ls.points = o3d.utility.Vector3dVector(np.asarray(pts))
    ls.lines  = o3d.utility.Vector2iVector([[i,(i+1)%segments] for i in range(segments)])
    ls.colors = o3d.utility.Vector3dVector(np.tile(np.array(color).reshape(1,3),(segments,1)))
    return ls

# ---------------- main ----------------

def main():
    ap = argparse.ArgumentParser(description="RS-1 visualizer: PCD + vicinity OBB + control volume AABB + tubes (clipped).")
    ap.add_argument("--gap", required=True, help=r"Path to a gap folder (e.g., .\outputs\rs1\scenario_001\d0_10m\gap_0.50m)")
    ap.add_argument("--frame", default="0", help="PCD frame index (e.g., 0) or 'all'")
    ap.add_argument("--voxel", type=float, default=0.0, help="Voxel downsample size (m) (0=off)")
    ap.add_argument("--edge-radius", type=float, default=0.01, help="Cylinder radius for box edges (m)")
    ap.add_argument("--tube-rho-scale", type=float, default=1.0, help="Visual scale for tube radius r = s_start*tan(rho)*scale")
    ap.add_argument("--tube-radius-min", type=float, default=0.01, help="Clamp tube radius to at least this (m)")
    ap.add_argument("--tube-max", type=int, default=600, help="Max tubes to draw (None for all)")
    ap.add_argument("--no-points", action="store_true", help="Do not draw PCD points")
    ap.add_argument("--color-from-grid", action="store_true", help="Use grid_tubes_map.csv is_separable to color tubes")
    args = ap.parse_args()

    gap_dir = args.gap
    if not os.path.isdir(gap_dir):
        print(f"[ERROR] gap dir not found: {gap_dir}")
        return

    geoms: List[o3d.geometry.Geometry] = []

    # Axes
    geoms.append(o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5, origin=[0,0,0]))

    # Points
    if not args.no_points:
        pc = _read_pcds(gap_dir, frame=args.frame)
        if pc is not None:
            if args.voxel and args.voxel > 0: pc = pc.voxel_down_sample(args.voxel)
            pc.paint_uniform_color([0.70,0.70,0.70]); geoms.append(pc)
        else:
            print("[WARN] no PCDs loaded")

    # Vicinity OBB (green) + we keep the OBB for clipping
    vic_mesh_and_obb = _vicinity_obb_mesh(gap_dir, edge_radius=args.edge_radius, color=(0.05,0.85,0.25))
    obb_for_clip = None
    if vic_mesh_and_obb is not None:
        vic_mesh, obb_for_clip = vic_mesh_and_obb
        geoms.append(vic_mesh)

    # Control Volume AABB (red)
    cv_mesh = _control_volume_aabb_mesh(gap_dir, edge_radius=args.edge_radius, color=(0.95,0.15,0.15))
    if cv_mesh is not None:
        geoms.append(cv_mesh)

    # Tubes (blue/green/grey)
    if obb_for_clip is None:
        print("[WARN] Cannot draw tubes (need target_vicinity.json to clip)")
    else:
        max_tubes = None if args.tube_max is None or args.tube_max < 0 else args.tube_max
        geoms.extend(_tube_meshes(
            gap_dir,
            obb_for_clip=obb_for_clip,
            tube_rho_scale=float(args.tube_rho_scale),
            tube_radius_min=float(args.tube_radius_min),
            tube_max=max_tubes,
            color_from_grid=bool(args.color_from_grid)
        ))

    if not geoms:
        print("[ERROR] nothing to visualize"); return

    o3d.visualization.draw_geometries(geoms)

if __name__ == "__main__":
    main()
