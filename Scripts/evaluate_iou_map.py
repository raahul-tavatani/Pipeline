import json, math, os, csv, glob
import numpy as np

# -----------------------------
# Geometry: rotated rectangle IoU (BEV)
# -----------------------------
def _wrap_pi(a):
    a = (a + math.pi) % (2 * math.pi) - math.pi
    return a

def _rect_corners(cx, cy, l, w, yaw):
    c, s = math.cos(yaw), math.sin(yaw)
    hx, hy = l * 0.5, w * 0.5
    pts = [(-hx, -hy), ( hx, -hy), ( hx,  hy), (-hx,  hy)]
    out = []
    for x, y in pts:
        xr = c * x - s * y + cx
        yr = s * x + c * y + cy
        out.append((xr, yr))
    return out  # 4 vertices CCW

def _poly_area(poly):
    if len(poly) < 3: return 0.0
    s = 0.0
    for i in range(len(poly)):
        x1, y1 = poly[i]
        x2, y2 = poly[(i + 1) % len(poly)]
        s += x1 * y2 - x2 * y1
    return abs(s) * 0.5

def _clip_poly(subject, clip):
    # Sutherland–Hodgman polygon clipping (convex clip)
    def inside(p, a, b):
        return (b[0]-a[0])*(p[1]-a[1]) - (b[1]-a[1])*(p[0]-a[0]) >= 0.0
    def intersect(a1, a2, b1, b2):
        # segment intersection (a1->a2) with (b1->b2)
        x1,y1 = a1; x2,y2 = a2; x3,y3 = b1; x4,y4 = b2
        den = (x1-x2)*(y3-y4) - (y1-y2)*(x3-x4)
        if abs(den) < 1e-9:
            return a2  # nearly parallel; fall back
        px = ((x1*y2 - y1*x2)*(x3-x4) - (x1-x2)*(x3*y4 - y3*x4)) / den
        py = ((x1*y2 - y1*x2)*(y3-y4) - (y1-y2)*(x3*y4 - y3*x4)) / den
        return (px, py)

    output = subject[:]
    for i in range(len(clip)):
        input_list = output[:]
        output = []
        A = clip[i]
        B = clip[(i+1) % len(clip)]
        if not input_list:
            break
        S = input_list[-1]
        for E in input_list:
            if inside(E, A, B):
                if not inside(S, A, B):
                    output.append(intersect(S, E, A, B))
                output.append(E)
            elif inside(S, A, B):
                output.append(intersect(S, E, A, B))
            S = E
    return output

def iou_bev_rotated(box_a, box_b):
    # box = (x,y,z,l,w,h,yaw)
    xa, ya, _, la, wa, _, ra = box_a
    xb, yb, _, lb, wb, _, rb = box_b
    pa = _rect_corners(xa, ya, la, wa, _wrap_pi(ra))
    pb = _rect_corners(xb, yb, lb, wb, _wrap_pi(rb))
    area_a = _poly_area(pa)
    area_b = _poly_area(pb)
    inter_poly = _clip_poly(pa, pb)
    inter_a = _poly_area(inter_poly)
    union = max(area_a + area_b - inter_a, 1e-9)
    return inter_a / union

# -----------------------------
# Adapters for your JSON formats
# -----------------------------
def _yaw_normalize(yaw_val):
    y = float(yaw_val)
    # If gt yaw came in degrees, convert
    if abs(y) > math.pi * 1.01:
        y = math.radians(y)
    return _wrap_pi(y)

def load_gt_boxes(gt_json_path, flip_y=False):
    """
    Returns list[(box7, class_id)], where box7=(x,y,z,l,w,h,yaw).
    Assumes CARLA extents are half-dimensions; length=2*extent.x, width=2*extent.y.
    Maps all vehicles to class_id=1 ("Car") by default.
    """
    with open(gt_json_path, 'r') as f:
        data = json.load(f)
    out = []
    for obj in data:
        c = obj.get("center_lidar", {})
        e = obj.get("extent", {})
        r = obj.get("rotation", {})
        x, y, z = float(c.get("x", 0.0)), float(c.get("y", 0.0)), float(c.get("z", 0.0))
        if flip_y: y = -y  # flip CARLA y-right -> KITTI y-left if needed
        l = 2.0 * float(e.get("x", 0.0))
        w = 2.0 * float(e.get("y", 0.0))
        h = 2.0 * float(e.get("z", 0.0))
        yaw = _yaw_normalize(r.get("yaw", 0.0))
        # Simple class map: treat vehicles as 1; expand if needed
        cls = 1 if str(obj.get("type_id","")).startswith("vehicle.") else 255
        out.append(((x,y,z,l,w,h,yaw), cls))
    return out

def load_pred_boxes(pred_json_path, score_thr=0.0):
    """
    Expects {"boxes":[[x,y,z,l,w,h,yaw],...], "scores":[...], "labels":[...]}
    Returns list[(box7, class_id, score)] filtered by score_thr.
    """
    with open(pred_json_path, 'r') as f:
        p = json.load(f)
    boxes = p.get("boxes", [])
    labels = p.get("labels", [])
    scores = p.get("scores", [])
    out = []
    for b, c, s in zip(boxes, labels, scores):
        if float(s) < score_thr: 
            continue
        x,y,z,l,w,h,yaw = [float(v) for v in b]
        out.append(((x,y,z,l,w,h,_yaw_normalize(yaw)), int(c), float(s)))
    return out

# -----------------------------
# Greedy matching by IoU
# -----------------------------
def match_greedy_iou(preds, gts, cls_set={1}, iou_thr=0.50):
    """
    preds: list[(box7, class, score)]
    gts:   list[(box7, class)]
    Only classes in cls_set are evaluated.
    Returns: TP, FP, FN
    """
    P = [(b,c,s) for (b,c,s) in preds if c in cls_set]
    G = [(b,c) for (b,c) in gts if c in cls_set]
    if not P and not G:
        return 0,0,0

    used_g = set()
    tp = 0
    # Sort preds by score desc for stable greedy
    P = sorted(P, key=lambda t: t[2], reverse=True)
    for (pb, pc, ps) in P:
        best_iou, best_g = 0.0, -1
        for gi, (gb, gc) in enumerate(G):
            if gi in used_g: 
                continue
            iou = iou_bev_rotated(pb, gb)
            if iou > best_iou:
                best_iou, best_g = iou, gi
        if best_iou >= iou_thr and best_g >= 0:
            tp += 1
            used_g.add(best_g)
    fp = max(len(P) - tp, 0)
    fn = max(len(G) - tp, 0)
    return tp, fp, fn

# -----------------------------
# Main: walk saved_data/* scenarios
# -----------------------------
def evaluate_root(saved_root, iou_thr=0.50, score_thr=0.0, gt_flip_y=False, out_csv=None):
    scenario_dirs = [d for d in glob.glob(os.path.join(saved_root, "*")) if os.path.isdir(d)]
    rows = [["Folder","File","TP","FP","FN","Precision","Recall","F1-Score"]]
    for scen in sorted(scenario_dirs):
        scen_name = os.path.basename(scen)
        gt_dir   = os.path.join(scen, "json")
        pred_dir = os.path.join(scen, "pred_json")
        if not (os.path.isdir(gt_dir) and os.path.isdir(pred_dir)):
            continue
        gt_files = sorted(glob.glob(os.path.join(gt_dir, "*.json")))
        tot_tp = tot_fp = tot_fn = 0
        for gt_path in gt_files:
            fname = os.path.basename(gt_path)
            pred_path = os.path.join(pred_dir, fname)
            if not os.path.isfile(pred_path):
                continue
            gts = load_gt_boxes(gt_path, flip_y=gt_flip_y)
            preds = load_pred_boxes(pred_path, score_thr=score_thr)
            tp, fp, fn = match_greedy_iou(preds, gts, cls_set={1}, iou_thr=iou_thr)
            tot_tp += tp; tot_fp += fp; tot_fn += fn
            prec = (tp / (tp + fp)) if (tp + fp) > 0 else 0.0
            rec  = (tp / (tp + fn)) if (tp + fn) > 0 else 0.0
            f1   = (2*prec*rec / (prec+rec)) if (prec+rec) > 0 else 0.0
            rows.append([scen_name, fname, tp, fp, fn,
                        f"{prec:.3f}", f"{rec:.3f}", f"{f1:.3f}"])
        # totals per scenario
        prec = (tot_tp / (tot_tp + tot_fp)) if (tot_tp + tot_fp) > 0 else 0.0
        rec  = (tot_tp / (tot_tp + tot_fn)) if (tot_tp + tot_fn) > 0 else 0.0
        f1   = (2*prec*rec / (prec+rec)) if (prec+rec) > 0 else 0.0
        rows.append([scen_name, "[TOTAL]", tot_tp, tot_fp, tot_fn,
                    f"{prec:.3f}", f"{rec:.3f}", f"{f1:.3f}"])

    out_csv = out_csv or os.path.join(saved_root, "per_file_metrics.csv")
    with open(out_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(rows)
    print(f"Saved results -> {out_csv}")

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=r"C:\Pipeline\saved_data", help="saved_data root")
    ap.add_argument("--iou", type=float, default=0.50, help="IoU threshold (BEV)")
    ap.add_argument("--score_thr", type=float, default=0.0, help="min score for detections")
    ap.add_argument("--gt_flip_y", action="store_true", help="flip GT y if still in CARLA frame")
    ap.add_argument("--out", default="", help="output CSV path")
    args = ap.parse_args()
    evaluate_root(args.root, iou_thr=args.iou, score_thr=args.score_thr,
                  gt_flip_y=args.gt_flip_y, out_csv=args.out or None)
