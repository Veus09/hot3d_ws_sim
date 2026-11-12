"""Monitor tương tác: replay 1 sequence HOT3D trong Isaac Sim (có cửa sổ) + bảng điều khiển để lọc task.

  python scripts/07_task_monitor.py --seq P0001_10a27bf7 [--fps 30] [--contact_cm 1.0]
  python scripts/07_task_monitor.py --seq P0001_10a27bf7 --selftest     # chạy không cửa sổ, kiểm tra rồi thoát

Bảng "HOT3D Task Monitor":
  - Phát như video: |< <<10 <1 Phát/Dừng 1> 10>> >|, thanh kéo frame, tốc độ 0.25-4x, lặp lại.
  - Nút bật/tắt theo frame: vật đang hiện -> "Ẩn từ frame này"; đang ẩn -> "Tắt ẩn từ frame này".
    "Ẩn/hiện chỉ frame này" đảo đúng 1 frame. "Mọi vật: tắt ẩn từ frame này".
  - Mỗi vật: tên, tay đang chạm (R / L / RL / -) ở frame hiện tại, các đoạn tay phải / trái chạm cả sequence,
    các đoạn đang GIỮ. Nút: Xóa vật (ẩn mọi frame) · Ẩn từ đây · Ẩn tới đây · Ẩn từ lúc sang tay trái
    (đoạn đầu tiên CHỈ tay trái chạm >= 0.3 s) · Khôi phục.
    Vật bị ẩn ở frame nào thì frame đó không hiện trong sim (dùng visibility, không xoá prim -> hoàn tác được).
  - Tách task: dừng ở chỗ cần cắt -> "Tách tại frame này". Mỗi đoạn giữa các điểm tách là 1 task: đặt tên, Xem,
    Phát (lặp riêng đoạn đó), Bỏ/Giữ, Gộp (bỏ điểm tách). "Lưu + xuất task" ghi mỗi task không bị bỏ thành
    outputs/tasks/<seq>/task_NN.{json,npz} (pose vật đã lọc, tay MANO, cổ tay, tay chạm vật; world Isaac Z-up)
    -> đọc bằng hot3d_sim.tasks.load_task() để cho model chạy từng task.
  - Lưu: outputs/tasks/<seq>.json (đoạn giữ theo frame / giây / timestamp ns, vật bị xoá, đoạn tay chạm).
    Mở lại sequence sẽ tự nạp file đã lưu.

Tay và vật KHÔNG chạy vật lý: đây là replay động học dữ liệu ground truth của HOT3D (motion capture có marker
quang học trên tay và vật -> pose vật dynamic_objects.csv + tham số MANO mano_hand_pose_trajectory.jsonl).
Mỗi frame script đặt lại ma trận pose của vật và toàn bộ 778 đỉnh MANO của tay. "Tay chạm vật" = khoảng cách
nhỏ nhất giữa đỉnh tay và bề mặt vật < --contact_cm (tính trước cho mọi frame trên GPU)."""
import argparse, bisect, collections, csv, json, os, sys, time
from isaaclab.app import AppLauncher

ap = argparse.ArgumentParser()
ap.add_argument("--seq", default="P0001_10a27bf7")
ap.add_argument("--fps", type=float, default=30.0, help="lưới thời gian phát (dữ liệu vật ~30 Hz, tay 60 Hz)")
ap.add_argument("--contact_cm", type=float, default=1.0, help="ngưỡng 'tay chạm vật' (cm)")
ap.add_argument("--selftest", action="store_true", help="không mở cửa sổ; thử các thao tác rồi thoát")
ap.add_argument("--export_only", action="store_true",
                help="không mở cửa sổ: đọc outputs/tasks/<seq>.json, xuất lại task_NN.{json,npz} rồi thoát")
AppLauncher.add_app_launcher_args(ap)
# Isaac Lab 3: không chọn visualizer thì chạy headless -> mặc định bật cửa sổ Kit
if not {"--selftest", "--export_only"} & set(sys.argv) and not any(a in sys.argv for a in ("--viz", "--visualizer", "--headless")):
    sys.argv += ["--viz", "kit"]
args = ap.parse_args()
if args.selftest or args.export_only:
    args.headless = True

ws = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
seq = os.path.join(ws, "data", "hot3d_dataset", args.seq)
if not os.path.isdir(seq):
    hits = sorted(d for d in os.listdir(os.path.join(ws, "data", "hot3d_dataset")) if d.startswith(args.seq[:9]))
    sys.exit(f"không thấy {seq}. Gần giống: {hits[:8]}")

# ------------------------------------------------------------------ đọc dữ liệu (trước khi bật Isaac)
poses = collections.defaultdict(list)  # uid -> [(ts, t, qwxyz)]
for r in csv.DictReader(open(os.path.join(seq, "dynamic_objects.csv"))):
    poses[r["object_uid"]].append((int(r["timestamp[ns]"]),
                                   [float(r[k]) for k in ("t_wo_x[m]", "t_wo_y[m]", "t_wo_z[m]")],
                                   [float(r[k]) for k in ("q_wo_w", "q_wo_x", "q_wo_y", "q_wo_z")]))
if not poses:
    sys.exit(f"{args.seq}: không có pose vật (tập test HOT3D)")
head = {}
for r in csv.DictReader(open(os.path.join(seq, "headset_trajectory.csv"))):
    head[int(r["timestamp[ns]"])] = [float(r[k]) for k in ("t_wo_x[m]", "t_wo_y[m]", "t_wo_z[m]")]
head_ts = sorted(head)
device = open(os.path.join(seq, "device.txt")).read().strip()
for uid in poses: poses[uid].sort()
obj_ts = {u: [p[0] for p in L] for u, L in poses.items()}
t_start = head_ts[0]; t_end = max(head_ts[-1], max(p[-1][0] for p in poses.values()))
K = int((t_end - t_start) / 1e9 * args.fps) + 1
grid_ts = [t_start + int(k / args.fps * 1e9) for k in range(K)]
names = {k: v["instance_name"] for k, v in json.load(open(os.path.join(ws, "data/hot3d_dataset/assets/instance.json"))).items()}
uids = list(poses)
task_file = os.path.join(ws, "outputs", "tasks", f"{args.seq}.json")

app = AppLauncher(args).app

import numpy as np, torch
import omni.usd
from pxr import Gf, Sdf, Usd, UsdGeom, Vt
from hot3d_sim.mano import ManoHands, load_mano_trajectory, vertices_batch


def nearest(ts_list, t):
    i = bisect.bisect_left(ts_list, t)
    if i == len(ts_list) or (i > 0 and t - ts_list[i - 1] < ts_list[i] - t): i -= 1
    return i


def mat(t, q):
    w, x, y, z = q
    R = np.array([[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
                  [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
                  [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]])
    M = np.eye(4); M[:3, :3] = R; M[:3, 3] = t
    return M


def gf(M):  # Gf.Matrix4d là row-vector -> chuyển vị
    return Gf.Matrix4d(*np.asarray(M).T.reshape(-1).tolist())


def usd_points(uid, n=1500):
    st = Usd.Stage.Open(os.path.join(ws, "assets_usd", "hot3d", f"o_{uid}.usd"))
    v = np.asarray(UsdGeom.Mesh(st.GetPrimAtPath(f"/o_{uid}/mesh")).GetPointsAttr().Get(), np.float64)
    return v[:: max(1, len(v) // n)]


def segs(mask):
    """bool array -> [[a, b], ...] (b gồm)"""
    out, s = [], None
    for i, m in enumerate(mask):
        if m and s is None: s = i
        if not m and s is not None: out.append([s, i - 1]); s = None
    if s is not None: out.append([s, len(mask) - 1])
    return out


def fmt_segs(mask, min_len=1):
    ss = [s for s in segs(mask) if s[1] - s[0] + 1 >= min_len]
    if not ss: return "—"
    txt = ", ".join(f"{a / args.fps:.1f}–{(b + 1) / args.fps:.1f}s" for a, b in ss[:8])
    return txt + (f" (+{len(ss) - 8})" if len(ss) > 8 else "")


R_ROOT = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0.0]]) if device == "quest" else np.eye(3)  # world HOT3D -> Isaac
t0 = time.time()

# ---- pose vật trên lưới (world HOT3D)
obj_M = {}
for uid, L in poses.items():
    idx = [nearest(obj_ts[uid], t) for t in grid_ts]
    obj_M[uid] = np.stack([mat(L[i][1], L[i][2]) for i in idx])

# ---- tay MANO trên lưới (world HOT3D), NaN khi thiếu
hands = ManoHands(os.path.join(ws, "data", "mano"))
rows = [r for r in load_mano_trajectory(os.path.join(seq, "mano_hand_pose_trajectory.jsonl")) if r[0] > 0]
hts = [r[0] for r in rows]
hand_v, hand_j, hand_idx = {}, {}, {}  # hand_idx[side][k] = chỉ số dòng trong rows (-1 khi thiếu tay)
for side in ("0", "1"):
    gi = [nearest(hts, t) for t in grid_ts]
    ok = [abs(hts[i] - t) < 5e7 and side in rows[i][1] for i, t in zip(gi, grid_ts)]
    hand_idx[side] = np.array([i if o else -1 for i, o in zip(gi, ok)])
    uniq = sorted({i for i, o in zip(gi, ok) if o})
    V, Jn = vertices_batch(hands, [rows[i][1][side] for i in uniq], side, return_joints=True)
    lut = {i: n for n, i in enumerate(uniq)}
    hv = np.full((K, 778, 3), np.nan, np.float32); hj_ = np.full((K, 16, 3), np.nan, np.float32)
    for k, (i, o) in enumerate(zip(gi, ok)):
        if o: hv[k] = V[lut[i]]; hj_[k] = Jn[lut[i]]
    hand_v[side], hand_j[side] = hv, hj_

# ---- khoảng cách tay-vật mỗi frame (GPU)
dev = "cuda:0" if torch.cuda.is_available() else "cpu"
pts = {uid: usd_points(uid, 600) for uid in uids}
dist = {uid: {} for uid in uids}  # uid -> side -> (K,) mét (inf khi không có tay)
for uid in uids:
    P = torch.tensor(pts[uid], dtype=torch.float32, device=dev)
    M = torch.tensor(obj_M[uid], dtype=torch.float32, device=dev)
    for side in ("0", "1"):
        H = torch.tensor(hand_v[side], dtype=torch.float32, device=dev)
        out = torch.full((K,), float("inf"), device=dev)
        for s in range(0, K, 256):
            Mw = M[s:s + 256]
            W = P[None] @ Mw[:, :3, :3].transpose(1, 2) + Mw[:, None, :3, 3]   # (B,P,3)
            Hb = H[s:s + 256]
            valid = ~torch.isnan(Hb[:, 0, 0])
            if valid.any():
                d = torch.cdist(W[valid], Hb[valid]).amin(dim=(1, 2))
                out[s:s + 256][valid] = d
        dist[uid][side] = out.cpu().numpy()
thr = args.contact_cm / 100.0
touch = {uid: {s: dist[uid][s] < thr for s in ("0", "1")} for uid in uids}
print(f"{args.seq} ({device}): {len(uids)} vật, {K} frame @ {args.fps:g} fps, tiền xử lý {time.time() - t0:.1f}s", flush=True)

# ---- bàn (giống 04) + camera
rest = []
for uid in uids:
    V = pts[uid]; Mk = obj_M[uid]
    for k in range(0, K, 10):
        j = min(K - 1, k + int(0.2 * args.fps))
        if np.linalg.norm(Mk[j, :3, 3] - Mk[k, :3, 3]) > 0.003: continue
        w = (R_ROOT @ (Mk[k, :3, :3] @ V.T + Mk[k, :3, 3:])).T
        rest.append(w[np.argmin(w[:, 2])])
rest = np.array(rest)
head_isaac = (R_ROOT @ np.array([head[t] for t in head_ts]).T).T
floor_z = 0.0 if device == "quest" else min(np.median(head_isaac[:, 2]) - 1.6, rest[:, 2].min() - 0.02 if len(rest) else 0)
tables = []
if len(rest):
    z = np.sort(rest[:, 2]); groups = [[z[0]]]
    for v in z[1:]:
        (groups[-1].append(v) if v - groups[-1][-1] < 0.02 else groups.append([v]))
    for g in groups:
        if len(g) < max(15, 0.04 * len(rest)): continue
        h = float(np.median(g))
        if h - floor_z < 0.08: continue
        p_ = rest[np.abs(rest[:, 2] - h) < 0.025]
        tables.append((h, p_[:, :2].min(0) - 0.12, p_[:, :2].max(0) + 0.12))

# ------------------------------------------------------------------ dựng cảnh
ctx = omni.usd.get_context()
stage = ctx.get_stage()
UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z); UsdGeom.SetStageMetersPerUnit(stage, 1.0)
UsdGeom.Xform.Define(stage, "/World")


def box(path, center, size, color):
    g = UsdGeom.Cube.Define(stage, path); g.GetSizeAttr().Set(1.0)
    x = UsdGeom.Xformable(g); x.AddTranslateOp().Set(Gf.Vec3d(*map(float, center))); x.AddScaleOp().Set(Gf.Vec3f(*map(float, size)))
    g.CreateDisplayColorAttr([Gf.Vec3f(*color)])


box("/World/floor", (0, 0, floor_z - 0.01), (8, 8, 0.02), (0.12, 0.13, 0.15))
for i, (h, lo, hi) in enumerate(tables):
    c = (lo + hi) / 2; s = hi - lo
    box(f"/World/table{i}", (c[0], c[1], h - 0.015), (s[0], s[1], 0.03), (0.78, 0.62, 0.42))
    box(f"/World/table{i}_base", (c[0], c[1], (h - 0.03 + floor_z) / 2),
        (max(0.05, s[0] - 0.2), max(0.05, s[1] - 0.2), h - 0.03 - floor_z), (0.35, 0.3, 0.25))
from pxr import UsdLux
UsdLux.DomeLight.Define(stage, "/World/dome").CreateIntensityAttr(250.0)
UsdLux.DistantLight.Define(stage, "/World/sun").CreateIntensityAttr(900.0)

root = UsdGeom.Xform.Define(stage, "/World/hot3d")
M_root = np.eye(4); M_root[:3, :3] = R_ROOT
UsdGeom.Xformable(root).AddTransformOp().Set(gf(M_root))
palette = [(0.9, 0.3, 0.3), (0.3, 0.7, 0.95), (0.95, 0.8, 0.2), (0.4, 0.85, 0.45), (0.8, 0.5, 0.9), (0.95, 0.6, 0.3)]
ops, prims = {}, {}
for i, uid in enumerate(uids):
    p = stage.DefinePrim(f"/World/hot3d/o_{uid}", "Xform")
    p.GetReferences().AddReference(os.path.join(ws, "assets_usd", "hot3d", f"o_{uid}.usd"))
    p.CreateAttribute("physics:kinematicEnabled", Sdf.ValueTypeNames.Bool).Set(True)
    ops[uid] = UsdGeom.Xformable(p).AddTransformOp(); prims[uid] = UsdGeom.Imageable(p)
    mesh = UsdGeom.Mesh(stage.GetPrimAtPath(f"/World/hot3d/o_{uid}/mesh"))
    if mesh: mesh.CreateDisplayColorAttr([Gf.Vec3f(*palette[i % len(palette)])])
hand_mesh = {}
for side, nm, col in (("0", "hand_left", (0.55, 0.75, 0.95)), ("1", "hand_right", (0.85, 0.6, 0.48))):
    m = UsdGeom.Mesh.Define(stage, f"/World/hot3d/{nm}")
    F = hands.faces[side]
    m.CreateFaceVertexCountsAttr(Vt.IntArray([3] * len(F)))
    m.CreateFaceVertexIndicesAttr(Vt.IntArray(F.reshape(-1).tolist()))
    m.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(np.zeros((778, 3), np.float32)))
    m.CreateSubdivisionSchemeAttr("none"); m.CreateDoubleSidedAttr(True)
    m.CreateDisplayColorAttr([Gf.Vec3f(*col)])
    hand_mesh[side] = m

# camera nhìn giống video batch (04); người dùng vẫn xoay/zoom tự do trong viewport
c = np.median(rest, 0) if len(rest) else np.zeros(3)
hm = np.median(head_isaac, 0)
d = hm[:2] - c[:2]; d = d / (np.linalg.norm(d) + 1e-9)
eye = Gf.Vec3d(c[0] + 0.8 * d[0], c[1] + 0.8 * d[1], c[2] + 1.4)
tgt = Gf.Vec3d(c[0] + 0.2 * d[0], c[1] + 0.2 * d[1], c[2] + 0.12)
cam = UsdGeom.Camera.Define(stage, "/World/monitor_cam")
cam.CreateFocalLengthAttr(14.0); cam.CreateHorizontalApertureAttr(20.955); cam.CreateClippingRangeAttr(Gf.Vec2f(0.05, 30.0))
view = Gf.Matrix4d().SetLookAt(eye, tgt, Gf.Vec3d(0, 0, 1))
UsdGeom.Xformable(cam).AddTransformOp().Set(view.GetInverse())
if not (args.selftest or args.export_only):
    try:
        from omni.kit.viewport.utility import get_active_viewport
        get_active_viewport().camera_path = "/World/monitor_cam"
    except Exception as e:  # noqa: BLE001
        print("không đặt được camera viewport:", e)

# ------------------------------------------------------------------ trạng thái lọc
keep = {uid: np.ones(K, bool) for uid in uids}
removed = {uid: False for uid in uids}
if os.path.exists(task_file):
    try:
        J = json.load(open(task_file))
        for uid, o in J.get("objects", {}).items():
            if uid in keep:
                keep[uid][:] = False
                for a, b in o.get("keep_frames", []): keep[uid][a:b + 1] = True
                removed[uid] = bool(o.get("removed", False))
        print("đã nạp", task_file)
    except Exception as e:  # noqa: BLE001
        print("không đọc được", task_file, e)

SPL = []      # điểm tách (frame bắt đầu của mỗi task, trừ task đầu tiên bắt đầu ở 0)
SEGMETA = {}  # frame bắt đầu -> {"name": str, "skip": bool}
if os.path.exists(task_file):
    try:
        J = json.load(open(task_file))
        SPL[:] = sorted(int(x) for x in J.get("splits", []) if 0 < int(x) < K)
        for t in J.get("tasks", []):
            SEGMETA[int(t["start_frame"])] = {"name": t.get("name", ""), "skip": bool(t.get("skip", False))}
    except Exception as e:  # noqa: BLE001
        print("không đọc được điểm tách:", e)
S = {"seg": None, "k": 0, "playing": False, "speed": 1.0, "loop": True, "acc": 0.0, "last": time.time(),
     "show": {"0": True, "1": True}, "busy": False}


def visible(uid, k):
    return (not removed[uid]) and bool(keep[uid][k])


def apply_frame(k):
    k = int(max(0, min(K - 1, k))); S["k"] = k
    for uid in uids:
        ops[uid].Set(gf(obj_M[uid][k]))
        (prims[uid].MakeVisible() if visible(uid, k) else prims[uid].MakeInvisible())
    for side, m in hand_mesh.items():
        v = hand_v[side][k]
        if np.isnan(v[0, 0]) or not S["show"][side]:
            m.MakeInvisible(); continue
        m.GetPointsAttr().Set(Vt.Vec3fArray.FromNumpy(v)); m.MakeVisible()
    refresh_labels()


def holder(uid, k):
    s = ("R" if touch[uid]["1"][k] else "") + ("L" if touch[uid]["0"][k] else "")
    return s or "-"


def save():
    os.makedirs(os.path.dirname(task_file), exist_ok=True)
    objs = {}
    for uid in uids:
        kf = [] if removed[uid] else segs(keep[uid])
        objs[uid] = {"name": names.get(uid, uid), "removed": removed[uid],
                     "keep_frames": kf,
                     "keep_seconds": [[round(a / args.fps, 3), round((b + 1) / args.fps, 3)] for a, b in kf],
                     "keep_timestamps_ns": [[grid_ts[a], grid_ts[b]] for a, b in kf],
                     "right_touch_seconds": [[round(a / args.fps, 3), round((b + 1) / args.fps, 3)] for a, b in segs(touch[uid]["1"])],
                     "left_touch_seconds": [[round(a / args.fps, 3), round((b + 1) / args.fps, 3)] for a, b in segs(touch[uid]["0"])]}
    J = {"seq": args.seq, "device": device, "fps": args.fps, "n_frames": K, "t_start_ns": t_start,
         "contact_cm": args.contact_cm, "saved": time.strftime("%Y-%m-%d %H:%M:%S"), "objects": objs,
         "splits": list(SPL),
         "tasks": [{"index": i + 1, "start_frame": a, "end_frame": b, "seconds": [round(a / args.fps, 3), round((b + 1) / args.fps, 3)],
                    "name": meta(a)["name"], "skip": meta(a)["skip"]} for i, (a, b) in enumerate(segments())]}
    json.dump(J, open(task_file, "w"), indent=1, ensure_ascii=False)
    status(f"Đã lưu {os.path.relpath(task_file, ws)}")
    return J


# ---- thao tác lọc
def act(uid, what):
    k = S["k"]
    if what == "remove": removed[uid] = True
    elif what == "restore": removed[uid] = False; keep[uid][:] = True
    elif what == "hide_from": keep[uid][k:] = False
    elif what == "hide_until": keep[uid][:k + 1] = False
    elif what == "toggle_from":  # đang hiện -> ẩn từ đây; đang ẩn -> tắt ẩn từ đây
        if removed[uid]:
            removed[uid] = False; keep[uid][:k] = False; keep[uid][k:] = True
        else:
            keep[uid][k:] = not keep[uid][k]
    elif what == "toggle_frame":
        if removed[uid]: removed[uid] = False; keep[uid][:] = False
        keep[uid][k] = not keep[uid][k]
    elif what == "hide_left":
        k0 = left_switch(uid)
        if k0 is None: status(f"{names.get(uid, uid)}: không có đoạn chỉ tay trái cầm ≥ {LEFT_MIN_S}s"); return
        keep[uid][k0:] = False
    status(f"{names.get(uid, uid)}: {what} @ {k / args.fps:.2f}s")
    apply_frame(k)


LEFT_MIN_S = 0.3


def left_switch(uid):
    """frame đầu tiên của đoạn CHỈ tay trái chạm (tay phải không chạm) dài >= LEFT_MIN_S; None nếu không có.
    Tay trái chạm thoáng qua khi tay phải vẫn đang cầm thì không tính."""
    only_l = touch[uid]["0"] & ~touch[uid]["1"]
    for a, b in segs(only_l):
        if (b - a + 1) / args.fps >= LEFT_MIN_S: return a
    return None


def show_all_from():
    k = S["k"]
    for uid in uids:
        if removed[uid]: removed[uid] = False; keep[uid][:k] = False
        keep[uid][k:] = True
    status(f"Mọi vật: tắt ẩn từ {k / args.fps:.2f}s"); apply_frame(k)


def act_all_left():
    n = 0
    for uid in uids:
        k0 = left_switch(uid)
        if k0 is not None: keep[uid][k0:] = False; n += 1
    status(f"Ẩn từ lúc chuyển sang tay trái: {n} vật"); apply_frame(S["k"])



# ---- tách task
def segments():
    st = [0] + SPL
    return [(a, (st[i + 1] - 1) if i + 1 < len(st) else K - 1) for i, a in enumerate(st)]


def meta(a):
    return SEGMETA.setdefault(a, {"name": "", "skip": False})


def seg_of(k):
    for i, (a, b) in enumerate(segments()):
        if a <= k <= b: return i, a, b
    return 0, 0, K - 1


def split_here():
    k = S["k"]; S["playing"] = False
    if k <= 0 or k in SPL:
        status("Không tách được ở frame 0 hoặc nơi đã có điểm tách"); return
    SPL.append(k); SPL.sort(); meta(k)
    i, a, b = seg_of(k)
    status(f"Đã tách tại {k / args.fps:.2f}s -> task #{i + 1} bắt đầu từ đây ({len(SPL) + 1} task)")
    tasks_changed()


def merge_prev(a):
    if a in SPL:
        SPL.remove(a); SEGMETA.pop(a, None)
        status(f"Đã bỏ điểm tách {a / args.fps:.2f}s (gộp với task trước)")
    tasks_changed()


def toggle_skip(a):
    meta(a)["skip"] = not meta(a)["skip"]; tasks_changed()


def play_seg(a, b):
    S["seg"] = (a, b); apply_frame(a); S["playing"] = True; S["last"] = time.time(); S["acc"] = 0.0
    status(f"Phát riêng task {a / args.fps:.1f}-{(b + 1) / args.fps:.1f}s (lặp). Bấm 'Phát toàn bộ' để thoát.")
    refresh_labels()


def play_all():
    S["seg"] = None; status("Phát toàn bộ sequence"); refresh_labels()


def tasks_changed():
    if "tasks_frame" in W: W["tasks_frame"].rebuild()
    refresh_labels()


def export_tasks(save_first=True):
    """Ghi mỗi task (không bị bỏ) thành <out>/task_NN.json + .npz; toạ độ world Isaac (Z-up, mét)."""
    if save_first: save()
    out = os.path.join(os.path.dirname(task_file), os.path.splitext(os.path.basename(task_file))[0])
    os.makedirs(out, exist_ok=True)
    for f in os.listdir(out):
        if f.startswith("task_"): os.remove(os.path.join(out, f))
    Mroot = np.eye(4); Mroot[:3, :3] = R_ROOT
    n = 0
    for i, (a, b) in enumerate(segments()):
        m = meta(a)
        if m["skip"]: continue
        n += 1; sl = slice(a, b + 1); T = b - a + 1
        objs = [u for u in uids if not removed[u] and keep[u][sl].any()]
        npz = {"t": (np.arange(T) / args.fps).astype(np.float32), "timestamps_ns": np.array(grid_ts[a:b + 1], np.int64),
               "obj_uids": np.array(objs),
               "obj_pose": np.stack([Mroot @ obj_M[u][sl] for u in objs]).astype(np.float32) if objs else np.zeros((0, T, 4, 4), np.float32),
               "obj_visible": np.stack([keep[u][sl] for u in objs]) if objs else np.zeros((0, T), bool)}
        for side, nm in (("1", "right"), ("0", "left")):
            npz[f"{nm}_hand_verts"] = (hand_v[side][sl] @ R_ROOT.T).astype(np.float32)
            npz[f"{nm}_hand_joints"] = (hand_j[side][sl] @ R_ROOT.T).astype(np.float32)
            idx = hand_idx[side][sl]; valid = idx >= 0
            npz[f"{nm}_hand_valid"] = valid
            W_ = np.full((T, 4, 4), np.nan, np.float32); X_ = np.full((T, 4, 4), np.nan, np.float32); P_ = np.full((T, 15), np.nan, np.float32); B_ = np.full((T, 10), np.nan, np.float32)
            for j, r in enumerate(idx):
                if r < 0: continue
                hj = rows[r][1][side]; w = hj["wrist_xform"]
                X_[j] = Mroot @ mat(w["t_xyz"], w["q_wxyz"]); P_[j] = hj["pose"]; B_[j] = hj["betas"]
                W_[j] = X_[j]; W_[j][:3, 3] = R_ROOT @ hand_j[side][a + j][0]  # vị trí = khớp cổ tay MANO thật
            npz[f"{nm}_wrist_pose"], npz[f"{nm}_mano_wrist_xform"] = W_, X_
            npz[f"{nm}_mano_pose_pca15"], npz[f"{nm}_mano_betas"] = P_, B_
            npz[f"{nm}_touch"] = np.stack([touch[u][side][sl] for u in objs]) if objs else np.zeros((0, T), bool)
        base = os.path.join(out, f"task_{n:02d}")
        np.savez_compressed(base + ".npz", **npz)
        rel = lambda mask: [[round(x / args.fps, 3), round((y + 1) / args.fps, 3)] for x, y in segs(mask)]
        J = {"seq": args.seq, "device": device, "task": n, "segment": i + 1, "name": m["name"], "fps": args.fps,
             "frames": [a, b], "seconds_in_seq": [round(a / args.fps, 3), round((b + 1) / args.fps, 3)],
             "timestamps_ns": [grid_ts[a], grid_ts[b]], "n_frames": T,
             "objects": [{"uid": u, "name": names.get(u, u), "usd": f"assets_usd/hot3d/o_{u}.usd",
                          "visible_seconds": rel(keep[u][sl]), "right_touch_seconds": rel(touch[u]["1"][sl]),
                          "left_touch_seconds": rel(touch[u]["0"][sl])} for u in objs],
             "scene": {"world": "Isaac Z-up, m", "floor_z": float(floor_z),
                       "tables": [{"top_z": float(h), "xy_min": lo.tolist(), "xy_max": hi.tolist()} for h, lo, hi in tables],
                       "camera_eye": list(eye), "camera_target": list(tgt)},
             "npz": os.path.basename(base) + ".npz",
             "npz_keys": {"t": "(T,) giây từ đầu task", "obj_pose": "(N_obj,T,4,4) T_world_object", "obj_visible": "(N_obj,T)",
                          "<side>_hand_verts": "(T,778,3) MANO, NaN khi thiếu",
                          "<side>_hand_joints": "(T,16,3) khớp MANO (0 = cổ tay)",
                          "<side>_wrist_pose": "(T,4,4) xoay = global_orient MANO, vị trí = khớp cổ tay",
                          "<side>_mano_wrist_xform": "(T,4,4) wrist_xform gốc của HOT3D (tịnh tiến = transl MANO, KHÔNG phải cổ tay)",
                          "<side>_mano_pose_pca15": "(T,15) PCA HOT3D", "<side>_mano_betas": "(T,10)",
                          "<side>_touch": "(N_obj,T) tay chạm vật", "side": "right | left"}}
        json.dump(J, open(base + ".json", "w"), indent=1, ensure_ascii=False)
    status(f"Đã xuất {n} task -> {os.path.relpath(out, ws)}/")
    return out, n

# ------------------------------------------------------------------ giao diện
W = {}

def asc(t):
    """omni.ui chỉ có glyph Latin cơ bản -> bỏ dấu tiếng Việt, thay ký hiệu lạ."""
    import unicodedata
    t = str(t).replace("đ", "d").replace("Đ", "D").replace("–", "-").replace("—", "-")
    return unicodedata.normalize("NFKD", t).encode("ascii", "ignore").decode()



def status(msg):
    print(msg, flush=True)
    if "status" in W: W["status"].text = asc(msg)


def refresh_labels():
    if "frame" not in W: return
    k = S["k"]
    W["frame"].text = asc(f"frame {k}/{K - 1}   t = {k / args.fps:6.2f} s / {(K - 1) / args.fps:.1f} s")
    if not S["busy"]:
        S["busy"] = True; W["slider"].model.set_value(k); S["busy"] = False
    W["play"].text = asc("Dừng" if S["playing"] else "Phát")
    i, a, b = seg_of(k); m = meta(a)
    W["seg"].text = asc(f"Task hiện tại: #{i + 1}/{len(SPL) + 1}  {a / args.fps:.1f}-{(b + 1) / args.fps:.1f}s"
                        + ("  [BỎ]" if m["skip"] else "") + (f"  '{m['name']}'" if m["name"] else "")
                        + ("   (đang phát riêng task)" if S["seg"] else ""))
    for uid in uids:
        st = "ĐÃ XÓA" if removed[uid] else ("hiện" if keep[uid][k] else "ẩn ở frame này")
        W[f"cur_{uid}"].text = asc(f"tay: {holder(uid, k):<2}   {st}")
        if f"tog_{uid}" in W:
            on = visible(uid, k)
            W[f"tog_{uid}"].text = asc("Ẩn từ frame này" if on else "Tắt ẩn từ frame này")
            W[f"tog_{uid}"].set_style({} if on else {"color": 0xFF80FF80})
        W[f"keep_{uid}"].text = asc("giữ: " + ("- (đã xóa)" if removed[uid] else fmt_segs(keep[uid])))


def build_task_list(ui):
    with ui.VStack(spacing=2, height=0):
        for i, (a, b) in enumerate(segments()):
            m = meta(a)
            with ui.HStack(height=24, spacing=3):
                ui.Label(f"#{i + 1}  {a / args.fps:.1f}-{(b + 1) / args.fps:.1f}s", width=110,
                         style={"color": 0xFF777777 if m["skip"] else 0xFFDDDDDD})
                f = ui.StringField(width=ui.Fraction(1))
                f.model.set_value(m["name"])
                f.model.add_value_changed_fn(lambda mdl, a=a: (meta(a).__setitem__("name", mdl.get_value_as_string()), refresh_labels()))
                ui.Button("Xem", width=40, clicked_fn=lambda a=a: (S.update(playing=False), apply_frame(a)))
                ui.Button("Phát", width=40, clicked_fn=lambda a=a, b=b: play_seg(a, b))
                ui.Button("Giữ" if m["skip"] else "Bỏ", width=40, clicked_fn=lambda a=a: toggle_skip(a))
                if a > 0:
                    ui.Button("Gộp", width=40, clicked_fn=lambda a=a: merge_prev(a), tooltip=asc("gộp với task trước (bỏ điểm tách)"))
                else:
                    ui.Spacer(width=40)


def build_ui():
    import types
    import omni.ui as _ui  # chỉ có khi chạy có cửa sổ
    ui = types.SimpleNamespace(**{k: getattr(_ui, k) for k in dir(_ui) if not k.startswith("__")})
    ui.Label = lambda text="", **kw: _ui.Label(asc(text), **kw)
    ui.Button = lambda text="", **kw: _ui.Button(asc(text), **kw)
    ui.CollapsableFrame = lambda title="", **kw: _ui.CollapsableFrame(asc(title), **kw)
    win = ui.Window("HOT3D Task Monitor", width=600, height=800)
    win.deferred_dock_in("Stage", ui.DockPolicy.TARGET_WINDOW_IS_ACTIVE)  # cột phải, không che viewport
    with win.frame:
        with ui.ScrollingFrame():
            with ui.VStack(spacing=6, height=0):
                ui.Label(f"{args.seq}  ({device}, {len(uids)} vật, {args.fps:g} fps)", style={"font_size": 17})
                W["frame"] = ui.Label("")
                W["slider"] = ui.IntSlider(min=0, max=K - 1, height=22)
                W["slider"].model.add_value_changed_fn(
                    lambda m: None if S["busy"] else (S.update(playing=False), apply_frame(m.get_value_as_int())))
                with ui.HStack(height=28, spacing=4):
                    ui.Button("|<", width=36, clicked_fn=lambda: apply_frame(0))
                    ui.Button("<< 10", width=52, clicked_fn=lambda: (S.update(playing=False), apply_frame(S["k"] - 10)))
                    ui.Button("< 1", width=40, clicked_fn=lambda: (S.update(playing=False), apply_frame(S["k"] - 1)))
                    W["play"] = ui.Button("Phát", width=80, clicked_fn=toggle_play)
                    ui.Button("1 >", width=40, clicked_fn=lambda: (S.update(playing=False), apply_frame(S["k"] + 1)))
                    ui.Button("10 >>", width=52, clicked_fn=lambda: (S.update(playing=False), apply_frame(S["k"] + 10)))
                    ui.Button(">|", width=36, clicked_fn=lambda: apply_frame(K - 1))
                with ui.HStack(height=24, spacing=6):
                    ui.Label("Tốc độ", width=50)
                    speeds = [0.25, 0.5, 1.0, 2.0, 4.0]
                    cb = ui.ComboBox(2, *[f"{s:g}x" for s in speeds], width=80)
                    cb.model.get_item_value_model().add_value_changed_fn(
                        lambda m: S.update(speed=speeds[m.get_value_as_int()]))
                    lp = ui.CheckBox(width=20); lp.model.set_value(True)
                    lp.model.add_value_changed_fn(lambda m: S.update(loop=m.get_value_as_bool()))
                    ui.Label("Lặp lại", width=60)
                    for side, nm in (("1", "Tay phải"), ("0", "Tay trái")):
                        c_ = ui.CheckBox(width=20); c_.model.set_value(True)
                        c_.model.add_value_changed_fn(lambda m, s=side: (S["show"].__setitem__(s, m.get_value_as_bool()), apply_frame(S["k"])))
                        ui.Label(nm, width=60)
                ui.Separator()
                with ui.CollapsableFrame("Tách task", height=0):
                    with ui.VStack(spacing=4, height=0):
                        W["seg"] = ui.Label("", style={"color": 0xFF80D0FF})
                        with ui.HStack(height=30, spacing=4):
                            ui.Button("Tách tại frame này", clicked_fn=split_here, style={"color": 0xFF80D0FF})
                            ui.Button("Phát toàn bộ", clicked_fn=play_all)
                            ui.Button("Lưu + xuất task", clicked_fn=export_tasks, style={"color": 0xFF80FF80})
                        W["tasks_frame"] = ui.Frame(height=0, build_fn=lambda: build_task_list(ui))
                ui.Separator()
                ui.Label(f"Vật thể — 'tay' = tay đang chạm (< {args.contact_cm:g} cm): R phải, L trái", style={"color": 0xFFAAAAAA})
                for i, uid in enumerate(uids):
                    col = palette[i % len(palette)]
                    argb = 0xFF000000 | (int(col[2] * 255) << 16) | (int(col[1] * 255) << 8) | int(col[0] * 255)
                    with ui.CollapsableFrame(f"{names.get(uid, uid)}  ({uid})", height=0):
                        with ui.VStack(spacing=3, height=0):
                            with ui.HStack(height=18):
                                ui.Rectangle(width=14, height=14, style={"background_color": argb})
                                ui.Spacer(width=6)
                                W[f"cur_{uid}"] = ui.Label("")
                            ui.Label("tay phải chạm: " + fmt_segs(touch[uid]["1"], 3), word_wrap=True, style={"color": 0xFF7FD0A0})
                            ui.Label("tay trái chạm: " + fmt_segs(touch[uid]["0"], 3), word_wrap=True, style={"color": 0xFFE0B070})
                            W[f"keep_{uid}"] = ui.Label("", word_wrap=True)
                            with ui.HStack(height=26, spacing=3):
                                ui.Button("Xóa vật", clicked_fn=lambda u=uid: act(u, "remove"), style={"color": 0xFF8080FF})
                                W[f"tog_{uid}"] = ui.Button("Ẩn từ frame này", clicked_fn=lambda u=uid: act(u, "toggle_from"))
                                ui.Button("Ẩn tới đây", clicked_fn=lambda u=uid: act(u, "hide_until"))
                            with ui.HStack(height=26, spacing=3):
                                ui.Button("Ẩn/hiện chỉ frame này", clicked_fn=lambda u=uid: act(u, "toggle_frame"))
                            with ui.HStack(height=26, spacing=3):
                                ui.Button("Ẩn từ lúc sang tay trái", clicked_fn=lambda u=uid: act(u, "hide_left"))
                                ui.Button("Khôi phục", clicked_fn=lambda u=uid: act(u, "restore"))
                ui.Separator()
                ui.Button("Mọi vật: ẩn từ lúc chuyển sang tay trái", clicked_fn=act_all_left, height=28)
                ui.Button("Mọi vật: tắt ẩn từ frame này", clicked_fn=show_all_from, height=28)
                ui.Button("Lưu task", clicked_fn=save, height=32, style={"color": 0xFF80FF80})
                W["status"] = ui.Label("", word_wrap=True, style={"color": 0xFFAAAAAA})
    return win


def toggle_play():
    S["playing"] = not S["playing"]; S["last"] = time.time(); S["acc"] = 0.0
    if S["playing"] and S["k"] >= K - 1: apply_frame(0)
    refresh_labels()


# ------------------------------------------------------------------ chạy
if args.export_only:
    if not os.path.exists(task_file):
        print("không có", task_file); app.close(); sys.exit(1)
    out_dir, n = export_tasks(save_first=False)  # không ghi đè file mô tả task
    app.close(); sys.exit(0)

if args.selftest:
    for u in uids: keep[u][:] = True; removed[u] = False  # không dùng file lọc thật của người dùng
    SPL.clear(); SEGMETA.clear()
    apply_frame(0)
    for _ in range(5): app.update()
    mug = next((u for u in uids if "mug" in names.get(u, "")), uids[0])
    print("tay chạm", names.get(mug), "| phải:", fmt_segs(touch[mug]["1"], 3), "| trái:", fmt_segs(touch[mug]["0"], 3))
    for k in range(0, K, 37):
        apply_frame(k); app.update()
    u2 = uids[0]
    apply_frame(300); act(u2, "toggle_from"); assert not keep[u2][300:].any() and keep[u2][:300].all()
    apply_frame(390); act(u2, "toggle_from"); assert keep[u2][390:].all() and not keep[u2][300:390].any()
    act(u2, "toggle_frame"); assert not keep[u2][390]; act(u2, "toggle_frame"); assert keep[u2][390]
    act(u2, "remove"); apply_frame(500); act(u2, "toggle_from"); assert not removed[u2] and keep[u2][500:].all() and not keep[u2][:500].any()
    act(u2, "restore")
    act(mug, "hide_left"); act(uids[-1], "remove")
    real_file = task_file; task_file = os.path.join(ws, "outputs", "tasks", f"_selftest_{args.seq}.json")
    SPL.clear(); SEGMETA.clear()
    for k in (600, 1200, 1800, 2400): apply_frame(k); split_here()
    merge_prev(2400); assert segments() == [(0, 599), (600, 1199), (1200, 1799), (1800, K - 1)], segments()
    meta(0)["name"] = "cam coc tay phai"; toggle_skip(600)
    out_dir, n = export_tasks(); assert n == 3, n
    import glob
    T1 = np.load(os.path.join(out_dir, "task_01.npz")); J1 = json.load(open(os.path.join(out_dir, "task_01.json")))
    print("task_01:", J1["name"], J1["seconds_in_seq"], {k_: T1[k_].shape for k_ in ("obj_pose", "right_hand_verts", "right_wrist_pose", "right_touch")},
          [o["name"] for o in J1["objects"]])
    assert T1["obj_pose"].shape[1] == 600 and len(glob.glob(os.path.join(out_dir, "task_*"))) == 6
    J = save()
    assert [t["skip"] for t in J["tasks"]] == [False, True, False, False]
    import shutil; shutil.rmtree(out_dir)
    print("selftest:", json.dumps({u: (o["name"], o["removed"], o["keep_seconds"][:3]) for u, o in J["objects"].items()}, ensure_ascii=False))
    os.remove(task_file)
    print("SELFTEST OK")
    app.close(); sys.exit(0)

win = build_ui()
apply_frame(0)
for _ in range(10): app.update()
win.focus()  # tab monitor hiện lên trước tab Stage
status("Sẵn sàng. Kéo thanh frame hoặc bấm Phát. Lọc xong bấm Lưu task.")
while app.is_running():
    if S["playing"]:
        now = time.time(); S["acc"] += (now - S["last"]) * args.fps * S["speed"]; S["last"] = now
        step = int(S["acc"])
        if step:
            S["acc"] -= step; k = S["k"] + step
            lo, hi = S["seg"] if S["seg"] else (0, K - 1)
            if k > hi:
                if S["loop"]: k = lo
                else: k = hi; S["playing"] = False
            apply_frame(k)
    app.update()
app.close()
