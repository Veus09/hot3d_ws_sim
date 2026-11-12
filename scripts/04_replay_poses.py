"""Replay pose GT của vật thể HOT3D + tay MANO trong Isaac Sim -> outputs/replay/<seq>/frames/%05d.png.
  python 04_replay_poses.py --seq P0003_d0ad9b47 [--fps 15] [--start_s S] [--dur_s D] [--no_hands] [--video] [--resume]

- Lấy mẫu theo lưới thời gian cố định (--fps) tính từ frame đầu của headset_trajectory (= đầu recording, cùng mốc
  với preview_rgb.mp4). Frame k <-> t = k/fps giây. Mỗi luồng (vật, tay) lấy pose gần nhất.
- Trục: Quest = world Y-up; Aria (MPS) = world Z-up. Isaac Z-up -> Quest bọc trong Xform gốc xoay +90° quanh X
  (x,y,z)->(x,-z,y); Aria giữ nguyên.
- Bàn: HOT3D không có mesh phòng -> tự ước lượng: điểm thấp nhất của vật lúc nằm yên, gom theo độ cao (mỗi cụm là
  1 mặt bàn), mặt bàn phủ vùng các điểm đó + lề. KHÔNG phải số đo thật.
- Camera: phía sau đầu người (theo vị trí kính trung bình), nhìn xuống cụm vật.
- Vật thể kinematic (không rơi). Sequence không có pose vật (tập test HOT3D) -> thoát mã 3."""
import argparse, bisect, collections, csv, json, os, subprocess, sys
from isaaclab.app import AppLauncher

ap = argparse.ArgumentParser()
ap.add_argument("--seq", default="P0003_d0ad9b47")
ap.add_argument("--fps", type=float, default=15.0)
ap.add_argument("--start_s", type=float, default=0.0)
ap.add_argument("--dur_s", type=float, default=0.0, help="0 = tới hết sequence")
ap.add_argument("--no_hands", action="store_true")
ap.add_argument("--video", action="store_true", help="ghép frames -> outputs/replay/<seq>.mp4 (VP9 trong mp4)")
ap.add_argument("--resume", action="store_true", help="không xoá frames cũ; bỏ qua frame đã có")
ap.add_argument("--out_dir", default="", help="mặc định outputs/replay/<seq>/frames")
ap.add_argument("--width", type=int, default=960); ap.add_argument("--height", type=int, default=540)
AppLauncher.add_app_launcher_args(ap)
args = ap.parse_args(); args.headless = True; args.enable_cameras = True

ws = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
seq = os.path.join(ws, "data", "hot3d_dataset", args.seq)

# ---- đọc dữ liệu (trước khi bật Isaac, để sequence không có GT thoát nhanh)
poses = collections.defaultdict(list)  # uid -> [(ts, t, qwxyz)]
for r in csv.DictReader(open(os.path.join(seq, "dynamic_objects.csv"))):
    poses[r["object_uid"]].append((int(r["timestamp[ns]"]),
                                   [float(r[k]) for k in ("t_wo_x[m]", "t_wo_y[m]", "t_wo_z[m]")],
                                   [float(r[k]) for k in ("q_wo_w", "q_wo_x", "q_wo_y", "q_wo_z")]))
if not poses:
    print(f"{args.seq}: không có pose vật (tập test HOT3D) -> bỏ qua"); sys.exit(3)
head_ts = sorted(int(r["timestamp[ns]"]) for r in csv.DictReader(open(os.path.join(seq, "headset_trajectory.csv"))))
head = {}
for r in csv.DictReader(open(os.path.join(seq, "headset_trajectory.csv"))):
    head[int(r["timestamp[ns]"])] = [float(r[k]) for k in ("t_wo_x[m]", "t_wo_y[m]", "t_wo_z[m]")]
device = open(os.path.join(seq, "device.txt")).read().strip()
t_start = head_ts[0]; t_end = max(head_ts[-1], max(p[-1][0] for p in poses.values()))
n_all = int((t_end - t_start) / 1e9 * args.fps) + 1
k0 = int(round(args.start_s * args.fps))
k1 = n_all if args.dur_s <= 0 else min(n_all, k0 + int(round(args.dur_s * args.fps)))
grid = [(k, t_start + int(k / args.fps * 1e9)) for k in range(k0, k1)]
out = args.out_dir or os.path.join(ws, "outputs", "replay", args.seq, "frames")
os.makedirs(out, exist_ok=True)
if not args.resume:
    for f in os.listdir(out): os.remove(os.path.join(out, f))

app = AppLauncher(args).app
import numpy as np, torch, omni.usd
from PIL import Image
from pxr import Gf, Sdf, Usd, UsdGeom, Vt
import isaaclab.sim as sim_utils
from isaaclab.sensors.camera import Camera, CameraCfg

for uid in poses: poses[uid].sort()
obj_ts = {u: [p[0] for p in L] for u, L in poses.items()}


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


R_ROOT = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0.0]]) if device == "quest" else np.eye(3)  # world HOT3D -> Isaac

hands = None
if not args.no_hands and os.path.exists(os.path.join(ws, "data", "mano", "MANO_RIGHT.pkl")):
    from hot3d_sim.mano import ManoHands, load_mano_trajectory
    hands = ManoHands(os.path.join(ws, "data", "mano"))
    hand_rows = [r for r in load_mano_trajectory(os.path.join(seq, "mano_hand_pose_trajectory.jsonl")) if r[0] > 0]
    hand_ts = [r[0] for r in hand_rows]
names = {k: v["instance_name"] for k, v in json.load(open(os.path.join(ws, "data/hot3d_dataset/assets/instance.json"))).items()}


def usd_points(uid):
    st = Usd.Stage.Open(os.path.join(ws, "assets_usd", "hot3d", f"o_{uid}.usd"))
    m = UsdGeom.Mesh(st.GetPrimAtPath(f"/o_{uid}/mesh"))
    v = np.asarray(m.GetPointsAttr().Get(), np.float64)
    return v[:: max(1, len(v) // 1500)]


# ---- ước lượng bàn (toạ độ Isaac, Z-up)
rest = []  # (x, y, z_thấp_nhất)
for uid, L in poses.items():
    V = usd_points(uid)
    P = np.array([p[1] for p in L]); T = np.array([p[0] for p in L])
    for i in range(0, len(L), 10):
        j = min(len(L) - 1, nearest(obj_ts[uid], T[i] + int(0.2e9)))
        if np.linalg.norm(P[j] - P[i]) > 0.003: continue  # đang di chuyển
        w = (R_ROOT @ (mat(P[i], L[i][2])[:3, :3] @ V.T + P[i][:, None])).T
        rest.append(w[np.argmin(w[:, 2])])
rest = np.array(rest)
head_isaac = (R_ROOT @ np.array(list(head.values())).T).T
floor_z = 0.0 if device == "quest" else min(np.median(head_isaac[:, 2]) - 1.6, rest[:, 2].min() - 0.02 if len(rest) else 0)
tables = []
if len(rest):
    z = np.sort(rest[:, 2]); groups = [[z[0]]]
    for v in z[1:]:
        (groups[-1].append(v) if v - groups[-1][-1] < 0.02 else groups.append([v]))
    for g in groups:
        if len(g) < max(15, 0.04 * len(rest)): continue
        h = float(np.median(g))
        if h - floor_z < 0.08: continue  # vật nằm trên sàn
        pts = rest[np.abs(rest[:, 2] - h) < 0.025]
        lo, hi = pts[:, :2].min(0) - 0.12, pts[:, :2].max(0) + 0.12
        tables.append((h, lo, hi))

sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=1 / 30, device="cuda:0", use_fabric=False))
stage = omni.usd.get_context().get_stage()
UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z); UsdGeom.SetStageMetersPerUnit(stage, 1.0)


def box(path, center, size, color):
    g = UsdGeom.Cube.Define(stage, path); g.GetSizeAttr().Set(1.0)
    x = UsdGeom.Xformable(g); x.AddTranslateOp().Set(Gf.Vec3d(*map(float, center))); x.AddScaleOp().Set(Gf.Vec3f(*map(float, size)))
    g.CreateDisplayColorAttr([Gf.Vec3f(*color)])


box("/World/floor", (0, 0, floor_z - 0.01), (8, 8, 0.02), (0.12, 0.13, 0.15))
for i, (h, lo, hi) in enumerate(tables):
    c = (lo + hi) / 2; s = hi - lo
    box(f"/World/table{i}", (c[0], c[1], h - 0.015), (s[0], s[1], 0.03), (0.78, 0.62, 0.42))
    box(f"/World/table{i}_base", (c[0], c[1], (h - 0.03 + floor_z) / 2), (max(0.05, s[0] - 0.2), max(0.05, s[1] - 0.2), h - 0.03 - floor_z),
        (0.35, 0.3, 0.25))
sim_utils.DomeLightCfg(intensity=250.0).func("/World/dome", sim_utils.DomeLightCfg(intensity=250.0))
sim_utils.DistantLightCfg(intensity=900.0).func("/World/sun", sim_utils.DistantLightCfg(intensity=900.0))

root = UsdGeom.Xform.Define(stage, "/World/hot3d")
M_root = np.eye(4); M_root[:3, :3] = R_ROOT
UsdGeom.Xformable(root).AddTransformOp().Set(gf(M_root))

palette = [(0.9, 0.3, 0.3), (0.3, 0.7, 0.95), (0.95, 0.8, 0.2), (0.4, 0.85, 0.45), (0.8, 0.5, 0.9), (0.95, 0.6, 0.3)]
ops = {}
for i, uid in enumerate(poses):
    p = stage.DefinePrim(f"/World/hot3d/o_{uid}", "Xform")
    p.GetReferences().AddReference(os.path.join(ws, "assets_usd", "hot3d", f"o_{uid}.usd"))
    p.CreateAttribute("physics:kinematicEnabled", Sdf.ValueTypeNames.Bool).Set(True)
    ops[uid] = UsdGeom.Xformable(p).AddTransformOp()
    mesh = UsdGeom.Mesh(stage.GetPrimAtPath(f"/World/hot3d/o_{uid}/mesh"))
    if mesh: mesh.CreateDisplayColorAttr([Gf.Vec3f(*palette[i % len(palette)])])

hand_mesh = {}
if hands is not None:
    for side, nm, col in (("0", "hand_left", (0.95, 0.72, 0.6)), ("1", "hand_right", (0.85, 0.6, 0.48))):
        m = UsdGeom.Mesh.Define(stage, f"/World/hot3d/{nm}")
        F = hands.faces[side]
        m.CreateFaceVertexCountsAttr(Vt.IntArray([3] * len(F)))
        m.CreateFaceVertexIndicesAttr(Vt.IntArray(F.reshape(-1).tolist()))
        m.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(np.zeros((778, 3), np.float32)))
        m.CreateSubdivisionSchemeAttr("none"); m.CreateDoubleSidedAttr(True)
        m.CreateDisplayColorAttr([Gf.Vec3f(*col)])
        hand_mesh[side] = m

# camera: sau đầu người, cao hơn mặt bàn, nhìn vào cụm vật
c = np.median(rest, 0) if len(rest) else (R_ROOT @ np.mean([p[1] for L in poses.values() for p in L], 0))
hm = np.median(head_isaac, 0)
d = hm[:2] - c[:2]; d = d / (np.linalg.norm(d) + 1e-9)
eye = np.array([c[0] + 0.8 * d[0], c[1] + 0.8 * d[1], c[2] + 1.4]); tgt = c + np.array([0.2 * d[0], 0.2 * d[1], 0.12])  # cao, dốc: tay đưa lên mặt ít che; nhìn lệch về phía người
cam = Camera(CameraCfg(prim_path="/World/cam", update_period=0, height=args.height, width=args.width, data_types=["rgb"],
                       spawn=sim_utils.PinholeCameraCfg(focal_length=14.0, horizontal_aperture=20.955, clipping_range=(0.05, 30.0))))
sim.reset()
cam.set_world_poses_from_view(torch.tensor([eye], dtype=torch.float32, device=sim.device),
                              torch.tensor([tgt], dtype=torch.float32, device=sim.device))
print(f"{args.seq} ({device}): {len(poses)} vật, frames {k0}..{k1 - 1} @ {args.fps:g} fps, "
      f"bàn {[round(t[0], 2) for t in tables]}, sàn {floor_z:.2f}, cam {eye.round(2)} -> {tgt.round(2)}, đầu {hm.round(2)}", flush=True)

for n, (k, t) in enumerate(grid):
    fn = os.path.join(out, f"{k:05d}.png")
    if args.resume and os.path.exists(fn) and n > 0: continue
    for uid, L in poses.items():
        _, tt, q = L[nearest(obj_ts[uid], t)]
        ops[uid].Set(gf(mat(tt, q)))
    if hand_mesh:
        i = nearest(hand_ts, t); ok = abs(hand_ts[i] - t) < 5e7
        for side, m in hand_mesh.items():
            hj = hand_rows[i][1].get(side) if ok else None
            if hj is None:
                m.MakeInvisible(); continue
            m.GetPointsAttr().Set(Vt.Vec3fArray.FromNumpy(hands.vertices(hj, side).astype(np.float32)))
            m.MakeVisible()
    for _ in range(3 if n == 0 else 1):  # frame đầu: render thêm để renderer ổn định
        sim.step(); cam.update(dt=sim.get_physics_dt())
    Image.fromarray(cam.data.output["rgb"][0, ..., :3].cpu().numpy().astype(np.uint8)).save(fn)
    if n % 300 == 0: print("frame", k, flush=True)
print("done", len(os.listdir(out)), "frames ->", out, flush=True)
if args.video:
    vid = os.path.join(ws, "outputs", "replay", f"{args.seq}.mp4")
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", f"{args.fps:g}", "-start_number", str(k0),
                    "-i", os.path.join(out, "%05d.png"), "-c:v", "libvpx-vp9", "-crf", "32", "-b:v", "0", "-row-mt", "1",
                    "-cpu-used", "4", "-pix_fmt", "yuv420p", "-movflags", "+faststart", vid], check=True)
    print("video ->", vid)
app.close()
