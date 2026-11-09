"""Replay pose GT của vật thể HOT3D trong Isaac Sim -> outputs/replay/frames/*.png (rồi ghép mp4 bằng ffmpeg).
  python 04_replay_poses.py [--seq P0003_d0ad9b47] [--step 2] [--max_frames N]
HOT3D: world Y-up, đơn vị mét, quaternion WXYZ, T_world_object. Isaac: Z-up -> bọc 1 Xform gốc xoay +90° quanh X
(x,y,z)->(x,-z,y). Mesh giữ nguyên toạ độ glb nên chỉ cần xoay ở gốc. Vật thể đặt kinematic (không rơi)."""
import argparse, csv, os, collections
from isaaclab.app import AppLauncher

ap = argparse.ArgumentParser()
ap.add_argument("--seq", default="P0003_d0ad9b47")
ap.add_argument("--step", type=int, default=2, help="lấy 1 trong N frame pose (pose 30 Hz)")
ap.add_argument("--max_frames", type=int, default=0)
ap.add_argument("--width", type=int, default=960); ap.add_argument("--height", type=int, default=540)
AppLauncher.add_app_launcher_args(ap)
args = ap.parse_args(); args.headless = True; args.enable_cameras = True
app = AppLauncher(args).app

import numpy as np, torch, omni.usd
from PIL import Image
from pxr import Gf, Sdf, UsdGeom, UsdPhysics, Vt
import isaaclab.sim as sim_utils
from isaaclab.sensors.camera import Camera, CameraCfg

ws = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
seq = os.path.join(ws, "data", "hot3d_dataset", args.seq)
out = os.path.join(ws, "outputs", "replay", "frames"); os.makedirs(out, exist_ok=True)
for f in os.listdir(out): os.remove(os.path.join(out, f))

# ---- đọc pose GT
poses = collections.defaultdict(dict)  # uid -> {ts: (t, qwxyz)}
for r in csv.DictReader(open(os.path.join(seq, "dynamic_objects.csv"))):
    poses[r["object_uid"]][int(r["timestamp[ns]"])] = (
        [float(r[k]) for k in ("t_wo_x[m]", "t_wo_y[m]", "t_wo_z[m]")],
        [float(r[k]) for k in ("q_wo_w", "q_wo_x", "q_wo_y", "q_wo_z")])
head = {}
for r in csv.DictReader(open(os.path.join(seq, "headset_trajectory.csv"))):
    head[int(r["timestamp[ns]"])] = (
        [float(r[k]) for k in ("t_wo_x[m]", "t_wo_y[m]", "t_wo_z[m]")],
        [float(r[k]) for k in ("q_wo_w", "q_wo_x", "q_wo_y", "q_wo_z")])
stamps = sorted({t for d in poses.values() for t in d})[::args.step]
if args.max_frames: stamps = stamps[:args.max_frames]
names = {k: v["instance_name"] for k, v in
         __import__("json").load(open(os.path.join(ws, "data/hot3d_dataset/assets/instance.json"))).items()}


def mat(t, q):
    w, x, y, z = q
    R = np.array([[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
                  [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
                  [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]])
    M = np.eye(4); M[:3, :3] = R; M[:3, 3] = t
    return M


def gf(M):  # Gf.Matrix4d là row-vector -> chuyển vị
    return Gf.Matrix4d(*M.T.reshape(-1).tolist())


sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=1 / 30, device="cuda:0", use_fabric=False))
stage = omni.usd.get_context().get_stage()
UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z); UsdGeom.SetStageMetersPerUnit(stage, 1.0)

fl = UsdGeom.Cube.Define(stage, "/World/floor"); fl.GetSizeAttr().Set(1.0)
x = UsdGeom.Xformable(fl); x.AddTranslateOp().Set(Gf.Vec3d(0, 0, -0.01)); x.AddScaleOp().Set(Gf.Vec3f(8, 8, 0.02))
fl.CreateDisplayColorAttr([Gf.Vec3f(0.12, 0.13, 0.15)])
sim_utils.DomeLightCfg(intensity=250.0).func("/World/dome", sim_utils.DomeLightCfg(intensity=250.0))
sim_utils.DistantLightCfg(intensity=900.0).func("/World/sun", sim_utils.DistantLightCfg(intensity=900.0))

root = UsdGeom.Xform.Define(stage, "/World/hot3d")  # Y-up -> Z-up
UsdGeom.Xformable(root).AddTransformOp().Set(gf(np.array([[1, 0, 0, 0], [0, 0, -1, 0], [0, 1, 0, 0], [0, 0, 0, 1.0]])))

# ---- bàn: HOT3D không kèm mesh phòng -> dựng xấp xỉ từ chính pose (toạ độ HOT3D Y-up, gốc Y=0 là sàn).
# Độ cao: đáy vật thể tụ ở y~0.72 (bàn chữ nhật) và y~0.42 (bàn tròn thấp). Kích thước/vị trí: bao quanh vùng
# vật thể từng nằm lên + lề ~10 cm; KHÔNG phải số đo thật (xem topdown.png). Chỉnh ở đây nếu cần.
TABLES = dict(
    rect=dict(center=(0.31, -0.325), size=(0.66, 0.66), top=0.72, thick=0.04, color=(0.78, 0.62, 0.42)),  # (x,z)
    round=dict(center=(0.51, 0.35), radius=0.30, top=0.42, thick=0.03, color=(0.60, 0.48, 0.34)),
)


def add_static(path, prim_type, pos, color, **kw):
    g = (UsdGeom.Cube if prim_type == "cube" else UsdGeom.Cylinder).Define(stage, path)
    x = UsdGeom.Xformable(g); x.AddTranslateOp().Set(Gf.Vec3d(*pos))
    if prim_type == "cube":
        g.GetSizeAttr().Set(1.0); x.AddScaleOp().Set(Gf.Vec3f(*kw["size"]))
    else:
        g.GetAxisAttr().Set("Y"); g.GetRadiusAttr().Set(kw["radius"]); g.GetHeightAttr().Set(kw["height"])
    g.CreateDisplayColorAttr([Gf.Vec3f(*color)]); UsdPhysics.CollisionAPI.Apply(g.GetPrim())


t = TABLES["rect"]; cx, cz = t["center"]; sx, sz = t["size"]
add_static("/World/hot3d/table_rect", "cube", (cx, t["top"] - t["thick"] / 2, cz), t["color"], size=(sx, t["thick"], sz))
for i, (dx, dz) in enumerate([(-1, -1), (-1, 1), (1, -1), (1, 1)]):  # 4 chân
    h = t["top"] - t["thick"]
    add_static(f"/World/hot3d/table_rect_leg{i}", "cube", (cx + dx * (sx / 2 - 0.05), h / 2, cz + dz * (sz / 2 - 0.05)),
               (0.35, 0.3, 0.25), size=(0.05, h, 0.05))
t = TABLES["round"]; cx, cz = t["center"]
add_static("/World/hot3d/table_round", "cyl", (cx, t["top"] - t["thick"] / 2, cz), t["color"], radius=t["radius"], height=t["thick"])
add_static("/World/hot3d/table_round_stem", "cyl", (cx, (t["top"] - t["thick"]) / 2, cz), (0.35, 0.3, 0.25),
           radius=0.03, height=t["top"] - t["thick"])

palette = [(0.9, 0.3, 0.3), (0.3, 0.7, 0.95), (0.95, 0.8, 0.2), (0.4, 0.85, 0.45), (0.8, 0.5, 0.9), (0.95, 0.6, 0.3)]
ops = {}
for i, uid in enumerate(poses):
    p = stage.DefinePrim(f"/World/hot3d/o_{uid}", "Xform")
    p.GetReferences().AddReference(os.path.join(ws, "assets_usd", "hot3d", f"o_{uid}.usd"))
    p.CreateAttribute("physics:kinematicEnabled", Sdf.ValueTypeNames.Bool).Set(True)
    ops[uid] = UsdGeom.Xformable(p).AddTransformOp()
    # tô màu riêng để phân biệt
    mesh = UsdGeom.Mesh(stage.GetPrimAtPath(f"/World/hot3d/o_{uid}/mesh"))
    if mesh: mesh.CreateDisplayColorAttr([Gf.Vec3f(*palette[i % len(palette)])])
    print(i, uid, names.get(uid))

# kính VR: hộp nhỏ + mũi tên hướng nhìn (trục -Z? ta chỉ vẽ hộp, hướng suy từ pose)
hm = UsdGeom.Cube.Define(stage, "/World/hot3d/headset"); hm.GetSizeAttr().Set(1.0)
hx = UsdGeom.Xformable(hm); hop = hx.AddTransformOp(); hx.AddScaleOp().Set(Gf.Vec3f(0.19, 0.08, 0.10))
hm.CreateDisplayColorAttr([Gf.Vec3f(0.95, 0.95, 0.95)])

# camera quan sát: nhìn từ phía trước-trên vào cụm vật thể
allp = np.array([v[0] for d in poses.values() for v in d.values()])
c_y = allp.mean(0); c = np.array([c_y[0], -c_y[2], c_y[1]])  # sang Z-up
cam = Camera(CameraCfg(prim_path="/World/cam", update_period=0, height=args.height, width=args.width, data_types=["rgb"],
                       spawn=sim_utils.PinholeCameraCfg(focal_length=18.0, horizontal_aperture=20.955, clipping_range=(0.05, 30.0))))
sim.reset()
eye = c + np.array([0.0, -1.7, 1.0]); tgt = c + np.array([0.0, 0.0, -0.05])
cam.set_world_poses_from_view(torch.tensor([eye], dtype=torch.float32, device=sim.device),
                              torch.tensor([tgt], dtype=torch.float32, device=sim.device))
print("centre", c.round(2), "frames", len(stamps))

for n, ts in enumerate(stamps):
    for uid, d in poses.items():
        if ts in d: ops[uid].Set(gf(mat(*d[ts])))
    if ts in head: hop.Set(gf(mat(*head[ts])))
    sim.step(); cam.update(dt=sim.get_physics_dt())
    if n < 2: continue  # bỏ 2 khung khởi động
    Image.fromarray(cam.data.output["rgb"][0, ..., :3].cpu().numpy().astype(np.uint8)).save(os.path.join(out, f"{n:05d}.png"))
    if n % 100 == 0: print("frame", n, flush=True)
print("done", len(os.listdir(out)), "frames ->", out)
app.close()
