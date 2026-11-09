"""Nạp các USD vật thể thành lưới trên 1 cái bàn + chụp ảnh bằng Isaac Lab Camera. Headless mặc định.
  python 03_view_objects.py --usd_dir assets_usd/proxy [--gui] [--drop]"""
import argparse, glob, os
from isaaclab.app import AppLauncher

ap = argparse.ArgumentParser()
ap.add_argument("--usd_dir", default="assets_usd/proxy")
ap.add_argument("--out", default="outputs/view_objects.png")
ap.add_argument("--gui", action="store_true")
ap.add_argument("--drop", action="store_true", help="thả vật thể xuống bàn (vật lý) thay vì đặt sẵn")
AppLauncher.add_app_launcher_args(ap)
args = ap.parse_args(); args.headless = not args.gui; args.enable_cameras = True
app = AppLauncher(args).app

import numpy as np
import torch
import omni.usd
from PIL import Image
from pxr import Gf, UsdGeom, UsdPhysics
import isaaclab.sim as sim_utils
from isaaclab.sensors.camera import Camera, CameraCfg

ws = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
usd_dir = args.usd_dir if os.path.isabs(args.usd_dir) else os.path.join(ws, args.usd_dir)
files = sorted(glob.glob(os.path.join(usd_dir, "*.usd")))
assert files, f"không có USD trong {usd_dir}"

sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=1 / 120, device="cuda:0"))
stage = omni.usd.get_context().get_stage()
UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z); UsdGeom.SetStageMetersPerUnit(stage, 1.0)


def box(path, pos, size, color):
    c = UsdGeom.Cube.Define(stage, path); c.GetSizeAttr().Set(1.0)
    x = UsdGeom.Xformable(c); x.AddTranslateOp().Set(Gf.Vec3d(*pos)); x.AddScaleOp().Set(Gf.Vec3f(*size))
    UsdPhysics.CollisionAPI.Apply(c.GetPrim()); c.CreateDisplayColorAttr([Gf.Vec3f(*color)])


box("/World/floor", (0, 0, -0.01), (6, 6, 0.02), (0.3, 0.3, 0.3))
box("/World/table", (0, 0, 0.74), (1.2, 0.8, 0.02), (0.55, 0.45, 0.35))  # mặt bàn z=0.75
sim_utils.DomeLightCfg(intensity=400.0).func("/World/dome", sim_utils.DomeLightCfg(intensity=400.0))
sim_utils.DistantLightCfg(intensity=1200.0).func("/World/sun", sim_utils.DistantLightCfg(intensity=1200.0))

cols = min(len(files), 6)
for i, f in enumerate(files):
    prim = stage.DefinePrim(f"/World/objs/{os.path.splitext(os.path.basename(f))[0]}", "Xform")
    prim.GetReferences().AddReference(f)
    x = (i % cols - (cols - 1) / 2) * 0.17; y = (i // cols) * 0.17 - 0.1
    UsdGeom.Xformable(prim).AddTranslateOp().Set(Gf.Vec3d(x, y, 0.80 + (0.15 if args.drop else 0.0)))

cam = Camera(CameraCfg(prim_path="/World/cam", update_period=0, height=720, width=1280, data_types=["rgb"],
                       spawn=sim_utils.PinholeCameraCfg(focal_length=24.0, horizontal_aperture=20.955,
                                                        clipping_range=(0.05, 20.0))))
sim.reset()
cam.set_world_poses_from_view(torch.tensor([[0.0, -0.9, 1.45]], device=sim.device),
                              torch.tensor([[0.0, 0.0, 0.78]], device=sim.device))
for _ in range(150 if args.drop else 30):
    sim.step(); cam.update(dt=sim.get_physics_dt())
img = cam.data.output["rgb"][0, ..., :3].cpu().numpy().astype(np.uint8)
out = args.out if os.path.isabs(args.out) else os.path.join(ws, args.out)
os.makedirs(os.path.dirname(out), exist_ok=True)
Image.fromarray(img).save(out); print("saved", out, img.shape, "mean", img.mean().round(1))
if args.gui:
    while app.is_running():
        sim.step()
app.close()
