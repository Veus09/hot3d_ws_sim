"""HOT3D object library (assets/*.glb|obj|ply) -> USD. Dùng: python 01_convert_assets.py [--src DIR] [--scale S]
HOT3D mesh theo mét; nếu bbox lớn bất thường (>2 m) tự đoán là mm và chia 1000 (có in cảnh báo)."""
import argparse, glob, os, sys
import numpy as np, trimesh, yaml
ws = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, os.path.join(ws, "src"))
from hot3d_sim.usd_utils import mesh_to_usd

ap = argparse.ArgumentParser()
ap.add_argument("--src", default=os.path.join(ws, "data", "hot3d_dataset", "assets"))
ap.add_argument("--out", default=os.path.join(ws, "assets_usd", "hot3d"))
ap.add_argument("--scale", type=float, default=None)
a = ap.parse_args()
cfg = yaml.safe_load(open(os.path.join(ws, "configs", "objects.yaml")))
files = sorted(sum([glob.glob(os.path.join(a.src, "**", e), recursive=True) for e in ("*.glb", "*.obj", "*.ply")], []))
if not files:
    sys.exit(f"Không thấy mesh trong {a.src}. Xem data/README.md (hoặc dùng 02_make_proxy_assets.py).")
os.makedirs(a.out, exist_ok=True)
for f in files:
    sc = trimesh.load(f, force="scene"); m = sc.to_geometry() if hasattr(sc, "to_geometry") else sc.dump(concatenate=True)
    s = a.scale or (0.001 if m.extents.max() > 2.0 else 1.0)
    if s != 1.0: print(f"[warn] {f}: bbox {m.extents.max():.2f} -> scale {s}")
    v = np.asarray(m.vertices) * s
    name = "o_" + os.path.splitext(os.path.basename(f))[0].replace("-", "_")
    k = {**cfg["default"], **cfg.get("overrides", {}).get(name, {})}
    mesh_to_usd(os.path.join(a.out, name + ".usd"), v, m.faces, name=name, mass=k["mass"],
                friction=k["friction"], restitution=k["restitution"], collision=k["collision"])
    print(name, (m.extents * s).round(3))
