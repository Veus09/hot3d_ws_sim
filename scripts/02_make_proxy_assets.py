"""Sinh vật thể proxy (hộp/trụ/cầu/...) -> assets_usd/proxy/*.usd để thử pipeline khi chưa có HOT3D."""
import os, sys
import trimesh
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from hot3d_sim.usd_utils import mesh_to_usd

OUT = os.path.join(os.path.dirname(__file__), "..", "assets_usd", "proxy")
os.makedirs(OUT, exist_ok=True)
items = {
    "box_cracker":  (trimesh.creation.box((0.16, 0.06, 0.21)), (0.8, 0.2, 0.2)),
    "can":          (trimesh.creation.cylinder(0.033, 0.12, sections=32), (0.2, 0.4, 0.8)),
    "ball":         (trimesh.creation.icosphere(subdivisions=3, radius=0.035), (0.9, 0.8, 0.2)),
    "mug":          (trimesh.creation.annulus(0.035, 0.04, 0.09, sections=32), (0.3, 0.7, 0.4)),
    "bottle":       (trimesh.creation.capsule(height=0.14, radius=0.03), (0.6, 0.3, 0.7)),
}
for n, (m, c) in items.items():
    p = mesh_to_usd(os.path.join(OUT, f"{n}.usd"), m.vertices, m.faces, c, name=n)
    print("wrote", p, "bbox(m)=", m.extents.round(3))
