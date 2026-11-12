"""Đọc task đã xuất từ scripts/07_task_monitor.py ("Lưu + xuất task").

    from hot3d_sim.tasks import load_task, list_tasks
    for path in list_tasks("P0001_10a27bf7"):
        T = load_task(path)
        T["name"], T["objects"]            # metadata (json)
        T["obj_pose"]                      # (N_obj, T, 4, 4) T_world_object, world Isaac Z-up, mét
        T["obj_visible"]                   # (N_obj, T) bool — vật bị lọc ở frame đó thì False
        T["right_hand_verts"]              # (T, 778, 3) MANO tay phải, NaN khi không có tay
        T["right_wrist_pose"]              # (T, 4, 4) xoay = global_orient MANO, vị trí = khớp cổ tay
        T["right_hand_joints"]             # (T, 16, 3) khớp MANO (0 = cổ tay)
        T["right_mano_wrist_xform"]        # (T, 4, 4) wrist_xform gốc HOT3D (tịnh tiến = transl MANO)
        T["right_mano_pose_pca15"], T["right_mano_betas"], T["right_touch"]  # (N_obj, T) tay phải chạm vật
        # tương tự left_*; T["t"] giây từ đầu task, T["timestamps_ns"] mốc gốc của HOT3D
"""
import glob
import json
import os

import numpy as np

WS = os.environ.get("HOT3D_WS", os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


def list_tasks(seq, root=None):
    root = root or os.path.join(WS, "outputs", "tasks")
    return sorted(glob.glob(os.path.join(root, seq, "task_*.json")))


def load_task(path):
    path = path[:-4] + ".json" if path.endswith(".npz") else path
    J = json.load(open(path))
    with np.load(os.path.join(os.path.dirname(path), J["npz"])) as z:
        J.update({k: z[k] for k in z.files})
    return J
