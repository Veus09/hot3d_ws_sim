"""Tay MANO cho HOT3D (giống hot3d/data_loaders/mano_layer.py của toolkit HOT3D).

- MANO_*.pkl gốc chứa chumpy (không chạy với numpy 2) -> chuyển 1 lần sang numpy thuần
  ở data/mano/plain/. File gốc giữ nguyên.
- Pose HOT3D: 15 hệ số PCA, betas 10, wrist_xform (q_wxyz, t_xyz) = T_world_wrist.
  Khoá "0" = tay trái, "1" = tay phải.
"""
import json
import pickle
from pathlib import Path

import numpy as np
import torch


class _Ch:  # thay cho lớp chumpy khi unpickle
    def __init__(self, *a, **k):
        self.st = {}

    def __setstate__(self, st):
        self.st = st


class _Unpickler(pickle.Unpickler):
    def find_class(self, module, name):
        return _Ch if module.startswith("chumpy") else super().find_class(module, name)


def _resolve(v):
    if not isinstance(v, _Ch):
        return v
    st = v.st
    if "x" in st:  # Ch lá
        return np.asarray(st["x"])
    if "a" in st and "idxs" in st:  # chumpy select
        a = np.asarray(_resolve(st["a"]))
        return a.ravel()[st["idxs"]].reshape(st["preferred_shape"])
    raise ValueError(f"không hiểu chumpy state: {list(st)}")


def plain_mano_dir(mano_dir):
    """Trả về thư mục chứa MANO_{LEFT,RIGHT}.pkl không có chumpy (tạo nếu chưa có)."""
    mano_dir = Path(mano_dir)
    out = mano_dir / "plain"
    out.mkdir(exist_ok=True)
    for side in ("LEFT", "RIGHT"):
        dst = out / f"MANO_{side}.pkl"
        if dst.exists():
            continue
        with open(mano_dir / f"MANO_{side}.pkl", "rb") as f:
            d = _Unpickler(f, encoding="latin1").load()
        d = {k: _resolve(v) for k, v in d.items()}
        with open(dst, "wb") as f:
            pickle.dump(d, f)
    return out


class ManoHands:
    """FK MANO cho cả 2 tay; trả vertices trong toạ độ world của HOT3D (mét, Y-up)."""

    def __init__(self, mano_dir):
        import smplx

        d = plain_mano_dir(mano_dir)
        kw = dict(model_type="mano", use_pca=True, num_pca_comps=15)
        self.layers = {
            "0": smplx.create(str(d / "MANO_LEFT.pkl"), is_rhand=False, **kw),
            "1": smplx.create(str(d / "MANO_RIGHT.pkl"), is_rhand=True, **kw),
        }
        L, R = self.layers["0"], self.layers["1"]
        # lỗi shapedirs tay trái (smplx issue #48), toolkit HOT3D cũng sửa như vậy
        if torch.sum(torch.abs(L.shapedirs[:, 0, :] - R.shapedirs[:, 0, :])) < 1:
            L.shapedirs[:, 0, :] *= -1
        self.faces = {k: np.asarray(l.faces, dtype=np.int64) for k, l in self.layers.items()}

    @torch.no_grad()
    def vertices(self, hand_json, side):
        """hand_json = record["hand_poses"][side] của mano_hand_pose_trajectory.jsonl -> (778,3)."""
        from scipy.spatial.transform import Rotation

        w = hand_json["wrist_xform"]
        qw, qx, qy, qz = w["q_wxyz"]
        rotvec = Rotation.from_quat([qx, qy, qz, qw]).as_rotvec()
        f = lambda x: torch.tensor(np.asarray(x, dtype=np.float32))[None]
        out = self.layers[side](
            betas=f(hand_json["betas"]),
            global_orient=f(rotvec),
            hand_pose=f(hand_json["pose"]),
            transl=f(w["t_xyz"]),
            return_verts=True,
        )
        return out.vertices[0].numpy()


def load_mano_trajectory(path):
    """-> list[(timestamp_ns, {side: hand_json})], theo thứ tự thời gian."""
    rows = []
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            rows.append((r["timestamp_ns"], r["hand_poses"]))
    rows.sort(key=lambda x: x[0])
    return rows


@torch.no_grad()
def vertices_batch(hands, hand_jsons, side, batch=1024, return_joints=False):
    """Như ManoHands.vertices nhưng cho cả list hand_json (cùng 1 tay) -> (N,778,3), nhanh hơn gọi từng frame.
    return_joints=True -> thêm (N,16,3) khớp MANO (0 = cổ tay)."""
    from scipy.spatial.transform import Rotation

    if not hand_jsons:
        e = np.zeros((0, 778, 3), np.float32)
        return (e, np.zeros((0, 16, 3), np.float32)) if return_joints else e
    q = np.array([h["wrist_xform"]["q_wxyz"] for h in hand_jsons], np.float64)
    rot = Rotation.from_quat(q[:, [1, 2, 3, 0]]).as_rotvec()
    f = lambda x: torch.tensor(np.asarray(x, dtype=np.float32))
    betas, pose = f([h["betas"] for h in hand_jsons]), f([h["pose"] for h in hand_jsons])
    transl, rot = f([h["wrist_xform"]["t_xyz"] for h in hand_jsons]), f(rot)
    out, jo = [], []
    for s in range(0, len(hand_jsons), batch):
        sl = slice(s, s + batch)
        o = hands.layers[side](betas=betas[sl], global_orient=rot[sl], hand_pose=pose[sl], transl=transl[sl])
        out.append(o.vertices.numpy()); jo.append(o.joints[:, :16].numpy())
    return (np.concatenate(out), np.concatenate(jo)) if return_joints else np.concatenate(out)
