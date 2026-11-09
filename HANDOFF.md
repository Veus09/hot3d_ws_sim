# HANDOFF — hot3d_ws_sim

Cập nhật: 2026-11-09. Mục tiêu: đưa dữ liệu HOT3D vào Isaac Sim. Giai đoạn 1 (vật thể + pose replay + bàn xấp xỉ) đã chạy được. Tay MANO và camera Aria/Quest chưa làm.

## Trạng thái

| Hạng mục | Trạng thái |
|---|---|
| Chuyển object library (35 file trong `assets/`) sang USD (`01`) | Xong, ra `assets_usd/hot3d/o_<uid>.usd` |
| Vật thể proxy khi chưa có dataset (`02`) | Xong |
| Xem/thả vật thể (`03`) | Xong, ra `outputs/view_objects.png` |
| Replay pose GT vật thể + kính VR của sequence `P0003_d0ad9b47` (`04`) | Chạy được. Đã có ảnh/video thử trong `outputs/replay/` |
| Bàn (rect + round) | Xấp xỉ, dựng từ pose vật thể, KHÔNG phải số đo thật (xem `TABLES` trong `04`, và `topdown.png`) |
| Tay MANO | Chưa làm |
| Camera Aria/Quest (render theo `camera_models.json`) | Chưa làm |

## Chạy lại

```
source env.sh      # venv Isaac Sim 6.0 + Isaac Lab 3.0b2 ở ~/fall26_envtest/.venv
python scripts/01_convert_assets.py
python scripts/04_replay_poses.py --seq P0003_d0ad9b47 --step 2   # ghi outputs/replay/frames/*.png (xoá frame cũ)
ffmpeg ...  # ghép frames thành mp4 (lệnh ffmpeg dùng trước đó không được lưu lại)
```

Chỉ chạy 1 tiến trình Isaac Sim (8 GB VRAM). Đĩa `/` còn ít (~9 GB).

## Dữ liệu (không có trong repo)

Repo không chứa dữ liệu. Phải đặt thủ công, vì license HOT3D/MANO riêng và các file JSON link tải có chữ ký, hết hạn sau 14 ngày:
- `data/hot3d_dataset/assets/`: object library (`*.glb`, `instance.json`). `04` đọc `instance.json` để lấy tên vật thể.
- `data/hot3d_dataset/P0003_d0ad9b47/`: sequence Quest. Có `dynamic_objects.csv`, `headset_trajectory.csv`, `mano_hand_pose_trajectory.jsonl`, `camera_models.json`, `box2d_*.csv`, `masks/`, `preview_rgb.mp4`.
- `data/dl/`: các file zip/mp4 đã tải.
- `data/mano/MANO_RIGHT.pkl`, `MANO_LEFT.pkl`: chưa có, cần đăng ký license MANO.
- `assets_usd/hot3d/` là bản chuyển từ asset có license, nên không đưa lên git. Tạo lại bằng `01`.

## Quy ước quan trọng (dễ sai)

- HOT3D: world Y-up, mét, quaternion WXYZ, `T_world_object`. Isaac: Z-up. Mọi thứ nằm dưới Xform gốc `/World/hot3d` xoay +90° quanh X, ánh xạ (x,y,z)->(x,-z,y). Toạ độ bàn, kính và vật thể trong `04` đều là toạ độ HOT3D (Y-up) bên trong Xform đó.
- `Gf.Matrix4d` dùng row-vector, nên hàm `gf()` trong `04` chuyển vị ma trận.
- Với script dùng simulator, tạo `AppLauncher` trước khi import `pxr`, `omni`, `isaaclab.sim`.
- Vật thể replay là kinematic (không rơi). Hộp kính chỉ vẽ vị trí và hướng suy ra từ pose, chưa có mũi tên hướng nhìn.

## Việc tiếp theo (gợi ý theo lộ trình README)

1. Tay MANO: xin MANO pkl vào `data/mano/`, đọc `mano_hand_pose_trajectory.jsonl`, dựng mesh tay và gắn vào replay trong `04`.
2. Camera Aria/Quest: dùng `camera_models.json` + `headset_trajectory.csv` để render đúng góc nhìn, so với `preview_rgb.mp4`.
3. Bàn/phòng: thay bàn xấp xỉ bằng số đo thật nếu có.
4. Lưu lại lệnh ffmpeg ghép video vào một script (hiện chưa có).
5. Không có test hay CI. Kiểm tra bằng cách xem ảnh/video trong `outputs/`.
