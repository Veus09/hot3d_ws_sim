# hot3d_ws_sim

Đưa dataset **HOT3D** (Meta, tương tác tay–vật nhìn từ kính Aria / Quest 3) vào **NVIDIA Isaac Sim**:
replay pose ground-truth của vật thể + tay MANO trên bàn ước lượng, rồi render video sim cho từng sequence
để chọn/lọc tác vụ (ví dụ: tác vụ chỉ dùng tay phải, dùng làm dữ liệu cho robot bắt chước).


Đã làm được:
- Chuyển 33 vật thể của HOT3D sang USD có vật lý (khối lượng, ma sát, collision).
- Replay pose vật thể (kinematic) + 2 tay MANO theo lưới thời gian cố định, cho cả sequence Aria và Quest.
- Ước lượng bàn từ vị trí vật đứng yên (HOT3D không có mesh bàn).
- Render hàng loạt 294 sequence có ground truth thành mp4 + `index.csv` để lọc.
- Video so sánh thật / sim cạnh nhau (khi có video preview thật).

Chưa làm: render đúng mô hình camera fisheye của Aria/Quest (camera sim hiện đặt cố định phía sau bàn).

## Yêu cầu

| Thành phần | Phiên bản đã dùng |
|---|---|
| Ubuntu | 22.04 / 24.04 |
| GPU NVIDIA | RTX, ≥ 8 GB VRAM (1 tiến trình render ~4 GB) |
| Isaac Sim | 6.0 (pip) |
| Isaac Lab | 3.0b2 |
| Python | 3.12 (venv của Isaac Sim) |
| ffmpeg | có `libvpx-vp9` (video xuất ra là VP9 trong `.mp4`) |
| Thư viện thêm trong venv | `trimesh`, `pyyaml`, `scipy`, `pillow`, `smplx` (`pip install --no-deps smplx`) |

`projectaria-tools` và toolkit `hot3d` **không** cần cho pipeline này (chỉ cần nếu muốn đọc VRS thô) — xem `requirements-extra.txt`.

## Cài đặt

```bash
git clone git@github.com:Veus09/hot3d_ws_sim.git && cd hot3d_ws_sim
# venv đã cài Isaac Sim 6.0 + Isaac Lab 3.0b2; mặc định ~/fall26_envtest/.venv
HOT3D_VENV=/đường/dẫn/tới/.venv source env.sh     # kích hoạt venv, đặt HOT3D_WS, thêm src/ vào PYTHONPATH
pip install --no-deps smplx
```

Luôn `source env.sh` trước khi chạy script. Mọi script chạy từ gốc repo: `python scripts/NN_*.py`.

## Dữ liệu (không có trong repo)

1. **HOT3D**: đăng ký tại https://www.projectaria.com/datasets/hot3D/ để nhận 3 file JSON link tải
   (`Hot3DAssets_download_urls.json`, `Hot3DAria_download_urls.json`, `Hot3DQuest_download_urls.json`; link có hạn dùng).
   Đặt chúng ở gốc repo (đã nằm trong `.gitignore`).
2. **MANO**: đăng ký https://mano.is.tue.mpg.de/, chép `MANO_RIGHT.pkl`, `MANO_LEFT.pkl` vào `data/mano/`.

License HOT3D và MANO là riêng, chỉ dùng cho nghiên cứu; repo này không phân phối dữ liệu.

## Pipeline

| Bước | Lệnh | Kết quả |
|---|---|---|
| 0. Tải dữ liệu | `python scripts/00_download.py` | `data/hot3d_dataset/{assets,<seq>/}` — assets + ground truth + tay của 424 sequence (~2.7 GB). Kiểm tra sha1, chạy lại an toàn. VRS thô (~715 GB) không tải. |
| 1. Vật thể → USD | `python scripts/01_convert_assets.py` | `assets_usd/hot3d/o_<tên>.usd`. Thông số vật lý trong `configs/objects.yaml`. |
| 2. Vật proxy (không cần dataset) | `python scripts/02_make_proxy_assets.py` | `assets_usd/proxy/*.usd` — để thử môi trường. |
| 3. Xem vật thể | `python scripts/03_view_objects.py [--usd_dir assets_usd/hot3d] [--drop] [--gui]` | `outputs/view_objects.png` |
| 4. Replay 1 sequence | `python scripts/04_replay_poses.py --seq P0003_d0ad9b47 [--fps 15] [--start_s S --dur_s D] [--no_hands] [--video]` | `outputs/replay/<seq>/frames/*.png`, `--video` → `outputs/replay/<seq>.mp4` |
| 5. So sánh thật / sim | `python scripts/05_compare_video.py --seq <seq>` | video ghép trái (thật) / phải (sim). Cần `preview_rgb.mp4` của sequence. |
| 6. Render hàng loạt | `python scripts/06_render_all.py [--workers 3] [--seqs ...]` | `outputs/replay/sim/<seq>.mp4` + `index.csv` (thiết bị, người, thời lượng, vật thể), log ở `outputs/replay/sim/logs/` |
| 7. Lọc + tách task tương tác | `python scripts/07_task_monitor.py --seq <seq>` | Mở Isaac Sim có cửa sổ + bảng *HOT3D Task Monitor*: phát/tua như video, xem tay nào đang chạm vật, ẩn vật theo đoạn thời gian (vd. từ lúc chuyển sang tay trái), dừng ở đâu bấm **Tách** ở đó để chia sequence thành task. *Lưu + xuất task* → `outputs/tasks/<seq>.json` + `outputs/tasks/<seq>/task_NN.{json,npz}`, đọc bằng `hot3d_sim.tasks.load_task()`. |

Thử nhanh không cần dataset:

```bash
source env.sh
python scripts/02_make_proxy_assets.py
python scripts/03_view_objects.py --drop
```

Chạy đầy đủ cho 1 sequence:

```bash
python scripts/00_download.py --seqs P0003_d0ad9b47
python scripts/01_convert_assets.py
python scripts/04_replay_poses.py --seq P0003_d0ad9b47 --video
```

### Ghi chú khi chạy

- `04` thoát với mã 3 nếu sequence không có pose vật (130 sequence test của HOT3D) — `06` tự bỏ qua.
- `06` chạy lại được: video đã có thì bỏ qua. 3 worker dùng ~13 GB VRAM, mỗi worker ~1–2× thời gian thực.
- Video là **VP9 trong .mp4** (không phải H.264) vì máy Ubuntu mặc định thiếu codec H.264; mở được bằng Totem, VLC, trình duyệt.
- Frame ghi ngay trước khi máy sập có thể là PNG rỗng; `06` kiểm tra frame trước khi gọi ffmpeg.

## Quy ước toạ độ

- Đơn vị mét, quaternion WXYZ, pose dạng `T_world_object`.
- **Quest**: world Y-up → root Xform xoay +90° quanh X, tức (x, y, z) → (x, −z, y).
- **Aria (MPS)**: world đã Z-up, gốc gần vị trí đầu lúc bắt đầu → sàn ước lượng = đầu − 1.6 m.
- Lưới thời gian: frame k = k/fps giây sau timestamp đầu tiên của headset; mỗi luồng (vật, tay) lấy pose gần nhất.
- Bàn: ước lượng từ đỉnh thấp nhất của vật khi đứng yên, gom theo độ cao — **không phải số đo thật**.
- Camera sim: cách tâm các vật đang nằm yên 0.8 m về phía sau và cao 1.4 m (ở phía người ngồi), nhìn lệch 0.2 m về phía người.
- Vật replay phải giữ `physics:kinematicEnabled`, nếu không `sim.reset()` làm mất prim.

## Cấu trúc

```
env.sh                     kích hoạt venv + PYTHONPATH
configs/objects.yaml       vật lý mặc định + override theo từng vật (khoá: o_<tên file>, '-' -> '_')
scripts/00..06_*.py        pipeline (đánh số theo thứ tự chạy)
src/hot3d_sim/
  tasks.py                 list_tasks(), load_task(): đọc task đã xuất từ 07
  usd_utils.py             mesh_to_usd(): ghi mesh thành USD có rigid body, mass, collision, physics material
  mano.py                  ManoHands: MANO qua smplx, giống mano_layer.py của toolkit hot3d
                           (15 PCA, wrist_xform -> global_orient + transl, sửa shapedirs tay trái); khoá "0" = trái, "1" = phải
assets_usd/proxy/          vật proxy (commit sẵn)
data/                      dữ liệu tải về (gitignore, xem data/README.md)
outputs/                   ảnh, frame, video (gitignore), trừ outputs/tasks/*.json + README (task đã lọc)
```

Script dùng simulator phải tạo `AppLauncher` của Isaac Lab **trước** khi import `pxr`, `omni`, `isaaclab.sim`.

## Tái hiện task đã lọc

`outputs/tasks/` chứa các task đã lọc thủ công (danh sách ở [`outputs/tasks/README.md`](outputs/tasks/README.md)):
mỗi `<seq>.json` ghi vật bị xóa, đoạn thời gian giữ từng vật, điểm tách và tên task. Repo **chỉ** có các file mô tả
này; dữ liệu pose/tay của từng task được tạo lại từ HOT3D trên máy bạn (HOT3D và MANO không cho phát tán lại).

```bash
# 1. Cài đặt như mục "Cài đặt", đặt 3 file JSON link HOT3D ở gốc repo, MANO_{LEFT,RIGHT}.pkl vào data/mano/
source env.sh

# 2. Chỉ tải các sequence có task (+ thư viện vật thể), ~0.2 GB thay vì 2.7 GB
python scripts/00_download.py --datasets assets aria --seqs P0001_23fa0ee8 P0001_10a27bf7

# 3. Vật thể -> USD
python scripts/01_convert_assets.py

# 4. Tạo lại dữ liệu task (không mở cửa sổ, ~1 phút / sequence, không sửa file mô tả)
python scripts/07_task_monitor.py --seq P0001_23fa0ee8 --export_only
#    -> outputs/tasks/P0001_23fa0ee8/task_01.json + task_01.npz, task_02...

# 5. (tuỳ chọn) xem / sửa trong Isaac Sim: phát từng task bằng nút "Phat" ở mục "Tach task"
python scripts/07_task_monitor.py --seq P0001_23fa0ee8
```

Đọc task trong Python (cho model chạy từng task):

```python
from hot3d_sim.tasks import list_tasks, load_task
for path in list_tasks("P0001_23fa0ee8"):
    T = load_task(path)
    T["name"], T["seconds_in_seq"], [o["name"] for o in T["objects"]]
    T["obj_pose"]          # (N_obj, T, 4, 4) pose vật, world Isaac (Z-up, mét)
    T["obj_visible"]       # (N_obj, T)  False ở frame vật bị lọc
    T["right_wrist_pose"]  # (T, 4, 4)   cổ tay phải (xoay = global_orient MANO, vị trí = khớp cổ tay)
    T["right_hand_joints"] # (T, 16, 3)  khớp MANO;  right_hand_verts (T, 778, 3) lưới tay
    T["right_touch"]       # (N_obj, T)  tay phải chạm vật (< 1 cm)
    # tương tự left_*; T["t"] giây từ đầu task; T["timestamps_ns"] mốc gốc HOT3D
```

Mô tả đầy đủ các khoá nằm trong `npz_keys` của mỗi `task_NN.json` và trong `src/hot3d_sim/tasks.py`.
Kết quả tạo lại giống hệt với cùng dữ liệu HOT3D, cùng `--fps` (mặc định 30) và `--contact_cm` (mặc định 1.0).

Tự lọc / tách sequence khác: `python scripts/07_task_monitor.py --seq <seq>` (bảng *HOT3D Task Monitor*, xem bước 7).

## Lọc video

Video trong `outputs/replay/sim/` được duyệt và cắt frame bằng web tool [**sim_dataset_verify**](https://github.com/Veus09/sim_dataset_verify) (repo riêng):
xem toàn bộ frame, đánh dấu xóa, xem trước, lưu đè mp4, loại video; nhật ký ghi ở `outputs/replay/sim/edits.csv`.

## Tài liệu khác

- `HANDOFF.md`: trạng thái chi tiết, việc còn dở.
- `CLAUDE.md`: ghi chú kỹ thuật cho trợ lý code.

## Trích dẫn

- HOT3D: Banerjee et al., *HOT3D: Hand and Object Tracking in 3D from Egocentric Multi-View Videos*, 2024.
- MANO: Romero et al., *Embodied Hands: Modeling and Capturing Hands and Bodies Together*, SIGGRAPH Asia 2017.
