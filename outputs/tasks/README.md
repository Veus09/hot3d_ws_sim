# Task đã lọc từ HOT3D

Mỗi file `<seq>.json` mô tả cách lọc 1 sequence: vật bị xóa, đoạn thời gian giữ từng vật, điểm tách và tên task.
Dữ liệu từng task (`<seq>/task_NN.{json,npz}`) **không** nằm trong repo (giấy phép HOT3D/MANO) — tạo lại bằng:

```bash
python scripts/07_task_monitor.py --seq <seq> --export_only
```

Xem hướng dẫn đầy đủ ở mục *Tái hiện task đã lọc* trong README gốc.

## P0001_10a27bf7 (aria)

| task | thời gian trong sequence | tên |
|---|---|---|
| 01 | 0.00 – 124.20 s | (cả sequence, chưa tách) |

- Vật bị xóa: keyboard
- Vật chỉ giữ một phần: mug_white (0-62.4s, 66.167-124.2s); mug_patterned (0-36.9s, 41.8-80.5s, 83.767-124.2s); cellphone (0-33.133s, 35.233-124.2s)

## P0001_23fa0ee8 (aria)

| task | thời gian trong sequence | tên |
|---|---|---|
| 01 | 6.43 – 13.50 s | cam_chuot_1 |
| 02 | 22.53 – 24.10 s | cam_dien_thoai |
| 03 | 32.23 – 49.20 s | cam_but_viet_1 |
| 04 | 49.20 – 58.67 s | cam_xoa_bang_1 |
| 05 | 58.67 – 60.67 s | cam_coc_1 |
| 06 | 67.77 – 70.93 s | cam_coc_2 |
| 07 | 81.20 – 89.80 s | cam_chuot_2 |
| 08 | 101.80 – 116.57 s | cam_but_viet_2 |
| 09 | 116.57 – 122.27 s | cam_xoa_bang_2 |

- Vật bị xóa: keyboard
- Vật chỉ giữ một phần: holder_black (0-60.533s, 62.567-124.967s); cellphone (0-24.067s, 32.267-89.967s)
