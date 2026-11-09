# hot3d_ws_sim
Đưa vật thể HOT3D vào Isaac Sim (giai đoạn 1: chỉ vật thể; chưa có tay/bàn thật).
```
source env.sh
python scripts/02_make_proxy_assets.py            # thử không cần dataset
python scripts/03_view_objects.py                 # ảnh -> outputs/view_objects.png  (--drop thả vật lý, --gui mở cửa sổ)
# khi có data/hot3d_dataset/assets:
python scripts/01_convert_assets.py && python scripts/03_view_objects.py --usd_dir assets_usd/hot3d
```
Lưu ý: chỉ chạy 1 tiến trình Isaac Sim (8 GB VRAM); đĩa còn ít (xem scripts/00_download_hint.sh); license HOT3D/MANO riêng.
Lộ trình: pose replay (04) -> tay MANO -> camera Aria/Quest -> bàn/phòng xấp xỉ.
