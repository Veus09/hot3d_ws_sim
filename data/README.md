# Dữ liệu HOT3D (không commit)
1. Đăng ký tại https://www.projectaria.com/datasets/hot3D/ -> nhận file JSON link (hết hạn 14 ngày).
2. Đĩa `/` chỉ còn ~9 GB: chỉ tải object library (`--sequence_name all`, data type Assets) + 1 sequence nhỏ.
3. Đặt vào `data/hot3d_dataset/{assets,<sequence>}`. Xem `scripts/00_download_hint.sh`.
4. Tay MANO cần MANO_RIGHT.pkl / MANO_LEFT.pkl (license MANO riêng) -> `data/mano/`.
