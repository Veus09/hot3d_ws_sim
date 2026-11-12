# data/ (không commit)

1. Đăng ký tại https://www.projectaria.com/datasets/hot3D/ -> nhận 3 file JSON link (`Hot3DAssets_download_urls.json`, `Hot3DAria_download_urls.json`, `Hot3DQuest_download_urls.json`; link có hạn dùng). Đặt ở gốc repo (đã gitignore).
2. `python scripts/00_download.py` -> `data/hot3d_dataset/{assets,<sequence>/}` (assets + ground truth + tay của 424 sequence, ~2.7 GB nén). VRS thô ~715 GB không tải.
3. Tay MANO: đăng ký https://mano.is.tue.mpg.de/, chép `MANO_RIGHT.pkl`, `MANO_LEFT.pkl` vào `data/mano/`. Lần chạy đầu tự chuyển sang numpy thuần ở `data/mano/plain/` (pkl gốc chứa chumpy, không chạy với numpy 2).
