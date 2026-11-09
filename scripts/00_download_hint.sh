#!/bin/bash
# In lệnh tải (cần JSON link từ projectaria.com và toolkit hot3d đã clone). Không tự tải.
df -h / | tail -1
cat <<'TXT'
git clone https://github.com/facebookresearch/hot3d ~/hot3d_toolkit
python ~/hot3d_toolkit/hot3d/dataset_downloader/dataset_downloader_base_main.py \
  -c <assets_json_link>.json -o $HOT3D_WS/data --sequence_name all
# 1 sequence (ví dụ):  --sequence_name P0003_c701bd11 --data_types <xem README toolkit>
TXT
