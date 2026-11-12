"""Tải HOT3D từ các file JSON link (projectaria.com) vào data/hot3d_dataset/.

Mặc định chỉ tải phần nhẹ cần cho bài toán điểm chạm: assets + ground_truth + hand_data
của mọi sequence Aria + Quest (~2.7 GB). VRS thô (~715 GB) không tải.

  python scripts/00_download.py                                  # assets + GT + tay, mọi sequence
  python scripts/00_download.py --types video_main_rgb --seqs P0003_d0ad9b47
  python scripts/00_download.py --dry_run                         # chỉ in dung lượng

Chạy lại an toàn: sequence nào đã giải nén xong (có file .done_<type>) thì bỏ qua.
"""
import argparse, hashlib, json, os, shutil, sys, urllib.request, zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

WS = Path(os.environ.get("HOT3D_WS", Path(__file__).resolve().parents[1]))
JSONS = {"assets": "Hot3DAssets_download_urls.json",
         "aria": "Hot3DAria_download_urls.json",
         "quest": "Hot3DQuest_download_urls.json"}


def sha1(path):
    h = hashlib.sha1()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def fetch(item, dst_dir, tag, keep_zip):
    dst_dir.mkdir(parents=True, exist_ok=True)
    done = dst_dir / f".done_{tag}"
    if done.exists():
        return f"skip {dst_dir.name}/{tag}"
    dl = WS / "data" / "dl" / item["filename"]
    dl.parent.mkdir(parents=True, exist_ok=True)
    if not (dl.exists() and dl.stat().st_size == item["file_size_bytes"] and sha1(dl) == item["sha1sum"]):
        tmp = dl.with_suffix(dl.suffix + ".part")
        with urllib.request.urlopen(item["download_url"], timeout=120) as r, open(tmp, "wb") as f:
            shutil.copyfileobj(r, f, 1 << 20)
        if sha1(tmp) != item["sha1sum"]:
            tmp.unlink()
            raise RuntimeError(f"sha1 sai: {item['filename']}")
        tmp.rename(dl)
    if dl.suffix == ".zip":
        with zipfile.ZipFile(dl) as z:
            root = dst_dir.resolve()
            for m in z.namelist():  # chặn path traversal
                if not (root / m).resolve().is_relative_to(root):
                    raise RuntimeError(f"đường dẫn lạ trong zip: {m}")
            z.extractall(dst_dir)
        if not keep_zip:
            dl.unlink()
    else:  # mp4 / vrs: chuyển thẳng vào thư mục sequence
        name = "preview_rgb.mp4" if tag == "video_main_rgb" else item["filename"]
        shutil.move(dl, dst_dir / name)
    done.touch()
    return f"ok   {dst_dir.name}/{tag} ({item['file_size_bytes'] / 1e6:.1f} MB)"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=["assets", "aria", "quest"])
    ap.add_argument("--types", nargs="+", default=["ground_truth", "hand_data"],
                    help="data type cho sequence (assets luôn tải nếu có trong --datasets)")
    ap.add_argument("--seqs", nargs="*", default=None, help="mặc định: mọi sequence")
    ap.add_argument("--out", default=str(WS / "data" / "hot3d_dataset"))
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--keep_zip", action="store_true")
    ap.add_argument("--dry_run", action="store_true")
    a = ap.parse_args()
    out = Path(a.out)

    jobs = []
    for ds in a.datasets:
        d = json.load(open(WS / JSONS[ds]))
        for seq, items in d["sequences"].items():
            if ds == "assets":
                jobs.append((items["assets"], out / "assets_zip_root", "assets"))
                continue
            if a.seqs and seq not in a.seqs:
                continue
            for t in a.types:
                if t in items:
                    jobs.append((items[t], out / seq, t))
            meta = out / seq / "device.txt"  # ghi lại thiết bị (Aria/Quest) cho loader sau này
            if not a.dry_run:
                meta.parent.mkdir(parents=True, exist_ok=True)
                meta.write_text(ds + "\n")

    total = sum(j[0]["file_size_bytes"] for j in jobs)
    free = shutil.disk_usage(WS).free
    print(f"{len(jobs)} file, {total / 1e9:.2f} GB; đĩa trống {free / 1e9:.1f} GB")
    if a.dry_run:
        return
    if total * 2.2 > free:  # zip + bản giải nén
        sys.exit("Không đủ chỗ trống, giảm --types/--seqs.")

    errs = 0
    with ThreadPoolExecutor(a.workers) as ex:
        futs = {ex.submit(fetch, *j, a.keep_zip): j for j in jobs}
        for i, f in enumerate(as_completed(futs), 1):
            try:
                print(f"[{i}/{len(jobs)}] {f.result()}", flush=True)
            except Exception as e:
                errs += 1
                print(f"[{i}/{len(jobs)}] LỖI {futs[f][1].name}/{futs[f][2]}: {e}", flush=True)

    # assets zip có thư mục con; đưa về data/hot3d_dataset/assets
    z = out / "assets_zip_root"
    if z.exists():
        inner = next((p.parent for p in z.rglob("instance.json")), None)
        if inner and not (out / "assets").exists():
            shutil.move(str(inner), out / "assets")
    print("Xong." if not errs else f"Xong, {errs} lỗi (chạy lại để thử tiếp).")


if __name__ == "__main__":
    main()
