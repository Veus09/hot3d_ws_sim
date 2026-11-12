"""Render hàng loạt: mọi sequence có pose GT -> video Isaac Sim ở outputs/replay/sim/<seq>.mp4 (chỉ sim, không cần
video gốc), kèm bảng outputs/replay/sim/index.csv (thiết bị, người, độ dài, vật thể) để lọc task.

  python scripts/06_render_all.py [--seqs ...] [--workers 3] [--fps 15] [--keep_frames]

Mỗi sequence: 04 (process riêng) -> kiểm tra frame -> ghép mp4 (VP9, có ghi tên sequence + thời gian) -> xoá frames.
Chạy lại an toàn: mp4 đã có thì bỏ qua. Log ở outputs/replay/sim/logs/.
"""
import argparse, csv, json, os, shutil, subprocess, sys, time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

WS = Path(__file__).resolve().parents[1]
DATA = WS / "data" / "hot3d_dataset"
OUT = WS / "outputs" / "replay" / "sim"
FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"


def gt_seqs():
    out = []
    for d in sorted(DATA.glob("P0*")):
        f = d / "dynamic_objects.csv"
        if f.exists() and sum(1 for _ in open(f)) > 1:
            out.append(d.name)
    return out


def frames_ok(fr):
    from PIL import Image
    files = sorted(fr.glob("*.png"))
    if not files:
        return False
    for f in files:  # PNG ghi dở / toàn 0 khi crash
        if f.stat().st_size < 1000:
            return False
    try:
        Image.open(files[-1]).load()
    except Exception:
        return False
    return True


def run(seq, a):
    vid = OUT / f"{seq}.mp4"
    if vid.exists():
        return seq, "skip"
    fr = WS / "outputs" / "replay" / seq / "frames"
    log = open(OUT / "logs" / f"{seq}.log", "w")
    t = time.time()
    r = subprocess.run([sys.executable, str(WS / "scripts" / "04_replay_poses.py"), "--seq", seq, "--fps", str(a.fps),
                        "--out_dir", str(fr)], stdout=log, stderr=subprocess.STDOUT)
    if r.returncode != 0 or not frames_ok(fr):
        return seq, f"LỖI 04 (exit {r.returncode}), xem {log.name}"
    tmp = OUT / f"{seq}.part.mp4"
    txt = (f"drawtext=fontfile={FONT}:text='{seq}  t=%{{eif\\:t\\:d}}.%{{eif\\:mod(t*10\\,10)\\:d}} s':"
           "x=(w-tw)/2:y=h-th-12:fontsize=22:fontcolor=white:box=1:boxcolor=black@0.55:boxborderw=6")
    r = subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", f"{a.fps:g}", "-start_number", "0",
                        "-i", str(fr / "%05d.png"), "-vf", txt, "-c:v", "libvpx-vp9", "-crf", "34", "-b:v", "0",
                        "-row-mt", "1", "-deadline", "good", "-cpu-used", "5", "-pix_fmt", "yuv420p",
                        "-movflags", "+faststart", str(tmp)], stdout=log, stderr=subprocess.STDOUT)
    if r.returncode != 0:
        return seq, f"LỖI ffmpeg (exit {r.returncode}), xem {log.name}"
    tmp.rename(vid)
    if not a.keep_frames:
        shutil.rmtree(fr.parent)
    return seq, f"ok {time.time() - t:.0f}s"


def write_index(seqs):
    rows = []
    for s in seqs:
        d = DATA / s
        m = json.load(open(d / "metadata.json"))
        hs = [int(r["timestamp[ns]"]) for r in csv.DictReader(open(d / "headset_trajectory.csv"))]
        rows.append(dict(seq=s, device=open(d / "device.txt").read().strip(), participant=m.get("participant_id", s[:5]),
                         duration_s=round((max(hs) - min(hs)) / 1e9, 1), n_objects=len(m.get("object_names", [])),
                         objects=" ".join(m.get("object_names", [])), video=f"{s}.mp4",
                         rendered=(OUT / f"{s}.mp4").exists()))
    with open(OUT / "index.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seqs", nargs="*")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--fps", type=float, default=15.0)
    ap.add_argument("--keep_frames", action="store_true")
    a = ap.parse_args()
    (OUT / "logs").mkdir(parents=True, exist_ok=True)
    seqs = a.seqs or gt_seqs()
    write_index(seqs)
    todo = [s for s in seqs if not (OUT / f"{s}.mp4").exists()]
    print(f"{len(seqs)} sequence có GT, còn {len(todo)} cần render", flush=True)
    done, fails = 0, []
    with ThreadPoolExecutor(a.workers) as ex:
        for seq, msg in ex.map(lambda s: run(s, a), todo):
            done += 1
            if not msg.startswith("ok") and msg != "skip": fails.append(seq)
            print(f"[{done}/{len(todo)}] {seq}: {msg}", flush=True)
            if done % 10 == 0: write_index(seqs)
    write_index(seqs)
    print("xong. index ->", OUT / "index.csv", "| lỗi:", len(fails), fails, flush=True)


if __name__ == "__main__":
    main()
