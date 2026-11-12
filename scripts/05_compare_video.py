"""Video so sánh 2 khung: video gốc (preview_rgb.mp4 của HOT3D) | Isaac Sim (frames của 04), cùng mốc thời gian.
  python scripts/05_compare_video.py --seq P0003_d0ad9b47 [--start 0] [--dur 0] [--fps 15] [--out FILE]
Cần chạy 04 trước (frames trong outputs/replay/<seq>/frames, frame k <-> k/fps giây từ đầu recording).
--dur 0 = tới frame sim cuối cùng. Mặc định ra outputs/replay/<seq>_compare_<a>s-<b>s.mp4 (VP9 trong mp4).
Khớp thời gian: preview_rgb.mp4 và lưới thời gian của 04 đều bắt đầu ở frame đầu của headset_trajectory.
"""
import argparse, glob, os, subprocess

ap = argparse.ArgumentParser()
ap.add_argument("--seq", default="P0003_d0ad9b47")
ap.add_argument("--start", type=float, default=0.0, help="giây bắt đầu")
ap.add_argument("--dur", type=float, default=0.0, help="độ dài (giây), 0 = tới hết")
ap.add_argument("--fps", type=float, default=15.0, help="--fps đã dùng khi chạy 04")
ap.add_argument("--frames", default="", help="mặc định outputs/replay/<seq>/frames")
ap.add_argument("--out", default="")
ap.add_argument("--height", type=int, default=540)
a = ap.parse_args()

ws = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
seq_dir = os.path.join(ws, "data", "hot3d_dataset", a.seq)
real = os.path.join(seq_dir, "preview_rgb.mp4")
device = open(os.path.join(seq_dir, "device.txt")).read().strip()
frames = a.frames or os.path.join(ws, "outputs", "replay", a.seq, "frames")
have = sorted(int(os.path.basename(f)[:-4]) for f in glob.glob(os.path.join(frames, "*.png")))
if not have:
    raise SystemExit(f"không có frame sim trong {frames}; chạy 04 trước")
k0 = round(a.start * a.fps)
n = (have[-1] - k0 + 1) if a.dur <= 0 else round(a.dur * a.fps)
hs = set(have)
missing = [k for k in range(k0, k0 + n) if k not in hs]
if missing:
    raise SystemExit(f"thiếu {len(missing)} frame sim (vd {missing[:3]}); chạy 04 cho đoạn này trước")
end = a.start + n / a.fps
out = a.out or os.path.join(ws, "outputs", "replay", f"{a.seq}_compare_{a.start:g}s-{end:g}s.mp4")
os.makedirs(os.path.dirname(out), exist_ok=True)

font = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
lbl = lambda t: (f"drawtext=fontfile={font}:text='{t}':x=12:y=10:fontsize=22:fontcolor=white:"
                 "box=1:boxcolor=black@0.55:boxborderw=6")
dev_name = "Aria" if device == "aria" else "Quest 3"
H = a.height
fg = (f"[0:v]fps={a.fps:g},scale=-2:{H},setsar=1,{lbl(f'Video gốc ({dev_name}, HOT3D)')}[r];"
      f"[1:v]scale=-2:{H},setsar=1,{lbl('Isaac Sim (pose GT + tay MANO)')}[s];"
      f"[r][s]hstack=inputs=2,"
      f"drawtext=fontfile={font}:text='{a.seq}  t=%{{eif\\:t+{a.start:g}\\:d}}.%{{eif\\:mod((t+{a.start:g})*10\\,10)\\:d}} s':"
      "x=(w-tw)/2:y=h-th-12:fontsize=22:fontcolor=white:box=1:boxcolor=black@0.55:boxborderw=6,"
      "pad=ceil(iw/2)*2:ceil(ih/2)*2[v]")
cmd = ["ffmpeg", "-y", "-loglevel", "error",
       "-ss", f"{a.start}", "-t", f"{n / a.fps}", "-i", real,
       "-framerate", f"{a.fps:g}", "-start_number", str(k0), "-i", os.path.join(frames, "%05d.png"),
       "-filter_complex", fg, "-map", "[v]", "-frames:v", str(n),
       "-c:v", "libvpx-vp9", "-crf", "34", "-b:v", "0", "-row-mt", "1", "-deadline", "good", "-cpu-used", "5",
       "-pix_fmt", "yuv420p", "-movflags", "+faststart", out]
subprocess.run(cmd, check=True)
print("video ->", out)
