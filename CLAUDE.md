# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

Brings HOT3D into Isaac Sim: object pose replay with MANO hands and estimated tables, plus real-vs-sim comparison videos for every GT sequence (used to hand-pick ~1000 evaluation tasks). Not done: rendering from the real Aria/Quest camera models. Sibling repo for ARCTIC: `~/artic_ws_sim`. README and comments are in Vietnamese. Not a git repo; no tests or linter.

## Environment & commands

- `source env.sh` first. It activates the shared venv at `~/fall26_envtest/.venv` (Isaac Sim 6.0 + Isaac Lab 3.0b2), sets `HOT3D_WS`, and adds `src/` to `PYTHONPATH`. Scripts import `hot3d_sim` from there.
- Scripts are numbered and run in order, each as `python scripts/NN_*.py`:
  - `00_download.py [--datasets assets aria quest --types ground_truth hand_data --seqs ...]`: downloads from the `Hot3D*_download_urls.json` link files in the repo root (links expire 2026-12-02), checks sha1, unzips to `data/hot3d_dataset/<seq>/`, writes `device.txt` (aria/quest). Re-runnable. Default = assets + GT + hand data for all 424 sequences (~2.7 GB); VRS (~715 GB) is not downloaded.
  - `01_convert_assets.py [--src DIR --out DIR --scale S]`: HOT3D meshes (glb/obj/ply) -> `assets_usd/hot3d/o_<name>.usd`. It auto-guesses mm when the bbox is over 2 m and divides by 1000.
  - `02_make_proxy_assets.py`: builds proxy assets into `assets_usd/proxy` for use without the dataset.
  - `03_view_objects.py [--usd_dir DIR --drop --gui]`: renders `outputs/view_objects.png`. `--drop` runs physics, and `--gui` opens a window.
  - `04_replay_poses.py [--seq P0003_d0ad9b47 --fps 15 --start_s S --dur_s D --no_hands --video --resume --out_dir DIR]`: replays GT object poses plus MANO hands on a fixed time grid (frame k = k/fps s after the first headset timestamp, nearest pose per stream) -> `outputs/replay/<seq>/frames/%05d.png` (wiped unless `--resume`). Handles both devices: Quest world is Y-up (root Xform rotated +90° about X), Aria (MPS) world is Z-up (no rotation; origin near the start head pose, so floor is estimated as head − 1.6 m). Tables are estimated automatically from where objects rest (lowest vertex while still, clustered by height). Camera sits 0.8 m behind and 1.4 m above the resting-object centre (on the side of the median head position), aimed 0.2 m toward the person, so hands raised to the face don't fill the frame. Exits with code 3 when the sequence has no object GT (the 130 HOT3D test sequences). Objects must keep `physics:kinematicEnabled` or `sim.reset()` invalidates their prims.
  - `05_compare_video.py --seq S [--start 0 --dur 0 --fps 15 --frames DIR --out FILE]`: side-by-side video, real `preview_rgb.mp4` (left; Quest 512x320, Aria 1408x1408 fisheye) vs sim frames (right), same timeline. `--dur 0` = all sim frames.
  - `06_render_all.py [--seqs ... --workers 3 --fps 15]`: batch over all 294 GT sequences (list also in `data/gt_seqs.txt`): 04 -> frame check -> ffmpeg (sim only, seq name + time overlay) -> delete frames. Output `outputs/replay/sim/<seq>.mp4`, `index.csv` (device, participant, duration, objects) for filtering, logs in `outputs/replay/sim/logs/`. Re-runnable (skips existing mp4). The user does not want the real preview videos downloaded (disk space), so the batch is sim-only; 05 still works for a sequence whose preview is present (only P0003). ~1-2x realtime per worker; 3 workers use ~13 GB VRAM.
- Frame-cutting/review web tool lives in `~/sim_dataset_verify` (`python3 ~/sim_dataset_verify/server.py`): it overwrites `outputs/replay/sim/<seq>.mp4`, backs originals up to `_orig/`, moves rejected videos to `_rejected/`, and records status + removed original-frame ranges in `edits.json`/`edits.csv` there. Re-running the batch render skips existing mp4s, so it won't undo cuts.
- Videos are `.mp4` with VP9 inside (`libvpx-vp9`, `-movflags +faststart`), not H.264. The Ubuntu box lacks `gstreamer1.0-libav`, so its default player (Totem) can't decode H.264, but it can decode VP9 (`qtdemux` + `vp9dec`). The user wants mp4 files.
- A full-sequence render once ended with the machine going down; the user says that was their own hard shutdown, not RAM (machine has 32 GB). No memory cap is needed; rendering a whole sequence in one process is fine. `--resume --start_s/--dur_s` still exist for chunked runs. Frames written just before a crash can be zero-filled PNGs, so validate them before ffmpeg.
- Extra deps (`requirements-extra.txt`): `projectaria-tools` and the `hot3d` toolkit are only needed to read real VRS data.

## Architecture

- `src/hot3d_sim/usd_utils.py`: `mesh_to_usd(...)` writes a mesh as USD with physics (mass, friction, restitution, collision approximation). It is shared by 01 and 02.
- `src/hot3d_sim/mano.py`: `ManoHands` wraps `smplx` MANO the same way the hot3d toolkit's `mano_layer.py` does (15 PCA comps, `wrist_xform` -> global_orient + transl, left-hand shapedirs fix). Hand key `"0"` = left, `"1"` = right. The original MANO pkl files contain chumpy, which breaks on numpy 2, so they are converted once to plain numpy in `data/mano/plain/`. `smplx` is installed in the venv with `--no-deps`.
- `configs/objects.yaml`: default physics (`default`) plus per-object `overrides`, keyed by the USD name (`o_<file stem>` with `-` replaced by `_`). It is read by 01.
- Scripts that use the simulator create the app with Isaac Lab's `AppLauncher` before importing `pxr`, `omni`, or `isaaclab.sim`. Keep that ordering. 04 forces `headless=True` and `enable_cameras=True`, and uses `isaaclab.sensors.camera.Camera`.
- Coordinate conventions (04): metres, quaternion WXYZ, `T_world_object`. Quest world is Y-up -> root Xform maps (x,y,z)->(x,-z,y); Aria world is already Z-up. Aria pose timestamps come in pairs 0.1-0.2 ms apart (effectively 30 Hz). Replayed objects are kinematic.
- Sequence data comes from `data/hot3d_dataset/<seq>/` (`dynamic_objects.csv`, `headset_trajectory.csv`, ...). The object library is in `data/hot3d_dataset/assets`. `data/` is gitignored (except its README), as is `outputs/`. Raw downloads and link JSONs are in `data/dl/` and `data/*.json`. HOT3D and MANO licences are separate, and MANO pkl files go in `data/mano/`.
