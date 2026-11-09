# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

Brings HOT3D dataset objects into Isaac Sim (phase 1: objects only; no hands/table yet). Roadmap: pose replay (04) -> MANO hands -> Aria/Quest cameras -> approximate table/room. README and comments are in Vietnamese. Not a git repo; no tests or linter.

## Environment & commands

- `source env.sh` first. It activates the shared venv at `~/fall26_envtest/.venv` (Isaac Sim 6.0 + Isaac Lab 3.0b2), sets `HOT3D_WS`, and adds `src/` to `PYTHONPATH`. Scripts import `hot3d_sim` from there.
- Scripts are numbered and run in order, each as `python scripts/NN_*.py`:
  - `01_convert_assets.py [--src DIR --out DIR --scale S]`: HOT3D meshes (glb/obj/ply) -> `assets_usd/hot3d/o_<name>.usd`. It auto-guesses mm when the bbox is over 2 m and divides by 1000.
  - `02_make_proxy_assets.py`: builds proxy assets into `assets_usd/proxy` for use without the dataset.
  - `03_view_objects.py [--usd_dir DIR --drop --gui]`: renders `outputs/view_objects.png`. `--drop` runs physics, and `--gui` opens a window.
  - `04_replay_poses.py [--seq P0003_d0ad9b47 --step 2 --max_frames N]`: replays GT object poses and writes PNG frames to `outputs/replay/frames/`. It wipes that dir on each run. Stitch the frames into mp4 with ffmpeg.
- Only one Isaac Sim process at a time (8 GB VRAM). Disk is tight (~9 GB), so see `scripts/00_download_hint.sh`, which only prints download commands.
- Extra deps (`requirements-extra.txt`): `projectaria-tools` and the `hot3d` toolkit are only needed to read real VRS data.

## Architecture

- `src/hot3d_sim/usd_utils.py`: `mesh_to_usd(...)` writes a mesh as USD with physics (mass, friction, restitution, collision approximation). It is shared by 01 and 02.
- `configs/objects.yaml`: default physics (`default`) plus per-object `overrides`, keyed by the USD name (`o_<file stem>` with `-` replaced by `_`). It is read by 01.
- Scripts that use the simulator create the app with Isaac Lab's `AppLauncher` before importing `pxr`, `omni`, or `isaaclab.sim`. Keep that ordering. 04 forces `headless=True` and `enable_cameras=True`, and uses `isaaclab.sensors.camera.Camera`.
- Coordinate conventions (04): HOT3D is world Y-up, in metres, quaternion WXYZ, `T_world_object`. Isaac is Z-up. The scene is wrapped in a root Xform rotated +90° about X, mapping (x,y,z)->(x,-z,y), so meshes keep their glb coordinates. Replayed objects are kinematic.
- Sequence data comes from `data/hot3d_dataset/<seq>/` (`dynamic_objects.csv`, `headset_trajectory.csv`, ...). The object library is in `data/hot3d_dataset/assets`. `data/` is gitignored (except its README), as is `outputs/`. Raw downloads and link JSONs are in `data/dl/` and `data/*.json`. HOT3D and MANO licences are separate, and MANO pkl files go in `data/mano/`.
