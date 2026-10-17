# AGENTS.md

Video2VMD extracts a 24-joint body skeleton from monocular video or a still and
retargets it directly to MMD VMD or BVH. CLI: `v2vmd`; package: `video2vmd`.

## Setup

```bash
pip install --upgrade pip setuptools wheel
pip install "numpy>=1.26"
pip install --no-build-isolation "chumpy==0.70"
pip install -e .
```

The VMD codec is a normal pinned dependency. PyTorch/GVHMR run in the user's
external Python environment; CUDA-enabled PyTorch is needed for `--device cuda`.
SMPL-X is user-supplied at `assets/body_models/smplx/SMPLX_NEUTRAL.npz`.
PMX targets are optional (`assets/mmd/`); input media live in `assets/video/` and
`assets/image/`. Inference weights auto-download to `weights/` on first extraction.
All user assets and generated outputs remain gitignored.

## Commands

```bash
v2vmd doctor
v2vmd run --video assets/video/dance.mp4 --device cuda --foot-lock
v2vmd run assets/video/dance.mp4 --device cuda --foot-lock --inspect
v2vmd run assets/video/dance.mp4 --device cuda --depth-cleanup --foot-lock --inspect
v2vmd inspect outputs/<run>  # reopen saved motion + original video in a local browser
v2vmd run --image assets/image/pose.jpg --model assets/mmd/character.pmx
v2vmd doctor --skeleton outputs/<run>/skeleton_motion.npz
v2vmd run --skeleton outputs/<run>/skeleton_motion.npz --model character.pmx --foot-lock
v2vmd run --skeleton outputs/<run>/skeleton_motion.npz --format bvh --no-preview
v2vmd run --output-dir DIR --progress-jsonl FILE --no-preview
```

VMD is the default; only VMD and BVH are supported. `--rig`/GLB and the legacy
Mixamo code were removed. Cached-skeleton exports need no source video, SMPL-X,
or inference weights. Extraction is expensive; reuse the NPZ when changing targets.
`--ik-mode ik` requires a supported PMX and VMD output; baked FK is the default.

## Outputs and conventions

- `motion_raw.vmd` / `motion_corrected.vmd`: before/after optional depth + foot cleanup.
- `motion.vmd`: exact corrected copy; BVH runs instead write `motion.bvh` and metadata.
- `skeleton_motion.npz`: reusable 24 SMPL joints, Y-up meters.
  New extractions also save `camera_joints_3d` (OpenCV camera coordinates),
  `camera_intrinsics` (full staged-image pixels), `camera_image_size` (W,H),
  and per-frame `camera_world_to_camera` rotations for overlap inspection.
- `target_motion_raw.npz` / `target_motion.npz`: target hierarchy, meter offsets,
  local xyzw quaternions, timestamps, profile hash and fitted-channel mask.
- `contacts.npz`, `quality.json`, `run.json`, optional skeleton preview and
  optional canonical 404-column `motion_encoder.npy`.
- With `--depth-cleanup`: `skeleton_motion_corrected.npz`, `target_motion_depth.npz`,
  and `depth_cleanup.json` / `.npz`. Preserve original source and raw exports.

Keep fingers/face neutral until an observation backend exists. The neck uses a
rest-relative chest-to-head axis; the head follows it without independent gaze
tracking. Do not aim along SMPL's short skull-center offset. Keep source timing,
coordinate bases, root ownership and quaternion conventions explicit.

## Code map

- `video2vmd/cli.py`, `paths.py`: CLI, preflight and locations.
- `video2vmd/extract.py`: GVHMR video/still extraction and input staging.
- `video2vmd/depth_cleanup.py`: optional static-camera metric root-depth regularizer.
  Translate all joints together, preserve timestamps/articulation, protect coherent
  motion, bound reprojection, and evaluate all stages on the same raw contact mask.
- `video2vmd/mmd/`: profiles, body retargeting, contacts, export, preview, pipeline.
- `video2vmd/inspection/`: localhost viewer, source proxy, HTTP byte ranges,
  raw/corrected FK buffers, estimated/manual camera overlap and video-clock-driven
  canvas playback. No training imports. Letterboxing must match the media's
  contained image rectangle; camera edits must never affect exported transforms.
- `blender_extension/video2vmd_blender/`: thin Blender UI and subprocess services.
- `docs/vmd-retargeting.md`: format contract, controls and known limits.

## Blender extension

Must not import PyTorch/GVHMR/OpenCV into Blender's Python. Launch
`<external-python> -m video2vmd.cli ...` with shell=False.
Build `dist/video2vmd-0.2.0.zip` with `python scripts/package_blender_extension.py`.
The ZIP has its manifest and __init__.py directly at root.

Blender 4.2+: Preferences > Get Extensions > repository dropdown > Install from
Disk. Enable Video2VMD; the panel is Viewport > N sidebar > Video2VMD. Configure
External Python and Project Directory. VMD targets are optional PMX files; BVH
imports a skeleton directly. VMD import needs MMD Tools and an active PMX armature.
Jobs write `outputs/blender_jobs/<id>/`. Rebuild/reinstall after add-on changes.
Disable the old extension first; quit/reopen Blender if old registrations persist.
See `blender_extension/README.md` and `docs/blender-extension.md`.

## Verification

Run `python scripts/verify_mmd.py`, `python scripts/verify_video_input.py`, and
`python scripts/verify_blender_extension.py`, plus `python scripts/verify_inspection.py`
for viewer data/HTTP/CLI changes. Independent VMD/BVH import scripts
run in fresh background Blender processes, never the user's open scene. Shared
codec changes belong in the separate vmd2png repository; pin a published tested
revision and leave the motion_encoder checkout unchanged.

Run `node scripts/verify_inspection_projection.cjs` for overlap projection changes.
Run `python scripts/verify_depth_cleanup.py` for depth-cleanup or related pipeline changes.
Run `python scripts/verify_feet.py` for contact, heel/toe pitch or leg-solver changes.
Run `python scripts/verify_noise_analysis.py` for batch-analysis metric changes.
Run `python scripts/verify_hand_motion.py` for hand observation/cache or gap-interpolation changes.
See `docs/noise-analysis.md` for reproducible multi-video processing and ablations.
`scripts/cache_inspection_camera.py` can add camera arrays to an older run using
its matching trusted local GVHMR cache; it needs SMPL-X but does not rerun inference.
