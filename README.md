# Video2VMD

Extract body motion from a single-person video or still image and export MMD
**VMD** or **BVH** animation. Command: `v2vmd`.

```text
video / image -> GVHMR + SMPL-X -> reusable human skeleton
                                      |
                         direct MMD retargeting
                                      |
                       optional foot-contact cleanup
                                      |
                    raw VMD + corrected VMD / BVH
```

## Install

Python 3.11+, Git, and an environment with PyTorch for your device. For CUDA,
use a CUDA-enabled PyTorch installation before selecting `--device cuda`.

```powershell
git clone https://github.com/alloystorm/video2vmd.git
cd video2vmd
python -m pip install --upgrade pip setuptools wheel
python -m pip install "numpy>=1.26"
# chumpy is a legacy GVHMR dependency that needs pip during its build.
python -m pip install --no-build-isolation "chumpy==0.70"
python -m pip install -e .
```

VMD support, including the pinned shared `vmd2png` codec, is installed by default.
Add ffmpeg to PATH for input normalization and browser-compatible previews.
Inference weights download to `weights/` on first extraction (about 5 GB).

## Inputs

- **Video/still extraction:** supply the licensed SMPL-X model at
  `assets/body_models/smplx/SMPLX_NEUTRAL.npz`, from
  [SMPL-X](https://smpl-x.is.tue.mpg.de/). Use one clearly visible person, head to toe.
- **Cached skeleton:** reuse a previous `skeleton_motion.npz`; the original video,
  SMPL-X model and inference weights are not needed for retargeting.
- **Optional PMX:** `--model character.pmx` fits the target's skeleton/proportions.
  Without it, the shared standard MMD A-pose is used. Extract ZIP archives first.

User assets and generated outputs are gitignored. See [asset locations](assets/README.md).

## Run

```powershell
v2vmd doctor
v2vmd run --video assets/video/dance.mp4 --device cuda --foot-lock
v2vmd run --image assets/image/pose.jpg --model assets/mmd/character.pmx

# Reuse extraction and change the target quickly.
v2vmd doctor --skeleton outputs/previous/skeleton_motion.npz
v2vmd run --skeleton outputs/previous/skeleton_motion.npz --model assets/mmd/character.pmx --foot-lock

# BVH or canonical motion_encoder data.
v2vmd run --skeleton outputs/previous/skeleton_motion.npz --format bvh --no-preview
v2vmd run --skeleton outputs/previous/skeleton_motion.npz --motion-encoder
```

`v2vmd run` defaults to VMD and the newest video in `assets/video/`, falling back
to `assets/image/`. `--video` and `--image` are mutually exclusive. Use
`--output-dir DIR --progress-jsonl FILE` for automation. The module entry point is
`python -m video2vmd` (or `python -m video2vmd.cli`).

Each VMD run writes `motion_raw.vmd` and `motion_corrected.vmd`; `motion.vmd` is
an exact copy of the corrected file. Raw means before optional foot cleanup;
both use the same retargeting and neck mapping. Without `--foot-lock`, they are
identical. Runs also save reusable skeletons, target motion, contact evidence,
quality metrics and a front/side skeleton preview. Default output directory:
`outputs/<timestamp>_<source>/`.

See [format conventions, IK mode and quality limits](docs/vmd-retargeting.md).
Foot cleanup is conservative. Fingers, face, gaze and secondary animation are
not captured; head orientation follows a coarse fitted neck axis.

## Blender

The [Video2VMD extension](blender_extension/README.md) runs the external Python
pipeline and displays progress. It imports BVH skeletons directly. VMD can be
applied to a selected PMX armature with MMD Tools enabled. Blender's Python does
not load PyTorch, GVHMR or OpenCV.

## Development

```powershell
python scripts/verify_mmd.py
python scripts/verify_video_input.py
python scripts/verify_blender_extension.py
python scripts/package_blender_extension.py
```

See [AGENTS.md](AGENTS.md), the [implementation plan](docs/vmd-retargeting-plan.md)
and the [Blender smoke checklist](docs/blender-extension.md).

## Migration and credits

Version 0.2 renames `m2mr` to `v2vmd` and the Python package to `video2vmd`.
Reinstall this checkout in your environment; uninstall the old `motion2mixamorig`
distribution to remove its obsolete console command. GLB output, `--rig`, and the
Mixamo exporter have been removed. Existing `skeleton_motion.npz` files work.
Disable the old Blender extension and install `dist/video2vmd-0.2.0.zip`.

This fork builds on [MaxLiu22/Motion2MixamoRig](https://github.com/MaxLiu22/Motion2MixamoRig).
Older release notes and translated upstream READMEs are historical. See
[LICENSE](LICENSE) and [NOTICE.md](NOTICE.md) for upstream licensing and attribution.
