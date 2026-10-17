# Batch motion noise analysis

Use the extraction environment (`p11` or the project's `.venv`) from the project
root. GPU extraction needs the normal SMPL-X model and inference weights.

```powershell
python scripts/process_noise_study.py --output outputs/diagnostics/noise_study_20261017
python scripts/analyze_motion_noise.py outputs/diagnostics/noise_study_20261017
python scripts/verify_noise_analysis.py
```

The processing script runs all videos in `assets/video` sequentially, with a fresh
process for each clip to release GPU memory. It reuses an existing extraction
when a cache's source size and nanosecond modification time match a renamed video,
then verifies five decoded frames against the staged video. Source SHA-256,
reuse provenance and pixel comparisons are saved per clip. Changed source content
or target geometry requires a fresh study directory. Expensive intermediate
extraction caches and completed processing are retained for retries.

All clips use the same PMX (default: the local AB7 skeleton; change `--model` to
use another character). No PMX mesh is rendered. Baked FK motion is evaluated at
30 fps. Each clip contains four ablations:

- Raw retargeting, before optional cleanup.
- Depth cleanup only.
- Foot cleanup only, using the original raw contact evidence.
- Depth plus foot cleanup, using that same evidence.

Raw/corrected VMD exports and the original/corrected source and target NPZs live
under each clip's `combined/`. `target_motion_feet.npz` and `motion_feet.vmd` save
the additional foot-only ablation. Analysis does not alter those transforms.

The report contains `index.html`, `analysis.md`, `findings.md`, summary CSV/JSON,
PNG/SVG charts, per-clip target `positions.csv`, source/camera `source_camera.csv`,
frequency spectra and background tracking caches. Start a localhost file server
in the report directory to browse the HTML. Per-clip `analysis.json` includes
2/3/5 Hz sensitivity, axis RMS, lag-one correlation, peak inspection windows,
edit sizes, contact coverage and image displacement under the estimated camera.

Hip means the target `HipMaster` bone; feet position plots show ankle centers.
Heel/toe proxies are separate and supply contact slip and floor metrics. Target
positions use the exported MMD basis (meters, Y up). Source/camera plots use the
original extraction basis and source body scale; their traces have individual
origins removed for comparison.

The nominal 3 Hz high-pass is an order-four zero-phase Butterworth filter. RMS
omits one second at both ends, or 20% for short clips. Welch spectra use linear
detrending and up to 128 samples per segment. High-frequency attenuation is a
noise **proxy**, not proof of better reconstruction: real impacts, fast footwork
and camera movement also contribute. Hip-relative foot trajectories separate
articulation from common translation; depth-only processing must preserve them.

Foot quality is scored on the same raw mask at every stage. It is inferred from
the reconstruction, not independently annotated support. Near-floor unlocked
frames include genuine steps/slides and must not automatically be treated as
missed plants. Clearing skeletal proxies does not prove mesh clearance.

All current extractions use static-camera recovery and estimated intrinsics.
The camera center derived from world/camera body predictions is not independent
SLAM, so correlations do not establish causal contributions to noise. Background
LK flow with affine RANSAC excludes the estimated person region and offers an
independent image-motion check at approximately 10 Hz and 480 pixels longest
side. Poor fits and possible cuts are reported. It cannot reliably identify
optical-axis camera movement or recover a metric trajectory. Target image
displacement measures changes under the estimated camera, not error against
observed landmarks; the depth-only reprojection budget does not constrain feet.

## Foot-transition comparison

Compare an existing study's saved combined motion against the current foot solver
without rerunning inference or changing its contact evidence:

```powershell
python scripts/compare_foot_transitions.py outputs/diagnostics/noise_study_20261017 --output outputs/diagnostics/foot_transitions_new --model assets/mmd/ab7_160404/ab7.pmx --plots
```

Use a fresh output directory outside the original study. `--clip v01` selects a
clip (repeatable); omit `--model` for NPZ-only results. `--plots` writes trajectory
plots around each clip's largest previous horizontal correction acceleration.
The report includes unchanged-mask hold/transition slip, correction derivatives,
floor penetration, anchor distance, body jerk and added image displacement under
the estimated camera. The same saved raw pelvis anchors both camera projections.
Input hashes and the solver hash are recorded; originals remain intact. This
comparison expects cached depth-stage and combined target motion from a study.

On the six-video AB7 study, the v3 horizontal fit reduced pair-weighted transition
speed from 118.0 to 60.4 mm/s and hold speed from 14.9 to 8.1 mm/s. Maximum actual
horizontal correction acceleration was at most 12 m/s²; all targets were reachable
and no optimizer iteration limits were reached. Mean distance from the original
median anchors increased by 2.7–12.4 mm depending on the clip. Additional estimated
camera displacement stayed below 0.86% of image height. Body jerk changed by less
than 0.5% and slightly increased; v03 hold slip rose from 3.5 to 5.2 mm/s. These are
solver comparisons on inferred contacts, not proof of better reconstruction.

## Authored VMD reference

```powershell
python scripts/compare_reference_vmd.py outputs/diagnostics/noise_study_20261017 --archive "D:\studio\Unity\content\motion\fav\Hibikase.zip"
```

This adds `reference_hibikase/` and a link in the main report. By default it selects
the largest VMD in the archive (the dance, excluding separate camera/face VMDs);
`--member` selects an explicit ZIP member. Only that VMD and the English readme
are copied to ignored outputs. Keep the readme credit: original motion Miku VR,
converted/fixed by GentleWhiteCat. The input archive is read-only.

`sample_reference_vmd_blender.py` evaluates the same AB7 PMX armature in a fresh
background Blender process, with no character geometry or physics. MMD Tools
creates an empty bookkeeping mesh. All authored foot/toe IK is honored in the
primary variant; another variant mutes native IK and IK knee-limit constraints.
World joint positions and physical bone rotations are converted to MMD Y-up
meters. The external project environment handles analysis; Blender imports no
Torch/GVHMR/project training packages. `--blender`, `--mmd-tools` and `--model`
select other local paths; reference and generated runs must share the PMX hash.

Reference sampling is cached by input VMD hash and PMX profile hash; `--resample`
forces fresh evaluation. Figures include reference XYZ positions, 3/5 Hz summary
comparisons, frequency spectra, equal seven-second window distributions and an
activity plot. Seven-second windows stride by 3.5 seconds; their overlap means
they are not independent observations. No time warping or pose alignment is
applied. CSV/JSON files preserve positions, metrics and provenance.

The reference is a different dance, with different tempo and activity, and was
authored for a different original character. On AB7 its floor/IK behavior also
reflects model adaptation. It is a descriptive reference, not paired ground truth,
a noise-free ideal, or a target level of smoothing. A generated trajectory with
lower high-frequency RMS is not necessarily more accurate or more natural.
