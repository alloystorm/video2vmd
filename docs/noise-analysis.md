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
