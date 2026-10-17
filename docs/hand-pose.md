# Hand-pose feasibility and proposed integration

Status (2026-10-17): MediaPipe and official WiLoR observation probes implemented
and compared; hand-pose gap completion is implemented. Production finger
retargeting is not enabled. Body extraction and existing VMD/BVH output are unchanged.

## What is already available

- Standard and AB7 PMX profiles each contain 30 finger controls: three joints per
  digit, including `Thumb0/1/2`, on both sides. VMD name mappings already exist.
- VMD/BVH exporters can serialize animated finger rotations. A new codec is not
  required for the basic hand pass.
- The installed GVHMR predicts 21 body-joint rotations, not articulated fingers.
  The cached 24-joint skeleton's coarse hand points cannot recover finger motion.
- Saved camera observations and the original 2D COCO-17 caches provide candidate
  wrist regions. Original videos are available at 1080x1920 or 2160x3840.
- The PMX reader already exposes bone tails, local/fixed axes and inheritance.
  Our profile adapter must retain/validate hand-specific geometry when animating
  those controls. AB7's finger controls have tails and no inherited/fixed-axis
  metadata. Terminal phalanges need their tail directions; do not assume every
  rig's final finger segment points along its preceding segment.

## Local MediaPipe experiment

An isolated environment lives at `outputs/dependencies/hand_probe_env` with
MediaPipe 1.1.1. The existing extraction environment was not modified. The official
full hand-landmarker bundle was downloaded to
`weights/mediapipe/hand_landmarker.task`; its SHA-256 is recorded in the report.

`scripts/probe_hand_pose.py` samples 12 evenly spaced frames per clip. It compares
full-image inference with two original-resolution crops guided by projected
body wrists/elbows. The model runs in IMAGE mode with detection/presence thresholds
of 0.5. Candidates are matched to the body wrist within 0.6 forearm lengths
(minimum 20 pixels). These are proposal-match counts, not annotated recall,
pose accuracy, left/right identity validation, or temporal tracking results.
Occluded/out-of-frame hands are still included in the denominator.

| Clip | Full-frame matches / 24 hand slots | Guided-crop matches / 24 |
|---|---:|---:|
| v01 | 2 | 6 |
| v02 | 0 | 8 |
| v03 | 4 | 12 |
| v04 | 1 | 4 |
| v05 | 2 | 3 |
| v06 | 1 | 14 |
| Total | 10 / 144 (6.9%) | 47 / 144 (32.6%) |

Raw full-image inference produced 17 hand proposals, of which 10 passed the
body-wrist association. Cropped inference produced 53 proposals, of which 47
passed. Body association can itself reject correct hands or assign the wrong
side; matched hands are not automatically safe to animate.

Contact-sheet inspection shows clear hands with useful landmarks, but also
covered fingers, motion blur, side-on hands, cut-off crops and ambiguous overlapping
hands. Some matched skeletons are visibly inaccurate. Only 37/47 matched crop
labels agreed with the body side; disagreement can arise from either backend,
mirroring or association and must not be resolved by blindly trusting one label.

Replacing projected wrists with the cached 2D body detections on v02/v04 did not
improve this small sample: crop matches were 8/24 and 2/24 respectively, versus
8/24 and 4/24 with reconstructed wrists. A single crop strategy is insufficient.

Outputs: `outputs/diagnostics/hand_pose_probe_20261017/` contains per-clip JSON,
21-landmark proposals in image/hand coordinates, and annotated contact sheets.
The 2D-guided comparison is under `hand_pose_probe_2d_20261017/`.

```powershell
outputs/dependencies/hand_probe_env/Scripts/python.exe scripts/probe_hand_pose.py outputs/diagnostics/noise_study_20261017 --output outputs/diagnostics/new_hand_probe
```

Use `--clip v06` to select a clip; `--samples` changes sample count. Optional
`--body-keypoints DIR` accepts per-clip numeric NPZs containing `keypoints`
(T,17,3; full staged-image x/y/confidence), `timestamps` and `image_size` (W,H).
The probe never deserializes training-model caches or changes source assets.

## Backend decision

Keep MediaPipe as a lightweight diagnostic baseline. Its current sparse coverage
does not support enabling automatic fingers by default on these dance clips.
Test full-rate VIDEO tracking and multiple crop scales before drawing a conclusion
about its best achievable coverage.

WiLoR has now been tested on the same frames and a short contiguous sequence,
using the user's
`models/mano_v1_2.zip`, extracting `mano_v1_2/models/MANO_RIGHT.pkl` and its license
to ignored `outputs/dependencies/wilor_mano/`. The model archives in `models/`
are gitignored. Both official setups require this separately supplied MANO model.

### WiLoR comparison

Official source revision: `fcb911312a38fa8badd30d9656a167485d61b8f9`.
The source lives in `outputs/dependencies/WiLoR/`; official detector and
reconstruction weights live in `weights/wilor/`. Missing import dependencies
were installed into `outputs/dependencies/wilor_extra`, leaving the main
extraction environment's installed packages unchanged. The probe uses float32
inference on the RTX 5070 Ti, a 640-pixel detector input, official detector
confidence 0.3, crop rescale 2 and the exact same frames/body gates/crops as the
MediaPipe baseline. Detector scores from different backends are not calibrated
to each other. Checkpoint keys matched the official architecture exactly.

| Clip | MediaPipe full / crop | WiLoR full / crop |
|---|---:|---:|
| v01 | 2 / 6 | 13 / 10 |
| v02 | 0 / 8 | 8 / 17 |
| v03 | 4 / 12 | 12 / 17 |
| v04 | 1 / 4 | 10 / 13 |
| v05 | 2 / 3 | 13 / 14 |
| v06 | 1 / 14 | 16 / 19 |
| Total (144 slots) | 10 / 47 | 72 / 90 |

WiLoR has more geometric matches in this experiment. Taking the union of its
full-frame/crop matches gives 111/144 slots; this is proposal availability, not a
validated fusion policy or pose-accuracy score. Only 54/90 crop labels agree with
the body side. Neither backend's side labels should be used without association
and temporal checks. Contact-sheet review shows useful finger estimates on clear
hands, but also proposals on clothing/skin and inaccurate estimates near faces,
blurred hands and occlusion. Do not enable raw WiLoR fingers by default.

A second experiment covers all 60 source frames of v06 at 10–12 seconds (30 fps):
MediaPipe matches 0/120 full-frame and 101/120 crop slots; WiLoR matches 110/120
full-frame and 116/120 crop slots. These are independent framewise detections,
without temporal model inference, tracking or smoothing. The interval was chosen
after inspecting sparse samples and is not held-out evidence.

WiLoR crop observations cover 57/60 left and 59/60 right frames, with longest gaps
of two/one frames. Its handedness labels flip twice on the left and zero times on
the right. Across consecutive matched frames, 95th-percentile absolute finger-bend
steps are 23.2° left and 15.5° right, with maxima of 86.0°/51.1°. Corresponding
MediaPipe values are 23.7°/26.8° (maxima 72.1°/73.8°), with two/six label flips.
These steps include genuine articulation, estimator jitter and association errors;
they are not measured reconstruction error. Better observation coverage alone
does not solve temporal stability.

Sparse outputs: `outputs/diagnostics/hand_pose_wilor_20261017/`.
Continuous outputs and side-by-side video:
`outputs/diagnostics/hand_pose_sequence_wilor_20261017/`.
The JSON retains 21 camera-basis joints, projected landmarks, raw canonical MANO
pose matrices/betas and detector scores. Left-hand display joints reflect X as
in the official demo; saved MANO rotation parameters still use the canonical
right-hand convention and must not be applied directly to left PMX controls.

```powershell
python scripts/probe_wilor_hands.py outputs/diagnostics/hand_pose_probe_20261017 --output outputs/diagnostics/new_wilor_probe
outputs/dependencies/hand_probe_env/Scripts/python.exe scripts/probe_hand_pose.py outputs/diagnostics/noise_study_20261017 --clip v06 --range 10 12 --output outputs/diagnostics/new_sequence_mp
python scripts/probe_wilor_hands.py outputs/diagnostics/new_sequence_mp --output outputs/diagnostics/new_sequence_wilor
```

With gap completion available, the next implementation work is stable
body-linked identity, geometric/temporal rejection and optional palm-relative
finger retargeting. WiLoR is the stronger
observation candidate on these clips, while MediaPipe remains a useful baseline.

### Missing-hand interpolation

The user's chosen policy is to interpolate missing hand poses across every bounded
gap, regardless of duration. `video2vmd/hand_motion.py` implements timestamp-based
shortest-path quaternion SLERP for all 15 local MANO finger rotations and palm
orientation, independently for the left/right body-associated tracks. Observed
poses are preserved as rotations; only equivalent quaternion signs may change.
MANO shape coefficients interpolate linearly. No smoothing or rejection of
observed pose keys is applied by this stage.

Leading/trailing missing frames hold the nearest observed pose. A hand with a
single observation holds that pose throughout. A hand with no observations stays
identity/zero and is marked unavailable. There is no automatic neutral fade or
gap-length cutoff. Detector handedness labels are retained as raw metadata and
do not exchange body-side tracks.

New WiLoR probe runs automatically write `<clip>_hand_motion.npz`. Existing JSON
observations can be completed without inference:

```powershell
python scripts/complete_hand_motion.py outputs/diagnostics/hand_pose_sequence_wilor_20261017 --output outputs/diagnostics/completed_hands
python scripts/complete_hand_motion.py observations/v06.json --output outputs/completed_hands --timeline outputs/body_run/target_motion.npz
python scripts/verify_hand_motion.py
```

The default candidate source is the crop track used in the comparison; pass
`--candidate-source full` to complete the full-frame track instead. The optional
`--timeline` resamples one clip onto an NPZ's exact `timestamps`. Relative rotations
interpolate in local MANO axes, not by blending 3D/image landmark positions.

The version-1 numeric `hand_motion.npz` contains:

- `timestamps`, `quat_xyzw` (F,2,16,4) and `betas` (F,2,10). Channel 0 is the MANO
  palm orientation; channels 1–15 are its canonical local finger rotations.
- `status` (F,2): 0 unavailable, 1 observed, 2 interpolated, 3 held at an edge.
  `observed` and `available` expose those masks directly.
- `source_timestamps`, `source_observed` and `source_detector_right` preserve the
  original evidence even when output timing changes. `metadata` records the
  coordinate convention and interpolation policy. The cache loads with
  `allow_pickle=False` and needs no inference runtime.

These rotations remain in canonical right-MANO axes for both body-side tracks;
the future PMX hand retargeter must convert rest axes and left-hand parity.

On the 60-frame v06 sequence, completion retains 116 observed hand samples,
interpolates three missing left-hand samples and holds one leading right-hand
sample, giving 120/120 available poses. Source observations are preserved within
1.7e-7 per rotation-matrix element. Outputs and a joint-angle plot are under
`outputs/diagnostics/hand_pose_interpolated_sequence_20261017/`. The six sparse
probe caches were also completed under `hand_pose_interpolated_sparse_20261017/`;
those remain sparse experiments, not full-video hand inference.

- [MediaPipe Python task](https://developers.google.com/edge/mediapipe/solutions/vision/hand_landmarker/python)
  returns 21 image landmarks, centered metric hand landmarks and handedness.
  Handedness score is classification confidence, not per-finger reliability.
- [WiLoR official setup](https://github.com/rolpotamias/WiLoR) specifies pretrained
  detector/reconstruction weights and a separately acquired MANO model. Its
  [repository license](https://github.com/rolpotamias/WiLoR/blob/main/license.txt)
  is CC BY-NC-ND 4.0; keep model distribution separate from this project.
- [HaMeR official setup](https://github.com/geopavlakos/hamer) also requires
  pretrained weights and `MANO_RIGHT.pkl`.

## First production increment

1. Add an optional external-Python hand observation stage that can run against an
   existing body cache and its original video. Preserve source timestamps and
   cache observations separately in `hand_motion.npz`. Re-exporting a new target
   must not rerun body or hand inference. Blender imports no observation packages.
2. Track hand identity jointly with body wrists and temporal continuity. Preserve
   raw handedness labels, association decisions and rejection reasons. Use multiple
   crop scales/fallback detection when body wrist guidance is wrong. Deduplicate
   detections from overlapping left/right crops.
3. Store raw 21-point image and hand-relative 3D observations, presence/validity,
   timestamps, crop transforms, backend/model hashes and source fingerprint. Keep
   detector confidence, handedness confidence and derived pose-quality tests as
   distinct fields. Do not fabricate a per-landmark confidence from handedness.
4. Fit finger articulation in a palm-relative frame against each target's rest
   pose. Preserve bone lengths and explicit thumb geometry. Anchor the wrist at
   the existing body solution. Initially preserve body wrist orientation; add
   independently gated palm orientation after coordinate/parity tests pass.
   MediaPipe's centered "world landmarks" are not the GVHMR world frame, and its
   normalized image z is not metric camera depth. Never splice these into body
   joint positions or mirror quaternion components.
5. Use the implemented gap completion to resample onto body output timestamps,
   SLERP all bounded gaps and hold clip edges. Keep observed/interpolated/held
   masks distinct. Any later smoothing/joint-limit stage must be separately
   evaluated; gap completion preserves current observed pose keys.
6. Write dense finger keys, including neutral/reset keys for opted-in hand tracks.
   Update the fitted-channel mask and `unobserved_hands` metadata. Preserve the
   existing all-neutral behavior when the hand option is off. Extend the optional
   standard-profile motion-encoder path as well as PMX/VMD/BVH output.
7. Add a hand close-up overlay to inspection, with accepted/rejected/missing states,
   and verify terminal finger transforms in a fresh Blender process.

## Acceptance checks

- Known open hand, fist, spread fingers and thumb opposition on both sides; target
  rest pose must remain neutral and scale changes must not change joint angles.
- Mirrored inputs, crossing/overlapping hands, body turns, fast wrist rotation,
  motion blur, sleeves, hands leaving the frame and reacquisition after gaps.
- No left/right swaps or wrist translation changes; legs/root/body transforms must
  be identical with only finger articulation enabled.
- Bone-length preservation, valid local xyzw rotations, bounded temporal changes,
  unchanged source timing and explicit missing-observation masks.
- Coverage and longest missing runs on independently annotated visible-hand spans,
  plus overlay agreement and temporal stability. Do not score quality solely by
  proposal count or smoothness. Keep a few clips held out when tuning thresholds.
- VMD readback and independent PMX/BVH playback, including terminal phalanges and
  neutral resets; cache reuse without inference dependencies.
