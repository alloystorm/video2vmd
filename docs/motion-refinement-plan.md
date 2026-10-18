# Learned motion refinement plan

Prepared 2026-10-18. Status: proposed experiment; implementation and training
have not started. Model sizes and dataset volumes below are planning estimates,
to be revised using measured learning curves and held-out results.

Renderer progress (2026-10-18): the dxr_native CLI has passed a short
render/sidecar/reconstruct/export smoke test. See [the verified workflow and
results](dxr-rendering.md). Model training and the reference-to-SMPL joint adapter
remain outstanding.

## Objective

Train a model to recover accurate body motion from the errors introduced by our
video reconstruction pipeline. Generate supervised pairs by rendering known
motion, reconstructing the rendered video with GVHMR, and using the original
evaluated motion as the target.

The objective is fidelity to the original movement, including fast footwork,
impacts, timing and travel. Reduced high-frequency energy alone is insufficient:
an excessively smoothed or stationary output can score well on that proxy.

The first version covers the existing 24-joint body skeleton. Fingers and face
remain outside this experiment. Normal use accepts cached skeleton motion;
rendering and video reconstruction are dataset-generation steps.

```text
Known motion -> character playback -> rendered video -> existing extraction
                    |                                         |
          evaluated reference skeleton                 noisy skeleton
                    |                                         |
                    +------------ supervised pair ------------+
                                                              |
                                                   refinement model
                                                              |
                                                   corrected skeleton
                                                              |
                                                retargeting -> VMD/BVH
```

## Evidence and starting assumptions

The [noise analysis](noise-analysis.md) separates shared root-depth motion from
relative articulation. Existing depth cleanup preserves relative joint motion;
foot cleanup acts on inferred support and floor collisions. Neither is a general
body-pose restoration model.

The six-video study and Hibikase comparison under
`outputs/diagnostics/noise_study_20261017/` are useful diagnostics. Hibikase is a
different dance and is not paired ground truth for those videos. Rendering it
would create a paired test of its own movement, subject to correct playback and
skeleton mapping. One choreography is sufficient to debug that process, but
insufficient to establish generalization.

The initial deployment target is offline processing of clips lasting a few
seconds to a few minutes. Future frames are available. Long clips can use a
fixed-size model through overlapping windows.

## Initial configuration

| Decision | Starting choice |
| --- | --- |
| Representation | 24 joint positions, Y-up meters, explicit timestamps |
| Network input | Root-relative pose and a separate root trajectory; optional velocity, camera and validity features |
| Network output | Residual corrections to root trajectory and relative pose |
| Architecture | Small temporal convolutional network with separate root and pose heads |
| Main model budget | Approximately 2–3 million parameters |
| Smaller baseline | Approximately 0.5–1 million parameters |
| Initial model window | 128 frames, approximately 4.3 seconds at 30 fps |
| Window ablation | 32, 64 and 128 frames |
| Reconstruction sources | Usually 20–60 seconds, plus complete multi-minute examples |
| Pilot dataset | Approximately 1 hour of distinct original motion, six variants each |
| Expansion target | 3–5 hours of distinct original motion, six variants each |
| Inference | Overlapping windows, initially 50% overlap, blending corrections |

The temporal receptive field must actually cover the selected window; a nominal
128-frame input does not guarantee 128 frames of useful context. Record the exact
architecture, parameter count and receptive field when implementing the model.

Float32 weights for 2–3 million parameters occupy approximately 8–12 MB. Training
requires additional memory for activations, gradients and optimizer state.
Benchmark batch size, peak memory and throughput on the available GPU before
estimating training time. Rendering and extraction have separate compute costs.

## Dataset generation and correctness

1. Select good-quality reference motions with known provenance and suitable use
   terms. Cover dance, locomotion, turns, holds, jumps, crouches and fast footwork.
   Include both stationary support and genuine slides or pivots.
2. Validate the reference playback and skeleton mapping. Initially prefer a
   compatible SMPL-family character so the reference and reconstructed joints
   have matching definitions. Broader character rigs follow after mapping checks.
3. Render variants with recorded camera intrinsics, extrinsics, frame times,
   body shape, ground plane and playback settings.
4. Save the reference skeleton actually evaluated on the rendered character,
   after retargeting, constraints and IK. Do not use pre-retargeting source keys
   as if they were the final rendered motion. Skeletal evaluation alone is not a
   render: extraction needs the character mesh in an image sequence or video.
5. Run the existing extraction pipeline on each complete rendered sequence.
   Preserve raw reconstruction before depth or foot cleanup.
6. Convert reference and reconstruction into an explicitly shared convention,
   validate temporal correspondence, then sample paired training windows.

Run reconstruction on longer sequences before cropping motion windows. Running
the extractor independently on every four-second crop could produce different
errors from normal long-video use.

Begin with fixed cameras matching the current static-camera path. Add moving
cameras as a separately labeled experiment once the reconstruction mode and
coordinate treatment are validated. Vary viewpoint, distance, body shape,
appearance, lighting, background, compression and partial occlusion. Include
human-like appearances to investigate transfer to real footage; anime-only
renders could produce a narrow reconstruction-error distribution.

Correctness requirements:

- Use the same joint definitions and order, metric units and timestamps on both
  sides. A simple bone-name mapping does not establish matching joint centers.
- Evaluate both skeletons on the same time grid using presentation times. Never
  align by unconstrained time warping, which could hide timing errors.
- Record any sequence-level world-origin or heading alignment. Report its effect
  and keep absolute root errors separate from root-relative pose errors. Avoid
  per-frame pose alignment in the primary score.
- Distinguish known render calibration from estimated extractor calibration.
  If an experiment supplies known intrinsics to extraction, label that condition
  separately from normal estimated-intrinsics extraction.
- Handle body proportions explicitly. Start with matched geometry; later record
  how shape-normalized inputs are restored to the intended skeleton proportions.
- Define contact labels from the evaluated reference and ground geometry, with
  uncertainty for ambiguous intervals. Stationary plants, slides and pivots must
  not all receive a stationary-foot constraint.
- Record extraction failures and exclusions by category. A failed track must not
  silently become an ordinary valid training sample.

## Data volume and splits

| Stage | Distinct original motion | Paired reconstruction footage at six variants | Purpose |
| --- | --- | --- | --- |
| Pilot | 30–60 minutes | 3–6 hours | Validate pairs and establish benefit over filtering |
| First useful model | 3–5 hours | 18–30 hours | Cover the main intended motion cases |
| Broader coverage, if justified | 20–50 hours | 120–300 hours | Extend movement and filming diversity |

For example, 360 original clips averaging 30 seconds contain three hours of
motion. Six variants produce 18 hours, or approximately 1.94 million paired frames
at 30 fps. These totals include held-out material; the training split is smaller.
Variants and overlapping windows are correlated, not independent choreography.

Assign approximately 80%/10%/10% train/validation/test splits at the original-motion
level before rendering. Keep every variant, crop, duplicate recording and related
choreography together. Where available, group performer/session identities to
test generalization beyond the people represented in training. Reserve unseen
characters or render conditions for a separate transfer test.

Maintain a fixed real-video evaluation set, including the existing six-video
study and additional unseen footage. Without paired ground truth, real footage
supports visual and consistency assessment, not an absolute accuracy claim.

## Model and training

Use a residual temporal network so already-good input can receive a near-zero
correction. Separate root translation from root-relative articulation to avoid
mistaking shared translation errors for errors in every limb.

Start with joint positions: 24 × 3 = 72 values per frame before extra features.
Fit the output to a consistent skeleton with explicit bone-length constraints;
unconstrained position regression can otherwise stretch limbs. Keep normalization
and the root reference consistent across windows and invert normalization before
writing corrected motion.

Train using:

- Position error against the reference, with root and relative-pose terms.
- Velocity and acceleration error against reference derivatives, computed using
  actual time intervals. Penalizing acceleration magnitude alone would suppress
  legitimate sharp motion.
- Bone-length consistency and reference-aware foot-contact terms.
- Clean-input examples whose desired correction is zero, to measure and resist
  damage to good motion.
- Validity masks for missing observations and padding; exclude invalid samples
  from derivative losses as well as position losses.

Optional synthetic corruptions can supplement real extractor errors, but actual
render-to-extractor pairs remain the main evidence. Select loss weights and
sampling balance on validation data. Record random seeds and compare repeated
runs before interpreting small gains.

Cache numeric pairs so training does not invoke Blender or GVHMR. Use the user's
external Python environment for training and extraction. Blender rendering and
reference sampling use separate background processes; do not import training
packages into Blender or alter the user's open scene.

## Short clips, long clips and boundaries

For clips shorter than the training window, use explicit padding and masks.
Include short examples and clip boundaries during training; report their quality
separately because they have less temporal context.

For longer clips, begin with 50% overlap and normalized tapered blending of
predicted corrections. Ensure endpoint frames receive valid nonzero coverage.
Preserve one root reference for the whole sequence; avoid independent window
recentering that introduces position or heading jumps. Measure seams against
ordinary interior frames, and never blend across detected cuts or track changes.

Local windows cannot reliably diagnose drift accumulated over minutes. If
long-sequence evaluation establishes this problem, add a separate coarse root
trajectory pass using downsampled full-sequence context. More parameters alone
do not supply missing temporal context. Absolute depth/scale ambiguity also
remains when the observations do not determine it.

## Evaluation and decision criteria

Compare raw extraction, current depth/foot cleanup, conventional temporal filters,
the small learned baseline, and the proposed model. Tune baseline settings on
validation data too. Evaluate learned refinement alone and with existing cleanup
to determine ordering and avoid double correction.

Primary paired metrics:

- Root-relative joint position error and per-joint error distributions.
- Root trajectory error, travel amplitude and drift over full sequences.
- Velocity and acceleration error against the reference.
- Foot slip on fixed reference stationary-support intervals, with separate
  transition, slide and pivot results; floor penetration and bone-length error.
- Event timing, peak-motion attenuation and changes to already-good input.
- Window seams and errors at short-clip boundaries.

Report full-sequence results and failures by motion and render condition. Use
original motions as the aggregation unit; overlapping windows must not inflate
the apparent sample count. Keep high-frequency RMS and jerk as diagnostics.
For synthetic image agreement, project with known render cameras; on real footage,
changes under estimated cameras are not measured landmark accuracy.

Review synchronized reference/raw/refined playback, including the eventual
retargeted VMD/BVH. Use complete 1–5 minute sequences for long-duration validation.

Before scaling beyond the pilot, require better held-out paired accuracy than
the tuned conventional baseline, preservation of fast movement and timing, and
acceptable clean-input, contact and boundary behavior. Set numerical acceptance
tolerances after measuring baseline variability, before selecting the final model.
If gains exist only on synthetic renders, improve the data/domain coverage before
increasing model capacity.

## Milestones

1. **Prove pair correctness.** Render a few representative motions, save evaluated
   ground truth, reconstruct them, and verify mapping, scale, coordinates and
   timing. Deliver a reproducible manifest and side-by-side report.
2. **Build the pilot.** Expand to approximately one hour of distinct motion with
   six variants, fixed source-level splits and resumable extraction caches.
   Establish conventional-filter and existing-cleanup baselines.
3. **Train and compare.** Compare the small and 2–3M parameter models, window
   lengths, identity behavior and root/pose losses. Plot validation performance
   against nested subsets of distinct training motions.
4. **Test real footage and long sequences.** Inspect transfer, fast footwork,
   boundaries, full-sequence root drift and compatibility with cleanup.
5. **Expand if supported.** Grow to 3–5 hours of distinct motion and target the
   failure categories identified by evaluation. Larger datasets/models or a
   coarse trajectory pass need evidence from these results.
6. **Integrate as an optional stage.** Preserve raw caches and exports; save a
   separate corrected skeleton and model/provenance report. Determine cleanup
   ordering from the ablations, then verify VMD/BVH and inspection behavior.

## Artifacts and pipeline integration

Keep user motions, rendered media, generated pairs and checkpoints in gitignored
locations. Track source/configuration manifests and implementation in the repo.
Fingerprint source motion, evaluated skeleton, character/profile, renderer,
cameras, extraction version/weights, sampling settings and seeds so changed
inputs invalidate only the relevant caches.

Proposed integration point: after body extraction and before target retargeting.
No CLI flag or refinement output format is implemented by this document.
The implementation must retain original timestamps and declare ownership of root
translation so it is not applied twice. Preserve original camera observations as
evidence; label refined geometry separately rather than silently replacing camera
arrays with inconsistent values. Hand extraction and retargeting interactions
need evaluation if their body-guided inputs change.

Run the existing verification scripts required by AGENTS.md when their associated
pipeline, viewer, contact or export code changes. New refinement checks should
cover identity input, temporal correspondence, bone lengths, root ownership,
short clips, masks, window seams and deterministic checkpoint loading. Independent
Blender playback validation runs in fresh background processes.

## References

- [SmoothNet paper](https://arxiv.org/abs/2112.13715) and
  [official implementation](https://github.com/cure-lab/SmoothNet): temporal pose
  refinement and compact window-based baselines.
- [Learning from Synthetic Humans / SURREAL](https://openaccess.thecvf.com/content_cvpr_2017/html/Varol_Learning_From_Synthetic_CVPR_2017_paper.html):
  rendering known human motion to generate labeled imagery.
- [AMASS](https://amass.is.tue.mpg.de/): a possible source of diverse reference
  motions, subject to selecting suitable data and checking applicable terms.
- [Noise-analysis guide](noise-analysis.md): current ablations, conventions and
  limits of the existing motion-quality proxies.
