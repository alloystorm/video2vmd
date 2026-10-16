# VMD retargeting plan

Prepared 2026-10-16 for `alloystorm/video2vmd`, after reviewing the local
`D:/studio/Projects/motion_encoder` checkout. The design below is retained as the
roadmap; the first implementation now exists. Current commands, validation and
limits are in [VMD/BVH usage](vmd-retargeting.md).

## Implementation status (2026-10-16)

- Implemented: shared codec IK-property/name fixes and PMX metadata, standard and
  PMX profiles, direct body FK retargeting, temporal heel/toe anchors, bounded
  pelvis/leg correction, VMD, BVH, canonical 404-column NPY and skeleton diagnostics.
- Verified: deterministic fixtures, independent Blender VMD/BVH import, supplied
  AB7 PMX in FK/foot-IK modes, full 817-frame extraction/export, existing GLB path,
  and motion_encoder feature slicing. The other project's checkout is unchanged.
- Still open: a second PMX, broader real-footage contact validation, user contact
  overrides, more general PMX control/inheritance support, and more reliable
  limb twist/head orientation. Cleanup remains conservative and opt-in.
- Later work: observed fingers/hands, learned refinement evaluation, downloads,
  trimming and a unified input-conversion workflow.

The modules live under `motion2mixamorig/mmd/` rather than being split into
format-wide top-level packages before another target needs those abstractions.

## Direction

Keep GVHMR/SMPL-X as the initial video-to-body-motion backend. Add a direct
human-to-MMD retargeter and use `vmd2png` as the shared MMD format/skeleton layer.
An MMD export must not depend on a Mixamo character or pass through Mixamo rotations.

Start with the same standard A-pose skeleton used by `motion_encoder`, baked FK
motion, and explicit disabling of model leg/toe IK. Add contact correction to
that working baseline, then add model-specific profiles and an optional IK-driven
VMD mode. Preserve the existing Mixamo/GLB path throughout.

The first milestone is a dense, 30 fps body-motion VMD that plays correctly on a
validated MMD reference model and can be inspected by `vmd2png`.

## What the other project already provides

Reviewed revisions:

- `motion_encoder`: `fd8514e8964938c43f329c238290e7fc6876a3e4`, branch `main`.
- Its `vmd2png` submodule: `a400b89b9d867419f453f7b7bad1f598aa034a7a`, branch `r6d`.
- Read `AGENTS.md`, the current journal, relevant design notes, format conversion,
  skeleton/FK/IK code, pose conversion, foot analysis, hand compression, and tests.

Paths in the following table are relative to the `motion_encoder` checkout.

| Existing component | Reuse and qualification |
| --- | --- |
| `vmd2png/src/vmd2png/vmd.py` | `parse_vmd`, `write_vmd`, Japanese bone-name mapping, playback and leg IK. Reuse the codec through a small adapter, with the limitations below fixed first. |
| `vmd2png/src/vmd2png/skeleton.py` | Reference skeleton: 73 total bones, 65 exported by default, including 30 finger joints. Its arms slope down about 26.6 degrees: this is an A-pose, not Mixamo's T-pose. |
| `vmd2png/src/vmd2png/bone.py` | Hierarchical FK and quaternion/6D conversion. Quaternions are `xyzw`; the 6D representation concatenates the first two matrix columns. |
| `vmd2png/src/vmd2png/pose_convert.py` | Existing T-to-A arm conversion and fixtures. Useful for verifying calibration; a new direct retargeter should solve against the target's actual rest pose, without applying this correction twice. |
| `vmd2png/src/vmd2png/ik.py` and `vmd.py` | Reach clamping, hinge-constrained two-bone geometry, and an iterative off-plane leg solver. Reuse the geometry behind a skeleton adapter and validate tighter export tolerances. |
| `balance_analysis.py` | Heel/toe contact proxies, airborne detection, floor estimates, and `pin_feet`. Useful algorithms and fixtures, but currently coupled to training/inspection code. |
| `motion_fk.py` | Differentiable FK for training metrics. Its own documentation identifies it as an approximation that omits separate Center/Groove/Waist translations. Do not use it as the playback correctness oracle. |
| `motion_refine.py` | A learned restoration model for degraded/VQ motion. Evaluate later on video-derived motion; its presence does not establish that existing checkpoints fix GVHMR errors. |
| `hand_compress.py` / `hand_ae.py` | Existing hand representation/compression work. These operate on motion; they do not extract fingers from video. |
| `utils/bvh.py` | Parsing and rigging helpers, not an existing BVH exporter. A BVH writer remains separate work. |

Verified during this review: 4 pose-conversion tests and 6 leg-IK tests passed.
A direct writer/parser probe also preserved a meter-space root translation using
the codec's `unit=0.085` setting. These are component checks, not validation of a
video-to-VMD implementation or of playback on an arbitrary PMX model.

## Compatibility issues to settle first

1. **IK state is missing from the codec.** The writer currently writes zero IK
   property frames; the parser stops after camera frames. Playback decides whether
   to solve a leg from the count of IK-bone keyframes, not VMD IK enable flags.
   This is insufficient for reliable FK export: an MMD model's active IK can
   override the baked leg pose. Extend read/write and playback state handling.
2. **The name map is incomplete.** Ten default export bones currently lack a
   Japanese mapping: `Torso1`, `Eyes`, `LeftEye`, `RightEye`, `LeftBreast`,
   `RightBreast`, `LeftArmTwist`, `LeftHandTwist`, `RightArmTwist`, and
   `RightHandTwist`. Some are optional/helper concepts on real models. A profile
   must map, compose away, or explicitly omit each one; do not export unmatched
   English names and assume the target will use them. Composing away an animated
   helper must preserve its effect on descendants.
3. **Helper topology is not a PMX specification.** For example,
   `LeftLegIKParent` is mapped to a toe-IK name in the codec. Validate control
   semantics against the actual target rather than inferring them from that alias.
4. **Foot correction is presently position-only.** `pin_feet` drops contacted feet
   vertically and returns corrected joint positions for an inspector ghost. It
   does not persist horizontal contact anchors, adjust the pelvis, or return VMD
   rotation channels. Those are required additions for exported foot planting.
5. **Encoding is not the retargeting contract.** Raw arrays are 404 values/frame:
   root offset 3 + scale 1 + 65 x 6 rotations + camera 10. Learning features are
   393 values. Keep these formats stable; use an adapter, not a changed bone list.
6. **Writer failures need to reach the caller.** `write_vmd` catches errors and
   returns a boolean. The integration must check it, verify the output, and report
   export failure instead of claiming success.

## Architecture and ownership

```text
video / still                   existing skeleton_motion.npz
     |                                      |
GVHMR + SMPL-X                              |
     +--------------> SourceMotion <--------+
                           |
                timing / floor / contact analysis
                           |
                target profile + MMD retarget
                           |
                target-space contact / IK cleanup
                           |
                      TargetMotion
                       /        \
             VMD exporter       standard-skeleton feature adapter
                  |                         |
             motion.vmd            motion_encoder.npy (optional)
```

The shared motion/profile boundary should also accommodate a later BVH exporter
and optional hand estimator. It must remain independent of either target format.

Proposed additions in this repository:

| Module | Responsibility |
| --- | --- |
| `motion.py` | Versioned source/target motion structures, explicit conventions and validation. Adapt existing `HumanMotion` rather than introduce two competing contracts. |
| `rigs/profile.py` | Skeleton hierarchy, rest transforms, semantic joints, names, units, constraints and control ownership. |
| `mmd/profile.py` | Adapter for the pinned standard skeleton; later PMX-derived profiles. |
| `mmd/retarget.py` | Body-frame construction, local bone rotations, root policy and continuity. |
| `postprocess/contacts.py` | Floor/contact estimation and persistent contact intervals. |
| `postprocess/foot_lock.py` | Target-space foot constraints, bounded pelvis correction and leg solving. |
| `exporters/vmd.py` | Validated, dense keyframe generation using the shared codec. |
| `adapters/motion_encoder.py` | Optional canonical 404-value array export, preserving existing ordering/scaling. |

Keep format-wide fixes in the separate `alloystorm/vmd2png` repository and commit
them separately. Install that package normally and pin a tested revision/release
in the MMD dependency extra. Do not add runtime imports from the local
`D:/studio/Projects/motion_encoder` path, copy its training stack, or depend on
that parent project's `.src` import workaround. Review lazy preview imports so
format-only use need not initialize a graphical viewer.

## Motion and coordinate contracts

- Preserve the existing 24-joint NPZ as an accepted source. Retain timestamps,
  source FPS/duration, root translation, and the available root orientation.
  Leave room for confidence/missing-data masks, local body rotations and hands.
- Use physical meters and seconds internally, with a declared axis basis and
  quaternion convention. Make `wxyz`/`xyzw` conversion explicit at boundaries.
- Record profile identity, bone order, rest transforms, coordinate conversion,
  scale, contact evidence and postprocessing settings in the run artifacts.
- The standard target places anatomical left in +X and toes toward -Z. Verify
  the source-to-target transform with asymmetric fixtures; derive the complete
  basis conversion instead of guessing a mirror or swapping left/right labels.
  Apply position and rotation conversions consistently (`p' = B p`,
  `R' = B R B^-1`, with any origin/scale operations declared separately).
- Treat `0.085` as the existing profile's meters-per-VMD-unit convention, not a
  universal physical scale for every model. The writer divides meter-space
  translations by this value; do not apply that conversion twice.
- Resample by timestamp to VMD's 30 fps timeline. Do not simply relabel a source
  sequence's FPS. Document end-frame semantics and test duration within one frame.

## Implementation sequence

### 1. Establish profiles, codec guarantees and fixtures

- Pin the reviewed `vmd2png` baseline and capture its standard bone order/rest pose.
- Define a validated standard MMD output profile and its serialized names. Retain
  the 65-bone internal/learning skeleton, even when the file profile omits helpers.
- Extend the shared codec with show/IK property read/write, explicit state-based
  playback, and a documented compatibility policy for old inputs without flags.
  Preserve meaningful stationary IK targets; keyframe count is not an on/off flag.
- Validate names against the chosen encoding and VMD byte limits. Reject empty,
  colliding or silently truncated names. Use known interpolation bytes; initially
  write a key every frame, without keyframe reduction.
- Create asymmetric rest-pose, raised-arm, bent-knee, turning-root and translating-
  root fixtures. Check exports with an independent importer/reference MMD player;
  an exporter and its own parser can share the same bug.

**Exit:** the target contract is fixed and a hand-authored FK fixture plays with
leg/toe IK explicitly disabled, correct names, scale, facing and timing.

### 2. Implement direct body-to-MMD retargeting

- Solve against the target A-pose directly. Use pelvis/chest anatomical frames,
  limb directions, bend planes, previous-frame continuity and reachable lengths.
- Define root ownership once: the initial standard profile puts world trajectory
  and heading on `Master`; keeps Center/Groove/Waist translations neutral; and
  assigns pelvis articulation and spine motion through an explicit profile map.
  Verify that heading/translation are not applied twice across the root hierarchy.
- Map source pelvis/spine to the lower-body and upper-body branches; distribute
  spine motion deliberately and avoid copying the same full rotation to every
  torso control. Fit shoulders, arms, legs and ankles in world space, then express
  the result as parent-relative local rotations.
- Treat limb-axis twist as underconstrained by 24 joint positions. Use stable bend
  planes and conservative twist handling; allow future source local rotations to
  improve this without changing the exporter. Preserve helper-chain composition.
- Keep fingers, eyes, facial expressions and unsupported secondary controls at a
  documented neutral pose. Mark unobserved channels; do not label them captured.
- Update CLI/preflight/pipeline dispatch so VMD requires no Mixamo FBX. A
  skeleton-only VMD run must require neither SMPL-X nor the original video;
  source-video comparison previews may require the original clip separately.
- Write a body VMD and a skeleton comparison preview before enabling cleanup.

**Exit:** an existing extracted dance can export directly to a validated MMD
profile, with repeatable body poses and no dependence on the Mixamo path.

### 3. Add measurable foot-contact correction

- Extract portable numerical helpers from `balance_analysis.py`; keep training,
  dashboard and Matplotlib dependencies outside the runtime solver.
- Estimate a floor from reliable evidence. Reuse heel/toe proxies, dwell filtering
  and airborne detection, but evaluate their assumptions on video-derived motion.
  The existing lowest-foot rule is an inspection heuristic, not ground truth for
  kicks, crouches, pivots or jumps. Permit explicit floor/contact overrides.
- Classify planted, rolling/pivoting, swinging and airborne intervals. For a planted
  interval establish a stable world-space contact anchor; avoid freezing intentional
  pivots or the whole foot during heel/toe rolls. Use smooth contact transitions.
- Solve on target proportions after retargeting. Combine bounded pelvis/root
  correction with hinge-constrained leg IK, ankle orientation and floor constraints.
  Handle double support jointly, preserve segment lengths and report unreachable
  contacts instead of stretching limbs or forcing every sample onto the floor.
- Convert corrections back to local rotation/translation channels. For the initial
  FK mode, bake the solved legs and keep player IK disabled. Verify the written and
  reloaded animation, not just an in-memory corrected point cloud.
- Save contacts and before/after diagnostics: horizontal slip during planted
  intervals, floor penetration, reach residuals, root displacement and pose jerk.

**Exit:** deterministic planted-foot fixtures stay planted, and real examples show
reduced contact slip/penetration without erasing jumps, steps or turns.

### 4. Validate model-specific playback and optional IK export

- Add profile overrides and then PMX skeleton/control ingestion. No PMX reader was
  found in the reviewed shared core; this is additional work. Mesh loading is only
  necessary for a character preview, not for writing motion.
- Resolve supported names, rest proportions, local/fixed axes, inherited rotations,
  twist helpers and foot/toe control topology from the target. Unsupported models
  should produce an actionable compatibility report.
- Provide separate FK-baked and IK-control modes. In IK mode export targets in the
  correct parent space, synchronize foot orientation and property flags, and verify
  that MMD and local playback solve the same pose. Avoid conflicting simultaneous
  IK targets and independently authored FK leg corrections.
- Initially keep IK state stable over the clip; add switches only with continuity
  fixtures. Validate a second model/rest pose before claiming broader portability.

**Exit:** standard-profile playback is proven on a reference model; a distinct
model profile has its own calibration/compatibility tests. Arbitrary PMX support
is not implied by a successful standard-skeleton preview.

### 5. Connect to motion_encoder and extend formats

- Add optional NPY export in the established 404-value layout. Rotations follow
  `root.export_bones()` and `quat_to_6d`; root normalization is meters / 32.768.
  Keep scale and camera columns separate from the 393-value learning features.
- For this adapter, start from motion on the canonical 65-bone skeleton. A
  model-specific pose must first be retargeted/baked into that skeleton. Explicitly
  flatten any Center/Groove/Waist translation into the existing world-offset policy;
  do not silently drop it or rotate it twice.
- Assert exported features agree with `extract_motion_features` and preserve
  timestamps, bone ordering, root travel and pose on a roundtrip.
- Add BVH as a separate skeleton/channel exporter with declared Euler order,
  frame time and unwrapping. Reuse the target-motion layer, not VMD serialization.
- Add optional hand observation/retargeting behind the same motion contract. The
  canonical target already has 30 finger joints; video hand estimation, confidence,
  wrist alignment and temporal recovery still need implementation.
- Treat downloads, transcoding and trimming as a separate ingestion layer with
  source/timestamp provenance. They should not become responsibilities of the
  retargeter. Learned motion refinement is a later measured option, not a required
  dependency for the first working VMD export.

## Proposed user flow and outputs

These interfaces are now available (see the usage guide for additional flags):

```powershell
m2mr run --video assets/video/dance.mp4 --format vmd --mmd-profile standard --device cuda
m2mr run --skeleton outputs/<run>/skeleton_motion.npz --format vmd --mmd-profile standard --foot-lock --no-preview
# PMX model-profile support:
m2mr run --skeleton outputs/<run>/skeleton_motion.npz --format vmd --model character.pmx
```

Suggested artifacts: `motion.vmd`, a versioned target-motion NPZ, `run.json`,
`quality.json`, optional contact intervals and comparison previews, and optional
`motion_encoder.npy`. Record the exact target profile and shared-codec revision.
Keep the current GLB defaults compatible until a deliberate CLI migration.

## Validation and acceptance

Proposed deterministic-fixture targets, to be confirmed during implementation:

- Identity/rest FK error below `1e-6 m`; finite unit quaternions and constant bone
  lengths. Correct left/right anatomy under asymmetric poses and 90/180-degree turns.
- VMD translation/rotation roundtrip within float32 precision; duration within one
  30 fps frame. Include 29.97/60 fps sources, sparse clips and single-frame poses.
- Reachable planted-foot fixtures: at most `5 mm` horizontal drift and `5 mm`
  floor penetration after write/read/playback. Report clamped unreachable cases.
- Explicit IK flags respected by an independent player; FK-only motion unchanged
  when the local preview enables its IK subsystem. Test static IK targets as well.
- A-pose/T-pose and helper-chain fixtures catch double correction, doubled root
  rotation and lost twist inheritance. Unknown serialized bone names fail validation.
- Real clips cover standing, walking, sideways travel, crouching, a jump, turning,
  toe/heel support and partial occlusion. Compare raw versus corrected output and
  assess contact transitions visually; synthetic tolerances are not claims about
  monocular extraction accuracy.
- Retain a Mixamo/GLB regression and verify skeleton-only VMD export without a
  Mixamo asset or SMPL-X installed. Keep cached extraction reusable.
- Use the shared skeleton FK and independent model playback for final correctness;
  use the training FK for compatible learning-space metrics only.

## Recommended first implementation slice

Implement steps 1 and 2 as the first vertical slice: hardened shared codec, a
validated standard profile, direct FK retargeting, and an existing-NPZ-to-VMD
command. Then implement step 3 against that baseline. This creates a measurable
reference before introducing contact correction, model-specific IK, additional
formats, or learned refiners.

Before model-specific validation, select a redistributable or locally supplied
reference PMX and its rest pose. The standard skeleton is available now, so its
contract, mathematical fixtures and codec work can proceed without that asset.
