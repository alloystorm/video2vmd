# VMD and BVH body-motion export

`v2vmd` can retarget an extracted human skeleton directly to MMD bones and write
VMD or BVH. It uses a canonical MMD A-pose or a supplied PMX skeleton's proportions.
VMD is the default output; BVH is the alternative. The Blender extension launches the same external pipeline.

## Install and run

Activate the project's Python environment, then install the format dependencies:

```powershell
python -m pip install -e .
```

This pins `alloystorm/vmd2png` at
`3c8837e005456b7081bd1a1366aaa755eb0d0c5d` (0.2.1), which adds IK property keys,
strict CP932 names and PMX skeleton metadata. The training project is not a runtime
dependency. Git must be available for the pinned dependency installation.

```powershell
# Extract once, then write a standard-profile VMD.
v2vmd run --video assets/video/dance.mp4 --format vmd --device cuda --foot-lock

# Reuse extraction with your PMX character. Extract ZIP archives first.
v2vmd run --skeleton outputs/previous/skeleton_motion.npz --format vmd --model assets/mmd/character.pmx --foot-lock

# Optional PMX foot-IK controls; baked FK is the default.
v2vmd run --skeleton outputs/previous/skeleton_motion.npz --format vmd --model assets/mmd/character.pmx --foot-lock --ik-mode ik

# BVH on the standard MMD hierarchy, without rendering a preview.
v2vmd run --skeleton outputs/previous/skeleton_motion.npz --format bvh --foot-lock --no-preview

# Also produce the motion_encoder canonical 404-value representation.
v2vmd run --skeleton outputs/previous/skeleton_motion.npz --format vmd --motion-encoder
```

Video/still extraction still needs the licensed
SMPL-X model and existing inference weights. A cached-skeleton export needs neither
SMPL-X nor the source video. `--image` also works through the existing still-hold
extraction. `--output-dir` and `--progress-jsonl` work for automation.

`--mmd-profile standard` is the default; `--model` selects PMX-derived geometry.
`--floor 0` overrides contact analysis with a target-space floor in meters. It
does not force every frame onto the floor. Foot locking is opt-in.

## Files and conventions

Each run writes:

| File | Contents |
| --- | --- |
| `motion.vmd` or `motion.bvh` | Dense 30 fps body animation |
| `motion_raw.vmd` / `motion_corrected.vmd` | VMD before and after optional depth/contact cleanup; both always written for VMD runs |
| `skeleton_motion.npz` | Reusable source human joints |
| `target_motion.npz` | Version 1 target hierarchy, rest positions, local `xyzw` quaternions, animated local translations, timestamps, profile hash and fitted-channel mask |
| `target_motion_raw.npz` | Matching target motion before depth/contact cleanup |
| `contacts.npz` | Heel/toe contact mask, persistent anchors, transition weights, floor and thresholds; per-foot support codes 0=swing/unknown, 1=heel, 2=toe, 3=both |
| `quality.json` | Before/after contact speed, anchor error, penetration, root travel, body jerk and reach residuals |
| `run.json` | Inputs, format, profile identity, codec version/pin, calibration, timing and output listing |
| `videos/target_skeleton.mp4` | Raw/corrected front and side skeleton views; silent, omitted by `--no-preview` |
| `motion_encoder.npy` | Optional canonical 404-column array |

`motion.vmd` remains an exact copy of `motion_corrected.vmd` for existing consumers.
With both `--depth-cleanup` and `--foot-lock` disabled, raw and corrected VMD files are identical. Both use the
same retargeting, neck mapping and selected FK/IK mode; "raw" means before optional
cleanup, not the original human skeleton or an older retargeter. In IK mode,
both variants include the necessary PMX knee-hinge compatibility conversion.

Internally positions/translations are meters, time is seconds, and rotations are
parent-relative `xyzw`. Source positions are mapped with `diag(-1,1,1)` into the
MMD basis (anatomical left +X, up +Y, anterior -Z in the reference pose). Geometry
is solved in that basis; quaternion components are not mirrored individually.
Global source heading and travel remain animated.

Source timestamps drive 30 fps resampling. A clip's duration includes the last
sample's interval; output frame count is its duration times 30, rounded to the
nearest whole frame. A 60 fps input is resampled, rather than relabeled 30 fps.

Root trajectory/heading live on `Master` for the standard profile, or `Center`
when the PMX has no Master. Other root translation controls remain neutral.
Pelvis articulation and chest rotation have separate ownership. The unspecifiable
synthetic `Torso1` helper remains neutral and is omitted from VMD. No fingers,
facial expressions, eye movement, hair or clothing animation are estimated.
The neck fits a virtual upper-chest-to-head axis against the target rest pose.
SMPL's short neck-to-head offset points toward the skull center; aiming the MMD
neck along it would add an anatomical bias as a large forward bend. The head
stays neutral locally and follows the fitted neck, without a second opposite
rotation to force world-upright heading. This remains a coarse position-based
estimate, not independent head/gaze tracking. Wrists use only the available
coarse hand direction. Joint-position-only input cannot resolve full limb twist
or detailed head/hand orientation.

VMD translations use the existing shared codec's 0.085 meters-per-unit convention.
PMX geometry uses that same scale; a PMX authored with unusual units needs a new
profile convention. Names are checked against CP932 and VMD byte limits. Export
uses a temporary file and verifies parser readback before replacing the result.

BVH uses meter offsets, Y up, intrinsic Z-X-Y rotation channels, root XYZ
translation and explicit frame time. Euler channels are unwrapped over time.
The BVH root OFFSET is zero and its position channels contain absolute world
position, including the PMX root's rest height.
BVH has no universal physical unit declaration: import at scale 1 for meters.
Blender's default BVH orientation is forward -Z/up Y; it places the first motion
sample at frame 1. Model-specific auxiliary Japanese names become `bone_<index>`;
the exact name map remains in `target_motion.npz`.
`motion.bvh.json` records hierarchy order and neutral auxiliary bones omitted
outside the body root tree, such as disconnected PMX IK controllers. Animated
controls outside that tree cause an error instead of being silently lost.

## Foot cleanup and PMX compatibility

`--depth-cleanup` optionally regularizes the source pelvis's combined metric
camera depth before producing the corrected target. Raw extraction/retargeting
stay intact. It requires static-camera observations in the source NPZ; the CLI
gives upgrade instructions for older caches. It does not filter bounding-box
size or weak-perspective camera scale independently.

The offline sparse least-squares regularizer combines a data term, adaptive
acceleration penalty (3 Hz base), and supported-foot horizontal velocity evidence.
Coherent travel reduces smoothing. A shared translation along the pelvis viewing
ray preserves its image position and all joint-relative vectors; world changes
use the saved camera rotation. Edits taper to zero at clip endpoints, are capped
at 4 cm in depth, and stay within 0.1% image-height RMS / 0.2% maximum per-joint
reprojection displacement. These are displacement budgets, not ground-truth error.

`skeleton_motion_corrected.npz` retains camera observations with the corrected
joint translations; `target_motion_depth.npz` captures the stage before foot lock.
`depth_cleanup.json` and `.npz` record actual edits, reprojection and motion/contact
weights. Refiltering an already cleaned skeleton is rejected; use the original
cache when comparing settings. `--depth-cleanup --foot-lock` applies the existing
leg correction after depth cleanup, using the original contact evidence. Quality
reports keep that mask fixed across raw/depth/final stages. Real fast movement
can still be attenuated, so the option defaults off and needs inspection.

Contact detection requires a heel/toe proxy within 3.5 cm of the floor, moving
slower than 0.35 m/s for at least 0.1 seconds. Each interval gets a persistent
horizontal anchor with a short transition ramp. Heel/toe tests are separate so
rolling support need not lock both points. The leg solver preserves segment
lengths and raw ankle orientation, lowers the pelvis by at most 8 cm when needed,
and reports unreachable contacts. Fast steps, uncertain contacts and airborne
samples remain uncorrected. Contact labels are heuristics, not observed ground
truth; monocular extraction errors, missing floor evidence and residual sliding
remain possible. Root and jerk diagnostics make adverse cleanup changes visible.

Default VMD **FK mode** disables foot/toe IK explicitly and bakes solved legs.
Optional **IK mode** requires a PMX with direct thigh/knee/ankle chains, matching
foot-IK targets, and supported local-X knee hinge limits. The baked legs provide
a solver-consistent seed; foot controls follow the solved ankles in their actual
parent space. Foot IK stays enabled and toe IK stays disabled throughout the clip.
This avoids a second conflicting leg solution but does not guarantee identical
results in every MMD player's iterative solver.

PMX metadata ingestion covers parents, rest positions, fixed/local axes,
inheritance and IK limits. Inherited transforms on body controls/ancestors are
currently rejected, as are fixed-axis body controls. Neutral twist helpers can
pass through. Unusual control chains and physics-driven animation need further
profile work. Only the supplied AB7 PMX has been independently tested; arbitrary
PMX portability and a second model remain validation work.

The canonical NPY adapter always retargets to the shared **65-bone standard**
skeleton, even when the VMD uses PMX proportions. Layout is root world offset
divided by 32.768 (3), scale (1), 65 local 6D rotations (390), camera defaults (10).
Each 6D rotation concatenates the first two rotation-matrix columns. The other
project's `extract_motion_features` can slice it into 393 motion features. This
does not run its learned refinement model or change that project's checkout.

## Validation and development

```powershell
python scripts/verify_mmd.py
python scripts/verify_video_input.py
```

Fifteen deterministic retarget/export/contact tests cover rest/asymmetric anatomy,
root heading/travel, resampling/fractional FPS, planted feet, airborne preservation, encoder
agreement, BVH timing, knee hinges, skull-center neck offsets, coherent upper-body
lean, raw/corrected pipeline output and asset-independent skeleton preflight.
The shared codec has 11 unittest checks; the motion_encoder pose/leg regressions
(4 + 6) also pass against the new codec.

Independent import checks run in a fresh background Blender process, so they do
not modify an open scene. `verify_mmd_blender.py` additionally needs a local
[MMD-Blender/blender_mmd_tools](https://github.com/MMD-Blender/blender_mmd_tools)
checkout. It loads only the model armature, imports VMD and compares sampled
joint positions against target FK. `verify_bvh_blender.py` uses Blender's bundled
BVH importer.

```powershell
blender --background --factory-startup --python-exit-code 1 --python scripts/verify_mmd_blender.py -- --mmd-tools path/to/blender_mmd_tools --model character.pmx --run-dir outputs/run
blender --background --factory-startup --python-exit-code 1 --python scripts/verify_mmd_blender.py -- --mmd-tools path/to/blender_mmd_tools --model character.pmx --run-dir outputs/run --variant raw
blender --background --factory-startup --python-exit-code 1 --python scripts/verify_bvh_blender.py -- --run-dir outputs/bvh_run
```

Local validation on the supplied AB7 model and an 817-frame video: sampled FK
playback error below 0.001 mm; optional IK mode below 3.5 mm. Standard-profile BVH
sampled playback error below 0.001 mm. These measure export/playback agreement,
not accuracy of the person reconstructed from video. On detected contacts in
that clip, mean horizontal speed fell from 0.270 to 0.082 m/s; only 84/817 frames
qualified. Maximum remaining floor penetration was 14 mm. The separate reachable
planted fixture meets the 5 mm drift/penetration targets. Broader walking, jumping,
occlusion and pivot footage still needs review before stronger quality claims.

Shared codec development is isolated on `codex/vmd-retargeting` in a separate git
worktree at `outputs/worktrees/vmd2png`; the motion_encoder submodule checkout was
left untouched. That branch is published in `alloystorm/vmd2png`, and installation
of the exact pinned GitHub commit has been verified in this workspace's `.venv`.
A built local codec wheel is also available under `dist/dependencies/`. The local
worktree is only needed for further codec development, not normal runtime use.
