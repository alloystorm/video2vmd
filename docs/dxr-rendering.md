# Verify motion rendering with dxr_native

The dxr_native batch CLI supplies the rendering half of the
[motion-refinement experiment](motion-refinement-plan.md). It renders VMD motion
on a character and writes the evaluated skeleton and camera alongside the video.
The application and its CLI live in the separate `D:/studio/Unity/dxr_native`
checkout; this project consumes their output.

## Verified path (2026-10-18)

An end-to-end smoke test used dxr_native 1.2.971, the complete AB7 PMX asset and
the Hibikase dance, recording motion seconds 12–18 with a fixed camera. Outputs
are under `outputs/diagnostics/dxr_render_e2e_20261018/` (gitignored). The source
VMD readme credits original motion Miku VR, converted/fixed by GentleWhiteCat;
the reference assets and their readmes remain local.

| Check | Result |
| --- | --- |
| Batch render | Two jobs succeeded, exit 0; 1280×720 and 640×360 H.264 |
| Video timing | 180 decoded frames each at 30 fps; six seconds |
| Ground truth | 180 rows, 62 mapped target joints per frame, finite positions and unit xyzw world quaternions |
| Camera consistency | Independent reprojection differed by at most 0.000106 pixels at 720p |
| Framing | All 11,160 joint samples were inside the frame and in front of the camera |
| Resolution repeat | Joint positions bit-identical; quaternion components differed by at most 1.17e-9 |
| Resume | Both jobs skipped, exit 0; all six video/sidecar files retained their hashes and modification times |
| Settings isolation | User `bin/Settings/app_globals.json` hash unchanged |
| Normal extraction | GVHMR CUDA run completed; skeleton shape `(180, 24, 3)` at 30 fps |
| Calibrated extraction | Renderer intrinsics supplied through a companion JSON; saved K matched within 4.6e-8 pixels |
| Exports | VMD, cached target/skeleton NPZs, `(180, 404)` motion-encoder array; BVH from cached skeleton |
| Independent playback | Fresh Blender/MMD Tools VMD and Blender BVH imports passed numerical FK checks |
| Inspection | Both normal and calibrated exports loaded as 180-frame raw/corrected inspection data |

The numerical projection check covers every saved frame/joint. Body overlays
were visually checked at frames 0, 60, 120 and 179 against decoded video. This
is a file, geometry and timing smoke test; it does not establish reconstruction
accuracy, training quality or generalization to other motions and characters.
Moving cameras, other frame rates, audio, PNG output and multi-minute rendering
were not exercised by this test.

The local PMX used for earlier skeleton-only analysis lacked its textures. The
first render therefore appeared white. The successful textured test extracted
the complete original asset into ignored outputs; its PMX hash matched the
analysis copy. Use complete character assets when generating training videos.

## Render a batch

Create a job file with absolute paths to the complete character and motion.
Recording outputs are relative to the job file, not the current directory.
For example:

```json
{
  "format": "dxr_native-batch/1",
  "jobs": [{
    "name": "dance_front_v001",
    "output": "front.mp4",
    "steps": [
      {"cmd": "load", "path": "D:/assets/character/character.pmx", "skipSavedSettings": true},
      {"cmd": "load", "path": "D:/assets/dance.vmd"},
      {"cmd": "place", "index": 0, "position": [0, 0, 0], "yaw": 0},
      {
        "cmd": "record", "output": "front.mp4",
        "width": 1280, "height": 720, "fps": 30,
        "start": 12, "frames": 180, "warmup": 30,
        "audio": false, "metadata": true,
        "camera": {"keys": [{
          "t": 0, "eye": [0, 1, -4.2], "target": [0, 0.9, 0], "fov": 40
        }]}
      }
    ]
  }]
}
```

Run from this project using the external extraction environment:

```powershell
$env:DXR_NATIVE_FFMPEG = 'D:\studio\Tools\ffmpeg\bin\ffmpeg.exe'
.venv/Scripts/python.exe D:/studio/Unity/dxr_native/tools/run_batch.py outputs/my_render/jobs.json
```

The wrapper provides an isolated settings directory and the batch exits when
finished. The renderer requires an un-minimized window, though the batch can run
without focus. Run rendering and GPU reconstruction sequentially.

The output consists of `front.mp4`, `front.meta.json`, `front.frames.jsonl`, an
encoder log and an appended batch manifest. Per-frame positions come from the
final target skeleton after retargeting, deform and IK. Preserve both sidecars.

## Verify the recording

```powershell
.venv/Scripts/python.exe scripts/verify_dxr_recording.py outputs/my_render/front.mp4
# Optional: compare the same scene, motion and camera rendered at another size.
.venv/Scripts/python.exe scripts/verify_dxr_recording.py outputs/my_render/front.mp4 --compare outputs/my_render/front_half.mp4
```

The verifier requires NumPy, OpenCV, Pillow and ffprobe. Use `--ffprobe PATH` if
ffprobe is not on PATH. It writes `front.verification.json` and an overlaid
contact sheet, `front.verification.png`. Missing or inconsistent sidecars,
incorrect timestamps, invalid joint arrays and projection mismatches fail the
command. The in-frame fraction is reported rather than universally required;
partial crops may be deliberate, and rig controls may extend past the body.

World coordinates are dxr_native's left-handed, Y-up meters. Quaternions are
world xyzw, pixels have a top-left origin, and camera forward has positive view
depth. `cross(up, dir)` supplies screen-right. The separate `camera` control
command has a documented FOV discrepancy; the `record.camera` track uses the
true angle, and saved `fx/fy/cx/cy` describe the actual projection.

## Reconstruct and export

```powershell
.venv/Scripts/python.exe -m video2vmd.cli run --video outputs/my_render/front.mp4 --device cuda --model assets/mmd/ab7_160404/ab7.pmx --motion-encoder --no-preview --output-dir outputs/my_render/reconstructed
.venv/Scripts/python.exe -m video2vmd.cli run --skeleton outputs/my_render/reconstructed/skeleton_motion.npz --format bvh --no-preview --output-dir outputs/my_render/bvh
```

Normal extraction estimates camera intrinsics. To test known calibration, write
`front.intrinsics.json` beside the input video with the renderer's `width`,
`height`, `fx`, `fy`, `cx`, `cy`. For a fixed lens, these are scalars from the
frame camera. Use a separate fresh output/cache directory for the calibrated
experiment. The existing extractor discovers this companion automatically;
it does not read dxr_native's `.meta.json` directly. In the test, estimated
focal length was 1468.60 pixels versus the renderer's measured 989.09 pixels.
Changing the calibration is a distinct reconstruction condition.

## Before using the pairs for training

- Target-rig joints and the extractor's 24 SMPL joints have different centers,
  topology and proportions. Build and verify a mapping/geometry adapter; renaming
  62 joints into 24 is not sufficient. Actor `root` is scene placement, not
  automatically the anatomical pelvis or the extractor's root translation.
- Preserve coordinate conversion, camera transforms, reference motion time and
  video presentation time explicitly. The source dance begins at 12 seconds in
  this test, while extraction timestamps begin at zero.
- Keep frame rate and warm-up settings fixed for paired comparisons. The
  renderer documents stateful grounding/physics differences across those settings.
- Resume checks successful job names and primary output existence. Source review
  shows it does not hash changed inputs or require valid sidecars. Use versioned
  job names/fresh manifests or `--no-resume` after changes, and run the verifier
  before accepting a cached pair. The saved provenance records input hashes for
  this smoke test; a dataset generator should do this for every pair.
- Current verification establishes the render/reconstruct/export path. Training,
  the 24-joint reference adapter and automatic dataset generation remain separate
  milestones in the refinement plan.
