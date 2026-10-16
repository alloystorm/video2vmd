# Video2VMD Blender extension

A thin Blender 4.2+ UI for the external `v2vmd` pipeline. Blender's Python does
not load PyTorch, GVHMR or OpenCV.

## Install

1. Install this project in your external Python environment: `pip install -e .`.
   Video/still extraction needs the licensed SMPL-X asset; cached skeletons do not.
2. Build the add-on: `python scripts/package_blender_extension.py`.
3. Disable the previous Motion2MixamoRig add-on, if installed.
4. In Blender: Edit > Preferences > Get Extensions > repository dropdown arrow >
   Install from Disk. Choose `dist/video2vmd-0.2.0.zip` and enable Video2VMD.
5. Open 3D Viewport > N sidebar > Video2VMD.

Configure External Python (`.venv/Scripts/python.exe` on Windows or
`.venv/bin/python` on Unix, not Blender's interpreter), Project Directory and
Device. Choose video, image or cached skeleton, then Check Environment.

## Export and import

Select a source file, optional PMX target, output format and optional Foot Contact
Cleanup. VMD is the default; PMX Foot IK needs a supported PMX target. Generate
Motion launches `<external-python> -m video2vmd.cli run ...` and streams progress.
Jobs write `outputs/blender_jobs/<job-id>/` including a log and run.json.

VMD runs write raw and corrected files; `motion.vmd` is the corrected copy.
VMD export succeeds without any Blender MMD plugin. To import it, enable
[MMD Tools](https://github.com/MMD-Blender/blender_mmd_tools), load your PMX at
scale 0.085, select its armature and press Import Motion. This applies the
corrected motion to that armature. Use Open Output to access the raw file.

BVH jobs automatically import the generated skeleton into a `V2VMD_<source>`
collection. BVH uses meters, Y up, Z-X-Y rotations and 30 fps. The first sample is
at Blender frame 1. All imports set the scene FPS and clip length from run.json.

The extension cancels only its owned subprocess. Logs remain available on errors.
Rebuild and reinstall after code changes; an installed extension is not live from
this checkout. If stale registration errors occur, disable it, quit Blender fully,
reopen and install again.

See the [smoke checklist](../docs/blender-extension.md) and
[format/quality guide](../docs/vmd-retargeting.md).
