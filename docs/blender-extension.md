# Blender extension smoke checklist

Build with `python scripts/package_blender_extension.py`. Validate the ZIP using
Blender's extension validator; the manifest ID is `video2vmd`, version 0.2.0.

- Enable/disable/re-enable in a fresh Blender 4.2+ process. Confirm the Video2VMD
  N-panel and persisted External Python / Project Directory preferences.
- Confirm Blender never imports torch, gvhmr or cv2. It starts the external
  interpreter with `-m video2vmd.cli` and list arguments, including paths with spaces.
- Check Environment reports missing extraction assets without requiring any FBX.
  Cached Skeleton mode checks only its NPZ and format dependencies.
- Generate VMD with the standard profile, then a supported PMX. Both raw/corrected
  VMDs exist and run.json lists them; completion does not require MMD Tools.
- With MMD Tools enabled, import the corrected VMD onto the selected PMX armature.
  Missing MMD Tools / active armature gives an actionable import error while
  retaining generated files. Matching PMX import scale is 0.085.
- Generate BVH and confirm automatic skeleton import, meter scale, frame 1 start,
  30 fps and clip length. No glTF/FBX importer or character asset is needed.
- Toggle foot cleanup and preview output. Existing source NPZs remain reusable.
- Cancel a running owned job; confirm the process stops and logs remain.
- Invalid External Python/source/PMX and unavailable CUDA produce useful messages.

Automated service/package checks: `python scripts/verify_blender_extension.py`.
Background Blender registration/import checks are exposed by that script's
`--blender-smoke` mode (see its module docstring). Use a fresh background process,
not the user's open scene. Rebuild/reinstall the ZIP after every code change.
