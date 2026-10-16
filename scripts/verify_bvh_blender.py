"""Validate exported BVH with Blender's independent importer in a fresh process.

blender --background --factory-startup --python scripts/verify_bvh_blender.py --
  --run-dir OUTPUT
"""
import argparse
import json
from pathlib import Path
import sys

import bpy
import numpy as np
from bpy_extras.io_utils import axis_conversion
from mathutils import Quaternion


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args(sys.argv[sys.argv.index("--")+1:])
    import io_anim_bvh
    if not io_anim_bvh.ImportBVH.is_registered:
        io_anim_bvh.register()
    result = bpy.ops.import_anim.bvh(filepath=str((args.run_dir/"motion.bvh").resolve()),
                                     frame_start=1, global_scale=1., axis_forward="-Z", axis_up="Y",
                                     update_scene_fps=True, rotate_mode="QUATERNION")
    assert result == {"FINISHED"}, result
    arm = next(obj for obj in bpy.context.scene.objects if obj.type == "ARMATURE")
    target = np.load(args.run_dir/"target_motion.npz")
    names, parents, rest = target["bone_names"].tolist(), target["parents"], target["rest_positions_m"]
    q, t = target["quat_xyzw"], target["translations_m"]
    basis = np.array(axis_conversion(from_forward="-Z", from_up="Y").to_3x3())
    errors = []
    for frame in (0, len(q)//2, len(q)-1):
        local = np.array([np.array(Quaternion((v[3],v[0],v[1],v[2])).to_matrix()) for v in q[frame]])
        world, position = [], []
        for i, parent in enumerate(parents):
            offset = rest[i] if parent < 0 else rest[i]-rest[parent]
            if parent < 0:
                world.append(local[i]); position.append(offset+t[frame,i])
            else:
                world.append(world[parent]@local[i]); position.append(position[parent]+world[parent]@(offset+t[frame,i]))
        bpy.context.scene.frame_set(frame+1)
        bpy.context.view_layer.update()
        for i, name in enumerate(names):
            name = name if name.isascii() and name.replace("_", "").isalnum() else f"bone_{i}"
            if name not in arm.pose.bones:
                continue
            actual = np.array(arm.matrix_world @ arm.pose.bones[name].head)
            errors.append({"frame": frame, "bone": name, "error_m": float(np.linalg.norm(actual-basis@position[i])),
                           "actual": actual.tolist(), "expected": (basis@position[i]).tolist()})
    report = {"blender": bpy.app.version_string, "imported_bones": len(arm.pose.bones),
              "max_error_m": max(e["error_m"] for e in errors), "fps": bpy.context.scene.render.fps/bpy.context.scene.render.fps_base,
              "samples": errors}
    (args.run_dir/"blender_validation.json").write_text(json.dumps(report, indent=2)+"\n", encoding="utf-8")
    print(json.dumps({k:v for k,v in report.items() if k != "samples"}, indent=2))
    assert report["max_error_m"] < 1e-5, "Independent Blender BVH playback disagrees with target FK"
    assert abs(report["fps"]-float(target["fps"])) < .001


if __name__ == "__main__":
    main()
