"""Independent validation in a fresh background Blender process, not the user's scene.

blender --background --factory-startup --python scripts/verify_mmd_blender.py --
  --mmd-tools PATH --model character.pmx --run-dir OUTPUT
"""
import argparse
import json
from pathlib import Path
import sys
import zipfile

import bpy
import numpy as np
from mathutils import Quaternion


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mmd-tools", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args(sys.argv[sys.argv.index("--")+1:])
    sys.path.insert(0, str(args.mmd_tools.resolve()))
    vendor = (args.run_dir/"validation_dependencies").resolve()
    vendor.mkdir(parents=True, exist_ok=True)
    for wheel in (args.mmd_tools/"mmd_tools"/"wheels").glob("*.whl"):
        with zipfile.ZipFile(wheel) as archive:
            for member in archive.namelist():
                if not (vendor/member).resolve().is_relative_to(vendor):
                    raise ValueError("Wheel contains an escaping path")
            archive.extractall(vendor)
    sys.path.insert(0, str(vendor))
    import mmd_tools
    mmd_tools.register()
    result = bpy.ops.mmd_tools.import_model(filepath=str(args.model.resolve()), types={"ARMATURE"},
                                             scale=.085, rename_bones=False, fix_ik_links=False,
                                             apply_bone_fixed_axis=False, save_log=False)
    assert result == {"FINISHED"}, result
    arm = next(obj for obj in bpy.context.scene.objects if obj.type == "ARMATURE")
    bpy.ops.object.select_all(action="DESELECT")
    arm.select_set(True)
    bpy.context.view_layer.objects.active = arm
    bpy.context.scene.frame_set(0)
    result = bpy.ops.mmd_tools.import_vmd(filepath=str((args.run_dir/"motion.vmd").resolve()),
                                        scale=.085, margin=0, bone_mapper="PMX", use_pose_mode=False)
    assert result == {"FINISHED"}, result
    target = np.load(args.run_dir/"target_motion.npz")
    names, parents, rest = target["bone_names"].tolist(), target["parents"], target["rest_positions_m"]
    q, t = target["quat_xyzw"], target["translations_m"]
    from mmd_tools.core.vmd import File
    independent = File()
    independent.load(filepath=str(args.run_dir/"motion.vmd"))
    ik = independent.propertyAnimation[0].ik_states
    mode = json.loads((args.run_dir/"run.json").read_text())["ik_mode"]
    assert ik
    assert all(bool(enabled) == (mode == "ik" and name in ("左足ＩＫ", "右足ＩＫ")) for name, enabled in ik), ik
    # The PMX importer swaps Y/Z to map MMD coordinates to Blender coordinates.
    mapping = {bone.mmd_bone.name_j: bone.name for bone in arm.pose.bones}
    english_to_jp = {"Center":"センター", "HipMaster":"下半身", "Torso":"上半身", "Torso2":"上半身2",
                     "Neck":"首", "Head":"頭", "LeftArm":"左腕", "RightArm":"右腕", "LeftElbow":"左ひじ", "RightElbow":"右ひじ",
                     "LeftWrist":"左手首", "RightWrist":"右手首", "LeftLeg":"左足", "RightLeg":"右足",
                     "LeftKnee":"左ひざ", "RightKnee":"右ひざ", "LeftAnkle":"左足首", "RightAnkle":"右足首"}
    errors = []
    for frame in (0, len(q)//2, len(q)-1):
        rotation = np.array([np.array(Quaternion((v[3],v[0],v[1],v[2])).to_matrix()) for v in q[frame]])
        world, position = [], []
        for i, parent in enumerate(parents):
            offset = rest[i] if parent < 0 else rest[i]-rest[parent]
            if parent < 0:
                world.append(rotation[i]); position.append(offset+t[frame,i])
            else:
                world.append(world[parent]@rotation[i]); position.append(position[parent]+world[parent]@(offset+t[frame,i]))
        bpy.context.scene.frame_set(frame)
        bpy.context.view_layer.update()
        for name, jp in english_to_jp.items():
            if name not in names or jp not in mapping:
                continue
            actual = np.array(arm.matrix_world @ arm.pose.bones[mapping[jp]].head)
            expected = np.asarray(position[names.index(name)])[[0,2,1]]
            errors.append({"frame":frame,"bone":name,"error_m":float(np.linalg.norm(actual-expected)),
                           "actual":actual.tolist(),"expected":expected.tolist()})
    report = {"blender":bpy.app.version_string, "imported_bones":len(arm.pose.bones),
              "vmd_ik_states":ik,"max_error_m":max(e["error_m"] for e in errors),"samples":errors}
    (args.run_dir/"blender_validation.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({k:v for k,v in report.items() if k!="samples"}, indent=2))
    assert report["max_error_m"] < .005, "Independent Blender/MMD playback disagrees with target FK"


if __name__ == "__main__":
    main()
