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
from mathutils import Matrix, Quaternion


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mmd-tools", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--variant", choices=("raw", "corrected"), default="corrected")
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
    vmd_path = args.run_dir/f"motion_{args.variant}.vmd"
    if args.variant == "corrected" and not vmd_path.exists():
        vmd_path = args.run_dir/"motion.vmd"  # Earlier runs used only this name.
    result = bpy.ops.mmd_tools.import_vmd(filepath=str(vmd_path.resolve()),
                                        scale=.085, margin=0, bone_mapper="PMX", use_pose_mode=False)
    assert result == {"FINISHED"}, result
    target = np.load(args.run_dir/("target_motion_raw.npz" if args.variant == "raw" else "target_motion.npz"))
    names, parents, rest = target["bone_names"].tolist(), target["parents"], target["rest_positions_m"]
    q, t = target["quat_xyzw"], target["translations_m"]
    from mmd_tools.core.vmd import File
    independent = File()
    independent.load(filepath=str(vmd_path))
    ik = independent.propertyAnimation[0].ik_states
    mode = json.loads((args.run_dir/"run.json").read_text())["ik_mode"]
    assert ik
    assert all(bool(enabled) == (mode == "ik" and name in ("左足ＩＫ", "右足ＩＫ")) for name, enabled in ik), ik
    # The PMX importer swaps Y/Z to map MMD coordinates to Blender coordinates.
    mapping = {bone.mmd_bone.name_j: bone.name for bone in arm.pose.bones}
    english_to_jp = {"Center":"センター", "HipMaster":"下半身", "Torso":"上半身", "Torso2":"上半身2",
                     "Neck":"首", "Head":"頭", "LeftArm":"左腕", "RightArm":"右腕", "LeftElbow":"左ひじ", "RightElbow":"右ひじ",
                     "LeftWrist":"左手首", "RightWrist":"右手首", "LeftLeg":"左足", "RightLeg":"右足",
                     "LeftKnee":"左ひざ", "RightKnee":"右ひざ", "LeftAnkle":"左足首", "RightAnkle":"右足首",
                     "LeftToe":"左つま先", "RightToe":"右つま先"}
    errors = []
    orientation_errors = []
    basis = np.eye(3)[[0,2,1]]
    frames = {0, len(q)//2, len(q)-1}
    raw_path = args.run_dir/"target_motion_raw.npz"
    if args.variant == "corrected" and raw_path.exists():
        with np.load(raw_path) as raw:
            # Include largest edited leg poses and adjacent frames, not just
            # sparse samples that could miss the entire grounding correction.
            for name in ("LeftAnkle", "RightAnkle", "LeftKnee", "RightKnee"):
                if name in names:
                    i = names.index(name)
                    change = 1-np.abs(np.sum(q[:, i]*raw["quat_xyzw"][:, i], axis=1))
                    peak = int(np.argmax(change))
                    frames.update(range(max(0, peak-1), min(len(q), peak+2)))
    for frame in sorted(frames):
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
            if name in ("Neck", "Head", "LeftAnkle", "RightAnkle"):
                # Remove Blender's bone-display rest axes to compare the
                # physical rotation of each control in MMD world coordinates.
                rest_r = np.array(arm.data.bones[mapping[jp]].matrix_local.to_3x3())
                actual_r = np.array(arm.pose.bones[mapping[jp]].matrix.to_3x3()) @ rest_r.T
                expected_r = basis @ world[names.index(name)] @ basis.T
                delta = Matrix(expected_r.T@actual_r).to_quaternion().normalized()
                angle = float(np.degrees(2*np.arccos(np.clip(abs(delta.w),0.,1.))))
                orientation_errors.append({"frame":frame,"bone":name,"error_deg":angle})
    report = {"blender":bpy.app.version_string, "imported_bones":len(arm.pose.bones),
              "vmd_ik_states":ik,"max_error_m":max(e["error_m"] for e in errors),"samples":errors,
              "max_body_error_m":max(e["error_m"] for e in errors if not e["bone"].endswith("Toe")),
              "max_toe_error_m":max(e["error_m"] for e in errors if e["bone"].endswith("Toe")),
              "max_neck_head_orientation_error_deg":max(e["error_deg"] for e in orientation_errors if e["bone"] in ("Neck", "Head")),
              "max_ankle_orientation_error_deg":max(e["error_deg"] for e in orientation_errors if e["bone"].endswith("Ankle")),
              "sample_frames":sorted(frames),
              "orientation_samples":orientation_errors}
    report["variant"] = args.variant
    # Preserve the original 5 mm body-position contract. Newly measured toe
    # endpoints/orientations expose iterative IK drift even in the raw seed;
    # keep explicit model-specific budgets separate from baked FK agreement.
    report["playback_limits"] = {"body_m":.005, "toe_m":.010 if mode == "ik" else .005,
                                 "ankle_deg":3. if mode == "ik" else .1}
    report_name = "blender_validation_raw.json" if args.variant == "raw" else "blender_validation.json"
    (args.run_dir/report_name).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({k:v for k,v in report.items() if k not in ("samples","orientation_samples")}, indent=2))
    assert report["max_body_error_m"] < report["playback_limits"]["body_m"], "Independent Blender/MMD body playback disagrees with target FK"
    assert report["max_toe_error_m"] < report["playback_limits"]["toe_m"], "Independent Blender/MMD toe playback disagrees with target FK"
    assert report["max_neck_head_orientation_error_deg"] < .1, "Independent Blender neck/head orientation differs"
    assert report["max_ankle_orientation_error_deg"] < report["playback_limits"]["ankle_deg"], "Independent Blender foot orientation differs"


if __name__ == "__main__":
    main()
