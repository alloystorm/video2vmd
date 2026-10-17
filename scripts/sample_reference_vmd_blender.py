"""Sample a reference VMD's evaluated body in a FRESH background Blender scene.

No PyTorch, GVHMR or project imports. Called by compare_reference_vmd.py.
Only an armature is imported; no mesh, rigid bodies, camera or face animation.
"""
import argparse
import json
from pathlib import Path
import sys
import time
import zipfile

import bpy
import numpy as np


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config",type=Path,required=True)
    parser.add_argument("--modes",nargs="+",choices=("authored_ik","fk"))
    args=parser.parse_args(sys.argv[sys.argv.index("--")+1:])
    config=json.loads(args.config.read_text(encoding="utf-8"))
    # This process always starts with --factory-startup. Remove its default cube,
    # camera and light before importing the armature-only reference.
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)
    out=Path(config["output"]);out.mkdir(parents=True,exist_ok=True)
    dependency=Path(config["mmd_tools"])
    vendor=out/"validation_dependencies";vendor.mkdir(exist_ok=True)
    for wheel in (dependency/"mmd_tools/wheels").glob("*.whl"):
        with zipfile.ZipFile(wheel) as archive:
            for member in archive.namelist():
                if not (vendor/member).resolve().is_relative_to(vendor.resolve()):
                    raise ValueError("Escaping wheel member")
            archive.extractall(vendor)
    sys.path.insert(0,str(vendor));sys.path.insert(0,str(dependency))
    import mmd_tools
    mmd_tools.register()
    assert bpy.ops.mmd_tools.import_model(filepath=config["model"],types={"ARMATURE"},scale=config["unit"],
        rename_bones=False,fix_ik_links=False,apply_bone_fixed_axis=False,save_log=False)=={"FINISHED"}
    arm=next(obj for obj in bpy.context.scene.objects if obj.type=="ARMATURE")
    bpy.ops.object.select_all(action="DESELECT")
    arm.select_set(True);bpy.context.view_layer.objects.active=arm
    bpy.context.scene.frame_set(0)
    assert bpy.ops.mmd_tools.import_vmd(filepath=config["vmd"],scale=config["unit"],margin=0,
                                       bone_mapper="PMX",use_pose_mode=False)=={"FINISHED"}
    mapping={bone.mmd_bone.name_j:bone.name for bone in arm.pose.bones}
    names=config["sample_bones"]
    selected=[arm.pose.bones[mapping[config["serialized"][name]]] for name in names]
    ik=[constraint for bone in arm.pose.bones for constraint in bone.constraints if constraint.type=="IK"]
    original_mutes=[constraint.mute for constraint in ik]
    original_influences=[constraint.influence for constraint in ik]
    limits=[constraint for bone in arm.pose.bones for constraint in bone.constraints if constraint.type=="LIMIT_ROTATION"]
    original_limit_mutes=[constraint.mute for constraint in limits]
    basis=np.eye(3)[[0,2,1]]
    rest=[np.array(arm.data.bones[bone.name].matrix_local.to_3x3()) for bone in selected]
    bpy.context.scene.render.fps=30
    widgets=[obj for obj in bpy.context.scene.objects if obj.type=="MESH"]
    summary={"blender":bpy.app.version_string,"frames":config["frames"],"fps":30,
             "sample_bones":names,"ik_constraints":len(ik),"mesh_objects":sum(o.type=="MESH" for o in bpy.context.scene.objects),
             "mesh_details":[{"name":obj.name,"vertices":len(obj.data.vertices),"armature_bound":any(mod.type=="ARMATURE" for mod in obj.modifiers)} for obj in widgets],
             "unmatched_vmd_body_bones":[name for name in config["body_tracks"] if name not in mapping],"modes":{}}
    summary["authored_ik_influences"]=original_influences
    summary["vmd_sha256"]=config["vmd_sha256"]
    summary["authored_ik_toggles"]={name:bool(arm.pose.bones[mapping[jp]].mmd_ik_toggle)
        for name,jp in config["serialized"].items() if name in ("LeftLegIK","RightLegIK","LeftLegIKParent","RightLegIKParent") and jp in mapping}
    # ARMATURE-only import creates an EMPTY mesh for bone-order bookkeeping.
    # Display widgets/empty bookkeeping objects are not character geometry.
    print("ARMATURE-ONLY MESH BOOKKEEPING: "+json.dumps(summary["mesh_details"]),flush=True)
    assert not any(item["armature_bound"] and item["vertices"]>0 for item in summary["mesh_details"])
    previous=out/"blender_sampling.json"
    if previous.exists():
        old=json.loads(previous.read_text(encoding="utf-8"))
        if old.get("vmd_sha256",config["vmd_sha256"])==config["vmd_sha256"]:
            summary["modes"].update(old["modes"])
    for mode in args.modes or config["modes"]:
        start=time.monotonic()
        for constraint,original in zip(ik,original_mutes):
            constraint.mute=True if mode=="fk" else original
        for constraint,original in zip(limits,original_limit_mutes):
            constraint.mute=True if mode=="fk" else original
        positions=np.empty((config["frames"],len(names),3))
        rotations=np.empty((config["frames"],len(names),3,3))
        for frame in range(config["frames"]):
            bpy.context.scene.frame_set(frame)
            bpy.context.view_layer.update()
            for b,bone in enumerate(selected):
                positions[frame,b]=np.array(arm.matrix_world@bone.head)[[0,2,1]]
                physical=np.array(arm.matrix_world.to_3x3())@np.array(bone.matrix.to_3x3())@rest[b].T
                rotations[frame,b]=basis@physical@basis.T
            if frame%1000==0:
                print(f"SAMPLE {mode}: {frame}/{config['frames']}",flush=True)
        assert np.isfinite(positions).all() and np.isfinite(rotations).all()
        np.savez_compressed(out/(mode+"_world.npz"),positions_m=positions,world_rotations=rotations,
            names=np.array(names),fps=30.,timestamps=np.arange(config["frames"])/30,
            rest_positions_m=np.array(config["sample_rest"]),unit=config["unit"],mode=np.array(mode))
        summary["modes"][mode]={"elapsed_s":round(time.monotonic()-start,2),"samples":len(positions),
                                "native_ik_muted":mode=="fk","ik_limits_muted":mode=="fk"}
    (out/"blender_sampling.json").write_text(json.dumps(summary,indent=2)+"\n",encoding="utf-8")
    print("REFERENCE SAMPLING COMPLETE",flush=True)


if __name__=="__main__":
    main()
