"""VMD, BVH and canonical motion-encoder export from the target-motion contract."""
from pathlib import Path
import json
import warnings

import numpy as np
from scipy.spatial.transform import Rotation

from .profile import require_codec


def write_vmd(path, motion, ik_mode="fk"):
    if ik_mode not in ("fk", "ik"):
        raise ValueError("VMD IK mode must be fk or ik")
    codec = require_codec()
    profile = motion.profile
    frames = []
    for i, name in enumerate(profile.names):
        moving = np.any(np.abs(motion.rotations[:, i, :3]) > 1e-8) or np.any(np.abs(motion.translations[:, i]) > 1e-8)
        # Explicit neutral Head keys also replace an older counter-rotation
        # when this motion is loaded onto an already animated character.
        if not moving and name not in (profile.root, "Head"):
            continue
        if name not in profile.serialized:
            raise ValueError(f"Animated helper {name} has no VMD mapping; compose it into a supported control first")
        serialized = profile.serialized[name]
        codec.encode_name(serialized)
        for frame in range(motion.n_frames):
            frames.append({"name": serialized, "frame_num": frame,
                           "position": tuple(motion.translations[frame, i]),
                           "rotation": tuple(motion.rotations[frame, i]),
                           "bezier": codec.LINEAR_BONE_INTERPOLATION})
    if ik_mode == "ik":
        positions, world = motion.fk()
        for side in ("Left", "Right"):
            control = profile.index.get(side+"LegIK")
            if control is None:
                raise ValueError("IK-driven export requires a PMX profile with foot IK controls")
            chain = getattr(profile, "ik_chains", {}).get(side+"LegIK", {})
            if chain.get("target") != side+"Ankle" or chain.get("links") != [side+"Knee", side+"Leg"]:
                raise ValueError(f"Unsupported PMX {side} foot IK chain; expected ankle target and knee/thigh links")
            ankle = profile.index[side+"Ankle"]
            parent = profile.parents[control]
            for frame in range(motion.n_frames):
                parent_r = np.eye(3) if parent < 0 else world[frame, parent]
                parent_p = np.zeros(3) if parent < 0 else positions[frame, parent]
                translation = parent_r.T @ (positions[frame, ankle]-parent_p)-profile.offsets[control]
                frames.append({"name": profile.serialized[side+"LegIK"], "frame_num": frame,
                               "position": tuple(translation), "rotation": (0.,0.,0.,1.),
                               "bezier": codec.LINEAR_BONE_INTERPOLATION})
    ik_names = [profile.serialized.get(name, codec.bone_name_reverse_translation[name])
                for name in ("LeftLegIK", "RightLegIK", "LeftLegIKParent", "RightLegIKParent")]
    states = {name: ik_mode == "ik" and i < 2 for i, name in enumerate(ik_names)}
    for name in getattr(profile, "ik_chains", {}):
        states[profile.serialized[name]] = ik_mode == "ik" and name in ("LeftLegIK", "RightLegIK")
    animation = {"unit": profile.unit, "bone_frames": frames,
                 "ik_frames": [{"frame_num": 0, "visible": True, "states": states}]}
    path = Path(path)
    temporary = path.with_suffix(".pending.vmd")
    try:
        if not codec.write_vmd(temporary, animation, model_name="video2vmd"):
            raise RuntimeError("VMD writer rejected the animation")
        ok, loaded = codec.parse_vmd(temporary, unit=profile.unit)
        if not ok or len(loaded["bone_frames"]) != len(frames):
            raise RuntimeError("VMD export did not pass readback validation")
        expected_states = {codec.bone_name_translation.get(name, name): enabled for name, enabled in states.items()}
        if not loaded["ik_frames"] or loaded["ik_frames"][0]["states"] != expected_states:
            raise RuntimeError("VMD IK states did not survive readback")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)
    return path


def write_bvh(path, motion):
    profile = motion.profile
    root = profile.index[profile.root]
    order, lines = [], ["HIERARCHY"]
    def visit(i, depth):
        order.append(i)
        indent = "  " * depth
        name = profile.names[i]
        name = name if name.isascii() and name.replace("_", "").isalnum() else f"bone_{i}"
        lines.extend([f"{indent}{'ROOT' if depth == 0 else 'JOINT'} {name}", indent + "{"])
        # Zero root OFFSET avoids importer ambiguity over whether root position
        # channels include it. Animate its absolute world position instead.
        offset = np.zeros(3) if depth == 0 else profile.offsets[i]
        lines.append(indent + "  OFFSET " + " ".join(f"{x:.9g}" for x in offset))
        channels = "6 Xposition Yposition Zposition Zrotation Xrotation Yrotation" if depth == 0 else "3 Zrotation Xrotation Yrotation"
        lines.append(indent + "  CHANNELS " + channels)
        children = np.flatnonzero(profile.parents == i)
        for child in children:
            visit(int(child), depth+1)
        if not len(children):
            lines.extend([indent+"  End Site", indent+"  {", indent+"    OFFSET 0 0 0", indent+"  }"])
        lines.append(indent + "}")
    visit(root, 0)
    omitted = sorted(set(range(len(profile.names)))-set(order))
    if omitted and (np.any(np.abs(motion.rotations[:, omitted, :3]) > 1e-8)
                    or np.any(np.abs(motion.translations[:, omitted]) > 1e-8)):
        raise ValueError("BVH cannot discard animated controls outside the body root tree")
    for i in order[1:]:
        if np.any(abs(motion.translations[:, i]) > 1e-9):
            raise ValueError("BVH profile contains unsupported non-root translation channels")
    quats = motion.rotations[:, order]
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Gimbal lock detected.*")
        euler = Rotation.from_quat(quats.reshape(-1, 4)).as_euler("ZXY").reshape(motion.n_frames, len(order), 3)
    euler = np.degrees(np.unwrap(euler, axis=0))
    values = np.column_stack([motion.translations[:, root]+profile.rest[root], euler.reshape(motion.n_frames, -1)])
    lines += ["MOTION", f"Frames: {motion.n_frames}", f"Frame Time: {1/motion.fps:.12g}"]
    lines += [" ".join(f"{x:.9g}" for x in row) for row in values]
    Path(path).write_text("\n".join(lines)+"\n", encoding="utf-8")
    metadata = {"units": "meters", "up": "+Y", "rotation_order": "ZXY", "fps": motion.fps,
                "root_position_channels": "absolute_world_with_zero_root_offset",
                "bone_indices": order, "target_bone_names": [profile.names[i] for i in order],
                "omitted_neutral_auxiliary_bones": [profile.names[i] for i in omitted]}
    Path(str(path)+".json").write_text(json.dumps(metadata, indent=2)+"\n", encoding="utf-8")
    return Path(path)


def write_motion_encoder(path, motion):
    if motion.profile.name != "standard" or len(motion.profile.names) != 65:
        raise ValueError("Motion-encoder output requires retargeting to the canonical standard profile")
    from vmd2png.bone import quat_to_6d
    positions, _ = motion.fk()
    waist = motion.profile.index["Waist"]
    values = np.zeros((motion.n_frames, 404), dtype=np.float32)
    values[:, :3] = (positions[:, waist]-motion.profile.rest[waist]) / 32.768
    values[:, 3] = 1.
    for i in range(65):
        values[:, 4+6*i:10+6*i] = np.array([quat_to_6d(q) for q in motion.rotations[:, i]])
    values[:, 394:397] = [0., .03, -.1]
    values[:, 397] = 30./180
    values[:, 398:404] = [1., 0., 0., 0., 1., 0.]
    np.save(path, values)
    return Path(path)
