"""Direct joint-position retargeting against an MMD rest skeleton."""
import numpy as np
from scipy.spatial.transform import Rotation

from .motion import TargetMotion, resample_source

# Source SMPL: anatomical right +X, up +Y, anterior -Z. MMD reference:
# anatomical left +X, up +Y, anterior -Z. This reflection changes the basis;
# rotations are solved from transformed landmarks, never quaternion-component flips.
SOURCE_TO_MMD = np.diag([-1.0, 1.0, 1.0])


def unit(vector, fallback=None):
    norm = np.linalg.norm(vector)
    if norm > 1e-9:
        return vector / norm
    return np.array([0., 1., 0.]) if fallback is None else fallback


def align(a, b):
    a, b = unit(a), unit(b)
    cross, dot = np.cross(a, b), np.clip(np.dot(a, b), -1., 1.)
    if dot > 1 - 1e-10:
        return np.eye(3)
    if dot < -1 + 1e-10:
        axis = unit(np.cross(a, [1., 0., 0.] if abs(a[0]) < .9 else [0., 0., 1.]))
        return Rotation.from_rotvec(axis * np.pi).as_matrix()
    return Rotation.from_rotvec(unit(cross) * np.arccos(dot)).as_matrix()


def body_frame(left, up):
    y = unit(up)
    x = unit(left - np.dot(left, y) * y, np.array([1., 0., 0.]))
    z = unit(np.cross(x, y), np.array([0., 0., 1.]))
    return np.column_stack([x, np.cross(z, x), z])


def retarget(motion, profile):
    points = resample_source(motion) @ SOURCE_TO_MMD.T
    indices = {name: i for i, name in enumerate(motion.joint_names)}
    required = {"pelvis", "spine1", "spine3", "neck", "head"}
    required.update(side + "_" + part for side in ("left", "right")
                    for part in ("hip", "knee", "ankle", "foot", "collar", "shoulder", "elbow", "wrist"))
    missing = required - indices.keys()
    if missing:
        raise ValueError(f"Source is missing joints needed for MMD: {sorted(missing)}")
    def joint(name):
        return points[:, indices[name]]
    source_lengths = []
    target_lengths = []
    for side, lower in (("Left", "left"), ("Right", "right")):
        source_lengths.append(np.median(np.linalg.norm(joint(lower+"_hip")-joint(lower+"_knee"), axis=1)
                                        + np.linalg.norm(joint(lower+"_knee")-joint(lower+"_ankle"), axis=1)))
        target_lengths.append(np.linalg.norm(profile.position(side+"Leg")-profile.position(side+"Knee"))
                              + np.linalg.norm(profile.position(side+"Knee")-profile.position(side+"Ankle")))
    if min(source_lengths) < 1e-5:
        raise ValueError("Source leg geometry is degenerate")
    scale = float(np.mean(target_lengths) / np.mean(source_lengths))
    count, bones = len(points), len(profile.names)
    rotations = np.zeros((count, bones, 4)); rotations[:, :, 3] = 1
    translations = np.zeros((count, bones, 3))
    out = TargetMotion(profile, rotations, translations)
    root_i = profile.index[profile.root]
    pelvis = joint("pelvis")
    translations[:, root_i] = (pelvis - pelvis[0]) * scale
    # Root height varies relative to a robust standing level, retaining crouches/jumps.
    translations[:, root_i, 1] = (pelvis[:, 1] - np.percentile(pelvis[:, 1], 75)) * scale
    aims = {"Torso": ("pelvis", "spine3", "Torso2"),
            "Torso2": ("spine3", "neck", "Neck")}
    for side, lower in (("Left", "left"), ("Right", "right")):
        aims.update({side+"Shoulder": (lower+"_collar", lower+"_shoulder", side+"Arm"),
                     side+"Arm": (lower+"_shoulder", lower+"_elbow", side+"Elbow"),
                     side+"Elbow": (lower+"_elbow", lower+"_wrist", side+"Wrist"),
                     side+"Leg": (lower+"_hip", lower+"_knee", side+"Knee"),
                     side+"Knee": (lower+"_knee", lower+"_ankle", side+"Ankle"),
                     side+"Ankle": (lower+"_ankle", lower+"_foot", side+"Toe")})
    for frame in range(count):
        p = points[frame]
        j = lambda name: p[indices[name]]
        pelvis_frame = body_frame(j("left_hip")-j("right_hip"), j("spine1")-j("pelvis"))
        lateral = pelvis_frame[:, 0].copy(); lateral[1] = 0
        heading = body_frame(lateral, np.array([0., 1., 0.]))
        world = []
        for i, name in enumerate(profile.names):
            parent = profile.parents[i]
            parent_r = np.eye(3) if parent < 0 else world[parent]
            local = np.eye(3)
            if i == root_i:
                local = heading
            elif name == "HipMaster":
                local = parent_r.T @ pelvis_frame
            elif name == "Neck":
                # SMPL's short neck->head vector points toward the skull/jaw
                # center, not along the neck's anatomical up axis. Aiming the
                # target's neck at it adds a large permanent forward bend.
                # Fit a longer virtual chest->head axis against the same span
                # in the target rest pose, preserving the rig's own neck offset.
                rest_axis = profile.position("Head")-profile.position("Torso2")
                direction = j("head")-j("spine3")
                local = align(rest_axis, parent_r.T @ direction)
            elif name in aims:
                a, b, child = aims[name]
                rest_direction = profile.position(child)-profile.position(name)
                direction = j(b)-j(a)
                # Solve the bone in its parent's coordinates, retaining the target rest axes.
                local = align(rest_direction, parent_r.T @ direction)
                if name == "Torso2":
                    chest = body_frame(j("left_shoulder")-j("right_shoulder"), j("neck")-j("spine3"))
                    rest_chest = body_frame(profile.position("LeftShoulder")-profile.position("RightShoulder"),
                                            profile.position("Neck")-profile.position("Torso2"))
                    local = parent_r.T @ chest @ rest_chest.T
            elif name == "Head":
                # Follow the neck; body landmarks do not observe a separate
                # head orientation. World-upright heading would counter-rotate
                # the fitted neck and chest.
                local = np.eye(3)
            elif name in ("LeftWrist", "RightWrist"):
                lower = "left" if name.startswith("Left") else "right"
                hand = lower + "_hand"
                if hand in indices:
                    child = name.replace("Wrist", "MiddleFinger1")
                    if child in profile.index:
                        local = align(profile.position(child)-profile.position(name), parent_r.T @ (j(hand)-j(lower+"_wrist")))
            world.append(parent_r @ local)
            quat = Rotation.from_matrix(local).as_quat()
            if frame and np.dot(quat, rotations[frame-1, i]) < 0:
                quat = -quat
            rotations[frame, i] = quat
    positions, world = out.fk()
    heights = []
    for side in ("Left", "Right"):
        ankle, toe = [profile.index[side+part] for part in ("Ankle", "Toe")]
        heel_y = positions[:, ankle, 1] + world[:, ankle, 1, 1]*(profile.rest[toe, 1]-profile.rest[ankle, 1])
        heights.extend([heel_y, positions[:, toe, 1]])
    # One global floor offset; never independently ground every frame.
    translations[:, root_i, 1] -= np.percentile(np.stack(heights, axis=1).min(axis=1), 5)
    return out, {"source_to_target_basis": SOURCE_TO_MMD.tolist(), "translation_scale": scale,
                 "root_policy": profile.root, "fps": 30., "unobserved_fingers": True,
                 "neck_policy": "rest_relative_chest_to_head_axis",
                 "head_orientation": "inherits_neck_no_independent_observation"}
