"""Rest-axis conversion from cached canonical MANO rotations to MMD fingers."""
import hashlib

import numpy as np
from scipy.spatial.transform import Rotation

from .motion import TargetMotion
from .retarget import align, body_frame, SOURCE_TO_MMD

# Target control suffixes, MANO rotation channels, OpenPose 21-joint landmarks.
FINGERS = (("IndexFinger", (1, 2, 3), (5, 6, 7, 8), (1, 2, 3)),
           ("MiddleFinger", (4, 5, 6), (9, 10, 11, 12), (1, 2, 3)),
           ("Pinky", (7, 8, 9), (17, 18, 19, 20), (1, 2, 3)),
           ("RingFinger", (10, 11, 12), (13, 14, 15, 16), (1, 2, 3)),
           ("Thumb", (13, 14, 15), (1, 2, 3, 4), (0, 1, 2)))


def finger_names(side):
    return [side+finger+str(number) for finger, _, _, numbers in FINGERS for number in numbers]


def _palm(wrist, index, middle, pinky):
    if np.linalg.norm(middle-wrist) < 1e-7 or np.linalg.norm(np.cross(index-pinky, middle-wrist)) < 1e-10:
        raise ValueError("Degenerate hand rest-palm calibration")
    return body_frame(index-pinky, middle-wrist)


def retarget_hands(motion, hands, world_to_camera=None):
    """Fit fingers and, when camera geometry is available, the wrist orientation.

    Both cached tracks use canonical right MANO rotations. The left-hand basis
    is an improper orthogonal transform; matrix conjugation handles parity
    without reflecting individual quaternion components. Each bone additionally
    aligns its source rest segment to the target's actual rest/tail direction.
    """
    profile = motion.profile
    if len(hands["timestamps"]) != motion.n_frames or not np.allclose(hands["timestamps"], np.arange(motion.n_frames)/motion.fps, atol=1e-9, rtol=0):
        raise ValueError("Hand cache must be resampled onto the target motion timeline")
    rest = np.asarray(hands["mano_rest_joints_m"], dtype=float)
    source_palm = _palm(rest[0], rest[5], rest[9], rest[17])
    rotations = Rotation.from_quat(hands["quat_xyzw"].reshape(-1, 4)).as_matrix().reshape(motion.n_frames, 2, 16, 3, 3)
    out = TargetMotion(profile, motion.rotations.copy(), motion.translations.copy(), motion.fps)
    out.hand_channels = np.zeros(len(profile.names), dtype=bool)
    out.hand_status = np.array(hands["status"], copy=True)
    out.hand_tip_offsets = np.zeros((len(profile.names), 3))
    out.hand_wrist_status = np.zeros_like(out.hand_status)
    orient_wrists = world_to_camera is not None and "camera_palm_quat_xyzw" in hands
    if orient_wrists:
        world_to_camera = np.asarray(world_to_camera, dtype=float)
        if (world_to_camera.shape != (motion.n_frames, 3, 3) or not np.isfinite(world_to_camera).all()
                or not np.allclose(world_to_camera @ world_to_camera.swapaxes(-1, -2), np.eye(3), atol=1e-4)
                or not np.allclose(np.linalg.det(world_to_camera), 1., atol=1e-4)):
            raise ValueError("Invalid world-to-camera bridge for hand orientation")
        parent_rotations = motion.fk()[1]
        palm_camera = Rotation.from_quat(hands["camera_palm_quat_xyzw"].reshape(-1, 4)).as_matrix().reshape(motion.n_frames, 2, 3, 3)
        # The body's source-to-MMD basis is a reflection. Transform the two
        # measured palm rays, then reconstruct their cross product so the
        # resulting wrist rotation remains proper on BOTH anatomical sides.
        camera_to_target = SOURCE_TO_MMD @ world_to_camera.swapaxes(-1, -2)
    calibration = []
    for hand, side in enumerate(("Left", "Right")):
        active = hands["available"][:, hand]
        names = finger_names(side)
        missing = [name for name in names if name not in profile.index]
        if missing:
            if active.any():
                raise ValueError(f"Target lacks supported {side} finger controls: {missing}")
            continue
        unsupported = set(names) & set(getattr(profile, "unsupported_hand_controls", ()))
        if active.any() and unsupported:
            raise ValueError(f"Unsupported inherited/fixed-axis PMX hand controls: {sorted(unsupported)}")
        target_palm = _palm(*(profile.position(side+name) for name in ("Wrist", "IndexFinger1", "MiddleFinger1", "Pinky1")))
        if orient_wrists:
            wrist = profile.index[side+"Wrist"]
            parent = profile.parents[wrist]
            desired_rays = camera_to_target @ palm_camera[:, hand]
            desired_palm = np.stack((desired_rays[:, :, 0], desired_rays[:, :, 1],
                                    np.cross(desired_rays[:, :, 0], desired_rays[:, :, 1])), axis=-1)
            desired_world = desired_palm @ target_palm.T
            parent_world = parent_rotations[:, parent] if parent >= 0 else np.broadcast_to(np.eye(3), desired_world.shape)
            local = parent_world.swapaxes(-1, -2) @ desired_world
            palm_active = hands["palm_status"][:, hand] != 0
            if palm_active.any():
                out.rotations[palm_active, wrist] = Rotation.from_matrix(local[palm_active]).as_quat()
            out.hand_wrist_status[:, hand] = hands["palm_status"][:, hand]
        parity = np.diag([1., 1., -1. if side == "Left" else 1.])
        palm_basis = target_palm @ parity @ source_palm.T
        for finger, channels, points, numbers in FINGERS:
            for segment, (channel, number) in enumerate(zip(channels, numbers)):
                name = side+finger+str(number)
                bone = profile.index[name]
                parent_name = side+"Wrist" if segment == 0 else side+finger+str(numbers[segment-1])
                if profile.parents[bone] != profile.index[parent_name]:
                    raise ValueError(f"Unsupported finger hierarchy: {name} must be directly parented to {parent_name}")
                if segment < 2:
                    axis = profile.position(side+finger+str(numbers[segment+1]))-profile.position(name)
                else:
                    axis = getattr(profile, "hand_tail_offsets", {}).get(name)
                    if axis is None:
                        # Standard skeleton has no terminal landmark; use its
                        # preceding segment as an explicit straight-tip proxy.
                        axis = profile.position(name)-profile.position(parent_name)
                    out.hand_tip_offsets[bone] = axis
                source_axis = rest[points[segment+1]]-rest[points[segment]]
                if np.linalg.norm(axis) < 1e-7 or np.linalg.norm(source_axis) < 1e-7:
                    raise ValueError(f"Degenerate rest segment for {name}")
                basis = align(palm_basis @ source_axis, axis) @ palm_basis
                converted = basis @ rotations[:, hand, channel] @ basis.T
                out.rotations[:, bone] = [0., 0., 0., 1.]
                if active.any():
                    out.rotations[active, bone] = Rotation.from_matrix(converted[active]).as_quat()
                out.hand_channels[bone] = True
                calibration.append(basis)
    channels = out.hand_channels.copy()
    for side in ("Left", "Right"):
        channels[profile.index[side+"Wrist"]] = True
    for frame in range(1, out.n_frames):
        flip = np.sum(out.rotations[frame-1]*out.rotations[frame], axis=1) < 0
        # Preserve quaternion signs of all body channels except fitted wrists.
        active_channels = channels.copy()
        for hand, side in enumerate(("Left", "Right")):
            active_channels[profile.index[side+"Wrist"]] = out.hand_wrist_status[frame, hand] != 0
        out.rotations[frame, flip & active_channels] *= -1
    return out, {"enabled": True, "method": "mano_local_rest_axes_v1",
                 "wrist_policy": "camera_palm_orientation_preserve_position" if orient_wrists else "preserve_body",
                 "wrist_fallback": None if orient_wrists else "camera palm observations or body world-to-camera geometry unavailable",
                 "wrist_counts": {side.lower(): {name: int((out.hand_wrist_status[:, hand] == code).sum())
                                                 for code, name in enumerate(("body_fallback", "observed", "interpolated", "held"))}
                                  for hand, side in enumerate(("Left", "Right"))},
                 "left_policy": "matrix_conjugation_with_palm_parity", "finger_channels": int(out.hand_channels.sum()),
                 "calibration_sha256": hashlib.sha256(np.asarray(calibration).tobytes()).hexdigest(),
                 "counts": {side.lower(): {name: int((out.hand_status[:, hand] == code).sum())
                                          for code, name in enumerate(("unavailable", "observed", "interpolated", "held"))}
                            for hand, side in enumerate(("Left", "Right"))}}
