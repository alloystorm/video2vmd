"""Temporal contact anchors and baked target-space foot correction, in meters."""
import numpy as np
from scipy.spatial.transform import Rotation

from .motion import TargetMotion
from .retarget import align, unit


def foot_points(motion):
    positions, rotations = motion.fk()
    points = []
    for side in ("Left", "Right"):
        ankle, toe = (motion.profile.index[side + part] for part in ("Ankle", "Toe"))
        heel_offset = np.array([0., motion.profile.rest[toe, 1]-motion.profile.rest[ankle, 1], 0.])
        heel = positions[:, ankle] + np.einsum("tij,j->ti", rotations[:, ankle], heel_offset)
        points.extend([heel, positions[:, toe]])
    return np.stack(points, axis=1)


def runs(mask):
    padded = np.r_[False, mask, False].astype(int)
    starts = np.flatnonzero(np.diff(padded) == 1)
    ends = np.flatnonzero(np.diff(padded) == -1)
    return zip(starts, ends)


def detect_contacts(motion, floor=None, height=.035, speed=.35, dwell=.1):
    points = foot_points(motion)
    floor = float(np.percentile(points[:, :, 1].min(axis=1), 5)) if floor is None else float(floor)
    velocity = np.zeros_like(points) if len(points) < 2 else np.gradient(points, axis=0) * motion.fps
    contact = (points[:, :, 1] < floor + height) & (np.linalg.norm(velocity, axis=2) < speed)
    minimum = max(1, round(dwell * motion.fps))
    for i in range(4):
        for start, end in list(runs(contact[:, i])):
            if end-start < minimum:
                contact[start:end, i] = False
    anchors = points.copy()
    weights = contact.astype(float)
    for i in range(4):
        for start, end in runs(contact[:, i]):
            anchor = np.median(points[start:end, i], axis=0)
            anchor[1] = floor
            anchors[start:end, i] = anchor
            # Flat holds remain exact away from lift-off / touchdown boundaries.
            ramp = min(3, (end-start)//3)
            if start > 0 and ramp:
                weights[start:start+ramp, i] *= np.arange(1, ramp+1)/(ramp+1)
            if end < len(points) and ramp:
                weights[end-ramp:end, i] *= np.arange(ramp, 0, -1)/(ramp+1)
    support = contact[:, ::2].astype(np.int8) + 2*contact[:, 1::2].astype(np.int8)
    return {"floor_m": floor, "contact": contact, "anchors_m": anchors, "weights": weights,
            "support": support, "airborne_candidate": points[:, :, 1].min(axis=1) > floor+height,
            "height_threshold_m": height, "speed_threshold_m_s": speed, "minimum_dwell_s": dwell}


def solve_leg(hip, knee, ankle, target, pole):
    l1, l2 = np.linalg.norm(knee-hip), np.linalg.norm(ankle-knee)
    direction = unit(target-hip)
    distance = np.clip(np.linalg.norm(target-hip), abs(l1-l2)+1e-6, l1+l2-1e-6)
    reach = hip + direction * distance
    projection = (l1*l1 + distance*distance-l2*l2) / (2*distance)
    height = np.sqrt(max(0., l1*l1-projection*projection))
    bend = knee-hip - np.dot(knee-hip, direction)*direction
    if np.linalg.norm(bend) < 1e-5:
        bend = pole - np.dot(pole, direction)*direction
    bend = unit(bend, np.array([0., 0., -1.]))
    return hip + projection*direction + height*bend, reach


def _frame_fk(motion, frame):
    temporary = TargetMotion(motion.profile, motion.rotations[frame:frame+1], motion.translations[frame:frame+1])
    p, r = temporary.fk()
    return p[0], r[0]


def _foot_target(points, ankle, anchors, weights, floor):
    """Heel/toe residual pitch and translation, including up-only swing clearance.

    Adapted from dxr_native/rig/src/FeetSolver.cpp's two-contact decomposition.
    Use an exact rigid rotation here rather than its small-angle approximation.
    Contact evidence supplies plant weights; no COM/gravity motion is synthesized.
    """
    total = float(weights.sum())
    plant = min(1., total)
    low, high = np.argsort(points[:, 1])
    desired = points[:, 1].copy()
    desired[low] = max(floor, desired[low]) * (1-plant) + floor*plant
    desired[high] = max(floor, desired[high] + min(0., desired[low]-points[low, 1]))
    # Flatten only when BOTH endpoints have support evidence. A toe stand or
    # heel roll keeps the unsupported end raised.
    desired[high] += min(weights) * (floor-desired[high])
    vector = points[1]-points[0]
    length = np.linalg.norm(vector)
    horizontal = np.hypot(vector[0], vector[2])
    tilt, angle = np.eye(3), 0.
    if horizontal > 1e-6 and length > 1e-6:
        current = np.arctan2(vector[1], horizontal)
        wanted = np.arcsin(np.clip((desired[1]-desired[0])/length, -1., 1.))
        angle = float(np.clip(current-wanted, -.26, .26))
        axis = np.cross([0., 1., 0.], vector) / horizontal
        tilt = Rotation.from_rotvec(axis*angle).as_matrix()
    rotated = ankle + (points-ankle) @ tilt.T
    delta = np.zeros(3)
    if total > 0:
        delta = np.average(anchors-rotated, axis=0, weights=weights) * plant
        delta[1] = np.average(floor-rotated[:, 1], weights=weights) * plant
    # Swing feet can be pushed UP out of the floor, never pulled down to it.
    delta[1] = max(delta[1], floor-rotated[:, 1].min())
    slide = np.linalg.norm(delta[[0, 2]])
    if slide > .12:
        delta[[0, 2]] *= .12/slide
    return ankle+delta, tilt, angle, slide > .12


def correct_feet(motion, contacts):
    out = TargetMotion(motion.profile, motion.rotations.copy(), motion.translations.copy(), motion.fps)
    original_points = foot_points(motion)
    raw_pos, raw_rot = motion.fk()
    profile = motion.profile
    root = profile.index[profile.root]
    residuals, pitches = [], []
    collision_frames = corrected_frames = slide_limited = 0
    maximum_lowering = 0.
    for frame in range(motion.n_frames):
        targets = []
        collision = False
        for side_num, side in enumerate(("Left", "Right")):
            sl = slice(side_num*2, side_num*2+2)
            weights = contacts["weights"][frame, sl]
            total = weights.sum()
            penetrating = original_points[frame, sl, 1].min() < contacts["floor_m"]-1e-8
            if total <= 0 and not penetrating:
                continue
            collision |= penetrating and total <= 0
            ankle = profile.index[side+"Ankle"]
            target, tilt, pitch, limited = _foot_target(
                original_points[frame, sl], raw_pos[frame, ankle],
                contacts["anchors_m"][frame, sl], weights, contacts["floor_m"])
            targets.append((side, target, tilt))
            pitches.append(abs(pitch))
            slide_limited += int(limited)
        if not targets:
            continue
        corrected_frames += 1
        collision_frames += int(collision)
        # Solve the vertical reach bound exactly (a radial shortfall is not a
        # vertical displacement). Keep the 8 cm pelvis safety limit.
        needed = []
        for side, target, _ in targets:
            hi, ki, ai = [profile.index[side+part] for part in ("Leg", "Knee", "Ankle")]
            length = np.linalg.norm(raw_pos[frame, ki]-raw_pos[frame, hi])+np.linalg.norm(raw_pos[frame, ai]-raw_pos[frame, ki])
            difference = raw_pos[frame, hi]-target
            vertical = np.sqrt(max(0., (length-1e-6)**2-np.sum(difference[[0, 2]]**2)))
            needed.append(max(0., difference[1]-vertical))
        lower = min(.08, max(needed, default=0.))
        maximum_lowering = max(maximum_lowering, lower)
        out.translations[frame, root, 1] -= lower
        if lower > 0:
            # Keep the other foot on its observed swing path as the body lowers.
            for side in ("Left", "Right"):
                if not any(item[0] == side for item in targets):
                    targets.append((side, raw_pos[frame, profile.index[side+"Ankle"]], np.eye(3)))
        for side, target, tilt in targets:
            hi, ki, ai = [profile.index[side+part] for part in ("Leg", "Knee", "Ankle")]
            p, r = _frame_fk(out, frame)
            knee, reached = solve_leg(p[hi], p[ki], p[ai], target, r[root] @ np.array([0., 0., -1.]))
            desired_hip = align(p[ki]-p[hi], knee-p[hi]) @ r[hi]
            parent = profile.parents[hi]
            parent_r = np.eye(3) if parent < 0 else r[parent]
            out.rotations[frame, hi] = Rotation.from_matrix(parent_r.T @ desired_hip).as_quat()
            p2, r2 = _frame_fk(out, frame)
            desired_knee = align(p2[ai]-p2[ki], reached-p2[ki]) @ r2[ki]
            out.rotations[frame, ki] = Rotation.from_matrix(r2[profile.parents[ki]].T @ desired_knee).as_quat()
            p3, r3 = _frame_fk(out, frame)
            # Preserve toe-local articulation while applying the residual sole
            # pitch to the ankle in WORLD space (including turned characters).
            out.rotations[frame, ai] = Rotation.from_matrix(r3[profile.parents[ai]].T @ tilt @ raw_rot[frame, ai]).as_quat()
            residuals.append(float(np.linalg.norm(p3[ai]-target)))
    # Enforce the quaternion double-cover convention after correction.
    for frame in range(1, out.n_frames):
        flip = np.sum(out.rotations[frame-1]*out.rotations[frame], axis=1) < 0
        out.rotations[frame, flip] *= -1
    return out, {"method": "heel_toe_clearance_v2", "max_reach_residual_m": max(residuals, default=0.),
                 "corrected_frames": corrected_frames, "swing_clearance_frames": collision_frames,
                 "slide_limited_foot_frames": slide_limited, "max_foot_pitch_correction_rad": max(pitches, default=0.),
                 "max_pelvis_lowering_m": maximum_lowering}


def quality(motion, contacts):
    p = foot_points(motion)
    mask = contacts["contact"]
    anchor_error = np.linalg.norm((p-contacts["anchors_m"])[:, :, [0, 2]], axis=2)
    slip = []
    if motion.n_frames > 1:
        valid = mask[1:] & mask[:-1]
        speed = np.linalg.norm(np.diff(p[:, :, [0, 2]], axis=0), axis=2) * motion.fps
        slip = speed[valid]
    positions, _ = motion.fk()
    root = positions[:, motion.profile.index[motion.profile.root]]
    body = [i for i, name in enumerate(motion.profile.names) if name in
            ("HipMaster", "Torso2", "Head", "LeftWrist", "RightWrist", "LeftAnkle", "RightAnkle")]
    jerk = np.diff(positions[:, body], n=3, axis=0)*motion.fps**3
    return {"contact_samples": int(mask.sum()),
            "contact_frames": int(np.any(mask, axis=1).sum()),
            "total_frames": motion.n_frames,
            "mean_contact_anchor_error_m": float(np.mean(anchor_error[mask])) if mask.any() else 0.,
            "mean_contact_speed_m_s": float(np.mean(slip)) if len(slip) else 0.,
            "max_floor_penetration_m": float(max(0., contacts["floor_m"]-p[:, :, 1].min())),
            "root_displacement_m": (root[-1]-root[0]).tolist(),
            "root_path_length_m": float(np.linalg.norm(np.diff(root, axis=0), axis=1).sum()),
            "body_jerk_rms_m_s3": float(np.sqrt(np.mean(np.sum(jerk**2, axis=2)))) if len(jerk) else 0.}


def constrain_ik_knees(motion):
    """Express the fitted triangle using the PMX knee's actual local-X hinge.

    A free 3D knee swing can match positions yet be clamped differently by MMD.
    Solve the hinge angle from endpoint distance, then solve the thigh's twist
    from the desired knee plane. Preserve the ankle's world orientation.
    """
    from .retarget import body_frame
    profile = motion.profile
    out = TargetMotion(profile, motion.rotations.copy(), motion.translations.copy(), motion.fps)
    for frame in range(out.n_frames):
        for side in ("Left", "Right"):
            name = side+"Knee"
            if name not in getattr(profile, "knee_limits", {}):
                raise ValueError(f"IK export requires a supported PMX local-X hinge for {name}")
            hi, ki, ai = [profile.index[side+part] for part in ("Leg", "Knee", "Ankle")]
            if profile.parents[ki] != hi or profile.parents[ai] != ki:
                raise ValueError("IK export requires a direct thigh/knee/ankle chain")
            p, r = _frame_fk(out, frame)
            a, b = profile.offsets[ki], profile.offsets[ai]
            distance = np.linalg.norm(p[ai]-p[hi])
            u, v = a[1]*b[1]+a[2]*b[2], -a[1]*b[2]+a[2]*b[1]
            radius = np.hypot(u, v)
            phase = np.arctan2(v, u)
            cosine = np.clip(((distance*distance-np.dot(a,a)-np.dot(b,b))/2-a[0]*b[0])/max(radius,1e-12), -1., 1.)
            angle = np.arccos(cosine)
            lo, high = profile.knee_limits[name]
            candidates = [phase+sign*angle+turn*2*np.pi for sign in (-1,1) for turn in (-1,0,1)]
            valid = [theta for theta in candidates if lo-1e-6 <= theta <= high+1e-6]
            theta = min(valid, key=abs) if valid else min((lo,high), key=lambda t: min(abs(t-c) for c in candidates))
            knee_rotation = Rotation.from_rotvec([theta,0,0]).as_matrix()
            local_end = a+knee_rotation@b
            desired = p[ai]-p[hi]
            # body_frame constructs a frame from two non-collinear vectors.
            source_frame = body_frame(a, local_end)
            desired_frame = body_frame(p[ki]-p[hi], desired)
            hip_rotation = desired_frame @ source_frame.T
            out.rotations[frame, hi] = Rotation.from_matrix(r[profile.parents[hi]].T@hip_rotation).as_quat()
            out.rotations[frame, ki] = Rotation.from_matrix(knee_rotation).as_quat()
            _, updated = _frame_fk(out, frame)
            out.rotations[frame, ai] = Rotation.from_matrix(updated[ki].T@r[ai]).as_quat()
    for frame in range(1, out.n_frames):
        flip = np.sum(out.rotations[frame-1]*out.rotations[frame], axis=1) < 0
        out.rotations[frame, flip] *= -1
    return out
