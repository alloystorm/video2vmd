"""Numeric hand-pose gap filling; no inference, video or body-motion imports.

Rotations use MANO's canonical right-hand local axes and xyzw quaternions.
Left/right tracks are body-associated identities, independent of detector labels.
Conversion to target-rig rest axes belongs to a later retargeting stage.
"""
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation, Slerp

SIDES = ("left", "right")
UNAVAILABLE, OBSERVED, INTERPOLATED, HELD = range(4)
STATUS_NAMES = ("unavailable", "observed", "interpolated", "held")


def _times(values, name):
    values = np.asarray(values, dtype=float)
    if (values.ndim != 1 or not len(values) or not np.isfinite(values).all()
            or np.any(np.diff(values) <= 0)):
        raise ValueError(f"{name} must be nonempty, finite and strictly increasing")
    return values


def _continuous(quaternions):
    result = quaternions.copy()
    for frame in range(1, len(result)):
        flip = np.sum(result[frame-1]*result[frame], axis=-1) < 0
        result[frame, flip] *= -1
    return result


def fill_hand_gaps(timestamps, rotations, observed, *, betas=None, target_timestamps=None):
    """SLERP all bounded gaps; hold nearest pose at clip edges.

    ``rotations`` is (frames,2,joints,4), canonical local xyzw. An optional
    ``betas`` array is (frames,2,shape_components) and interpolates linearly.
    All channels of a hand share its observation mask. Unobserved inputs may be
    NaN; a hand with no observation remains identity/zero and unavailable.
    Valid observations are preserved as rotations; quaternion signs may change.
    There is no maximum gap duration or implicit smoothing of observed samples.
    """
    source_times = _times(timestamps, "timestamps")
    times = source_times.copy() if target_timestamps is None else _times(target_timestamps, "target_timestamps")
    rotations = np.asarray(rotations, dtype=float)
    observed = np.asarray(observed, dtype=bool)
    if (rotations.ndim != 4 or rotations.shape[0] != len(source_times)
            or rotations.shape[1] != 2 or rotations.shape[2] < 1 or rotations.shape[3] != 4
            or observed.shape != (len(source_times), 2)):
        raise ValueError("Expected (frames,2,joints,4) rotations and (frames,2) observations")
    accepted = rotations[observed]
    if (not np.isfinite(accepted).all()
            or not np.allclose(np.linalg.norm(accepted, axis=-1), 1., atol=1e-6, rtol=0)):
        raise ValueError("Observed hand rotations must be finite unit xyzw quaternions")
    if betas is not None:
        betas = np.asarray(betas, dtype=float)
        if (betas.ndim != 3 or betas.shape[:2] != observed.shape or betas.shape[2] < 1
                or not np.isfinite(betas[observed]).all()):
            raise ValueError("Observed betas must be finite (frames,2,components)")
    result = np.zeros((len(times),) + rotations.shape[1:])
    result[..., 3] = 1.
    status = np.full((len(times), 2), UNAVAILABLE, dtype=np.uint8)
    shapes = None if betas is None else np.zeros((len(times), 2, betas.shape[2]))
    for hand in range(2):
        indices = np.flatnonzero(observed[:, hand])
        if not len(indices):
            continue
        anchors = source_times[indices]
        values = _continuous(rotations[indices, hand])
        query = np.clip(times, anchors[0], anchors[-1])
        if len(indices) == 1:
            result[:, hand] = values[0]
        else:
            for joint in range(rotations.shape[2]):
                result[:, hand, joint] = Slerp(anchors, Rotation.from_quat(values[:, joint]))(query).as_quat()
        result[:, hand] = _continuous(result[:, hand])
        status[:, hand] = INTERPOLATED
        status[(times < anchors[0]) | (times > anchors[-1]), hand] = HELD
        # Exact source-time samples keep their observed label after resampling.
        # The tolerance is only for floating-point time representation.
        nearest = np.searchsorted(anchors, times)
        before = np.clip(nearest-1, 0, len(anchors)-1)
        after = np.clip(nearest, 0, len(anchors)-1)
        exact = np.minimum(abs(times-anchors[before]), abs(times-anchors[after])) <= 1e-9
        status[exact, hand] = OBSERVED
        if shapes is not None:
            for component in range(betas.shape[2]):
                shapes[:, hand, component] = np.interp(times, anchors, betas[indices, hand, component])
    payload = {"timestamps": times, "quat_xyzw": result, "status": status,
               "observed": status == OBSERVED, "available": status != UNAVAILABLE,
               "source_timestamps": source_times, "source_observed": observed.copy()}
    if shapes is not None:
        payload["betas"] = shapes
    return payload


def _matrix_quaternions(values, count):
    values = np.asarray(values, dtype=float).reshape(count, 3, 3)
    if (not np.isfinite(values).all()
            or not np.allclose(values @ values.swapaxes(-1, -2), np.eye(3), atol=1e-4)
            or not np.allclose(np.linalg.det(values), 1., atol=1e-4)):
        raise ValueError("Observed MANO matrices must be proper rotations")
    return Rotation.from_matrix(values).as_quat()


def complete_wilor_observations(document, *, target_timestamps=None, candidate_source="crop"):
    """Convert a saved probe to a numeric pose cache without rerunning inference.

    Channel 0 is the canonical MANO palm orientation; channels 1–15 are the
    canonical MANO finger rotations. Body side owns identity, not ``is_right``.
    The detector's label is retained separately for inspection.
    """
    if candidate_source not in ("crop", "full"):
        raise ValueError("candidate_source must be crop or full")
    frames = document["observations"]
    times = np.asarray([frame["time_s"] for frame in frames], dtype=float)
    rotations = np.full((len(frames), 2, 16, 4), np.nan)
    betas = np.full((len(frames), 2, 10), np.nan)
    observed = np.zeros((len(frames), 2), dtype=bool)
    detector_right = np.full((len(frames), 2), -1, dtype=np.int8)
    for frame_index, frame in enumerate(frames):
        tracks = {hand["side"]: hand for hand in frame["hands"]}
        if len(tracks) != len(frame["hands"]) or set(tracks) != set(SIDES):
            raise ValueError("Each frame must contain unique left/right body-associated tracks")
        for hand_index, side in enumerate(SIDES):
            track = tracks[side]
            match = track[candidate_source+"_match"]
            if match is None:
                continue
            candidates = track["crop_candidates"] if candidate_source == "crop" else frame["full_candidates"]
            if not isinstance(match, int) or not 0 <= match < len(candidates):
                raise ValueError("Hand match index is outside its candidate list")
            candidate = candidates[match]
            rotations[frame_index, hand_index, 0] = _matrix_quaternions(candidate["mano_global_orient_matrix"], 1)[0]
            rotations[frame_index, hand_index, 1:] = _matrix_quaternions(candidate["mano_pose_matrices"], 15)
            betas[frame_index, hand_index] = np.asarray(candidate["mano_betas"], dtype=float).reshape(10)
            observed[frame_index, hand_index] = True
            detector_right[frame_index, hand_index] = int(candidate["is_right"])
    payload = fill_hand_gaps(times, rotations, observed, betas=betas, target_timestamps=target_timestamps)
    payload["source_detector_right"] = detector_right
    payload["version"] = np.int32(1)
    payload["sides"] = np.array(SIDES)
    payload["metadata"] = np.array(json.dumps({
        "backend": "wilor", "clip": document.get("clip"), "quaternion": "xyzw",
        "rotation_space": "canonical_right_MANO_local_axes_for_both_body_side_tracks",
        "channels": "0=MANO palm global orientation; 1:16=MANO local finger rotations",
        "identity": "body-associated side; detector handedness does not switch tracks",
        "candidate_source": candidate_source, "gap_policy": "all_bounded_gaps_slerp; endpoints_hold_nearest",
        "unobserved_hand_policy": "identity_unavailable", "status_names": STATUS_NAMES,
    }))
    return payload


def save_hand_motion(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **payload)
    return path
