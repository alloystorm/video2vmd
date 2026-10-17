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
    palms = np.full((len(frames), 2, 1, 4), np.nan)
    palm_observed = np.zeros((len(frames), 2), dtype=bool)
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
            if "camera_relative_xyz_m" in candidate:
                points = np.asarray(candidate["camera_relative_xyz_m"], dtype=float)
                if points.shape != (21, 3) or not np.isfinite(points).all():
                    raise ValueError("Invalid WiLoR camera-space hand joints")
                # These joints already include the official detector-side X
                # reflection. Derive the palm AFTER reflection; canonical MANO
                # global quaternions alone do not encode left-hand parity.
                up, across = points[9]-points[0], points[5]-points[17]
                if np.linalg.norm(up) > 1e-7 and np.linalg.norm(np.cross(across, up)) > 1e-10:
                    y = up/np.linalg.norm(up)
                    x = across-np.dot(across, y)*y; x /= np.linalg.norm(x)
                    palms[frame_index, hand_index, 0] = Rotation.from_matrix(np.column_stack((x, y, np.cross(x, y)))).as_quat()
                    palm_observed[frame_index, hand_index] = True
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
    if "mano_rest_joints_m" in document:
        payload["mano_rest_joints_m"] = np.asarray(document["mano_rest_joints_m"], dtype=float)
    if palm_observed.any():
        palm = fill_hand_gaps(times, palms, palm_observed, target_timestamps=target_timestamps)
        payload.update(camera_palm_quat_xyzw=palm["quat_xyzw"][:, :, 0],
                       palm_status=palm["status"], source_palm_observed=palm_observed,
                       camera_palm_convention=np.array("opencv_xright_ydown_zforward; x=pinky_to_index; y=wrist_to_middle; z=cross(x,y)"))
    return payload


def save_hand_motion(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **payload)
    return path


def mano_rest_joints(model_path):
    """One-time licensed-model calibration; exports/viewers use the numeric cache."""
    import torch
    import smplx
    from smplx.vertex_ids import vertex_ids
    from gvhmr.utils._smpl_compat import apply
    apply()
    model = smplx.MANOLayer(model_path=str(model_path), is_rhand=True, use_pca=False)
    with torch.no_grad():
        neutral = model()
    extra = neutral.vertices[:, list(vertex_ids["mano"].values())]
    points = torch.cat((neutral.joints, extra), dim=1)
    order = [0, 13, 14, 15, 16, 1, 2, 3, 17, 4, 5, 6, 18, 10, 11, 12, 19, 7, 8, 9, 20]
    return points[0, order].cpu().numpy().astype(float)


def load_hand_motion(path, *, require_calibration=True):
    """Read a reusable numeric hand cache, rejecting unsupported coordinate bases."""
    with np.load(path, allow_pickle=False) as data:
        payload = {key: data[key] for key in data.files}
    if int(payload.get("version", -1)) != 1:
        raise ValueError("Unsupported hand-motion cache version")
    metadata = json.loads(str(payload["metadata"]))
    if metadata.get("rotation_space") != "canonical_right_MANO_local_axes_for_both_body_side_tracks":
        raise ValueError("Unsupported hand-motion rotation convention")
    if tuple(payload["sides"]) != SIDES:
        raise ValueError("Hand-motion cache must contain left/right body-associated tracks")
    q, times = np.asarray(payload["quat_xyzw"], dtype=float), _times(payload["timestamps"], "timestamps")
    if q.shape != (len(times), 2, 16, 4) or not np.isfinite(q).all() or not np.allclose(np.linalg.norm(q, axis=-1), 1., atol=1e-6, rtol=0):
        raise ValueError("Hand-motion cache has invalid local xyzw rotations")
    status = payload["status"]
    if status.shape != (len(times), 2) or not np.isin(status, range(4)).all():
        raise ValueError("Hand-motion cache has invalid observation states")
    if not np.array_equal(payload["available"], status != UNAVAILABLE) or not np.array_equal(payload["observed"], status == OBSERVED):
        raise ValueError("Hand-motion observation masks disagree with status")
    if "betas" in payload and (payload["betas"].shape != (len(times), 2, 10) or not np.isfinite(payload["betas"]).all()):
        raise ValueError("Hand-motion cache has invalid MANO shape coefficients")
    if "source_timestamps" in payload or "source_observed" in payload:
        source_times = _times(payload["source_timestamps"], "source_timestamps")
        if payload["source_observed"].shape != (len(source_times), 2):
            raise ValueError("Hand-motion source evidence has an incompatible timeline")
    if "mano_rest_joints_m" in payload:
        rest = np.asarray(payload["mano_rest_joints_m"], dtype=float)
        if rest.shape != (21, 3) or not np.isfinite(rest).all():
            raise ValueError("Invalid MANO rest-joint calibration")
    elif require_calibration:
        raise ValueError("Hand cache needs rest calibration: recreate it with scripts/complete_hand_motion.py --mano PATH/TO/MANO_RIGHT.pkl")
    if "camera_palm_quat_xyzw" in payload:
        palm = payload["camera_palm_quat_xyzw"]
        if (palm.shape != (len(times), 2, 4) or not np.isfinite(palm).all()
                or not np.allclose(np.linalg.norm(palm, axis=-1), 1., atol=1e-6, rtol=0)
                or payload["palm_status"].shape != (len(times), 2)
                or not np.isin(payload["palm_status"], range(4)).all()
                or payload["source_palm_observed"].shape != payload["source_observed"].shape
                or str(payload["camera_palm_convention"]) != "opencv_xright_ydown_zforward; x=pinky_to_index; y=wrist_to_middle; z=cross(x,y)"):
            raise ValueError("Invalid camera-space palm calibration or observation states")
    return payload


def resample_hand_motion(payload, timestamps):
    """Resample completed poses while preserving the original observation states."""
    result = fill_hand_gaps(payload["timestamps"], payload["quat_xyzw"], payload["available"],
                            betas=payload.get("betas"), target_timestamps=timestamps)
    original_times = np.asarray(payload.get("source_timestamps", payload["timestamps"]))
    original_mask = np.asarray(payload.get("source_observed", payload["observed"]))
    result["status"] = _evidence_status(original_times, original_mask, result["timestamps"])
    result["observed"] = result["status"] == OBSERVED
    result["available"] = result["status"] != UNAVAILABLE
    for key in ("source_timestamps", "source_observed", "source_detector_right", "version", "sides", "metadata", "mano_rest_joints_m"):
        if key in payload:
            result[key] = np.array(payload[key], copy=True)
    if "camera_palm_quat_xyzw" in payload:
        palm = fill_hand_gaps(payload["timestamps"], payload["camera_palm_quat_xyzw"][:, :, None],
                              payload["palm_status"] != UNAVAILABLE, target_timestamps=timestamps)
        result["camera_palm_quat_xyzw"] = palm["quat_xyzw"][:, :, 0]
        result["palm_status"] = _evidence_status(original_times, payload["source_palm_observed"], result["timestamps"])
        for key in ("source_palm_observed", "camera_palm_convention"):
            result[key] = np.array(payload[key], copy=True)
    return result


def _evidence_status(original_times, original_mask, query):
    status = np.full((len(query), 2), UNAVAILABLE, dtype=np.uint8)
    for hand in range(2):
        anchors = original_times[original_mask[:, hand]]
        if not len(anchors):
            continue
        status[:, hand] = INTERPOLATED
        status[(query < anchors[0]) | (query > anchors[-1]), hand] = HELD
        at = np.searchsorted(anchors, query)
        lo, hi = np.clip(at-1, 0, len(anchors)-1), np.clip(at, 0, len(anchors)-1)
        exact = np.minimum(abs(query-anchors[lo]), abs(query-anchors[hi])) <= 1e-9
        status[exact, hand] = OBSERVED
    return status


def world_to_camera_rotations(path, timestamps):
    """Numeric camera-to-world bridge on the same relative clock as body export."""
    with np.load(path, allow_pickle=False) as data:
        if "camera_world_to_camera" not in data:
            return None
        rotations = np.asarray(data["camera_world_to_camera"], dtype=float)
        times = _times(data["timestamps"], "camera timestamps")
    if (rotations.shape != (len(times), 3, 3) or not np.isfinite(rotations).all()
            or not np.allclose(rotations @ rotations.swapaxes(-1, -2), np.eye(3), atol=1e-4)
            or not np.allclose(np.linalg.det(rotations), 1., atol=1e-4)):
        raise ValueError("Invalid saved world-to-camera rotations for hand orientation")
    times = times-times[0]
    if len(times) == 1:
        return np.repeat(rotations, len(timestamps), axis=0)
    return Slerp(times, Rotation.from_matrix(rotations))(np.clip(timestamps, times[0], times[-1])).as_matrix()
