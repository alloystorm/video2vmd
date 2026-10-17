"""Optional body-guided WiLoR extraction, with numeric caches and no MediaPipe.

Configuration, preflight and cached exports do not import inference runtimes.
Camera observations live in the body's staged-image pixels; crops are scaled to
the upright source media. Video samples use presentation timestamps, not a
second assumed frame rate. Still images are observed once and held explicitly.
"""
from dataclasses import dataclass
import hashlib
import importlib.util
import json
from pathlib import Path
import time

import numpy as np

from . import paths
from .hand_motion import complete_wilor_observations, load_hand_motion, save_hand_motion


@dataclass(frozen=True)
class HandConfig:
    repo: Path
    extras: Path
    weights: Path
    mano: Path

    @classmethod
    def from_options(cls, *, repo=None, extras=None, weights=None, mano=None):
        default_mano = paths.BODY_MODELS/"mano/MANO_RIGHT.pkl"
        if not default_mano.is_file():
            default_mano = paths.OUTPUTS/"dependencies/wilor_mano/MANO_RIGHT.pkl"
        return cls(Path(repo or paths.OUTPUTS/"dependencies/WiLoR").resolve(),
                   Path(extras or paths.OUTPUTS/"dependencies/wilor_extra").resolve(),
                   Path(weights or paths.WEIGHTS/"wilor").resolve(), Path(mano or default_mano).resolve())

    def files(self):
        return {"checkpoint": self.weights/"wilor_final.ckpt", "detector": self.weights/"detector.pt",
                "mano": self.mano, "config": self.repo/"pretrained_models/model_config.yaml",
                "mean_params": self.repo/"mano_data/mano_mean_params.npz"}


def check_hand_runtime(config):
    """Return actionable setup problems without loading checkpoints or Torch."""
    problems = [f"Missing hand {name}: {path}" for name, path in config.files().items() if not path.is_file()]
    if not (config.repo/"wilor/models/__init__.py").is_file():
        problems.append(f"Missing official WiLoR source: {config.repo}; set --wilor-repo")
    if not config.extras.is_dir():
        problems.append(f"Missing WiLoR dependency directory: {config.extras}; set --wilor-extras")
    for module in ("torch", "cv2", "av"):
        if importlib.util.find_spec(module) is None:
            problems.append(f"Missing hand extraction Python module: {module}")
    return problems


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_hand_source(skeleton, source=None):
    with np.load(skeleton, allow_pickle=False) as data:
        recorded = [Path(str(data[key])) for key in ("image", "video") if key in data]
    if source is not None:
        selected = Path(source).resolve()
    else:
        selected = next((path.resolve() for path in recorded if path.is_file()), None)
    if selected is None or not selected.is_file():
        raise FileNotFoundError("--hands needs the original source video/image; pass --video or --image with --skeleton")
    if not (paths.is_image(selected) or selected.suffix.lower() in paths.VIDEO_SUFFIXES):
        raise ValueError("--hands source must be a supported video or image")
    # For a still, the recorded video is a deleted temporary hold clip.
    original = next((path.resolve() for path in recorded if path.is_file()), None)
    if original is not None and selected != original and sha256(selected) != sha256(original):
        raise ValueError("--hands source differs from the body cache's original media; use the matching skeleton/source")
    return selected


def load_hand_body(skeleton):
    required = ("camera_joints_3d", "camera_intrinsics", "camera_image_size", "camera_world_to_camera",
                "timestamps", "joint_names")
    with np.load(skeleton, allow_pickle=False) as data:
        if not set(required) <= set(data.files):
            raise ValueError("--hands needs saved body camera geometry; use a new extraction or scripts/cache_inspection_camera.py")
        body = {key: np.array(data[key]) for key in required}
    times = np.asarray(body["timestamps"], dtype=float)
    n = len(times)
    points, k, size, r = (body[key] for key in required[:4])
    names = list(body["joint_names"])
    if (times.ndim != 1 or n == 0 or np.any(np.diff(times) <= 0) or not np.isfinite(times).all()
            or points.shape != (n, len(names), 3) or k.shape not in ((3, 3), (n, 3, 3))
            or size.shape != (2,) or np.any(size <= 0) or r.shape != (n, 3, 3)
            or not all(np.isfinite(a).all() for a in (points, k, size, r))
            or not np.allclose(r @ r.swapaxes(-1, -2), np.eye(3), atol=1e-4)
            or not np.allclose(np.linalg.det(r), 1., atol=1e-4)
            or np.any(k[..., [0, 1], [0, 1]] <= 0)
            or any(side+"_"+part not in names for side in ("left", "right") for part in ("elbow", "wrist"))):
        raise ValueError("Invalid saved body camera geometry for --hands")
    body["timestamps"] = times-times[0]
    return body


def body_hand_crops(body, index, image_size):
    """Use the established wrist + quarter-forearm crop and wrist-distance gate."""
    camera, intrinsics, size = (body[key] for key in ("camera_joints_3d", "camera_intrinsics", "camera_image_size"))
    k = intrinsics if intrinsics.ndim == 2 else intrinsics[index]
    width, height = image_size
    if abs((width/height)/(size[0]/size[1])-1) > .02:
        raise ValueError("Hand source orientation/aspect differs from body camera geometry")
    projected = camera[index] @ k.T
    xy = np.divide(projected[:, :2], projected[:, 2, None], out=np.full_like(projected[:, :2], np.nan),
                   where=projected[:, 2, None] > 1e-6)*np.array(image_size)/size
    names = list(body["joint_names"])
    crops = []
    for side in ("left", "right"):
        wrist, elbow = (xy[names.index(side+"_"+part)] for part in ("wrist", "elbow"))
        row = {"side": side, "crop_valid": False, "crop_match": None, "full_match": None, "crop_candidates": []}
        if not np.isfinite([wrist, elbow]).all():
            row["rejection"] = "body_wrist_or_elbow_behind_camera"
        else:
            forearm = float(np.linalg.norm(wrist-elbow))
            center, radius = wrist+.25*(wrist-elbow), max(48., .85*forearm)
            low = np.clip(np.floor(center-radius), 0, image_size).astype(int)
            high = np.clip(np.ceil(center+radius), 0, image_size).astype(int)
            row.update(body_wrist_xy=wrist.tolist(), forearm_px=forearm, gate_px=max(20., .6*forearm))
            if np.any(high <= low):
                row["rejection"] = "hand_crop_outside_image"
            else:
                row.update(crop_valid=True, crop_xyxy=[int(low[0]), int(low[1]), int(high[0]), int(high[1])])
        crops.append(row)
    return crops


def associate_crops(crops):
    for row in crops:
        candidates = row["crop_candidates"]
        if not row["crop_valid"]:
            continue
        distances = [np.linalg.norm(np.asarray(c["image_xy_px"])[0]-row["body_wrist_xy"]) for c in candidates]
        if distances and min(distances) <= row["gate_px"]:
            row["crop_match"] = int(np.argmin(distances))
        else:
            row["rejection"] = "no_detection" if not distances else "outside_body_wrist_gate"
    if any(row["crop_match"] is None for row in crops):
        return
    a, b = (row["crop_candidates"][row["crop_match"]] for row in crops)
    # The same physical hand can be detected in both overlapping crops. Reject
    # ambiguous assignments, retaining candidates and reasons for audit.
    box_a, box_b = np.array(a["detector_xyxy_px"]), np.array(b["detector_xyxy_px"])
    intersection = np.prod(np.maximum(0., np.minimum(box_a[2:], box_b[2:])-np.maximum(box_a[:2], box_b[:2])))
    union = np.prod(box_a[2:]-box_a[:2])+np.prod(box_b[2:]-box_b[:2])-intersection
    separation = np.linalg.norm(np.array(a["image_xy_px"])[0]-np.array(b["image_xy_px"])[0])
    if union > 0 and intersection/union > .5 and separation < max(12., .2*min(row["forearm_px"] for row in crops)):
        for row in crops:
            row.update(crop_match=None, rejection="ambiguous_duplicate_hand")


def _source_frames(source, times):
    """Yield upright BGR frames nearest each requested presentation time."""
    if paths.is_image(source):
        import cv2
        frame = cv2.imread(str(source))
        if frame is None:
            raise ValueError(f"Cannot decode hand source image: {source}")
        yield frame, 0., 0
        return
    import av
    from gvhmr.utils.video_io_utils import get_video_rotation
    rotation = get_video_rotation(source)
    with av.open(str(source)) as container:
        if not container.streams.video:
            raise ValueError(f"No video stream in hand source: {source}")
        stream = container.streams.video[0]
        stream.thread_type = "AUTO"
        decoded = iter(container.decode(stream))
        origin, counter = None, 0
        def read():
            nonlocal origin, counter
            frame = next(decoded, None)
            if frame is None:
                return None
            if frame.time is None:
                raise ValueError("Hand source video needs presentation timestamps")
            stamp = float(frame.time)
            if origin is None:
                origin = stamp
            result = (frame, stamp-origin, counter)
            counter += 1
            return result
        previous, following = None, read()
        if following is None:
            raise ValueError(f"Cannot decode hand source video: {source}")
        for target in times:
            while following is not None and following[1] < target:
                previous, following = following, read()
            choices = [item for item in (previous, following) if item is not None]
            selected = min(choices, key=lambda item: abs(item[1]-target))
            frame = selected[0].to_ndarray(format="bgr24")
            if rotation:
                frame = np.ascontiguousarray(np.rot90(frame, rotation))
            yield frame, selected[1], selected[2]


def _cache_identity(source, body, config):
    digest = hashlib.sha256()
    for key, array in sorted(body.items()):
        digest.update(key.encode()); digest.update(array.dtype.str.encode()); digest.update(str(array.shape).encode())
        digest.update(array.tobytes())
    code = hashlib.sha256()
    for path in sorted((config.repo/"wilor").rglob("*.py")):
        code.update(path.relative_to(config.repo).as_posix().encode()); code.update(path.read_bytes())
    for filename in ("hand_extract.py", "wilor_backend.py", "hand_motion.py"):
        code.update(Path(__file__).with_name(filename).read_bytes())
    return {"version": 1, "source": str(source), "source_sha256": sha256(source),
            "body_camera_sha256": digest.hexdigest(), "implementation_sha256": code.hexdigest(),
            "models": {name: sha256(path) for name, path in config.files().items()}}


def extract_hand_motion(skeleton, cache_dir, *, source=None, config=None, device="cpu", progress=None):
    """Extract once, or reuse an exactly matching completed hand cache."""
    config = config or HandConfig.from_options()
    problems = check_hand_runtime(config)
    if problems:
        raise FileNotFoundError("; ".join(problems)+". See docs/hand-pose.md for WiLoR setup.")
    source = resolve_hand_source(skeleton, source)
    body = load_hand_body(skeleton)
    identity = _cache_identity(source, body, config)
    cache_dir = Path(cache_dir)
    manifest, cache = cache_dir/"identity.json", cache_dir/"hand_motion.npz"
    if manifest.is_file() and json.loads(manifest.read_text(encoding="utf-8")) != identity:
        raise ValueError("Hand extraction cache belongs to different media, camera geometry or models; use a fresh output directory")
    if manifest.is_file() and cache.is_file() and (cache_dir/"extraction.json").is_file():
        load_hand_motion(cache)
        report = json.loads((cache_dir/"extraction.json").read_text(encoding="utf-8"))
        return cache, {**report, "reused": True}
    cache_dir.mkdir(parents=True, exist_ok=True)
    manifest.write_text(json.dumps(identity, indent=2)+"\n", encoding="utf-8")
    from .wilor_backend import WiLoRBackend
    started = time.monotonic()
    backend = WiLoRBackend(config.repo, config.extras, config.weights, config.mano, device)
    times, rows = body["timestamps"], []
    total = 1 if paths.is_image(source) else len(times)
    for index, (frame, decoded_time, decoded_index) in enumerate(_source_frames(source, times)):
        crops = body_hand_crops(body, index, frame.shape[1::-1])
        for row in crops:
            if row["crop_valid"]:
                x0, y0, x1, y1 = row["crop_xyxy"]
                row["crop_candidates"] = backend.detect(frame[y0:y1, x0:x1], (x0, y0))
        associate_crops(crops)
        rows.append({"time_s": float(times[index]), "source_time_s": decoded_time, "frame_index": decoded_index,
                     "full_candidates": [], "hands": crops})
        if progress is not None and (index % 30 == 0 or index+1 == total):
            progress(index+1, total)
    if len(rows) != total:
        raise ValueError("Hand source decoder did not produce the requested timeline")
    document = {"clip": source.stem, "backend": "wilor", "source_video": str(source),
                "source_sha256": identity["source_sha256"], "observations": rows,
                "mano_rest_joints_m": backend.rest_joints().tolist()}
    observations = cache_dir/"observations.json"
    observations.write_text(json.dumps(document)+"\n", encoding="utf-8")
    payload = complete_wilor_observations(document, target_timestamps=times)
    pending = cache_dir/"hand_motion.pending.npz"
    save_hand_motion(pending, payload)
    load_hand_motion(pending)
    pending.replace(cache)
    report = {"enabled": True, "backend": "wilor", "reused": False, "source": str(source), "device": device,
              "frames": len(times), "observation_frames": total, "elapsed_s": round(time.monotonic()-started, 2),
              "sampling": "nearest source presentation timestamp; upright source pixels; still observed once",
              "association": "body-side wrist gate; ambiguous duplicate hands rejected",
              "models": identity["models"], "source_sha256": identity["source_sha256"],
              "counts": {side: {name: int((payload["status"][:, hand] == code).sum())
                                for code, name in enumerate(("unavailable", "observed", "interpolated", "held"))}
                         for hand, side in enumerate(("left", "right"))}}
    (cache_dir/"extraction.json").write_text(json.dumps(report, indent=2)+"\n", encoding="utf-8")
    return cache, report
