"""Target motion contract: meter translations, xyzw local rotations, seconds."""
from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from .profile import MMDProfile


@dataclass
class TargetMotion:
    profile: MMDProfile
    rotations: np.ndarray  # (T,B,4), xyzw
    translations: np.ndarray  # (T,B,3), animated local offsets, meters
    fps: float = 30.0

    def __post_init__(self):
        self.rotations = np.asarray(self.rotations, dtype=np.float64)
        self.translations = np.asarray(self.translations, dtype=np.float64)
        count = len(self.profile.names)
        if (self.rotations.ndim != 3 or self.rotations.shape[1:] != (count, 4)
                or len(self.rotations) == 0
                or self.translations.shape != self.rotations.shape[:2] + (3,)):
            raise ValueError("Target motion must have nonempty (frames,bones,4)/(frames,bones,3) arrays")
        if not np.isfinite(self.fps) or self.fps <= 0:
            raise ValueError("Target FPS must be finite and positive")
        if not np.isfinite(self.rotations).all() or not np.isfinite(self.translations).all():
            raise ValueError("Target transforms must be finite")
        if not np.allclose(np.linalg.norm(self.rotations, axis=2), 1., atol=1e-6):
            raise ValueError("Target rotations must be unit xyzw quaternions")

    @property
    def n_frames(self):
        return len(self.rotations)

    def fk(self):
        n, count = self.rotations.shape[:2]
        local = Rotation.from_quat(self.rotations.reshape(-1, 4)).as_matrix().reshape(n, count, 3, 3)
        positions = np.zeros((n, count, 3))
        world = np.zeros((n, count, 3, 3))
        for i, parent in enumerate(self.profile.parents):
            offset = self.profile.offsets[i] + self.translations[:, i]
            if parent < 0:
                positions[:, i] = offset
                world[:, i] = local[:, i]
            else:
                positions[:, i] = positions[:, parent] + np.einsum("tij,tj->ti", world[:, parent], offset)
                world[:, i] = world[:, parent] @ local[:, i]
        return positions, world

    def save(self, path: Path):
        fitted = {self.profile.root, "HipMaster", "Torso", "Torso2", "Neck"}
        fitted.update(side+part for side in ("Left", "Right") for part in
                      ("Shoulder", "Arm", "Elbow", "Wrist", "Leg", "Knee", "Ankle"))
        np.savez_compressed(path, version=np.int32(1), bone_names=np.array(self.profile.names),
                            parents=self.profile.parents, rest_positions_m=self.profile.rest,
                            quat_xyzw=self.rotations, translations_m=self.translations,
                            timestamps=np.arange(self.n_frames) / self.fps, fps=self.fps,
                            observed=np.array([n in fitted for n in self.profile.names]),
                            metadata=json.dumps({"profile": self.profile.name, "profile_hash": self.profile.identity,
                                                 "meters_per_vmd_unit": self.profile.unit, "quaternion": "xyzw",
                                                 "up": "+Y", "unobserved_hands": True,
                                                 "neck_policy": "rest_relative_chest_to_head_axis",
                                                 "head_orientation": "inherits_neck_no_independent_observation",
                                                 "observed_semantics": "fitted_from_body_landmarks_not_full_rotation_observations"}))


def resample_source(motion, fps=30.0):
    times = np.asarray(motion.timestamps, dtype=float)
    if times.ndim != 1 or len(times) != motion.n_frames or not np.isfinite(times).all() or (len(times) > 1 and np.any(np.diff(times) <= 0)):
        raise ValueError("Source timestamps must be finite and strictly increasing")
    if (motion.n_frames == 0 or not np.isfinite(motion.joints_3d).all()
            or not np.isfinite(motion.fps) or motion.fps <= 0 or not np.isfinite(fps) or fps <= 0):
        raise ValueError("Source motion must contain finite joints and a positive FPS")
    if (motion.joints_3d.shape != (motion.n_frames, len(motion.joint_names), 3)
            or len(set(motion.joint_names)) != len(motion.joint_names)):
        raise ValueError("Source must contain unique joint names and (frames,joints,3) positions")
    times = times - times[0]
    duration = times[-1] + (np.median(np.diff(times)) if len(times) > 1 else 1 / motion.fps)
    count = max(1, int(round(duration * fps)))
    target = np.arange(count) / fps
    flat = motion.joints_3d.reshape(motion.n_frames, -1)
    values = np.column_stack([np.interp(target, times, column) for column in flat.T])
    return values.reshape(count, -1, 3)
