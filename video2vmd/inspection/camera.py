"""Numeric-only bridge from saved GVHMR camera observations to viewer buffers."""
from dataclasses import dataclass

import numpy as np
from scipy.spatial.transform import Rotation, Slerp

from ..extract import SMPL_JOINT_NAMES

SMPL_PARENTS = [-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19, 20, 21]


@dataclass
class CameraInspection:
    metadata: dict
    targets: bytes
    reconstruction: bytes


def world_reconstruction(path, frames, fps):
    """The recovered source body is also inspectable without any camera estimate."""
    with np.load(path, allow_pickle=False) as data:
        if not {"joints_3d","joint_names","timestamps"} <= set(data.files):
            return None
        points=np.asarray(data["joints_3d"],dtype=float)
        times=np.asarray(data["timestamps"],dtype=float)
        names=[str(n) for n in data["joint_names"]]
    if (names != SMPL_JOINT_NAMES or points.shape != (len(times),24,3) or len(times)==0
            or not np.isfinite(points).all() or not np.isfinite(times).all()
            or (len(times)>1 and np.any(np.diff(times)<=0))):
        raise ValueError("Invalid source reconstruction arrays")
    points=points*[-1,1,1]  # Same display basis as target FK.
    target=np.arange(frames)/fps
    values=np.column_stack([np.interp(target,times-times[0],c) for c in points.reshape(len(times),-1).T]).reshape(frames,24,3)
    floor=float(np.percentile(values[:,[10,11],1],5))
    return {"names":names,"parents":SMPL_PARENTS,"root":0,"floor":floor,
            "height":max(.5,float(np.ptp(values[0,:,1]))+.18)},values.astype("<f4").tobytes()


def camera_inspection(path, motions):
    """Old NPZs have no camera geometry; manual overlap remains available."""
    with np.load(path, allow_pickle=False) as data:
        keys = {"camera_joints_3d", "camera_intrinsics", "camera_image_size", "camera_world_to_camera"}
        if not keys <= set(data.files):
            return None
        joints = np.asarray(data["camera_joints_3d"], dtype=float)
        rotations = np.asarray(data["camera_world_to_camera"], dtype=float)
        intrinsics = np.asarray(data["camera_intrinsics"], dtype=float)
        size = np.asarray(data["camera_image_size"], dtype=float)
        world = np.asarray(data["joints_3d"], dtype=float)
        names = [str(n) for n in data["joint_names"]]
        times = np.asarray(data["timestamps"], dtype=float)
    n = len(joints)
    if (n == 0 or names != SMPL_JOINT_NAMES or joints.shape != (n, 24, 3)
            or world.shape != joints.shape or rotations.shape != (n, 3, 3)
            or intrinsics.shape not in ((3, 3), (n, 3, 3)) or size.shape != (2,)
            or times.shape != (n,) or not all(np.isfinite(a).all() for a in
                (joints, rotations, intrinsics, size, times, world))
            or np.any(size <= 0) or (n > 1 and np.any(np.diff(times) <= 0))):
        raise ValueError("Invalid saved inspection camera arrays")
    if not np.allclose(rotations @ rotations.swapaxes(1,2), np.eye(3), atol=1e-4) or not np.allclose(np.linalg.det(rotations), 1, atol=1e-4):
        raise ValueError("Invalid world-to-camera rotations")
    if intrinsics.ndim == 2:
        intrinsics = np.repeat(intrinsics[None], n, axis=0)
    if np.any(intrinsics[:, [0,1], [0,1]] <= 0):
        raise ValueError("Invalid camera focal length")
    times = times-times[0]
    reference = motions.get("raw", motions["corrected"])
    target_times = np.arange(reference.n_frames) / reference.fps
    def resample(values):
        flat = values.reshape(n, -1)
        return np.column_stack([np.interp(target_times, times, column) for column in flat.T]).reshape((len(target_times),)+values.shape[1:])
    cam = resample(joints)
    k = resample(intrinsics)
    r = (Slerp(times, Rotation.from_matrix(rotations))(np.clip(target_times, times[0], times[-1])).as_matrix()
         if n > 1 else np.repeat(rotations, len(target_times), axis=0))
    source_lengths, target_lengths = [], []
    for side, lower in (("Left","left"), ("Right","right")):
        hip, knee, ankle = [names.index(lower+"_"+part) for part in ("hip","knee","ankle")]
        source_lengths.append(np.median(np.linalg.norm(world[:,hip]-world[:,knee],axis=1)+np.linalg.norm(world[:,knee]-world[:,ankle],axis=1)))
        a,b,c = [reference.profile.position(side+part) for part in ("Leg","Knee","Ankle")]
        target_lengths.append(np.linalg.norm(a-b)+np.linalg.norm(b-c))
    scale = np.mean(target_lengths) / np.mean(source_lengths)
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("Degenerate camera alignment skeleton")
    anchor = reference.fk()[0][:, reference.profile.index["HipMaster"]]
    targets = []
    for motion in motions.values():
        # Undo MMD's X reflection and retarget scale. Anchor both variants to
        # the SAME raw pelvis so corrected root displacement stays visible.
        local = (motion.fk()[0]-anchor[:,None]) * [-1,1,1] / scale
        targets.append((np.einsum("tij,tbj->tbi", r, local)+cam[:,0,None]).astype("<f4").tobytes())
    normalized = k[:, [0,1,0,1], [0,1,2,2]] / size[[0,1,0,1]]
    return CameraInspection({
        "intrinsics": normalized.tolist(), "convention": "OpenCV +X right, +Y down, +Z forward",
        "names": names, "parents": SMPL_PARENTS,
        "fov": float(np.degrees(2*np.arctan(.5/np.median(normalized[:,1])))),
        "distance": float(max(.2, np.median(cam[:,0,2]))),
        "alignment": "estimated camera; target proportions aligned to raw pelvis and source leg scale",
    }, b"".join(targets), cam.astype("<f4").tobytes())
