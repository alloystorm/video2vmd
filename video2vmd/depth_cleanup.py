"""Conservative, offline root-depth regularization with image/contact guards.

Works on combined metric pelvis depth, never on crop size or weak-perspective
scale separately. All joints receive one translation; articulation is preserved.
"""
from dataclasses import replace
import json
from pathlib import Path

import numpy as np
from scipy import sparse
from scipy.ndimage import gaussian_filter1d, uniform_filter1d
from scipy.signal import butter, sosfiltfilt
from scipy.sparse.linalg import spsolve

from .human_motion import load_human_motion

CAMERA_KEYS = {"camera_joints_3d", "camera_world_to_camera", "camera_intrinsics", "camera_image_size"}


def check_depth_input(path):
    try:
        with np.load(path, allow_pickle=False) as data:
            if not CAMERA_KEYS <= set(data.files):
                return ("--depth-cleanup needs saved camera observations. Use a new extraction, or add them with "
                        "scripts/cache_inspection_camera.py using this run's matching GVHMR cache.")
            if str(data.get("camera", "")) != "static":
                return "--depth-cleanup currently requires a static-camera extraction"
            if "depth_cleanup" in data:
                return "This skeleton already has depth cleanup; use the original skeleton_motion.npz to avoid filtering twice"
    except (OSError, ValueError) as exc:
        return str(exc)
    return None


def _project(points, intrinsics):
    pixels = np.einsum("tij,tbj->tbi", intrinsics, points)
    return pixels[:,:,:2] / pixels[:,:,2,None]


def _jitter(points, fps):
    if len(points) < 16:
        return None
    cutoff = min(3., fps*.2)
    try:
        residual = sosfiltfilt(butter(4,cutoff,fs=fps,btype="highpass",output="sos"), points,axis=0)
    except ValueError:
        return None
    trim=min(round(fps),len(points)//5)
    residual=residual[trim:-trim] if trim else residual
    return float(np.sqrt(np.mean(np.sum(residual**2,axis=1))))


def regularize_depth(human, camera, *, contacts=None, contact_fps=30.):
    """Return corrected HumanMotion, per-frame evidence, and a JSON-safe report.

    The data term resists edits, acceleration regularization adapts to coherent
    travel, and stable foot samples constrain horizontal contact velocity.
    Corrections are bounded and reprojected against the original camera joints.
    """
    points=np.asarray(human.joints_3d,dtype=float)
    times=np.asarray(human.timestamps,dtype=float)
    n=len(points)
    cam=np.asarray(camera["camera_joints_3d"],dtype=float)
    rotations=np.asarray(camera["camera_world_to_camera"],dtype=float)
    k=np.asarray(camera["camera_intrinsics"],dtype=float)
    size=np.asarray(camera["camera_image_size"],dtype=float)
    if (n==0 or not np.isfinite(human.fps) or human.fps<=0
            or points.shape!=(n,len(human.joint_names),3) or cam.shape!=points.shape
            or rotations.shape!=(n,3,3) or times.shape!=(n,) or size.shape!=(2,)
            or k.shape not in ((3,3),(n,3,3))
            or not all(np.isfinite(a).all() for a in (points,times,cam,rotations,k,size))
            or np.any(size<=0) or (n>1 and np.any(np.diff(times)<=0))):
        raise ValueError("Invalid source motion/camera arrays for depth cleanup")
    if (not np.allclose(rotations @ rotations.swapaxes(1,2),np.eye(3),atol=1e-4)
            or not np.allclose(np.linalg.det(rotations),1,atol=1e-4)):
        raise ValueError("Invalid world-to-camera rotations for depth cleanup")
    if k.ndim==2:
        k=np.repeat(k[None],n,axis=0)
    if np.any(k[:,[0,1],[0,1]]<=0) or np.any(cam[:,:,2]<=.1):
        raise ValueError("Depth cleanup needs positive focal lengths and joints in front of the camera")
    pelvis=human.index("pelvis")
    feet=[human.index(name) for name in ("left_ankle","left_foot","right_ankle","right_foot")]
    fps=1/np.median(np.diff(times)) if n>1 else human.fps
    if not np.isfinite(fps) or fps<=0:
        raise ValueError("Depth cleanup needs positive FPS")
    z=cam[:,pelvis,2]
    # Translation along the pelvis viewing ray keeps its image location exact.
    ray=cam[:,pelvis]/z[:,None]
    world_ray=np.einsum("tji,tj->ti",rotations,ray)
    support=np.zeros((n,4))
    if contacts is not None:
        values=np.asarray(contacts["weights"],dtype=float)
        if (values.ndim!=2 or values.shape[1]!=4 or len(values)==0
                or not np.isfinite(values).all() or np.any((values<0)|(values>1))
                or not np.isfinite(contact_fps) or contact_fps<=0):
            raise ValueError("Invalid foot-contact evidence for depth cleanup")
        contact_times=np.arange(len(values))/contact_fps
        support=np.column_stack([np.interp(times-times[0],contact_times,v) for v in values.T])
    delta=np.zeros(n)
    gate=np.zeros(n)
    contact_pairs=0
    if n>=max(8,round(fps*.4)):
        # Motion adaptation is measured on a modestly denoised, uniform grid.
        uniform_times=np.linspace(times[0],times[-1],max(n,round((times[-1]-times[0])*fps)+1))
        uniform_z=np.interp(uniform_times,times,z)
        velocity=np.gradient(gaussian_filter1d(uniform_z,max(1.,fps*.06)),uniform_times)
        difference=np.gradient(uniform_z,uniform_times)
        window=max(3,round(fps*.17))
        coherence=np.abs(uniform_filter1d(difference,window))/(uniform_filter1d(np.abs(difference),window)+1e-9)
        uniform_gate=np.clip(coherence,0,1)*np.clip((np.abs(velocity)-.12)/.5,0,1)
        gate=np.interp(times,uniform_times,uniform_gate)
        # Nonuniform second differences, normalized to the source frame rate.
        h=np.diff(times)*fps
        left=2/(h[:-1]*(h[:-1]+h[1:]))
        middle=-2/(h[:-1]*h[1:])
        right=2/(h[1:]*(h[:-1]+h[1:]))
        rows=np.repeat(np.arange(n-2),3)
        cols=np.column_stack([np.arange(n-2),np.arange(1,n-1),np.arange(2,n)]).ravel()
        d2=sparse.coo_matrix((np.column_stack([left,middle,right]).ravel(),(rows,cols)),shape=(n-2,n)).tocsr()
        cutoff=min(3.,fps*.2)
        lam=(np.sqrt(2)-1)/(16*np.sin(np.pi*cutoff/fps)**4)
        adaptive=np.sqrt(lam*(1-.85*gate[1:-1]))
        acceleration=sparse.diags(adaptive)@d2
        blocks=[sparse.eye(n,format="csr"),acceleration]
        observations=[np.zeros(n),-acceleration@z]
        # A contact is evidence, not a command to freeze the pelvis. Penalize
        # only horizontal motion of the same supported foot across neighbors.
        cr=[];cc=[];cv=[];target=[]
        for foot_num,index in enumerate(feet):
            weights=np.sqrt(support[1:,foot_num]*support[:-1,foot_num])*2
            for t in np.flatnonzero(weights>0)+1:
                contact_pairs+=1
                for axis in (0,2):
                    row=len(target);weight=weights[t-1]
                    cr.extend([row,row]);cc.extend([t-1,t])
                    cv.extend([-weight*world_ray[t-1,axis],weight*world_ray[t,axis]])
                    target.append(-weight*(points[t,index,axis]-points[t-1,index,axis]))
        if target:
            blocks.append(sparse.coo_matrix((cv,(cr,cc)),shape=(len(target),n)).tocsr())
            observations.append(np.asarray(target))
        matrix=sparse.vstack(blocks,format="csr")
        delta=spsolve((matrix.T@matrix).tocsc(),matrix.T@np.concatenate(observations))
        # Smooth saturation, then a smooth endpoint taper: no abrupt clipping
        # and no change to the clip's initial/final root position.
        delta=.04*np.tanh(delta/.04)
        taper=np.clip(np.minimum(times-times[0],times[-1]-times)/.2,0,1)
        delta*=taper*taper*(3-2*taper)
    original_pixels=_project(cam,k)
    # Budgets are fractions of image height, so proxies/resolution changes
    # cannot change the accepted correction. Tighten smoothly until satisfied.
    for _ in range(12):
        shifted=cam+delta[:,None,None]*ray[:,None]
        errors=np.linalg.norm(_project(shifted,k)-original_pixels,axis=2)/size[1]
        rms=np.sqrt(np.mean(errors**2,axis=1));maximum=errors.max(axis=1)
        scale=np.minimum(1,np.minimum(.001/(rms+1e-15),.002/(maximum+1e-15)))
        if np.all(scale>=1-1e-10):
            break
        delta*=np.where(scale<1,scale*.995,1)
    delta[np.abs(delta)<1e-12]=0
    delta_cam=delta[:,None]*ray
    delta_world=delta[:,None]*world_ray
    result=replace(human,joints_3d=points+delta_world[:,None],
                   root_translation=None if human.root_translation is None else human.root_translation+delta_world)
    cam_corrected=cam+delta_cam[:,None]
    errors=np.linalg.norm(_project(cam_corrected,k)-original_pixels,axis=2)/size[1]
    report={"method":"adaptive_metric_pelvis_depth_v1","frames":n,"fps":float(fps),
            "base_cutoff_hz":min(3.,float(fps)*.2),"max_depth_edit_m":.04,
            "reprojection_rms_budget_image_height":.001,"reprojection_max_budget_image_height":.002,
            "max_depth_correction_m":float(np.max(np.abs(delta))),
            "rms_depth_correction_m":float(np.sqrt(np.mean(delta**2))),
            "max_world_correction_m":float(np.linalg.norm(delta_world,axis=1).max()),
            "max_reprojection_rms_image_height":float(np.sqrt(np.mean(errors**2,axis=1)).max()),
            "max_reprojection_joint_image_height":float(errors.max()),
            "mean_reprojection_rms_pixels":float(np.sqrt(np.mean(errors**2,axis=1)).mean()*size[1]),
            "contact_pairs":contact_pairs,"motion_protection_mean":float(gate.mean()),
            "root_high_frequency_rms_before_m":_jitter(points[:,pelvis],fps),
            "root_high_frequency_rms_after_m":_jitter(result.joints_3d[:,pelvis],fps),
            "camera_depth_high_frequency_rms_before_m":_jitter(cam[:,pelvis,2,None],fps),
            "camera_depth_high_frequency_rms_after_m":_jitter(cam_corrected[:,pelvis,2,None],fps),
            "scope":"one translation per frame; local pose, timestamps and camera estimates unchanged",
            "limits":"high-frequency true motion can also be attenuated; enabled explicitly"}
    diagnostics={"timestamps":times,"depth_before_m":z,"depth_after_m":z+delta,
                 "delta_world_m":delta_world,"delta_camera_m":delta_cam,
                 "motion_protection":gate,"contact_weights":support,
                 "reprojection_rms_image_height":np.sqrt(np.mean(errors**2,axis=1))}
    return result,cam_corrected,diagnostics,report


def clean_skeleton(source: Path, destination: Path, *, contacts=None):
    if Path(source).resolve()==Path(destination).resolve():
        raise ValueError("Depth cleanup must write a separate corrected skeleton; raw extraction is preserved")
    problem=check_depth_input(source)
    if problem:
        raise ValueError(problem)
    human=load_human_motion(source)
    with np.load(source,allow_pickle=False) as data:
        payload={key:data[key] for key in data.files}
    result,cam,diagnostics,report=regularize_depth(human,payload,contacts=contacts)
    payload["joints_3d"]=result.joints_3d
    payload["camera_joints_3d"]=cam
    if result.root_translation is not None:
        payload["root_translation"]=result.root_translation
    payload["depth_cleanup"]=np.array(json.dumps(report))
    destination=Path(destination)
    np.savez_compressed(destination,**payload)
    result.path=destination
    return result,diagnostics,report
