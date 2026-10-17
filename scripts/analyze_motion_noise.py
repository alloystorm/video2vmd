"""Analyze cached hip/feet trajectories and four processing ablations.

No estimator imports. Frequencies are motion/noise proxies, never ground truth.
Example: python scripts/analyze_motion_noise.py outputs/diagnostics/noise_study_20261017
"""
import argparse
import csv
import html
import json
from pathlib import Path
import sys

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import butter, sosfiltfilt, welch
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from video2vmd.inspection.server import load_target
from video2vmd.inspection.camera import camera_inspection
from video2vmd.mmd.contacts import foot_points, quality, runs

STAGES = ("raw", "depth", "feet", "combined")
LABELS = {"raw":"Raw", "depth":"Depth only", "feet":"Feet only", "combined":"Depth + feet"}
COLORS = {"raw":"#8b8e98", "depth":"#cf8a27", "feet":"#7961b2", "combined":"#008e9c"}
BONES = ("HipMaster", "LeftAnkle", "RightAnkle")
BONE_LABELS = ("Hip", "Left foot (ankle)", "Right foot (ankle)")
plt.rcParams.update({"font.size":10, "axes.spines.top":False, "axes.spines.right":False,
                     "axes.grid":True, "grid.alpha":.18, "savefig.dpi":150})


def highpass(values, fps, cutoff=3.):
    """Zero-phase filter; at least 1 s is excluded from reported long-clip RMS."""
    return sosfiltfilt(butter(4, cutoff, btype="highpass", fs=fps, output="sos"), values, axis=0)


def interior(count, fps):
    trim = min(round(fps), count//5)
    return slice(trim, count-trim)


def vector_rms(values):
    return float(np.sqrt(np.mean(np.sum(np.asarray(values)**2, axis=-1))))


def correlation(a, b):
    if min(np.std(a), np.std(b)) < 1e-10:
        return None
    return float(np.corrcoef(a.ravel(), b.ravel())[0, 1])


def trajectory_metrics(points, fps):
    mid = interior(len(points), fps)
    result = {"range_m":np.ptp(points, axis=0).tolist(), "highpass_rms_mm":{},
              "velocity_p95_m_s":float(np.percentile(np.linalg.norm(np.diff(points, axis=0)*fps, axis=1),95)),
              "acceleration_rms_m_s2":vector_rms(np.diff(points,n=2,axis=0)*fps**2),
              "jerk_rms_m_s3":vector_rms(np.diff(points,n=3,axis=0)*fps**3)}
    for hz in (2, 3, 5):
        filtered = highpass(points, fps, hz)[mid]
        result["highpass_rms_mm"][str(hz)] = vector_rms(filtered)*1000
        if hz == 3:
            result["highpass_axis_rms_mm"] = (np.sqrt(np.mean(filtered**2,axis=0))*1000).tolist()
            result["lag1_highpass_correlation"] = correlation(filtered[:-1], filtered[1:])
    return result


def savefig(fig, folder, name):
    fig.savefig(folder/(name+".png"), bbox_inches="tight")
    fig.savefig(folder/(name+".svg"), bbox_inches="tight")
    plt.close(fig)


def background_tracking(video, cam, intrinsics, image_size, times, output):
    """Independent 10 Hz background flow; reject person features and weak fits.

    Measures image movement, not metric camera translation. Cumulative affine
    paths drift; invalid fits/cuts are counted instead of interpolated as truth.
    """
    cache = output/"background_tracking.npz"
    if cache.exists():
        return dict(np.load(cache, allow_pickle=False))
    cap = cv2.VideoCapture(str(video))
    native_fps = cap.get(cv2.CAP_PROP_FPS)
    ok, frame = cap.read()
    if not ok:
        raise ValueError(f"Cannot decode {video}")
    scale = min(1., 480/max(frame.shape[:2]))
    size = np.array([round(frame.shape[1]*scale), round(frame.shape[0]*scale)])
    stride = max(1, round(native_fps/10))
    def gray(im):
        return cv2.cvtColor(cv2.resize(im,tuple(size)), cv2.COLOR_BGR2GRAY)
    def box(time):
        i = min(len(times)-1, int(np.searchsorted(times,time)))
        k = intrinsics if intrinsics.ndim == 2 else intrinsics[i]
        projected = cam[i]@k.T
        projected = projected[:,:2]/projected[:,2,None]/image_size*size
        lo, hi = projected.min(axis=0), projected.max(axis=0)
        margin = np.maximum((hi-lo)*[.20,.12],[10,10])
        return np.r_[np.maximum(0,lo-margin),np.minimum(size,hi+margin)].astype(int)
    previous, last_box, index = gray(frame), box(0.), 0
    transform = np.eye(3)
    rows = []
    try:
        while cap.grab():
            index += 1
            if index % stride:
                continue
            ok, frame = cap.retrieve()
            if not ok:
                break
            current = gray(frame)
            now = box(index/native_fps)
            mask = np.full(previous.shape,255,np.uint8)
            cv2.rectangle(mask,tuple(last_box[:2]),tuple(last_box[2:]),0,-1)
            points = cv2.goodFeaturesToTrack(previous,maxCorners=350,qualityLevel=.012,minDistance=6,mask=mask)
            affine, count, n_inliers, residual = None, 0, 0, np.nan
            cut = float(np.median(np.abs(current.astype(float)-previous))) > 45.
            if points is not None and len(points)>=12 and not cut:
                tracked, status, error = cv2.calcOpticalFlowPyrLK(previous,current,points,None,winSize=(21,21),maxLevel=3)
                back, status2, _ = cv2.calcOpticalFlowPyrLK(current,previous,tracked,None,winSize=(21,21),maxLevel=3)
                a, b = points[:,0], tracked[:,0]
                outside = (b[:,0]<now[0])|(b[:,0]>now[2])|(b[:,1]<now[1])|(b[:,1]>now[3])
                valid = (status[:,0]>0)&(status2[:,0]>0)&(np.linalg.norm(back[:,0]-a,axis=1)<.5)&outside&(error[:,0]<20)
                a,b = a[valid], b[valid]; count=len(a)
                if count >= 12:
                    affine, inliers = cv2.estimateAffinePartial2D(a,b,method=cv2.RANSAC,ransacReprojThreshold=.7)
                    if affine is not None:
                        n_inliers = int(inliers.sum())
                        residual = float(np.median(np.linalg.norm((a@affine[:,:2].T+affine[:,2]-b)[inliers[:,0]>0],axis=1)))
            valid = affine is not None and n_inliers>=12 and n_inliers/max(1,count)>=.5
            if valid:
                pair = np.vstack([affine,[0,0,1]])
                center = np.r_[size/2,1.]
                delta = (pair@center-center)[:2]
                angle = np.degrees(np.arctan2(affine[1,0],affine[0,0]))
                zoom = np.hypot(affine[0,0],affine[1,0])
                transform = pair@transform
            else:
                delta = [np.nan,np.nan]; angle=zoom=np.nan
            displacement = (transform@np.r_[size/2,1.]-np.r_[size/2,1.])[:2]
            rows.append([index/native_fps,*delta,angle,zoom,*displacement,count,n_inliers,residual,int(valid),int(cut)])
            previous,last_box=current,now
    finally:
        cap.release()
    result = {"values":np.asarray(rows),"size":size,"columns":np.array(
        ["time_s","dx_px","dy_px","rotation_deg","scale","path_x_px","path_y_px","tracks","inliers","residual_px","valid","cut_candidate"])}
    np.savez_compressed(cache,**result)
    return result


def analyze_clip(folder, track_background=True):
    run = folder/"combined"
    meta = json.loads((run/"run.json").read_text(encoding="utf-8"))
    report = json.loads((run/"quality.json").read_text())
    motion = {"raw":load_target(run/"target_motion_raw.npz"), "depth":load_target(run/"target_motion_depth.npz"),
              "feet":load_target(folder/"target_motion_feet.npz"), "combined":load_target(run/"target_motion.npz")}
    contacts = dict(np.load(run/"contacts.npz",allow_pickle=False))
    fps, n = motion["raw"].fps, motion["raw"].n_frames
    t = np.arange(n)/fps
    mid = interior(n,fps)
    ids = [motion["raw"].profile.index[b] for b in BONES]
    positions = {stage:obj.fk()[0][:,ids] for stage,obj in motion.items()}
    for obj in motion.values():
        assert obj.n_frames == n and obj.fps == fps and obj.profile.identity == motion["raw"].profile.identity
    # Depth-only processing must leave relative articulation unchanged.
    np.testing.assert_allclose(positions["depth"][:,1:]-positions["depth"][:,:1],
                               positions["raw"][:,1:]-positions["raw"][:,:1],atol=1e-7)
    result = {"clip":folder.name,"frames":n,"fps":fps,"duration_s":n/fps,"profile":meta["profile"],
              "target_unit":"meters", "highpass_definition":"zero-phase order-4 Butterworth; 1 s trimmed at each end (short clips: 20%)",
              "stages":{}, "fixed_contact_samples":int(contacts["contact"].sum()),
              "fixed_contact_frames":int(contacts["contact"].any(axis=1).sum()),
              "contact_coverage_percent":100*float(contacts["contact"].any(axis=1).mean()),
              "depth_cleanup":report["depth_cleanup"], "foot_correction":report["correction"]}
    valid_pairs=contacts["contact"][1:]&contacts["contact"][:-1]
    full_pairs=valid_pairs&(contacts["weights"][1:]>1-1e-8)&(contacts["weights"][:-1]>1-1e-8)
    ramp_pairs=valid_pairs&~full_pairs
    result["contact_pair_counts"]={"all":int(valid_pairs.sum()),"full_strength":int(full_pairs.sum()),
                                   "transition":int(ramp_pairs.sum())}
    # Dense data for independent inspection, no filtering baked into exports.
    columns = ["time_s"]; values = [t]
    for stage in STAGES:
        result["stages"][stage] = {"quality":quality(motion[stage],contacts),"bones":{},"hip_relative_feet":{}}
        points=foot_points(motion[stage])
        speeds=np.linalg.norm(np.diff(points[:,:,[0,2]],axis=0),axis=2)*fps
        result["stages"][stage]["slip_by_weight_m_s"]={label:float(speeds[mask].mean()) if mask.any() else None
            for label,mask in (("full_strength",full_pairs),("transition",ramp_pairs))}
        if valid_pairs.any():
            assert abs(float(speeds[valid_pairs].mean())-result["stages"][stage]["quality"]["mean_contact_speed_m_s"])<1e-10
        for b, name in enumerate(BONES):
            result["stages"][stage]["bones"][name] = trajectory_metrics(positions[stage][:,b],fps)
            for axis, label in enumerate("xyz"):
                columns.append(f"{stage}_{name}_{label}_m"); values.append(positions[stage][:,b,axis])
        for b,name in enumerate(BONES[1:],start=1):
            result["stages"][stage]["hip_relative_feet"][name] = trajectory_metrics(positions[stage][:,b]-positions[stage][:,0],fps)
    for i,label in enumerate(("left_heel","left_toe","right_heel","right_toe")):
        columns.append(label+"_contact"); values.append(contacts["contact"][:,i].astype(int))
    with (folder/"positions.csv").open("w",newline="",encoding="utf-8") as stream:
        writer=csv.writer(stream);writer.writerow(columns);writer.writerows(np.column_stack(values))
    hp = {stage:highpass(p,fps) for stage,p in positions.items()}
    raw_hp = hp["raw"][mid]
    result["raw_hip_foot_highpass_correlation_xyz"] = {
        name:[correlation(raw_hp[:,0,a],raw_hp[:,b,a]) for a in range(3)] for b,name in enumerate(BONES[1:],1)}
    result["raw_left_right_highpass_correlation_xyz"] = [correlation(raw_hp[:,1,a],raw_hp[:,2,a]) for a in range(3)]
    delta = positions["combined"]-positions["raw"]
    result["combined_edits"] = {name:{"rms_mm":vector_rms(delta[:,b])*1000,
        "max_mm":float(np.linalg.norm(delta[:,b],axis=1).max()*1000),
        "below_2hz_change_rms_mm":vector_rms((delta[:,b]-highpass(delta[:,b],fps,2.))[mid])*1000}
        for b,name in enumerate(BONES)}
    _, depth_world = motion["depth"].fk()
    _, final_world = motion["combined"].fk()
    ankle_delta=final_world[:,ids[1:]]@depth_world[:,ids[1:]].swapaxes(-1,-2)
    angles=Rotation.from_matrix(ankle_delta.reshape(-1,3,3)).magnitude().reshape(n,2)
    result["foot_pitch_limit_foot_frames"]=int((angles>=.26-1e-6).sum())
    bridge=camera_inspection(run/"skeleton_motion.npz",{("corrected" if stage=="combined" else stage):motion[stage] for stage in STAGES})
    camera_positions=np.frombuffer(bridge.targets,dtype="<f4").reshape(4,n,len(motion["raw"].profile.names),3)[:,:,ids]
    intrinsics=np.asarray(bridge.metadata["intrinsics"])
    with np.load(run/"skeleton_motion.npz",allow_pickle=False) as data:
        image_size=np.asarray(data["camera_image_size"])
    projected=(camera_positions[:,:,:,:2]/camera_positions[:,:,:,2,None]*intrinsics[None,:,None,:2]+intrinsics[None,:,None,2:])*image_size
    image_displacement=np.linalg.norm(projected-projected[:1],axis=-1)
    result["target_image_displacement"]={stage:{name:{
        "rms_px":float(np.sqrt(np.mean(image_displacement[s,:,b]**2))),
        "p95_px":float(np.percentile(image_displacement[s,:,b],95)),
        "max_px":float(image_displacement[s,:,b].max()),
        "max_image_height_percent":float(image_displacement[s,:,b].max()/image_size[1]*100)}
        for b,name in enumerate(BONES)} for s,stage in enumerate(STAGES)}
    result["inspection_windows"] = []
    for b,name in enumerate(BONES):
        score=np.linalg.norm(hp["raw"][:,b],axis=1)
        candidates=score.copy();candidates[:mid.start]=-1;candidates[mid.stop:]=-1
        for _ in range(2):
            peak=int(np.argmax(candidates))
            result["inspection_windows"].append({"bone":name,"peak_s":float(t[peak]),
                "window_s":[float(max(0,t[peak]-.5)),float(min(n/fps,t[peak]+.5))],
                "raw_highpass_magnitude_mm":float(score[peak]*1000),
                "combined_highpass_magnitude_mm":float(np.linalg.norm(hp["combined"][peak,b])*1000),
                "heel_toe_contact":contacts["contact"][peak].astype(int).tolist()})
            candidates[max(0,peak-round(fps)):min(n,peak+round(fps)+1)]=-1
    sole = foot_points(motion["raw"])
    result["support"] = {}
    for side,i in (("Left",0),("Right",2)):
        planted = contacts["contact"][:,i:i+2].any(axis=1)
        near = sole[:,i:i+2,1].min(axis=1)<contacts["floor_m"]+contacts["height_threshold_m"]
        lengths=[(end-start)/fps for start,end in runs(planted)]
        result["support"][side] = {"contact_frames":int(planted.sum()),"near_floor_unlocked_frames":int((near&~planted).sum()),
            "intervals":len(lengths),"median_interval_s":float(np.median(lengths)) if lengths else 0.,
            "max_interval_s":max(lengths,default=0.)}
    fig, axes = plt.subplots(3,3,figsize=(16,9),sharex=True,layout="constrained")
    for b in range(3):
        for a in range(3):
            ax=axes[b,a]
            for stage in STAGES:
                ax.plot(t,positions[stage][:,b,a],label=LABELS[stage],color=COLORS[stage],linewidth=1.1,alpha=.85)
            if b:
                mask=contacts["contact"][:,(b-1)*2:(b-1)*2+2].any(axis=1)
                for start,end in runs(mask):
                    ax.axvspan(t[start],min(n/fps,end/fps),color="#2e9f56",alpha=.10,lw=0)
            ax.set_ylabel(BONE_LABELS[b]+" (m)" if a==0 else "m")
            if b==0:ax.set_title(("X / lateral", "Y / height", "Z / world depth")[a])
            if b==2:ax.set_xlabel("Time (s)")
    axes[0,0].legend(fontsize=8)
    fig.suptitle(folder.name+": absolute positions; green spans = fixed raw foot-contact evidence")
    savefig(fig,folder,"positions")
    fig,axes=plt.subplots(3,2,figsize=(14,9),sharex=True,layout="constrained")
    for b in range(3):
        for stage in STAGES:
            axes[b,0].plot(t,hp[stage][:,b,2]*1000,color=COLORS[stage],label=LABELS[stage],lw=.9)
            vector=positions[stage][:,b] if b==0 else positions[stage][:,b]-positions[stage][:,0]
            axes[b,1].plot(t,np.linalg.norm(highpass(vector,fps),axis=1)*1000,color=COLORS[stage],lw=.9)
        axes[b,0].set_ylabel(BONE_LABELS[b]+" (mm)")
        for a in range(2):
            axes[b,a].axvspan(0,t[mid.start],color="black",alpha=.05)
            axes[b,a].axvspan(t[mid.stop-1],n/fps,color="black",alpha=.05)
            if b==2:axes[b,a].set_xlabel("Time (s)")
    axes[0,0].set_title("World Z above 3 Hz (signed)")
    axes[0,1].set_title("3D above 3 Hz: hip absolute; feet relative to hip")
    axes[0,0].legend(fontsize=8)
    fig.suptitle(folder.name+": high-frequency motion proxy; shaded ends excluded from RMS")
    savefig(fig,folder,"high_frequency")
    fig, axes=plt.subplots(1,3,figsize=(15,4),layout="constrained")
    spectra={}
    for b in range(3):
        for stage in STAGES:
            frequency,power=welch(positions[stage][mid,b],fs=fps,nperseg=min(128,mid.stop-mid.start),axis=0,detrend="linear")
            total=power.sum(axis=1)*1e6
            spectra[stage+"_"+BONES[b]]=total
            selected=frequency>=3.
            result["stages"][stage]["bones"][BONES[b]]["dominant_frequency_above_3hz"] = float(frequency[selected][np.argmax(total[selected])])
            axes[b].semilogy(frequency,np.maximum(total,1e-8),color=COLORS[stage],label=LABELS[stage])
        axes[b].axvline(3,color="black",ls="--",lw=.8);axes[b].set_xlim(.25,15)
        axes[b].set_title(BONE_LABELS[b]);axes[b].set_xlabel("Frequency (Hz)")
    axes[0].set_ylabel("Position power density (mm²/Hz)");axes[0].legend(fontsize=8)
    fig.suptitle(folder.name+": spectra (real footwork also contributes above 3 Hz)")
    savefig(fig,folder,"spectra")
    np.savez_compressed(folder/"spectra.npz",frequency_hz=frequency,**spectra)
    # Source/camera diagnosis uses original meters, separate from target scale.
    with np.load(run/"skeleton_motion.npz",allow_pickle=False) as source:
        world=np.asarray(source["joints_3d"],float);cam=np.asarray(source["camera_joints_3d"],float)
        times=np.asarray(source["timestamps"],float);times-=times[0]
        r=np.asarray(source["camera_world_to_camera"],float);k=np.asarray(source["camera_intrinsics"],float)
        size=np.asarray(source["camera_image_size"],float);names=source["joint_names"].tolist()
        video=Path(str(source["video"]));source_fps=float(source["fps"])
    sids=[names.index(name) for name in ("pelvis","left_ankle","right_ankle")]
    smid=interior(len(world),source_fps)
    hip=world[:,sids[0]]
    center=hip-np.einsum("tji,tj->ti",r,cam[:,sids[0]])
    camera_columns=["time_s"]
    camera_values=[times]
    for kind,array in (("source_world",world[:,sids]),("camera_space",cam[:,sids])):
        for b,name in enumerate(BONES):
            for a,axis in enumerate("xyz"):
                camera_columns.append(kind+"_"+name+"_"+axis+"_m");camera_values.append(array[:,b,a])
    for a,axis in enumerate("xyz"):
        camera_columns.append("derived_camera_center_"+axis+"_m");camera_values.append(center[:,a])
    with (folder/"source_camera.csv").open("w",newline="",encoding="utf-8") as stream:
        writer=csv.writer(stream);writer.writerow(camera_columns);writer.writerows(np.column_stack(camera_values))
    hhip=highpass(hip,source_fps)[smid];hcam=highpass(center,source_fps)[smid]
    chp=highpass(cam[:,sids[0]],source_fps)[smid]
    fixed=cam@r[0]
    result["source"]={name:trajectory_metrics(world[:,i],source_fps) for name,i in zip(BONES,sids)}
    result["camera"]={"derived_camera_hp_rms_mm":vector_rms(hcam)*1000,
        "pelvis_camera_hp_correlation_xyz":[correlation(hhip[:,a],hcam[:,a]) for a in range(3)],
        "camera_depth_fraction_of_pelvis_hp_energy":float(np.mean(chp[:,2]**2)/max(1e-20,np.mean(np.sum(chp**2,axis=1)))),
        "pelvis_hp_rms_mm":vector_rms(hhip)*1000,
        "pelvis_fixed_camera_rotation_hp_rms_mm":vector_rms(highpass(fixed[:,0],source_fps)[smid])*1000,
        "caveat":"Derived camera comes from the same body predictions, not independent SLAM; correlation is not causal attribution."}
    fig,axes=plt.subplots(4,3,figsize=(16,10),sharex=True,layout="constrained")
    for b,i in enumerate(sids):
        for a in range(3):
            axes[b,a].plot(times,world[:,i,a]-world[0,i,a],color="#3186b1",lw=1,label="Source joint")
            axes[b,a].plot(times,center[:,a]-center[0,a],color="#b96e44",lw=.9,alpha=.7,label="Derived camera center")
            if a==0:axes[b,a].set_ylabel(BONE_LABELS[b]+" Δ (m)")
            if b==0:axes[b,a].set_title("XYZ"[a]+" / source world")
    for a in range(3):
        axes[3,a].plot(times,cam[:,0,a]-cam[0,0,a],color="#655aa8",label="Hip in camera coordinates",lw=1)
        axes[3,a].set_xlabel("Time (s)")
    axes[3,0].set_ylabel("Camera-space hip Δ (m)");axes[0,0].legend(fontsize=8);axes[3,0].legend(fontsize=8)
    fig.suptitle(folder.name+": source joints and derived camera (independent origin removed per trace)")
    savefig(fig,folder,"source_camera")
    if track_background:
        print(f"Background tracking {folder.name}",flush=True)
        bg=background_tracking(video,cam,k,size,times,folder)
        rows=bg["values"];valid=rows[:,10]>0
        result["camera"]["background"]={"samples":len(rows),"valid_percent":float(valid.mean()*100),
            "analysis_image_size":bg["size"].tolist(),"cut_candidates":int(rows[:,11].sum()),
            "median_step_px":float(np.median(np.linalg.norm(rows[valid,1:3],axis=1))) if valid.any() else None,
            "p95_step_px":float(np.percentile(np.linalg.norm(rows[valid,1:3],axis=1),95)) if valid.any() else None,
            "median_fit_residual_px":float(np.median(rows[valid,9])) if valid.any() else None}
        fig,axes=plt.subplots(2,1,figsize=(13,5),sharex=True,layout="constrained")
        for a in range(2):
            axes[a].plot(rows[:,0],np.where(valid,rows[:,1+a],np.nan),lw=.8,label="XY"[a]+" frame-to-frame")
            axes[a].set_ylabel("Background flow (px)");axes[a].legend()
        axes[1].set_xlabel("Time (s)")
        fig.suptitle(folder.name+": independent background image movement at ~10 Hz / 480px longest side")
        savefig(fig,folder,"background")
    (folder/"analysis.json").write_text(json.dumps(result,indent=2,allow_nan=False)+"\n",encoding="utf-8")
    return result


def build_report(root, results):
    table=[]
    for r in results:
        raw=r["stages"]["raw"];final=r["stages"]["combined"]
        table.append([r["clip"],r["duration_s"],r["contact_coverage_percent"],
            *[raw["bones"][b]["highpass_rms_mm"]["3"] for b in BONES],
            *[final["bones"][b]["highpass_rms_mm"]["3"] for b in BONES],
            raw["quality"]["mean_contact_speed_m_s"],final["quality"]["mean_contact_speed_m_s"],
            raw["quality"]["max_floor_penetration_m"]*1000,final["quality"]["max_floor_penetration_m"]*1000])
    headings=["clip","duration_s","contact_coverage_percent","raw_hip_hf_mm","raw_left_hf_mm","raw_right_hf_mm",
              "final_hip_hf_mm","final_left_hf_mm","final_right_hf_mm","raw_contact_speed_m_s","final_contact_speed_m_s",
              "raw_floor_penetration_mm","final_floor_penetration_mm"]
    with (root/"summary.csv").open("w",newline="",encoding="utf-8") as stream:
        writer=csv.writer(stream);writer.writerow(headings);writer.writerows(table)
    (root/"summary.json").write_text(json.dumps(results,indent=2,allow_nan=False)+"\n",encoding="utf-8")
    fig,axes=plt.subplots(3,1,figsize=(14,10),layout="constrained")
    x=np.arange(len(results));width=.18
    for b in range(3):
        for s,stage in enumerate(STAGES):
            value=[r["stages"][stage]["bones"][BONES[b]]["highpass_rms_mm"]["3"] for r in results]
            axes[b].bar(x+(s-1.5)*width,value,width,label=LABELS[stage],color=COLORS[stage])
        axes[b].set_xticks(x,[r["clip"] for r in results]);axes[b].set_ylabel(BONE_LABELS[b]+" RMS (mm)")
    axes[0].legend(ncol=4);fig.suptitle("Motion above 3 Hz: four processing stages, identical skeleton per clip")
    savefig(fig,root,"overview")
    fig,axes=plt.subplots(1,3,figsize=(16,4.5),layout="constrained")
    for s,stage in enumerate(STAGES):
        for ax,key,factor in ((axes[0],"mean_contact_speed_m_s",1000),(axes[1],"max_floor_penetration_m",1000),
                              (axes[2],"body_jerk_rms_m_s3",1)):
            ax.bar(x+(s-1.5)*width,[r["stages"][stage]["quality"][key]*factor for r in results],width,
                   color=COLORS[stage],label=LABELS[stage])
            ax.set_xticks(x,[r["clip"] for r in results])
    for ax,label in zip(axes,("Detected-contact horizontal speed (mm/s)","Worst proxy floor penetration (mm)","Body jerk RMS (m/s³)")):
        ax.set_title(label)
    axes[0].legend(fontsize=8);savefig(fig,root,"contacts_overview")
    fig,axes=plt.subplots(2,1,figsize=(14,7),layout="constrained")
    for a,kind in enumerate(("full_strength","transition")):
        for s,stage in enumerate(STAGES):
            axes[a].bar(x+(s-1.5)*width,[r["stages"][stage]["slip_by_weight_m_s"][kind]*1000
                if r["stages"][stage]["slip_by_weight_m_s"][kind] is not None else np.nan for r in results],
                width,color=COLORS[stage],label=LABELS[stage])
        axes[a].set_xticks(x,[r["clip"]+"\n"+str(r["contact_pair_counts"][kind])+" pairs" for r in results])
        axes[a].set_ylabel("Horizontal speed (mm/s)")
        axes[a].set_title("Full-strength holds" if a==0 else "Touchdown / lift-off transitions")
    axes[0].legend(ncol=4);fig.suptitle("Anti-slide effectiveness depends strongly on contact weight")
    savefig(fig,root,"slip_by_weight")
    pooled={}
    for kind in ("full_strength","transition"):
        count=sum(r["contact_pair_counts"][kind] for r in results)
        pooled[kind]={"pairs":count,**{stage:sum(r["contact_pair_counts"][kind]*(r["stages"][stage]["slip_by_weight_m_s"][kind] or 0.)
                            for r in results)/count*1000 if count else None for stage in ("raw","combined")}}
    (root/"pooled_slip.json").write_text(json.dumps(pooled,indent=2)+"\n",encoding="utf-8")
    lines=["# Six-video motion noise study", "", "All clips use the same AB7 PMX geometry and baked FK at 30 fps.",
           "Four stages: raw, depth only, feet only, depth + feet. Each clip uses its original raw contact mask at every stage.",
           "Hip = HipMaster; feet = ankle joint centers. Floor/slip metrics use separate heel/toe proxies.",
           "", "High-frequency RMS is a zero-phase 3 Hz high-pass proxy, not measured reconstruction error. Real dance/impacts contribute.",
           "One second at each end is omitted from RMS (20% for very short clips). 2/5 Hz sensitivity and per-axis results are in JSON.",
           "Feet-relative-to-hip traces isolate articulation from common translation. Derived camera is not independent SLAM.",
           "Background flow excludes the person and measures image motion, not metric camera position. There is no ground truth.",
           "", "| Clip | Seconds | Contact frames % | Hip HF raw → final (mm) | Left foot (mm) | Right foot (mm) | Contact slip raw → final (mm/s) |",
           "|---|---:|---:|---:|---:|---:|---:|"]
    for row in table:
        lines.append(f"| {row[0]} | {row[1]:.1f} | {row[2]:.1f} | {row[3]:.2f} → {row[6]:.2f} | {row[4]:.2f} → {row[7]:.2f} | {row[5]:.2f} → {row[8]:.2f} | {row[9]*1000:.1f} → {row[10]*1000:.1f} |")
    lines.extend(["", "![High-frequency overview](overview.png)", "", "![Contact overview](contacts_overview.png)"])
    body=["<h1>Six-video motion noise study</h1>","<p>Raw · Depth only · Feet only · Depth + feet. Same AB7 PMX; fixed contact evidence per clip.</p>",
          "<p><strong>Above 3 Hz means high-frequency motion, not proven noise.</strong> Real footwork contributes. Camera centers are derived from body predictions; correlations do not establish causality. Ground clearance uses skeletal proxies.</p>",
          '<p><a href="summary.csv">Summary CSV</a> · <a href="summary.json">All metrics JSON</a> · <a href="analysis.md">Method and table</a></p>',
          '<nav>'+" · ".join(f'<a href="#{r["clip"]}">{r["clip"]}</a>' for r in results)+"</nav>",
          '<img src="overview.png"><img src="contacts_overview.png"><img src="slip_by_weight.png">']
    findings=["# Findings", "", "All changes below compare the same source, target geometry and contact evidence within each clip.",
        "High-frequency values include real motion. They measure attenuation, not accuracy improvement.", "",
        "## What the stages address", "",
        "Depth cleanup changes one shared translation per frame. Hip-relative foot trajectories are numerically unchanged in the depth-only stage (checked to 0.0001 mm). It therefore cannot remove leg-articulation jitter.",
        "Foot cleanup changes legs only at inferred support or floor collisions, with bounded pelvis reach assistance. It is not a whole-clip pose filter. Changes in ankle HF RMS can be small even when contact slip drops substantially.",
        "Foot speed is evaluated only on consecutive raw-labeled contacts. Near-floor unlocked frames are NOT proven missed contacts: they also contain real slides, steps and pivots.",
        "All six extractions use the static-camera path and estimated camera intrinsics. Background flow checks that assumption independently, but neither flow nor derived camera centers establish a true 3D camera path.", "",
        "## Holds versus transitions", "",
        f"Across the fixed contact masks, full-strength hold speed is {pooled['full_strength']['raw']:.1f} → {pooled['full_strength']['combined']:.1f} mm/s over {pooled['full_strength']['pairs']} heel/toe pairs. Transition speed is {pooled['transition']['raw']:.1f} → {pooled['transition']['combined']:.1f} mm/s over {pooled['transition']['pairs']} pairs. These are pair-weighted means, not ground-truth foot slip.",
        "The difference is material: the solver is much stronger during holds. A short ramp toward a distant interval-median anchor can itself add motion. Some clips also gain a small amount of hip/ankle HF energy from the foot-only stage, despite lower contact slip. Inspect transitions and correction velocities before extending locks or increasing smoothing.", "",
        "## Per-clip results", ""]
    for r in results:
        raw=r["stages"]["raw"];end=r["stages"]["combined"]
        def reduction(a,b):return 100*(1-b/a) if a>1e-10 else 0.
        hf=[(raw["bones"][b]["highpass_rms_mm"]["3"],end["bones"][b]["highpass_rms_mm"]["3"]) for b in BONES]
        slip0,slip1=[s["quality"]["mean_contact_speed_m_s"]*1000 for s in (raw,end)]
        jerk0,jerk1=[s["quality"]["body_jerk_rms_m_s3"] for s in (raw,end)]
        findings.extend([f"### {r['clip']}", "",
            f"- Above-3-Hz RMS: hip {hf[0][0]:.2f} → {hf[0][1]:.2f} mm ({reduction(*hf[0]):.1f}% reduction); left ankle {hf[1][0]:.2f} → {hf[1][1]:.2f} mm; right ankle {hf[2][0]:.2f} → {hf[2][1]:.2f} mm.",
            f"- Contact speed: {slip0:.1f} → {slip1:.1f} mm/s ({reduction(slip0,slip1):.1f}% reduction). Coverage: {r['contact_coverage_percent']:.1f}% of frames, {r['fixed_contact_samples']} heel/toe samples.",
            f"- Consecutive heel/toe contact pairs: {r['contact_pair_counts']['all']}; full-strength holds {r['contact_pair_counts']['full_strength']}, transitions {r['contact_pair_counts']['transition']}. Combined mean slip by class (m/s): {end['slip_by_weight_m_s']}.",
            f"- Body jerk: {jerk0:.1f} → {jerk1:.1f} m/s³ ({reduction(jerk0,jerk1):.1f}% reduction). Worst proxy penetration: {raw['quality']['max_floor_penetration_m']*1000:.2f} → {end['quality']['max_floor_penetration_m']*1000:.4f} mm.",
            f"- Raw hip-relative foot HF RMS: left {raw['hip_relative_feet']['LeftAnkle']['highpass_rms_mm']['3']:.2f} mm; right {raw['hip_relative_feet']['RightAnkle']['highpass_rms_mm']['3']:.2f} mm.",
            f"- Additional foot-pitch limit hits: {r['foot_pitch_limit_foot_frames']} foot-frames. Maximum leg reach residual: {r['foot_correction']['max_reach_residual_m']*1000:.4f} mm.",
            f"- Combined target ankle image displacement from raw: left max {r['target_image_displacement']['combined']['LeftAnkle']['max_image_height_percent']:.3f}% of image height; right max {r['target_image_displacement']['combined']['RightAnkle']['max_image_height_percent']:.3f}%. This is a change under the estimated camera, not reprojection error against observed landmarks. The depth-only budget does not constrain foot cleanup.",
            f"- Camera-depth component: {r['camera']['camera_depth_fraction_of_pelvis_hp_energy']*100:.1f}% of camera-space hip HF energy. Derived-camera/hip XYZ correlations: {r['camera']['pelvis_camera_hp_correlation_xyz']}. Correlations are not causal attribution."])
        bg=r["camera"].get("background")
        if bg:
            findings.append(f"- Background flow at ~10 Hz, 480px longest side: median {bg['median_step_px']:.3f}px, p95 {bg['p95_step_px']:.3f}px; {bg['valid_percent']:.1f}% valid fits; {bg['cut_candidates']} cut candidates. This does not capture translation along the optical axis reliably.")
        for side in ("Left","Right"):
            support=r["support"][side]
            findings.append(f"- {side}: {support['intervals']} contact intervals, median {support['median_interval_s']:.2f}s; {support['near_floor_unlocked_frames']} near-floor unlocked frames.")
        findings.extend(["", "Largest raw high-frequency windows (inspection candidates, not confirmed errors): " + "; ".join(
            f"{p['bone']} {p['window_s'][0]:.2f}–{p['window_s'][1]:.2f}s" for p in r["inspection_windows"]), ""])
    findings.extend(["## Next experiments supported by this study", "",
        "1. Inspect the listed peak windows against video to label real impacts/pivots versus estimator mistakes before increasing smoothing.",
        "2. Evaluate smoother, velocity/acceleration-bounded anchor transitions together with contact hysteresis/short-gap bridging on verified stationary feet; keep a held-out raw contact mask to avoid improving metrics by relabeling them.",
        "3. Evaluate bone-length-preserving local leg/foot regularization separately from common depth. Compare hip-relative trajectories and image agreement to protect real footwork.",
        "4. Review clips with substantial background movement before interpreting static-camera depth correction; calibrated intrinsics and moving-camera estimation need a separate experiment."])
    (root/"findings.md").write_text("\n".join(findings)+"\n",encoding="utf-8")
    body.insert(3,'<details><summary><strong>Findings and interpretation</strong></summary><pre style="white-space:pre-wrap;font:inherit">'+html.escape("\n".join(findings))+"</pre></details>")
    if (root/"reference_hibikase/index.html").exists():
        body.insert(3,'<p><strong><a href="reference_hibikase/">Compare with Hibikase reference VMD: hip/feet plots, spectra and equal-duration windows</a></strong></p>')
    for r in results:
        name=r["clip"]
        lines.extend(["",f"## {name}","",f"{r['frames']} frames; {r['fixed_contact_frames']} frames / {r['fixed_contact_samples']} heel-toe samples labeled contact.",
                      f"[Metrics]({name}/analysis.json) · [Position CSV]({name}/positions.csv)"])
        body.append(f'<section id="{name}"><h2>{name}</h2><p>{r["frames"]} frames; contact coverage {r["contact_coverage_percent"]:.1f}%. '
                    f'<a href="{name}/analysis.json">Metrics</a> · <a href="{name}/positions.csv">Positions CSV</a></p>')
        for plot in ("positions","high_frequency","spectra","source_camera","background"):
            if (root/name/(plot+".png")).exists():
                lines.extend(["",f"![{name} {plot}]({name}/{plot}.png)"])
                body.append(f'<a href="{name}/{plot}.png"><img loading="lazy" src="{name}/{plot}.png" alt="{html.escape(name+" "+plot)}"></a>')
        body.append("</section>")
    (root/"analysis.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
    (root/"index.html").write_text('<!doctype html><meta charset="utf-8"><title>Video2VMD noise study</title>'
        '<style>body{font:16px/1.5 system-ui;max-width:1500px;margin:30px auto;padding:0 24px;color:#24313c;background:#f6f8fa}img{width:100%;height:auto;background:white;margin:12px 0}nav{position:sticky;top:0;padding:12px;background:#fff;z-index:1}section{scroll-margin-top:60px}a{color:#006b8c}</style>'
        +"\n".join(body),encoding="utf-8")


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root",type=Path)
    parser.add_argument("--no-background",action="store_true")
    parser.add_argument("--clips",nargs="*")
    args=parser.parse_args()
    root=args.root.resolve()
    folders=sorted(p for p in root.iterdir() if p.is_dir() and (p/"combined/run.json").exists()
                   and (not args.clips or p.name in args.clips))
    results=[]
    for folder in folders:
        print("Analyzing "+folder.name,flush=True)
        results.append(analyze_clip(folder,not args.no_background))
    if not results:
        parser.error("No completed study runs found")
    build_report(root,results)
    print(f"Report: {root/'index.html'}",flush=True)


if __name__ == "__main__":
    main()
