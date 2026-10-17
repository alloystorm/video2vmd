"""Compare authored VMD playback with the saved six-video study, without ML.

Different choreography is a descriptive reference, never paired ground truth.
Sample in a fresh background Blender process; keep source/archive unmodified.
"""
import argparse
import csv
import hashlib
import html
import json
from pathlib import Path
import struct
import subprocess
import sys
from types import SimpleNamespace
import zipfile

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
from scipy.signal import welch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from analyze_motion_noise import BONES,BONE_LABELS,COLORS,highpass,interior,trajectory_metrics,savefig,vector_rms
from video2vmd.inspection.server import load_target
from video2vmd.mmd.contacts import detect_contacts,foot_points,quality
from video2vmd.mmd.profile import pmx_profile

REFERENCE_COLORS={"authored_ik":"#c64d36","fk":"#242933"}
REFERENCE_LABELS={"authored_ik":"Hibikase / authored IK","fk":"Hibikase / IK disabled"}


def prepare(archive,out,model,mmd_tools,member=None):
    out.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(archive) as z:
        candidates=[i for i in z.infolist() if i.filename.lower().endswith(".vmd")]
        selected=z.getinfo(member) if member else max(candidates,key=lambda i:i.file_size)
        data=z.read(selected)
        path=out/Path(selected.filename).name
        if not path.exists() or path.read_bytes()!=data:
            path.write_bytes(data)
        readmes=[i for i in z.infolist() if "english" in i.filename.lower() and i.filename.lower().endswith(".txt")]
        credit=""
        if readmes:
            raw=z.read(readmes[0]);(out/"Read Me(English).txt").write_bytes(raw)
            credit=raw.decode("utf-16" if raw[:2] in (b"\xff\xfe",b"\xfe\xff") else "utf-8",errors="replace")
    if not data.startswith(b"Vocaloid Motion Data 0002"):
        raise ValueError("Reference metadata probe supports VMD 0002")
    count=struct.unpack_from("<I",data,50)[0]
    if len(data)<54+count*111:
        raise ValueError("Truncated VMD bone records")
    tracks={}
    for offset in range(54,54+count*111,111):
        name=data[offset:offset+15].split(b"\0")[0].decode("cp932",errors="replace")
        frame=struct.unpack_from("<I",data,offset+15)[0]
        track=tracks.setdefault(name,{"keys":0,"first":frame,"last":frame})
        track["keys"]+=1;track["first"]=min(track["first"],frame);track["last"]=max(track["last"],frame)
    profile=pmx_profile(model)
    names=["HipMaster"]+[side+part for side in ("Left","Right") for part in ("Leg","Knee","Ankle","Toe")]
    frames=max(v["last"] for name,v in tracks.items() if name in profile.serialized.values())+1
    config={"vmd":str(path.resolve()),"model":profile.source,"unit":profile.unit,"profile_hash":profile.identity,
        "output":str(out.resolve()),"mmd_tools":str(mmd_tools.resolve()),"frames":frames,"sample_bones":names,
        "sample_rest":[profile.position(name).tolist() for name in names],"serialized":profile.serialized,
        "modes":["authored_ik","fk"],"body_tracks":[profile.serialized[name] for name in names if profile.serialized[name] in tracks],
        "archive":str(archive.resolve()),"vmd_sha256":hashlib.sha256(data).hexdigest(),"bone_keys":count,
        "model_header":data[30:50].split(b"\0")[0].decode("cp932",errors="replace"),
        "member":selected.filename,"readme":credit,
        "active_relevant_tracks":{name:tracks[jp] for name,jp in profile.serialized.items() if jp in tracks and name in names+["Center","Groove","Master","LeftLegIK","RightLegIK","LeftLegIKParent","RightLegIKParent"]}}
    (out/"sampling_config.json").write_text(json.dumps(config,indent=2)+"\n",encoding="utf-8")
    return config


class SampledWorldMotion:
    """Duck-typed world data for the same contact geometry/threshold functions."""
    def __init__(self,path):
        with np.load(path,allow_pickle=False) as d:
            self.positions=d["positions_m"];self.rotations=d["world_rotations"]
            names=d["names"].tolist();rest=d["rest_positions_m"];self.fps=float(d["fps"])
        self.profile=SimpleNamespace(names=names,index={name:i for i,name in enumerate(names)},rest=rest,root="HipMaster")
        self.n_frames=len(self.positions)
        if (self.positions.shape!=(self.n_frames,len(names),3) or self.rotations.shape!=(self.n_frames,len(names),3,3)
            or rest.shape!=(len(names),3) or self.n_frames<2 or not np.isfinite(self.fps) or self.fps<=0
            or not all(np.isfinite(v).all() for v in (self.positions,self.rotations,rest))
            or not np.allclose(self.rotations@self.rotations.swapaxes(-1,-2),np.eye(3),atol=1e-4)):
            raise ValueError("Invalid sampled reference world transforms")
    def fk(self):
        return self.positions,self.rotations


def window_metrics(points,fps=30.,seconds=7.,stride_seconds=3.5):
    """Unwarped, fixed-duration windows; return RMS and locomotion activity."""
    count=round(seconds*fps);stride=round(stride_seconds*fps)
    if len(points)<count:
        return []
    rows=[]
    for start in range(0,len(points)-count+1,stride):
        sample=points[start:start+count]
        hp=highpass(sample,fps)[interior(count,fps)]
        slow=sample-highpass(sample,fps,2.)
        speed=np.sqrt(np.mean(np.sum((np.diff(slow,axis=0)*fps)**2,axis=-1),axis=0))
        rows.append({"start_s":start/fps,"end_s":(start+count)/fps,
            "highpass_rms_mm":(np.sqrt(np.mean(np.sum(hp**2,axis=-1),axis=0))*1000).tolist(),
            "low_frequency_speed_rms_m_s":speed.tolist()})
    return rows


def describe_world(motion,contacts=None):
    positions,_=motion.fk();ids=[motion.profile.index[b] for b in BONES]
    points=positions[:,ids]
    contacts=detect_contacts(motion) if contacts is None else contacts
    q=quality(motion,contacts)
    # The sampler stores fewer body bones than target NPZs. Compare per-bone
    # jerk explicitly, not quality()'s hierarchy-dependent aggregate body jerk.
    keys=("contact_samples","contact_frames","total_frames","mean_contact_anchor_error_m",
          "mean_contact_speed_m_s","max_floor_penetration_m")
    return {"frames":motion.n_frames,"fps":motion.fps,"duration_s":motion.n_frames/motion.fps,
        "floor_m":float(contacts["floor_m"]),"contact":{k:q[k] for k in keys},
        "contact_coverage_percent":100*q["contact_frames"]/motion.n_frames,
        "bones":{name:trajectory_metrics(points[:,b],motion.fps) for b,name in enumerate(BONES)},
        "hip_relative_feet":{name:trajectory_metrics(points[:,b]-points[:,0],motion.fps) for b,name in enumerate(BONES[1:],1)},
        "windows":window_metrics(points,motion.fps)},points,contacts


def write_positions(path,points,fps):
    columns=["time_s"]+[name+"_"+axis+"_m" for name in BONES for axis in "xyz"]
    with path.open("w",newline="",encoding="utf-8") as stream:
        writer=csv.writer(stream);writer.writerow(columns)
        writer.writerows(np.column_stack([np.arange(len(points))/fps,points.reshape(len(points),-1)]))


def compare(study,out,config):
    results=json.loads((study/"summary.json").read_text(encoding="utf-8"))
    reference={};points={};contacts={};motions={}
    for mode in ("authored_ik","fk"):
        motions[mode]=SampledWorldMotion(out/(mode+"_world.npz"))
        reference[mode],points[mode],contacts[mode]=describe_world(motions[mode])
        write_positions(out/(mode+"_positions.csv"),points[mode],30.)
        np.savez_compressed(out/(mode+"_contacts.npz"),**contacts[mode])
    np.testing.assert_allclose(points["authored_ik"][:,0],points["fk"][:,0],atol=2e-6)
    datasets={}
    for mode in ("authored_ik","fk"):
        datasets["reference_"+mode]={"points":points[mode],"metrics":reference[mode],"label":REFERENCE_LABELS[mode]}
    for r in results:
        folder=study/r["clip"]/"combined"
        evidence=dict(np.load(folder/"contacts.npz",allow_pickle=False))
        for stage,file in (("raw","target_motion_raw.npz"),("combined","target_motion.npz")):
            obj=load_target(folder/file)
            if json.loads((folder/"run.json").read_text())["profile_hash"]!=config["profile_hash"]:
                raise ValueError("Reference/generated comparison must use identical PMX geometry")
            metrics,values,_=describe_world(obj,evidence)
            datasets[r["clip"]+"_"+stage]={"points":values,"metrics":metrics,"label":r["clip"]+" "+stage}
    overview=[]
    for key,d in datasets.items():
        metrics=d["metrics"]
        overview.append([key,metrics["duration_s"],*[metrics["bones"][b]["highpass_rms_mm"]["3"] for b in BONES],
            metrics["contact_coverage_percent"],metrics["contact"]["mean_contact_speed_m_s"]*1000])
    with (out/"comparison.csv").open("w",newline="",encoding="utf-8") as stream:
        writer=csv.writer(stream);writer.writerow(["dataset","seconds","hip_hf_mm","left_hf_mm","right_hf_mm","contact_coverage_percent","contact_speed_mm_s"]);writer.writerows(overview)
    credit=("Original motion: Miku VR; converted and fixed by GentleWhiteCat (archive readme)."
            if "gentlewhitecat" in config["readme"].lower() else "See the copied archive readme for attribution.")
    metadata={"reference":reference,"datasets":{k:v["metrics"] for k,v in datasets.items()},
        "archive":config["archive"],"member":config["member"],"vmd_sha256":config["vmd_sha256"],
        "credit":credit,
        "model_header":config["model_header"],"target_model":config["model"],"profile_hash":config["profile_hash"],
        "sampling":"Fresh Blender/MMD Tools armature evaluation, 30 fps, authored IK and native IK/IK-limits disabled; no physics, camera or mesh rendering.",
        "comparison_limits":"Different choreography and movement intensity; imported reference uses an AB7 adaptation of a different original model. Not paired ground truth or a noise floor.",
        "window_definition":"7 seconds, 3.5 second stride, no time warping; zero-phase 3 Hz high-pass with 1 second trimmed at both ends."}
    metadata["reference_window_comparison"]={}
    for b,name in enumerate(BONES):
        threshold=float(np.percentile([w["highpass_rms_mm"][b] for w in reference["authored_ik"]["windows"]],95))
        metadata["reference_window_comparison"][name]={"reference_p95_mm":threshold,"generated":{}}
        for r in results:
            rows=datasets[r["clip"]+"_combined"]["metrics"]["windows"]
            metadata["reference_window_comparison"][name]["generated"][r["clip"]]={"windows":len(rows),
                "above_reference_p95":sum(w["highpass_rms_mm"][b]>threshold for w in rows)}
    (out/"comparison.json").write_text(json.dumps(metadata,indent=2,allow_nan=False)+"\n",encoding="utf-8")
    windows=[]
    for key,d in datasets.items():
        for row in d["metrics"]["windows"]:
            windows.append([key,row["start_s"],row["end_s"],*row["highpass_rms_mm"],*row["low_frequency_speed_rms_m_s"]])
    with (out/"windows.csv").open("w",newline="",encoding="utf-8") as stream:
        writer=csv.writer(stream);writer.writerow(["dataset","start_s","end_s","hip_hf_mm","left_hf_mm","right_hf_mm","hip_slow_speed_m_s","left_slow_speed_m_s","right_slow_speed_m_s"]);writer.writerows(windows)
    fig,axes=plt.subplots(3,3,figsize=(16,9),sharex=True,layout="constrained")
    t=np.arange(len(points["fk"]))/30.
    for b in range(3):
        for a in range(3):
            for mode in ("authored_ik","fk"):
                axes[b,a].plot(t,points[mode][:,b,a],color=REFERENCE_COLORS[mode],label=REFERENCE_LABELS[mode],lw=.85,alpha=.8)
            if a==0:axes[b,a].set_ylabel(BONE_LABELS[b]+" (m)")
            if b==0:axes[b,a].set_title("XYZ"[a]+" / target world")
            if b==2:axes[b,a].set_xlabel("VMD time (s)")
    axes[0,0].legend(fontsize=8);fig.suptitle("Hibikase: full VMD hip and ankle positions on the same AB7 geometry")
    savefig(fig,out,"positions")
    for cutoff,filename in (("3","overview"),("5","overview_5hz")):
        fig,axes=plt.subplots(3,1,figsize=(14,10),layout="constrained")
        for b,ax in enumerate(axes):
            for i,mode in enumerate(("authored_ik","fk")):
                ax.bar(i,reference[mode]["bones"][BONES[b]]["highpass_rms_mm"][cutoff],color=REFERENCE_COLORS[mode],label=REFERENCE_LABELS[mode])
            for s,stage in enumerate(("raw","combined")):
                ax.bar(np.arange(len(results))+2+(s-.5)*.3,
                    [datasets[r["clip"]+"_"+stage]["metrics"]["bones"][BONES[b]]["highpass_rms_mm"][cutoff] for r in results],
                    .3,color=COLORS[stage],label="Generated "+stage)
            ax.set_xticks(np.arange(len(results)+2),["Hibikase\nIK","Hibikase\nFK"]+[r["clip"] for r in results])
            ax.set_ylabel(BONE_LABELS[b]+" HF RMS (mm)")
        axes[0].legend(ncol=4,fontsize=9);fig.suptitle(f"Full-clip motion above {cutoff} Hz: same PMX and units; different dances")
        savefig(fig,out,filename)
    fig,axes=plt.subplots(3,1,figsize=(14,9),layout="constrained")
    for b,ax in enumerate(axes):
        for i,mode in enumerate(("authored_ik","fk")):
            value=[w["highpass_rms_mm"][b] for w in reference[mode]["windows"]]
            artists=ax.boxplot([value],positions=[i],widths=.45,showfliers=False,patch_artist=True,manage_ticks=False)
            artists["boxes"][0].set_facecolor(REFERENCE_COLORS[mode])
        for i,r in enumerate(results):
            for s,stage in enumerate(("raw","combined")):
                value=[w["highpass_rms_mm"][b] for w in datasets[r["clip"]+"_"+stage]["metrics"]["windows"]]
                artists=ax.boxplot([value],positions=[i+2+(s-.5)*.32],widths=.27,showfliers=False,patch_artist=True,manage_ticks=False)
                artists["boxes"][0].set_facecolor(COLORS[stage])
        ax.set_xticks(np.arange(len(results)+2),["Reference IK\n"+str(len(reference["authored_ik"]["windows"]))+" windows","Reference FK\n"+str(len(reference["fk"]["windows"]))+" windows"]+
            [r["clip"]+"\n"+str(len(datasets[r["clip"]+"_raw"]["metrics"]["windows"]))+" windows" for r in results])
        ax.set_ylabel(BONE_LABELS[b]+" HF RMS (mm)")
    fig.suptitle("Equal 7-second windows: boxes = median/IQR; whiskers = 1.5 IQR; no phase or speed alignment")
    axes[0].legend(handles=[Patch(color=REFERENCE_COLORS[m],label=REFERENCE_LABELS[m]) for m in ("authored_ik","fk")]+
        [Patch(color=COLORS[s],label="Generated "+s) for s in ("raw","combined")],ncol=4,fontsize=8)
    savefig(fig,out,"window_distributions")
    fig,axes=plt.subplots(1,3,figsize=(16,4.7),layout="constrained")
    for b,ax in enumerate(axes):
        for i,r in enumerate(results):
            for stage in ("raw","combined"):
                p=datasets[r["clip"]+"_"+stage]["points"]
                f,power=welch(p[interior(len(p),30),b],fs=30,nperseg=128,axis=0,detrend="linear")
                ax.semilogy(f,np.maximum(power.sum(axis=1)*1e6,1e-8),color="#aaaaaa" if stage=="raw" else plt.cm.tab10(i),
                            lw=.65 if stage=="raw" else 1.,alpha=.4 if stage=="raw" else .7,label=r["clip"] if stage=="combined" else None)
        for mode in ("authored_ik","fk"):
            p=points[mode];f,power=welch(p[interior(len(p),30),b],fs=30,nperseg=128,axis=0,detrend="linear")
            ax.semilogy(f,np.maximum(power.sum(axis=1)*1e6,1e-8),color=REFERENCE_COLORS[mode],lw=2.,label=REFERENCE_LABELS[mode])
        ax.axvline(3,color="black",ls="--",lw=.7);ax.set_xlim(.25,15)
        ax.set_title(BONE_LABELS[b]);ax.set_xlabel("Frequency (Hz)")
    axes[0].set_ylabel("Position power density (mm²/Hz)");axes[0].legend(fontsize=7)
    fig.suptitle("Spectra: gray = generated raw; colored = generated corrected; bold = reference")
    savefig(fig,out,"spectra")
    fig,axes=plt.subplots(1,3,figsize=(16,4.7),layout="constrained")
    for b,ax in enumerate(axes):
        for mode in ("authored_ik","fk"):
            rows=reference[mode]["windows"]
            ax.scatter([w["low_frequency_speed_rms_m_s"][b] for w in rows],[w["highpass_rms_mm"][b] for w in rows],
                s=12,alpha=.4,color=REFERENCE_COLORS[mode],label=REFERENCE_LABELS[mode])
        for i,r in enumerate(results):
            rows=datasets[r["clip"]+"_combined"]["metrics"]["windows"]
            ax.scatter([w["low_frequency_speed_rms_m_s"][b] for w in rows],[w["highpass_rms_mm"][b] for w in rows],
                s=30,marker="x",color=plt.cm.tab10(i),label=r["clip"])
        ax.set_title(BONE_LABELS[b]);ax.set_xlabel("Low-frequency (<2 Hz) speed RMS (m/s)")
    axes[0].set_ylabel("Above-3-Hz position RMS (mm)");axes[0].legend(fontsize=7)
    fig.suptitle("Activity context: same-duration windows; activity is not choreography alignment")
    savefig(fig,out,"activity")
    build_comparison_report(out,config,metadata,overview)
    # Rebuild the original report's navigation with a persistent reference link.
    from analyze_motion_noise import build_report
    build_report(study,results)
    print(f"Reference comparison: {out/'index.html'}",flush=True)


def build_comparison_report(out,config,metadata,overview):
    lines=["# Hibikase VMD versus video-generated motion", "",metadata["credit"],"",
        f"Reference: `{config['member']}`; {config['frames']} samples / {config['frames']/30:.1f} seconds at 30 fps.",
        "All playback uses the same AB7 PMX geometry and 0.085 m/VMD-unit convention. The reference was authored for another model; native IK and IK-disabled playback are both included.",
        "No mesh, camera, face or physics is rendered. The camera VMDs in the ZIP are excluded. Bone world transforms come from a fresh Blender/MMD Tools scene.",
        "", "This is a **different dance**, with different movement intensity and duration. It is a descriptive authored-motion reference, not paired ground truth or a target noise floor.",
        "The original is captured Miku VR motion converted/fixed by GentleWhiteCat; dense keys do not imply an analytically smooth or noise-free curve.",
        "", "## Whole-motion comparison", "",
        "High-frequency RMS uses the same 3 Hz zero-phase high-pass, one-second endpoint trim and meter units as the six-video study.",
        "", "| Dataset | Seconds | Hip HF (mm) | Left ankle HF (mm) | Right ankle HF (mm) | Contact coverage % | Contact speed (mm/s) |",
        "|---|---:|---:|---:|---:|---:|---:|"]
    for row in overview:
        lines.append(f"| {row[0]} | "+" | ".join(f"{v:.2f}" for v in row[1:])+" |")
    ref=metadata["reference"]["authored_ik"]
    lines.extend(["", "## Interpretation", "",
        f"- Authored IK reference whole-motion HF RMS: hip {ref['bones']['HipMaster']['highpass_rms_mm']['3']:.2f} mm, left ankle {ref['bones']['LeftAnkle']['highpass_rms_mm']['3']:.2f} mm, right ankle {ref['bones']['RightAnkle']['highpass_rms_mm']['3']:.2f} mm.",
        f"- At 5 Hz, reference RMS is hip {ref['bones']['HipMaster']['highpass_rms_mm']['5']:.2f} mm, left ankle {ref['bones']['LeftAnkle']['highpass_rms_mm']['5']:.2f} mm, right ankle {ref['bones']['RightAnkle']['highpass_rms_mm']['5']:.2f} mm. The extra 5 Hz chart helps separate the faster residual component from movement close to 3 Hz; it still includes real motion.",
        "- The corrected generated clips can overlap this reference's HF levels. A lower value alone is not evidence of better motion: slower choreography, smaller travel and stronger smoothing all reduce it.",
        "- Check the equal-duration window distributions and the activity plot alongside the whole-clip averages. Each seven-second window has five seconds of RMS evaluation after endpoint trimming; windows overlap by 3.5 seconds and are not independent trials.",
        "- The activity plot compares high-frequency amplitude against low-frequency speed. It provides context, not matched poses, tempo normalization or proof that remaining differences are estimator noise.",
        "- Floor/contact measurements use the same skeletal proxies and thresholds, with a separate inferred floor/mask for each reference playback variant. Generated raw/corrected stages retain each video's original shared mask. Cross-dance contact-speed differences are descriptive, not controlled error reductions.",
        f"- The reference's authored IK playback has {ref['contact_coverage_percent']:.1f}% inferred contact coverage, {ref['contact']['mean_contact_speed_m_s']*1000:.1f} mm/s mean contact speed and {ref['contact']['max_floor_penetration_m']*1000:.1f} mm worst proxy penetration on AB7. These include model adaptation and IK-player effects; they are not a judgment of playback on its original character.",
        "- The generated processing intentionally clears floor proxies. Its better floor metric against this unmodified reference does not establish better authored animation.",
        "", "## Data and figures", "",
        "CSV files contain full reference hip/ankle XYZ trajectories and equal-duration window metrics. JSON includes 2/3/5 Hz sensitivity, per-axis RMS, hip-relative foot motion, activity and sampling provenance.",
        "Reference trajectories remain unfiltered; filtering is used only to calculate/plot analysis metrics."])
    for name,value in metadata["reference_window_comparison"].items():
        counts=", ".join(f"{clip}: {count['above_reference_p95']}/{count['windows']}" for clip,count in value["generated"].items())
        lines.append(f"- {name}: reference window P95 {value['reference_p95_mm']:.2f} mm; corrected generated windows above that level: {counts}. Different choreography and overlapping windows preclude a statistical accuracy claim.")
    body=["<h1>Hibikase reference VMD comparison</h1>","<p>"+html.escape(metadata["credit"])+"</p>",
        "<p><strong>Different choreography, same PMX and units. This is a descriptive reference, not paired ground truth.</strong></p>",
        '<p><a href="../">Six-video study</a> · <a href="analysis.md">Method and findings</a> · <a href="comparison.csv">Summary CSV</a> · <a href="comparison.json">Detailed JSON</a> · <a href="windows.csv">7-second windows CSV</a></p>',
        '<details><summary>Method, attribution and limits</summary><pre style="white-space:pre-wrap;font:inherit">'+html.escape("\n".join(lines))+"</pre></details>"]
    captions={"overview":"Whole-motion high-frequency RMS above 3 Hz", "overview_5hz":"Faster component: motion above 5 Hz", "window_distributions":"Equal-duration seven-second windows",
        "activity":"High-frequency motion versus low-frequency movement speed", "spectra":"Hip and ankle frequency spectra",
        "positions":"Reference hip and both ankles: full XYZ positions"}
    for plot,caption in captions.items():
        lines.extend(["",f"![{caption}]({plot}.png)"])
        body.append(f'<h2>{html.escape(caption)}</h2><a href="{plot}.png"><img src="{plot}.png" alt="{html.escape(caption)}"></a>')
    (out/"analysis.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
    (out/"index.html").write_text('<!doctype html><meta charset="utf-8"><title>Hibikase reference comparison</title>'
        '<style>body{font:16px/1.5 system-ui;max-width:1500px;margin:30px auto;padding:0 24px;background:#f6f8fa;color:#24313c}img{width:100%;height:auto;background:white;margin:12px 0}a{color:#006b8c}</style>'
        +"\n".join(body),encoding="utf-8")


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("study",type=Path)
    parser.add_argument("--archive",type=Path,required=True)
    parser.add_argument("--member",help="VMD ZIP member; default largest VMD, normally the dance")
    parser.add_argument("--model",type=Path,default=ROOT/"assets/mmd/ab7_160404/ab7.pmx")
    parser.add_argument("--mmd-tools",type=Path,default=ROOT/"outputs/dependencies/blender_mmd_tools")
    parser.add_argument("--blender",type=Path,default=Path("C:/Program Files/Blender Foundation/Blender 5.2/blender.exe"))
    parser.add_argument("--resample",action="store_true")
    args=parser.parse_args();study=args.study.resolve();out=study/"reference_hibikase"
    old_path=out/"sampling_config.json"
    old=json.loads(old_path.read_text(encoding="utf-8")) if old_path.exists() else {}
    config=prepare(args.archive.resolve(),out,args.model.resolve(),args.mmd_tools,args.member)
    reusable=old.get("vmd_sha256")==config["vmd_sha256"] and old.get("profile_hash")==config["profile_hash"]
    reusable &= all((out/(mode+"_world.npz")).exists() for mode in config["modes"])
    if args.resample or not reusable:
        with (out/"blender_sampling.log").open("w",encoding="utf-8") as log:
            subprocess.run([str(args.blender),"--background","--factory-startup","--python-exit-code","1","--python",
                str(ROOT/"scripts/sample_reference_vmd_blender.py"),"--","--config",str(out/"sampling_config.json")],
                stdout=log,stderr=subprocess.STDOUT,check=True,cwd=ROOT)
    for mode in config["modes"]:
        with np.load(out/(mode+"_world.npz"),allow_pickle=False) as data:
            assert len(data["positions_m"])==config["frames"]
            np.testing.assert_allclose(data["rest_positions_m"],config["sample_rest"],atol=1e-10)
    compare(study,out,config)


if __name__=="__main__":
    main()
