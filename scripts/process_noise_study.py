"""Resumable six-video noise study. Run each GPU extraction in a fresh process."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for stream in (sys.stdout, sys.stderr):
    if hasattr(stream, "reconfigure"):
        stream.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, str(ROOT))
from video2vmd.mmd.contacts import correct_feet, quality
from video2vmd.inspection.server import load_target
from video2vmd.mmd.export import write_vmd
from video2vmd.mmd.profile import pmx_profile
from video2vmd.mmd.pipeline import run_mmd_pipeline
from video2vmd.paths import VIDEO_SUFFIXES


def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b""):
            h.update(chunk)
    return h.hexdigest()


def validate_cached_video(video, staged):
    """Verify renamed sources against decoded cached pixels, not filename alone."""
    import cv2
    source, previous = cv2.VideoCapture(str(video)), cv2.VideoCapture(str(staged))
    rows = []
    try:
        count = previous.get(cv2.CAP_PROP_FRAME_COUNT)
        fps = previous.get(cv2.CAP_PROP_FPS)
        for fraction in (.02, .25, .5, .75, .98):
            time_s = fraction*(count-1)/fps
            frames=[]
            for cap in (source,previous):
                cap.set(cv2.CAP_PROP_POS_MSEC,time_s*1000)
                ok,frame=cap.read()
                if not ok:
                    raise ValueError("Cannot verify renamed input against staged cache")
                frames.append(cv2.resize(cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY),(64,64)).astype(float))
            error=float(np.mean(np.abs(frames[0]-frames[1])))
            corr=float(np.corrcoef(frames[0].ravel(),frames[1].ravel())[0,1])
            if error>10 or corr<.95:
                raise ValueError(f"Cached extraction pixel mismatch at {time_s:.2f}s: {error=}, {corr=}")
            rows.append({"time_s":time_s,"mean_absolute_gray_difference":error,"correlation":corr})
    finally:
        source.release();previous.release()
    return rows


def worker(video, out, model):
    out.mkdir(parents=True, exist_ok=True)
    extraction = out/"extraction"
    skeleton = extraction/"skeleton_motion.npz"
    if not skeleton.exists():
        extraction.mkdir(exist_ok=True)
        stat = video.stat()
        candidates = []
        for manifest in (ROOT/"outputs").rglob("input_source.json"):
            if out.parent in manifest.parents:
                continue
            info = json.loads(manifest.read_text(encoding="utf-8"))
            cached = manifest.parent.parent.parent/"skeleton_motion.npz"
            if info["size"] == stat.st_size and info["mtime_ns"] == stat.st_mtime_ns and cached.exists():
                candidates.append((manifest, cached, info))
        candidates.sort(key=lambda item: item[1].stat().st_mtime_ns, reverse=True)
        provenance = {"video":str(video), "video_sha256":sha256(video), "size":stat.st_size,
                      "mtime_ns":stat.st_mtime_ns, "reused":bool(candidates)}
        if candidates:
            manifest, cached, info = candidates[0]
            print(f"Reusing extraction: {cached}", flush=True)
            with np.load(cached, allow_pickle=False) as data:
                payload = {k:data[k] for k in data.files}
            payload["video"] = np.array(str(video))
            np.savez_compressed(skeleton, **payload)
            provenance.update(cached_skeleton=str(cached), cache_manifest=str(manifest),
                              previous_video=info["path"],
                              match="identical source byte size and nanosecond mtime after rename")
            if "camera_joints_3d" not in payload:
                subprocess.run([sys.executable, str(ROOT/"scripts/cache_inspection_camera.py"), str(extraction),
                                "--prediction",str(manifest.parent/"hmr4d_results.pt"),"--device","cpu"], check=True)
        else:
            from video2vmd.extract import extract_motion
            extract_motion(video, skeleton, work_dir=extraction/"cache", device="cuda")
        (out/"provenance.json").write_text(json.dumps(provenance, indent=2)+"\n", encoding="utf-8")
    provenance_path=out/"provenance.json"
    provenance=json.loads(provenance_path.read_text(encoding="utf-8"))
    if provenance["video_sha256"] != sha256(video):
        raise ValueError("Source video changed since extraction; use a fresh study directory")
    if provenance["reused"] and "pixel_verification" not in provenance:
        provenance["pixel_verification"]=validate_cached_video(video,Path(provenance["cache_manifest"]).parent/"0_input_video.mp4")
        provenance_path.write_text(json.dumps(provenance,indent=2)+"\n",encoding="utf-8")
    combined = out/"combined"
    if not (combined/"run.json").exists() or json.loads((combined/"run.json").read_text())["status"] != "completed":
        run_mmd_pipeline(skeleton_npz=skeleton, output_dir=combined, model=model,
                         device="cuda", depth_cleanup=True, foot_lock=True, no_preview=True)
    # Compute the feet-only ablation against EXACTLY the same raw target/mask.
    raw = load_target(combined/"target_motion_raw.npz")
    # Restore serialization mappings, which saved NPZ intentionally omits.
    raw.profile = pmx_profile(model)
    run_meta=json.loads((combined/"run.json").read_text(encoding="utf-8"))
    if run_meta["profile_hash"] != raw.profile.identity:
        raise ValueError("Target profile changed since processing; use a fresh study directory")
    contacts = dict(np.load(combined/"contacts.npz", allow_pickle=False))
    feet, diagnostics = correct_feet(raw, contacts)
    feet.save(out/"target_motion_feet.npz")
    write_vmd(out/"motion_feet.vmd", feet)
    (out/"feet_only_quality.json").write_text(json.dumps({"quality":quality(feet, contacts),
                                                          "correction":diagnostics}, indent=2)+"\n", encoding="utf-8")
    print(f"COMPLETE {video.name}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT/"outputs/diagnostics/noise_study_20261017")
    parser.add_argument("--model", type=Path, default=ROOT/"assets/mmd/ab7_160404/ab7.pmx")
    parser.add_argument("--worker", type=Path)
    args = parser.parse_args()
    args.output = args.output.resolve()
    if args.worker:
        worker(args.worker.resolve(), args.output/args.worker.stem, args.model.resolve())
        return
    args.output.mkdir(parents=True, exist_ok=True)
    videos = sorted(p for p in (ROOT/"assets/video").iterdir() if p.suffix.lower() in VIDEO_SUFFIXES)
    state = {}
    for video in videos:
        started = time.monotonic()
        print(f"START {video.name}", flush=True)
        with (args.output/(video.stem+".log")).open("w", encoding="utf-8") as log:
            process = subprocess.run([sys.executable,"-u",str(Path(__file__).resolve()),"--output",str(args.output),
                                      "--model",str(args.model),"--worker",str(video)], stdout=log,stderr=subprocess.STDOUT,
                                     cwd=ROOT)
        state[video.stem] = {"returncode":process.returncode,"elapsed_s":round(time.monotonic()-started,2)}
        (args.output/"processing.json").write_text(json.dumps(state, indent=2)+"\n",encoding="utf-8")
        print(f"END {video.name}: {state[video.stem]}",flush=True)
    if any(v["returncode"] for v in state.values()):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
