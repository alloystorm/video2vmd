"""Video or cached human skeleton -> MMD target motion -> VMD/BVH."""
from datetime import datetime
from importlib.metadata import version
import json
from pathlib import Path
import shutil
import sys
import time

import numpy as np

from .. import paths
from ..errors import classify_error
from ..human_motion import load_human_motion
from ..progress import ProgressWriter
from ..run_report import write_failed_report, write_run_json
from .profile import TESTED_CODEC_REVISION, pmx_profile, standard_profile
from .retarget import retarget
from .contacts import correct_feet, detect_contacts, quality, constrain_ik_knees
from .export import write_bvh, write_motion_encoder, write_vmd


def run_mmd_pipeline(video=None, *, image=None, skeleton_npz=None, output_dir=None,
                     device="cpu", output_format="vmd", model=None, foot_lock=False,
                     floor=None, motion_encoder=False, no_preview=False, progress=None, ik_mode="fk", depth_cleanup=False,
                     hand_motion=None):
    started, clock = datetime.now(), time.monotonic()
    source = Path(image or video or skeleton_npz).resolve()
    run_dir = Path(output_dir).resolve() if output_dir else paths.new_run_dir(source, started)
    run_dir.mkdir(parents=True, exist_ok=True)
    writer = progress or ProgressWriter()
    stage = "profile"
    try:
        profile = pmx_profile(model) if model else standard_profile()
        stage = "extract"
        writer.emit(stage, .15, "Loading or extracting human motion")
        motion_npz = run_dir / "skeleton_motion.npz"
        if skeleton_npz:
            if Path(skeleton_npz).resolve() != motion_npz.resolve():
                shutil.copyfile(skeleton_npz, motion_npz)
        else:
            from ..extract import extract_motion, temp_hold_video
            if image:
                with temp_hold_video(Path(image)) as hold:
                    extract_motion(hold, motion_npz, work_dir=run_dir/"cache", device=device, source_image=Path(image))
            else:
                extract_motion(Path(video), motion_npz, work_dir=run_dir/"cache", device=device)
        human = load_human_motion(motion_npz)
        stage = "retarget"
        writer.emit(stage, .55, "Retargeting body motion to MMD")
        raw, calibration = retarget(human, profile)
        if ik_mode == "ik":
            raw = constrain_ik_knees(raw)
        contacts = detect_contacts(raw, floor=floor)
        before = quality(raw, contacts)
        base=raw
        depth_report={"enabled": False}
        depth_outputs={}
        if depth_cleanup:
            from ..depth_cleanup import clean_skeleton
            stage="depth_cleanup"
            writer.emit(stage,.65,"Regularizing root depth with contact and reprojection guards")
            cleaned_path=run_dir/"skeleton_motion_corrected.npz"
            human,diagnostics,depth_report=clean_skeleton(motion_npz,cleaned_path,contacts=contacts)
            depth_report["enabled"]=True
            np.savez_compressed(run_dir/"depth_cleanup.npz",**diagnostics)
            (run_dir/"depth_cleanup.json").write_text(json.dumps(depth_report,indent=2)+"\n",encoding="utf-8")
            base,_=retarget(human,profile)
            if ik_mode=="ik":
                base=constrain_ik_knees(base)
            base.save(run_dir/"target_motion_depth.npz")
            depth_outputs={"skeleton_motion_corrected":cleaned_path.name,"depth_cleanup":"depth_cleanup.json",
                           "depth_cleanup_evidence":"depth_cleanup.npz","target_motion_depth":"target_motion_depth.npz"}
        corrected, correction = correct_feet(base, contacts) if foot_lock else (base, {"corrected_frames": 0})
        if ik_mode == "ik" and foot_lock:
            corrected = constrain_ik_knees(corrected)
        hand_report = {"enabled": False}
        hand_outputs = {}
        hand_payload = None
        if hand_motion:
            from ..hand_motion import load_hand_motion, resample_hand_motion, save_hand_motion, world_to_camera_rotations
            from .hands import retarget_hands
            stage = "hands"
            writer.emit(stage, .78, "Retargeting cached finger poses")
            hand_payload = resample_hand_motion(load_hand_motion(hand_motion), np.arange(raw.n_frames)/raw.fps)
            hand_camera = world_to_camera_rotations(motion_npz, hand_payload["timestamps"])
            corrected.save(run_dir/"target_motion_body.npz")
            raw, _ = retarget_hands(raw, hand_payload, hand_camera)
            corrected, hand_report = retarget_hands(corrected, hand_payload, hand_camera)
            if depth_cleanup:
                base, _ = retarget_hands(base, hand_payload, hand_camera)
                base.save(run_dir/"target_motion_depth.npz")
            save_hand_motion(run_dir/"hand_motion.npz", hand_payload)
            (run_dir/"hand_retargeting.json").write_text(json.dumps(hand_report, indent=2)+"\n", encoding="utf-8")
            hand_outputs = {"hand_motion": "hand_motion.npz", "hand_retargeting": "hand_retargeting.json",
                            "target_motion_body": "target_motion_body.npz"}
            calibration["hands"] = hand_report
        correction["max_root_correction_m"] = float(np.linalg.norm(corrected.translations-raw.translations, axis=2).max())
        after = quality(corrected, contacts)
        raw.save(run_dir/"target_motion_raw.npz")
        corrected.save(run_dir/"target_motion.npz")
        np.savez_compressed(run_dir/"contacts.npz", **contacts)
        metrics = {"before": before, "after_depth_cleanup":quality(base,contacts), "after": after, "correction": correction,
                   "depth_cleanup":depth_report, "hands": hand_report,
                   "floor_m": contacts["floor_m"], "foot_lock": foot_lock,
                   "contact_policy": {key: contacts[key] for key in
                                      ("height_threshold_m", "speed_threshold_m_s", "minimum_dwell_s")}}
        (run_dir/"quality.json").write_text(json.dumps(metrics, indent=2)+"\n", encoding="utf-8")
        stage = "export"
        writer.emit(stage, .85, f"Writing {output_format.upper()} motion")
        output = run_dir / f"motion.{output_format}"
        if output_format == "vmd":
            write_vmd(run_dir/"motion_raw.vmd", raw, ik_mode=ik_mode)
            write_vmd(run_dir/"motion_corrected.vmd", corrected, ik_mode=ik_mode)
            # Keep the existing filename as an exact copy of corrected output.
            shutil.copyfile(run_dir/"motion_corrected.vmd", output)
        else:
            write_bvh(output, corrected)
        outputs = {"motion": output.name, "skeleton_motion": motion_npz.name,
                   "target_motion_raw": "target_motion_raw.npz", "target_motion": "target_motion.npz",
                   "contacts": "contacts.npz", "quality": "quality.json"}
        outputs.update(depth_outputs)
        outputs.update(hand_outputs)
        if output_format == "vmd":
            outputs.update(motion_raw="motion_raw.vmd", motion_corrected="motion_corrected.vmd")
        if output_format == "bvh":
            outputs["bvh_metadata"] = output.name+".json"
        if motion_encoder:
            canonical = corrected if not model else retarget(human, standard_profile())[0]
            if model and foot_lock:
                canonical = correct_feet(canonical, detect_contacts(canonical))[0]
            if model and hand_payload is not None:
                from .hands import retarget_hands
                canonical, _ = retarget_hands(canonical, hand_payload, hand_camera)
            write_motion_encoder(run_dir/"motion_encoder.npy", canonical)
            outputs["motion_encoder"] = "motion_encoder.npy"
        if not no_preview:
            from .preview import render_preview
            stage = "preview"
            writer.emit(stage, .9, "Rendering skeleton diagnostics")
            render_preview(run_dir/"videos"/"target_skeleton.mp4", raw, corrected, contacts)
            outputs["preview"] = "videos/target_skeleton.mp4"
        write_run_json(run_dir, {"status": "completed", "started": started.isoformat(timespec="seconds"),
                                "elapsed_s": round(time.monotonic()-clock, 2), "command": sys.argv,
                                "format": output_format, "ik_mode": ik_mode, "profile": profile.name, "profile_hash": profile.identity,
                                "codec": {"name": "vmd2png", "version": version("vmd2png"), "pinned_revision": TESTED_CODEC_REVISION},
                                "model": str(Path(model).resolve()) if model else None,
                                "source": str(source), "skeleton_reused_from": str(skeleton_npz) if skeleton_npz else None,
                                "device": device, "fps": corrected.fps, "n_frames": corrected.n_frames,
                                "depth_cleanup":depth_report,
                                "hands":hand_report, "hand_motion_reused_from":str(Path(hand_motion).resolve()) if hand_motion else None,
                                "units": "meters", "calibration": calibration, "outputs": outputs})
        writer.emit("done", 1., "Completed")
        print(f"saved {output} ({corrected.n_frames} frames at {corrected.fps:g} fps)")
        return run_dir
    except Exception as exc:
        code, message = classify_error(exc)
        write_failed_report(run_dir, stage=stage, error_code=code, message=message)
        writer.emit("failed", .55 if stage == "retarget" else .15, message)
        raise
