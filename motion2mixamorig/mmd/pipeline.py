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
                     floor=None, motion_encoder=False, no_preview=False, progress=None, ik_mode="fk"):
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
        contacts = detect_contacts(raw, floor=floor)
        before = quality(raw, contacts)
        corrected, correction = correct_feet(raw, contacts) if foot_lock else (raw, {"corrected_frames": 0})
        if ik_mode == "ik":
            corrected = constrain_ik_knees(corrected)
        correction["max_root_correction_m"] = float(np.linalg.norm(corrected.translations-raw.translations, axis=2).max())
        after = quality(corrected, contacts)
        corrected.save(run_dir/"target_motion.npz")
        np.savez_compressed(run_dir/"contacts.npz", **contacts)
        metrics = {"before": before, "after": after, "correction": correction,
                   "floor_m": contacts["floor_m"], "foot_lock": foot_lock,
                   "contact_policy": {key: contacts[key] for key in
                                      ("height_threshold_m", "speed_threshold_m_s", "minimum_dwell_s")}}
        (run_dir/"quality.json").write_text(json.dumps(metrics, indent=2)+"\n", encoding="utf-8")
        stage = "export"
        writer.emit(stage, .85, f"Writing {output_format.upper()} motion")
        output = run_dir / f"motion.{output_format}"
        if output_format == "vmd":
            write_vmd(output, corrected, ik_mode=ik_mode)
        else:
            write_bvh(output, corrected)
        outputs = {"motion": output.name, "skeleton_motion": motion_npz.name,
                   "target_motion": "target_motion.npz", "contacts": "contacts.npz", "quality": "quality.json"}
        if output_format == "bvh":
            outputs["bvh_metadata"] = output.name+".json"
        if motion_encoder:
            canonical = corrected if not model else retarget(human, standard_profile())[0]
            if model and foot_lock:
                canonical = correct_feet(canonical, detect_contacts(canonical))[0]
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
                                "units": "meters", "calibration": calibration, "outputs": outputs})
        writer.emit("done", 1., "Completed")
        print(f"saved {output} ({corrected.n_frames} frames at {corrected.fps:g} fps)")
        return run_dir
    except Exception as exc:
        code, message = classify_error(exc)
        write_failed_report(run_dir, stage=stage, error_code=code, message=message)
        writer.emit("failed", .55 if stage == "retarget" else .15, message)
        raise
