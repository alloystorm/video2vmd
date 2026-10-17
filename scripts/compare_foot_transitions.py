"""Compare cached foot cleanup with the current solver on unchanged contact labels."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from video2vmd.inspection.server import load_target
from video2vmd.inspection.camera import camera_inspection
from video2vmd.mmd.contacts import correct_feet, foot_points, quality
from video2vmd.mmd.export import write_vmd
from video2vmd.mmd.profile import pmx_profile


def checksum(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def masks(contacts):
    contact, weights = contacts["contact"], contacts["weights"]
    pairs = contact[1:] & contact[:-1]
    holds = pairs & (weights[1:] >= 1.-1e-8) & (weights[:-1] >= 1.-1e-8)
    return {"hold": holds, "transition": pairs & ~holds, "all": pairs}


def metrics(motion, base, contacts):
    points = foot_points(motion)
    speed = np.linalg.norm(np.diff(points[:, :, [0, 2]], axis=0), axis=2)*motion.fps
    result = quality(motion, contacts)
    result["slip"] = {key: {"pairs": int(mask.sum()),
                            "mean_m_s": float(speed[mask].mean()) if mask.any() else None}
                      for key, mask in masks(contacts).items()}
    indices = [motion.profile.index[side+"Ankle"] for side in ("Left", "Right")]
    delta = (motion.fk()[0]-base.fk()[0])[:, indices][:, :, [0, 2]]
    for order, label in ((1, "speed_m_s"), (2, "acceleration_m_s2")):
        values = np.linalg.norm(np.diff(delta, n=order, axis=0), axis=2)*motion.fps**order
        result["horizontal_correction_"+label] = {
            "max": float(values.max()) if values.size else 0.,
            "rms": float(np.sqrt(np.mean(values**2))) if values.size else 0.}
    return result


def image_change(folder, old, current):
    raw = load_target(folder/"target_motion_raw.npz")
    bridge = camera_inspection(folder/"skeleton_motion.npz", {"raw": raw, "before": old, "corrected": current})
    if bridge is None:
        return None
    ids = [raw.profile.index[side+"Ankle"] for side in ("Left", "Right")]
    positions = np.frombuffer(bridge.targets, dtype="<f4").reshape(3, raw.n_frames, len(raw.profile.names), 3)[:, :, ids]
    intrinsics = np.asarray(bridge.metadata["intrinsics"])
    with np.load(folder/"skeleton_motion.npz", allow_pickle=False) as data:
        size = data["camera_image_size"]
    if np.any(positions[:, :, :, 2] <= 0):
        raise ValueError("Cannot compare projections behind the estimated camera")
    projected = (positions[:, :, :, :2]/positions[:, :, :, 2, None]
                 * intrinsics[None, :, None, :2] + intrinsics[None, :, None, 2:])*size
    delta = np.linalg.norm(projected[2]-projected[1], axis=2)/size[1]*100.
    return {"definition": "After versus before under estimated camera; displacement, not observed-landmark error",
            "rms_image_height_percent": float(np.sqrt(np.mean(delta**2))),
            "max_image_height_percent": float(delta.max())}


def plot_window(destination, old, current, base, contacts):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    indices = [old.profile.index[side+"Ankle"] for side in ("Left", "Right")]
    base_ankles = base.fk()[0][:, indices][:, :, [0, 2]]
    deltas = [(m.fk()[0][:, indices][:, :, [0, 2]]-base_ankles) for m in (old, current)]
    acceleration = [np.linalg.norm(np.diff(d, n=2, axis=0), axis=2)*old.fps**2 for d in deltas]
    peak, side = np.unravel_index(np.argmax(acceleration[0]), acceleration[0].shape)
    time = np.arange(old.n_frames)/old.fps
    window = (max(0., (peak+1)/old.fps-.6), min(time[-1], (peak+1)/old.fps+.6))
    visible = (time >= window[0]) & (time <= window[1])
    fig, axes = plt.subplots(3, 1, figsize=(10, 7), sharex=True, layout="constrained")
    points = [foot_points(m) for m in (base, old, current)]
    endpoint = side*2 + int(contacts["contact"][peak+1, side*2+1])
    origin = points[0][peak+1, endpoint, [0, 2]]
    for a, axis in enumerate((0, 2)):
        for p, label, color in zip(points, ("Before foot cleanup", "Previous solver", "New solver"), ("#999999", "#c15b32", "#007f86")):
            axes[a].plot(time[visible], (p[visible, endpoint, axis]-origin[a])*1000, label=label, color=color)
        axes[a].set_ylabel(f"Sole {'XZ'[a]} (mm)")
    for values, label, color in zip(acceleration, ("Previous solver", "New solver"), ("#c15b32", "#007f86")):
        axes[2].plot(time[1:-1][visible[1:-1]], values[visible[1:-1], side], label=label, color=color)
    axes[2].axhline(12., color="#555555", linestyle=":", label="Target correction limit")
    axes[2].set_ylabel("Correction accel. (m/s²)")
    axes[2].set_xlabel("Time (seconds)")
    for ax in axes:
        ax.set_xlim(window)
        ax.grid(alpha=.2)
    axes[0].legend(loc="upper left", ncol=3)
    fig.suptitle(f"{destination.name}: {'left' if side == 0 else 'right'} foot, largest previous correction acceleration\n"
                 "Trajectory diagnostics; the source video remains the reference for intent")
    fig.savefig(destination/"transition.png", dpi=140)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("study", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--clip", action="append")
    parser.add_argument("--model", type=Path)
    parser.add_argument("--plots", action="store_true")
    args = parser.parse_args()
    study, output = args.study.resolve(), args.output.resolve()
    if output == study or study in output.parents or output in study.parents:
        parser.error("Output must be separate from the original study")
    if output.exists():
        parser.error("Choose a fresh output directory to preserve comparisons")
    output.mkdir(parents=True)
    rows = []
    for folder in sorted(study.glob("v*/combined")):
        clip = folder.parent.name
        if args.clip and clip not in args.clip:
            continue
        started = time.monotonic()
        base_path, old_path = folder/"target_motion_depth.npz", folder/"target_motion.npz"
        contacts_path = folder/"contacts.npz"
        base, old = load_target(base_path), load_target(old_path)
        if (base.n_frames != old.n_frames or base.fps != old.fps
                or base.profile.identity != old.profile.identity):
            raise ValueError("Baseline and depth-stage motion must share timing and target geometry")
        # The inspection loader omits serialization mappings. Preserve the saved
        # profile identity when writing an NPZ without an optional PMX reload.
        with np.load(base_path, allow_pickle=False) as data:
            base.profile.identity = json.loads(str(data["metadata"]))["profile_hash"]
        with np.load(contacts_path, allow_pickle=False) as data:
            contacts = {key: data[key] for key in data.files}
        current, diagnostics = correct_feet(base, contacts)
        destination = output/clip
        destination.mkdir()
        if args.model:
            profile = pmx_profile(args.model)
            metadata = json.loads((folder/"run.json").read_text(encoding="utf-8"))
            if profile.identity != metadata["profile_hash"]:
                raise ValueError("Comparison model differs from cached target")
            current.profile = profile
            write_vmd(destination/"motion.vmd", current)
        current.save(destination/"target_motion.npz")
        if args.plots:
            plot_window(destination, old, current, base, contacts)
        row = {"clip": clip, "before": metrics(old, base, contacts),
               "after": metrics(current, base, contacts), "correction": diagnostics,
               "image_change": image_change(folder, old, current),
               "source_sha256": {str(path): checksum(path) for path in (base_path, old_path, contacts_path)},
               "elapsed_s": time.monotonic()-started}
        rows.append(row)
        (destination/"comparison.json").write_text(json.dumps(row, indent=2)+"\n", encoding="utf-8")
        print(f"{clip}: transition {row['before']['slip']['transition']['mean_m_s']} -> "
              f"{row['after']['slip']['transition']['mean_m_s']} m/s; "
              f"fit failures {diagnostics['transition_fit_failures']}; {row['elapsed_s']:.1f}s", flush=True)
    if not rows:
        raise ValueError("No matching cached clips")
    pooled = {}
    for stage in ("before", "after"):
        pooled[stage] = {}
        for kind in ("hold", "transition", "all"):
            entries = [row[stage]["slip"][kind] for row in rows]
            count = sum(entry["pairs"] for entry in entries)
            pooled[stage][kind] = {"pairs": count, "mean_m_s":
                sum(entry["pairs"]*(entry["mean_m_s"] or 0.) for entry in entries)/count if count else None}
    report = {"study": str(study), "solver_sha256": checksum(ROOT/"video2vmd/mmd/contacts.py"),
              "pooled": pooled, "clips": rows}
    (output/"comparison.json").write_text(json.dumps(report, indent=2)+"\n", encoding="utf-8")
    lines = ["# Foot-transition comparison", "",
             "Same cached depth-stage motion, original contact masks, weights and anchors. No inference rerun.",
             "Speed/acceleration bounds concern the horizontal correction, not the dancer's motion.",
             "These are solver diagnostics, not reconstruction accuracy or proof of visual improvement.", "",
             "| Clip | Transition speed before/after (mm/s) | Hold speed before/after (mm/s) | Correction acceleration max before/after (m/s²) |",
             "|---|---:|---:|---:|"]
    for row in rows:
        cells = [row["clip"]]
        for kind in ("transition", "hold"):
            cells.append(" / ".join(f"{row[s]['slip'][kind]['mean_m_s']*1000:.1f}" if row[s]['slip'][kind]['mean_m_s'] is not None
                                    else "n/a" for s in ("before", "after")))
        cells.append(" / ".join(f"{row[s]['horizontal_correction_acceleration_m_s2']['max']:.2f}" for s in ("before", "after")))
        lines.append("| " + " | ".join(cells) + " |")
    lines += ["", "Pair-weighted means:"]
    for kind in ("transition", "hold", "all"):
        lines.append(f"- {kind}: " + " -> ".join(f"{pooled[s][kind]['mean_m_s']*1000:.1f} mm/s" if pooled[s][kind]['mean_m_s'] is not None
                                                else "n/a" for s in ("before", "after")))
    lines += ["", "## Tradeoffs and artifacts", "",
              "The fit can move a planted foot away from the original interval-median anchor to reduce boundary pulls. "
              "This distance and estimated-camera image displacement are changes, not accuracy errors. "
              "Vertical floor clearance and foot pitch use the previous solver. The derivative bounds apply to planned ankle-target "
              "corrections; unreachable leg targets can differ after IK (see reach residuals).", "",
              "| Clip | Mean anchor distance before/after (mm) | Added image displacement max (% height) | Body jerk before/after (m/s³) |",
              "|---|---:|---:|---:|"]
    for row in rows:
        image = row["image_change"]
        lines.append("| " + " | ".join((row["clip"],
            " / ".join(f"{row[s]['mean_contact_anchor_error_m']*1000:.1f}" for s in ("before", "after")),
            f"{image['max_image_height_percent']:.3f}" if image else "n/a",
            " / ".join(f"{row[s]['body_jerk_rms_m_s3']:.1f}" for s in ("before", "after")))) + " |")
    for row in rows:
        clip = row["clip"]
        lines += ["", f"### {clip}", "", f"[Saved target motion]({clip}/target_motion.npz) · [Full metrics]({clip}/comparison.json)"]
        if args.model:
            lines.append(f"[New VMD]({clip}/motion.vmd)")
        if args.plots:
            lines += ["", f"![Transition diagnostics]({clip}/transition.png)"]
    (output/"comparison.md").write_text("\n".join(lines)+"\n", encoding="utf-8")


if __name__ == "__main__":
    main()
