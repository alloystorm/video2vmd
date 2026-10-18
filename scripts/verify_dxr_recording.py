"""Verify an existing dxr_native video/sidecar pair without inference or rendering.

Example:
  python scripts/verify_dxr_recording.py outputs/clip.mp4 --compare outputs/half.mp4

Checks decoded video count/timing, numeric sidecar data, and an independent
camera projection. Writes a JSON report and a contact sheet with body overlays.
These checks establish file/camera consistency, not reconstruction accuracy or
equivalence between target-rig joint centers and SMPL joints.
"""
import argparse
from fractions import Fraction
import json
from pathlib import Path
import shutil
import subprocess

import cv2
import numpy as np
from PIL import Image, ImageDraw


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sidecars(video):
    meta = json.loads(video.with_suffix(".meta.json").read_text(encoding="utf-8"))
    rows = [json.loads(line) for line in
            video.with_suffix(".frames.jsonl").read_text(encoding="utf-8").splitlines()]
    return meta, rows


def verify(video, ffprobe):
    meta, rows = sidecars(video)
    require(meta["format"] == "dxr_native-record/1", "Unsupported recording format")
    fps = Fraction(meta["fps"]["num"], meta["fps"]["den"])
    n, width, height = meta["frames"], meta["width"], meta["height"]
    require(n > 0 and fps > 0, "Empty recording or invalid rate")
    require(len(rows) == n, "Sidecar frame count differs from metadata")
    probe = json.loads(subprocess.check_output([
        ffprobe, "-v", "error", "-select_streams", "v:0", "-show_frames",
        "-show_entries", "stream=width,height,r_frame_rate,codec_name,pix_fmt,color_space:"
        "frame=best_effort_timestamp_time", "-of", "json", str(video)]))
    stream = probe["streams"][0]
    require((stream["width"], stream["height"]) == (width, height), "Video size mismatch")
    require(Fraction(stream["r_frame_rate"]) == fps, "Video frame rate mismatch")
    require(len(probe["frames"]) == n, "Decoded video frame count mismatch")
    times = np.arange(n) / float(fps)
    pts = np.array([float(f["best_effort_timestamp_time"]) for f in probe["frames"]])
    video_time_error = float(np.max(np.abs(pts - times)))
    require(video_time_error < 2e-6, "Video presentation timestamps mismatch")
    characters = meta["characters"]
    require(bool(characters), "No ground-truth actors")
    max_time = max_motion = max_clock = max_projection = max_depth = max_quat = 0.
    inside = total = 0
    for i, row in enumerate(rows):
        require(row["i"] == i, "Non-contiguous sidecar frame index")
        require(np.isfinite([row[k] for k in ("t", "motionTime", "clockTime")]).all(),
                "Non-finite sidecar time")
        max_time = max(max_time, abs(row["t"] - times[i]))
        max_motion = max(max_motion, abs(row["motionTime"] - meta["startSeconds"] - times[i]))
        max_clock = max(max_clock, abs(row["clockTime"] - row["motionTime"] - 1/float(fps)))
        require(len(row["actors"]) == len(characters), "Actor count changed")
        cam = row["camera"]
        eye, forward, up = (np.asarray(cam[k], dtype=float) for k in ("eye", "dir", "up"))
        require(all(v.shape == (3,) and np.isfinite(v).all() for v in (eye, forward, up)),
                "Invalid camera vectors")
        require(abs(np.linalg.norm(forward)-1) < 1e-5 and abs(np.linalg.norm(up)-1) < 1e-5
                and abs(up @ forward) < 1e-5, "Non-orthonormal camera basis")
        right = np.cross(up, forward)  # dxr_native's left-handed world
        intrinsics = np.array([cam[k] for k in ("fx", "fy", "cx", "cy")])
        require(np.isfinite(intrinsics).all() and np.all(intrinsics[:2] > 0),
                "Invalid camera intrinsics")
        expected_fov = np.degrees(2*np.arctan(height/(2*cam["fy"])))
        require(abs(expected_fov-cam["verticalFovDegrees"]) < 1e-4, "Lens/FOV mismatch")
        for actor, character in zip(row["actors"], characters):
            require(actor["index"] == character["index"], "Actor ordering mismatch")
            root = np.asarray(actor["root"], dtype=float)
            require(root.shape == (3,) and np.isfinite(root).all(), "Invalid actor placement root")
            ids = character["frameBones"]
            require(len(set(ids)) == len(ids), "Duplicate frame bone")
            lookup = {b["i"]: b for b in character["bones"]}
            require(all(j in lookup and lookup[j]["canonical"] for j in ids), "Invalid frame-bone map")
            count = len(ids)
            pos = np.asarray(actor["pos"], dtype=float).reshape(-1, 3)
            rot = np.asarray(actor["rot"], dtype=float).reshape(-1, 4)
            px = np.asarray(actor["px"], dtype=float).reshape(-1, 3)
            require(pos.shape == px.shape == (count, 3) and rot.shape == (count, 4),
                    "Joint array lengths differ from rig metadata")
            require(all(np.isfinite(v).all() for v in (pos, rot, px)), "Non-finite joint data")
            max_quat = max(max_quat, float(np.max(np.abs(np.linalg.norm(rot, axis=1)-1))))
            relative = pos-eye
            depth = relative @ forward
            require(np.all(np.abs(depth) > 1e-6), "Joint lies on camera plane")
            projected = np.column_stack((cam["cx"]+cam["fx"]*(relative @ right)/depth,
                                         cam["cy"]-cam["fy"]*(relative @ up)/depth))
            max_projection = max(max_projection, float(np.max(np.abs(projected-px[:, :2]))))
            max_depth = max(max_depth, float(np.max(np.abs(depth-px[:, 2]))))
            inside += int(((px[:, 0] >= 0) & (px[:, 0] < width) & (px[:, 1] >= 0)
                           & (px[:, 1] < height) & (depth > 0)).sum())
            total += count
    require(max_time < 1e-7 and max_motion < 1e-7 and max_clock < 1e-7,
            "Sidecar timestamps mismatch")
    require(max_projection < .01 and max_depth < 1e-5, "Independent camera projection mismatch")
    require(max_quat < 1e-4, "World quaternions are not unit length")
    return {"video": str(video.resolve()), "frames": n, "fps": str(fps), "size": [width, height],
            "codec": stream["codec_name"], "renderer": meta["renderer"],
            "actors": len(characters), "joints_per_actor": [len(c["frameBones"]) for c in characters],
            "video_timestamp_max_error_s": video_time_error,
            "sidecar_timestamp_max_error_s": max_time, "motion_timestamp_max_error_s": max_motion,
            "animation_clock_max_error_s": max_clock,
            "projection_max_error_px": max_projection, "depth_max_error_m": max_depth,
            "quaternion_norm_max_error": max_quat, "joints_in_frame": inside,
            "joint_samples": total, "all_joints_in_frame": inside == total}, meta, rows


def compare(video, other):
    a, x = sidecars(video)
    b, y = sidecars(other)
    require(a["frames"] == b["frames"] and a["fps"] == b["fps"]
            and a["startSeconds"] == b["startSeconds"], "Comparison timeline differs")
    require(a["characters"] == b["characters"], "Comparison rig/source differs")
    max_pos = max_rot = max_px = 0.
    for first, second in zip(x, y):
        for u, v in zip(first["actors"], second["actors"]):
            max_pos = max(max_pos, float(np.max(np.abs(np.array(u["pos"])-v["pos"]))))
            max_rot = max(max_rot, float(np.max(np.abs(np.array(u["rot"])-v["rot"]))))
            px = np.array(u["px"]).reshape(-1, 3)[:, :2]/[a["width"], a["height"]]
            py = np.array(v["px"]).reshape(-1, 3)[:, :2]/[b["width"], b["height"]]
            max_px = max(max_px, float(np.max(np.abs(px-py))))
    require(max_pos < 1e-6 and max_rot < 1e-6 and max_px < 1e-5,
            "Repeated recording differs beyond tolerance")
    return {"other": str(other.resolve()), "max_position_difference_m": max_pos,
            "max_quaternion_component_difference": max_rot, "max_normalized_pixel_difference": max_px}


def contact_sheet(video, meta, rows):
    indices = sorted(set([0, len(rows)//3, 2*len(rows)//3, len(rows)-1]))
    tile_w = min(960, meta["width"])
    scale = tile_w/meta["width"]
    tile_h = round(meta["height"]*scale)
    sheet = Image.new("RGB", (tile_w*2, (tile_h+28)*2), "#111111")
    main = {"HipMaster", "Torso", "Torso2", "Neck", "Head", "LeftShoulder", "RightShoulder",
            "LeftArm", "RightArm", "LeftElbow", "RightElbow", "LeftWrist", "RightWrist",
            "LeftLeg", "RightLeg", "LeftKnee", "RightKnee", "LeftAnkle", "RightAnkle",
            "LeftToe", "RightToe"}
    cap = cv2.VideoCapture(str(video))
    try:
        for k, n in enumerate(indices):
            cap.set(cv2.CAP_PROP_POS_FRAMES, n)
            ok, frame = cap.read()
            require(ok, f"Cannot decode overlay frame {n}")
            img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).resize((tile_w, tile_h))
            draw = ImageDraw.Draw(img)
            for character, actor in zip(meta["characters"], rows[n]["actors"]):
                lookup = {b["i"]: b for b in character["bones"]}
                pixels = {j: p for j, p in zip(character["frameBones"],
                          np.array(actor["px"]).reshape(-1, 3)) if lookup[j]["canonical"] in main and p[2] > 0}
                for j, p in pixels.items():
                    parent = lookup[j]["parent"]
                    visited = {j}
                    while parent in lookup and parent not in pixels and parent not in visited:
                        visited.add(parent)
                        parent = lookup[parent]["parent"]
                    x, y = p[:2]*scale
                    if parent in pixels:
                        q = pixels[parent][:2]*scale
                        draw.line((x, y, *q), fill="#ffc857", width=2)
                    draw.ellipse((x-3, y-3, x+3, y+3), fill="#e83f6f")
            left, top = k%2*tile_w, k//2*(tile_h+28)
            sheet.paste(img, (left, top))
            ImageDraw.Draw(sheet).text((left+8, top+tile_h+6),
                                      f"frame {n} | video {rows[n]['t']:.3f}s | motion {rows[n]['motionTime']:.3f}s",
                                      fill="white")
    finally:
        cap.release()
    output = video.with_suffix(".verification.png")
    sheet.save(output)
    return str(output.resolve())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video", type=Path)
    parser.add_argument("--compare", type=Path, help="same motion/camera/settings at another resolution")
    parser.add_argument("--ffprobe", default=shutil.which("ffprobe"))
    args = parser.parse_args()
    require(bool(args.ffprobe), "ffprobe is required; use --ffprobe PATH")
    report, meta, rows = verify(args.video, args.ffprobe)
    if args.compare:
        other, _, _ = verify(args.compare, args.ffprobe)
        report["comparison_recording"] = other
        report["repeat_comparison"] = compare(args.video, args.compare)
    report["overlay"] = contact_sheet(args.video, meta, rows)
    report["passed"] = True
    args.video.with_suffix(".verification.json").write_text(json.dumps(report, indent=2)+"\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
