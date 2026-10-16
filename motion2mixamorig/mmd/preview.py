"""Headless front/side skeleton diagnostics, independent of Mixamo and Blender."""
from pathlib import Path
import shutil
import subprocess

import numpy as np


def render_preview(path, raw, corrected, contacts):
    import cv2
    before, _ = raw.fk()
    after, _ = corrected.fk()
    profile = corrected.profile
    width, height = 960, 720
    core = {"Master", "Center", "Groove", "Waist", "HipMaster", "Torso", "Torso1", "Torso2", "Neck", "Head"}
    core.update(side+part for side in ("Left", "Right")
                for part in ("Shoulder", "Arm", "ArmTwist", "Elbow", "HandTwist", "Wrist", "Leg", "Knee", "Ankle", "Toe"))
    used = [i for i, n in enumerate(profile.names) if n in core]
    all_points = after[:, used].reshape(-1, 3)
    low, high = all_points.min(axis=0), all_points.max(axis=0)
    center = (low+high)/2
    scale = min(420/max(high[0]-low[0], high[2]-low[2], .5), 285/max(high[1]-low[1], .5))
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    intermediate = path.with_name(path.stem+".mp4v.mp4")
    writer = cv2.VideoWriter(str(intermediate), cv2.VideoWriter_fourcc(*"mp4v"), corrected.fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"Cannot open skeleton preview writer: {path}")
    try:
        for frame in range(corrected.n_frames):
            image = np.full((height, width, 3), 245, np.uint8)
            for panel, (points, title, axis) in enumerate(((before, "Raw / front", 0), (after, "Corrected / front", 0),
                                                         (before, "Raw / side", 2), (after, "Corrected / side", 2))):
                origin = np.array([240+(panel%2)*480, 195+(panel//2)*360])
                coords = points[frame]
                xy = np.column_stack([(coords[:, axis]-center[axis])*scale,
                                      -(coords[:, 1]-center[1])*scale]) + origin
                cv2.putText(image, title, (panel%2*480+15, panel//2*360+25), cv2.FONT_HERSHEY_SIMPLEX, .6, (35, 35, 35), 1)
                for i in used:
                    parent = profile.parents[i]
                    if parent >= 0:
                        a, b = tuple(np.int32(xy[parent])), tuple(np.int32(xy[i]))
                        cv2.line(image, a, b, (160, 90, 40), 2, cv2.LINE_AA)
                floor_y = int(origin[1]-(contacts["floor_m"]-center[1])*scale)
                cv2.line(image, (panel%2*480, floor_y), (panel%2*480+479, floor_y), (140, 140, 140), 1)
            cv2.putText(image, f"frame {frame+1}/{corrected.n_frames} | {profile.name} | meters, 30 fps",
                        (20, height-12), cv2.FONT_HERSHEY_SIMPLEX, .5, (30, 30, 30), 1)
            if frame == 0:
                cv2.imwrite(str(path.with_suffix(".png")), image)
            writer.write(image)
    finally:
        writer.release()
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg:
        subprocess.run([ffmpeg, "-y", "-loglevel", "error", "-i", str(intermediate), "-c:v", "libx264",
                        "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(path)], check=True)
        intermediate.unlink()
    else:
        intermediate.replace(path)
    return path
