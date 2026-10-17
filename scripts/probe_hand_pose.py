"""Sample cached videos to compare full-frame and body-guided hand detection.

Run in an isolated environment with mediapipe, numpy and opencv-contrib-python.
Writes observations and contact sheets only; never changes body/animation caches.
"""
import argparse
import hashlib
import json
from pathlib import Path
import time

import cv2
import mediapipe as mp
import numpy as np

CONNECTIONS = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8),
               (5, 9), (9, 10), (10, 11), (11, 12), (9, 13), (13, 14), (14, 15),
               (15, 16), (13, 17), (0, 17), (17, 18), (18, 19), (19, 20)]


def detect(detector, frame, origin=(0, 0)):
    result = detector.detect(mp.Image(image_format=mp.ImageFormat.SRGB,
                                     data=np.ascontiguousarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))))
    hands = []
    for landmarks, world, handedness in zip(result.hand_landmarks, result.hand_world_landmarks, result.handedness):
        xy = np.array([[point.x*frame.shape[1]+origin[0], point.y*frame.shape[0]+origin[1]] for point in landmarks])
        hands.append({"image_xy_px": xy.tolist(),
                      "world_xyz_m": [[point.x, point.y, point.z] for point in world],
                      "handedness_label": handedness[0].category_name,
                      "handedness_score": handedness[0].score})
    return hands


def nearest(hands, wrist, gate):
    if not hands:
        return None
    distances = [np.linalg.norm(np.asarray(hand["image_xy_px"])[0]-wrist) for hand in hands]
    index = int(np.argmin(distances))
    return index if distances[index] <= gate else None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("study", type=Path)
    parser.add_argument("--videos", type=Path, default=Path("assets/video"))
    parser.add_argument("--model", type=Path, default=Path("weights/mediapipe/hand_landmarker.task"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=12)
    parser.add_argument("--clip", action="append")
    parser.add_argument("--range", dest="time_range", nargs=2, type=float, metavar=("START", "END"),
                        help="Probe every source frame in this time interval instead of sparse samples")
    parser.add_argument("--body-keypoints", type=Path, help="Directory of numeric COCO-17 NPZs: keypoints, timestamps, image_size")
    args = parser.parse_args()
    if args.samples < 1:
        parser.error("--samples must be positive")
    if args.time_range and not (0 <= args.time_range[0] < args.time_range[1]):
        parser.error("--range requires 0 <= START < END")
    if args.output.exists():
        parser.error("Use a fresh output directory")
    args.output.mkdir(parents=True)
    options = mp.tasks.vision.HandLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=str(args.model.resolve())),
        running_mode=mp.tasks.vision.RunningMode.IMAGE, num_hands=2,
        min_hand_detection_confidence=.5, min_hand_presence_confidence=.5)
    summary = {"mediapipe_version": mp.__version__,
               "model_sha256": hashlib.sha256(args.model.read_bytes()).hexdigest(),
               "mode": "Independent IMAGE detections; not a temporal tracking evaluation",
               "score_semantics": "Handedness classification score is not landmark/pose confidence",
               "matching": "Nearest predicted wrist within 0.6 projected forearm lengths (minimum 20 px); not annotated ground truth",
               "crop_guidance": "observed COCO-17" if args.body_keypoints else "projected body reconstruction",
               "clips": []}
    with mp.tasks.vision.HandLandmarker.create_from_options(options) as detector:
        for folder in sorted(args.study.glob("v*/combined")):
            clip = folder.parent.name
            if args.clip and clip not in args.clip:
                continue
            videos = [path for path in args.videos.glob(clip+".*") if path.suffix.lower() in (".mp4", ".webm", ".mov")]
            if len(videos) != 1:
                raise ValueError(f"Expected one source video for {clip}")
            with np.load(folder/"skeleton_motion.npz", allow_pickle=False) as data:
                camera = data["camera_joints_3d"]
                intrinsics = data["camera_intrinsics"]
                size = data["camera_image_size"]
                timestamps = data["timestamps"]
                names = list(data["joint_names"])
            seeds = None
            if args.body_keypoints:
                with np.load(args.body_keypoints/(clip+".npz"), allow_pickle=False) as data:
                    seeds, seed_times, seed_size = data["keypoints"], data["timestamps"], data["image_size"]
            cap = cv2.VideoCapture(str(videos[0]))
            rows, tiles = [], []
            started = time.monotonic()
            try:
                fps = cap.get(cv2.CAP_PROP_FPS)
                frame_indices = ([round(float(t)*fps) for t in np.linspace(timestamps[0], timestamps[-1], args.samples+2)[1:-1]]
                                 if args.time_range is None else range(round(args.time_range[0]*fps),
                                     min(round(args.time_range[1]*fps), int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))))
                sheet_frames = set(np.linspace(0, len(frame_indices)-1, min(12, len(frame_indices))).astype(int))
                for frame_step, frame_index in enumerate(frame_indices):
                    if round(cap.get(cv2.CAP_PROP_POS_FRAMES)) != frame_index:
                        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
                    ok, frame = cap.read()
                    if not ok:
                        raise ValueError(f"Cannot decode {videos[0]} at frame {frame_index}")
                    sample = int(np.argmin(abs(timestamps-frame_index/fps)))
                    k = intrinsics if intrinsics.ndim == 2 else intrinsics[sample]
                    projected = camera[sample] @ k.T
                    projected = projected[:, :2]/projected[:, 2, None]*np.array([frame.shape[1], frame.shape[0]])/size
                    full = detect(detector, frame)
                    frame_rows = []
                    for side in ("left", "right"):
                        wrist, elbow = (projected[names.index(side+"_"+part)] for part in ("wrist", "elbow"))
                        if seeds is not None:
                            seed_frame = int(np.argmin(abs(seed_times-frame_index/fps)))
                            wi, ei = (9, 7) if side == "left" else (10, 8)
                            wrist, elbow = seeds[seed_frame, [wi, ei], :2]*np.array([frame.shape[1], frame.shape[0]])/seed_size
                        forearm = float(np.linalg.norm(wrist-elbow))
                        center = wrist+.25*(wrist-elbow)
                        radius = max(48., .85*forearm)
                        x0, y0 = np.maximum(0, np.floor(center-radius)).astype(int)
                        x1, y1 = np.minimum([frame.shape[1], frame.shape[0]], np.ceil(center+radius)).astype(int)
                        if x1 <= x0 or y1 <= y0:
                            frame_rows.append({"side": side, "crop_valid": False, "full_match": None, "crop_match": None})
                            continue
                        crop = frame[y0:y1, x0:x1]
                        candidates = detect(detector, crop, (int(x0), int(y0)))
                        gate = max(20., .6*forearm)
                        fi, ci = nearest(full, wrist, gate), nearest(candidates, wrist, gate)
                        row = {"side": side, "crop_valid": True, "crop_xyxy": [int(x0), int(y0), int(x1), int(y1)],
                               "body_wrist_xy": wrist.tolist(), "forearm_px": forearm,
                               "full_match": fi, "crop_match": ci, "crop_candidates": candidates}
                        frame_rows.append(row)
                        if frame_step not in sheet_frames:
                            continue
                        tile = cv2.resize(crop, (256, 256))
                        for index, hand in enumerate(candidates):
                            xy = (np.asarray(hand["image_xy_px"])-[x0, y0])/[x1-x0, y1-y0]*256
                            color = (0, 210, 70) if index == ci else (0, 150, 255)
                            for a, b in CONNECTIONS:
                                cv2.line(tile, tuple(np.int32(xy[a])), tuple(np.int32(xy[b])), color, 1, cv2.LINE_AA)
                            for p in xy:
                                cv2.circle(tile, tuple(np.int32(p)), 2, color, -1)
                        bw = (wrist-[x0, y0])/[x1-x0, y1-y0]*256
                        cv2.drawMarker(tile, tuple(np.int32(bw)), (255, 180, 0), cv2.MARKER_CROSS, 16, 2)
                        tile = cv2.copyMakeBorder(tile, 0, 40, 0, 0, cv2.BORDER_CONSTANT, value=(245, 245, 245))
                        cv2.putText(tile, f"{clip} {frame_index/fps:.2f}s {side} F:{fi is not None} C:{ci is not None}",
                                    (5, 280), cv2.FONT_HERSHEY_SIMPLEX, .42, (30, 30, 30), 1, cv2.LINE_AA)
                        tiles.append(tile)
                    # Ambiguous full-frame association is not evidence for both hands.
                    if len(frame_rows) == 2 and frame_rows[0]["full_match"] is not None and frame_rows[0]["full_match"] == frame_rows[1]["full_match"]:
                        for row in frame_rows:
                            row["full_match"] = None
                            row["full_assignment_ambiguous"] = True
                    rows.append({"time_s": frame_index/fps, "frame_index": frame_index,
                                 "full_candidates": full, "hands": frame_rows})
            finally:
                cap.release()
            slots = [hand for row in rows for hand in row["hands"]]
            result = {"clip": clip, "samples": len(rows), "hand_slots": len(slots),
                      "full_matches": sum(hand["full_match"] is not None for hand in slots),
                      "crop_matches": sum(hand["crop_match"] is not None for hand in slots),
                      "elapsed_s": time.monotonic()-started,
                      "source_video": str(videos[0].resolve()), "observations": rows}
            (args.output/(clip+".json")).write_text(json.dumps(result, indent=2)+"\n", encoding="utf-8")
            if tiles:
                while len(tiles) % 4:
                    tiles.append(np.full_like(tiles[0], 245))
                sheet = np.vstack([np.hstack(tiles[i:i+4]) for i in range(0, len(tiles), 4)])
                cv2.imwrite(str(args.output/(clip+".jpg")), sheet)
            summary["clips"].append({key: value for key, value in result.items() if key != "observations"})
            print(f"{clip}: full {result['full_matches']}/{len(slots)}, crops {result['crop_matches']}/{len(slots)}", flush=True)
    (args.output/"summary.json").write_text(json.dumps(summary, indent=2)+"\n", encoding="utf-8")


if __name__ == "__main__":
    main()
