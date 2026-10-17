"""Compare official WiLoR with saved MediaPipe samples; no animation is changed."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from video2vmd.hand_motion import complete_wilor_observations, save_hand_motion
from video2vmd.wilor_backend import WiLoRBackend as WiLoRProbe
CONNECTIONS = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8),
               (5, 9), (9, 10), (10, 11), (11, 12), (9, 13), (13, 14), (14, 15),
               (15, 16), (13, 17), (0, 17), (17, 18), (18, 19), (19, 20)]


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def nearest(hands, wrist, gate):
    if not hands:
        return None
    distances = [np.linalg.norm(np.asarray(hand["image_xy_px"])[0]-wrist) for hand in hands]
    best = int(np.argmin(distances))
    return best if distances[best] <= gate else None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("baseline", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repo", type=Path, default=ROOT/"outputs/dependencies/WiLoR")
    parser.add_argument("--extras", type=Path, default=ROOT/"outputs/dependencies/wilor_extra")
    parser.add_argument("--weights", type=Path, default=ROOT/"weights/wilor")
    parser.add_argument("--mano", type=Path, default=ROOT/"outputs/dependencies/wilor_mano/MANO_RIGHT.pkl")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--clip", action="append")
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Choose a fresh output directory")
    args.output.mkdir(parents=True)
    probe = WiLoRProbe(args.repo, args.extras, args.weights, args.mano, args.device)
    summary = {"backend": "official WiLoR, full float32, detector confidence 0.3, 640 detector image size, crop rescale 2",
               "source_revision": subprocess.check_output(["git", "-C", str(args.repo), "rev-parse", "HEAD"], text=True).strip(),
               "hashes": {name: sha256(path) for name, path in (
                   ("checkpoint", args.weights/"wilor_final.ckpt"), ("detector", args.weights/"detector.pt"),
                   ("mano", args.mano), ("probe", Path(__file__)))},
               "coordinate_note": "3D joints are MANO camera-basis predictions with X reflected for left hands; pose matrices retain the canonical right-hand model convention",
               "evaluation_note": "Same frames, body-wrist gates and crops as baseline. Matches are not annotated pose accuracy. Detector confidence is not finger confidence.",
               "clips": []}
    rest_joints = probe.rest_joints()
    for path in sorted(args.baseline.glob("v[0-9]*.json")):
        baseline = json.loads(path.read_text(encoding="utf-8"))
        clip = baseline["clip"]
        if args.clip and clip not in args.clip:
            continue
        started = time.monotonic()
        cap = cv2.VideoCapture(baseline["source_video"])
        rows, tiles = [], []
        try:
            sheet_frames = set(np.linspace(0, len(baseline["observations"])-1, min(12, len(baseline["observations"]))).astype(int))
            for frame_step, sample in enumerate(baseline["observations"]):
                if round(cap.get(cv2.CAP_PROP_POS_FRAMES)) != sample["frame_index"]:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, sample["frame_index"])
                ok, frame = cap.read()
                if not ok:
                    raise ValueError(f"Failed to read {clip} frame {sample['frame_index']}")
                full = probe.detect(frame)
                hands = []
                for previous in sample["hands"]:
                    if not previous["crop_valid"]:
                        hands.append({"side": previous["side"], "crop_valid": False, "full_match": None, "crop_match": None})
                        continue
                    x0, y0, x1, y1 = previous["crop_xyxy"]
                    crop = frame[y0:y1, x0:x1]
                    candidates = probe.detect(crop, (x0, y0))
                    wrist = np.asarray(previous["body_wrist_xy"])
                    gate = max(20., .6*previous["forearm_px"])
                    fi, ci = nearest(full, wrist, gate), nearest(candidates, wrist, gate)
                    hands.append({"side": previous["side"], "crop_valid": True, "full_match": fi, "crop_match": ci,
                                  "crop_candidates": candidates, "body_wrist_xy": wrist.tolist(),
                                  "crop_xyxy": previous["crop_xyxy"]})
                    if frame_step not in sheet_frames:
                        continue
                    tile = cv2.resize(crop, (256, 256))
                    for index, hand in enumerate(candidates):
                        xy = (np.asarray(hand["image_xy_px"])-[x0, y0])/[x1-x0, y1-y0]*256
                        color = (0, 210, 70) if index == ci else (0, 150, 255)
                        for a, b in CONNECTIONS:
                            cv2.line(tile, tuple(np.int32(xy[a])), tuple(np.int32(xy[b])), color, 1, cv2.LINE_AA)
                        for point in xy:
                            cv2.circle(tile, tuple(np.int32(point)), 2, color, -1)
                    tile = cv2.copyMakeBorder(tile, 0, 40, 0, 0, cv2.BORDER_CONSTANT, value=(245, 245, 245))
                    label = f"{clip} {sample['time_s']:.2f}s {previous['side']} MP:{previous['crop_match'] is not None} W:{ci is not None}"
                    cv2.putText(tile, label, (4, 280), cv2.FONT_HERSHEY_SIMPLEX, .38, (30, 30, 30), 1, cv2.LINE_AA)
                    tiles.append(tile)
                if len(hands) == 2 and hands[0]["full_match"] is not None and hands[0]["full_match"] == hands[1]["full_match"]:
                    for hand in hands:
                        hand["full_match"] = None
                        hand["full_assignment_ambiguous"] = True
                rows.append({"time_s": sample["time_s"], "frame_index": sample["frame_index"], "full_candidates": full, "hands": hands})
        finally:
            cap.release()
        slots = [hand for row in rows for hand in row["hands"]]
        result = {"clip": clip, "hand_slots": len(slots), "full_matches": sum(h["full_match"] is not None for h in slots),
                  "crop_matches": sum(h["crop_match"] is not None for h in slots),
                  "mediapipe_full_matches": baseline["full_matches"], "mediapipe_crop_matches": baseline["crop_matches"],
                  "elapsed_s": time.monotonic()-started, "source_sha256": sha256(baseline["source_video"]),
                  "baseline_sha256": sha256(path)}
        document = {**result, "observations": rows, "mano_rest_joints_m": rest_joints.tolist()}
        (args.output/(clip+".json")).write_text(json.dumps(document, indent=2)+"\n", encoding="utf-8")
        completed = complete_wilor_observations(document)
        completed["mano_rest_joints_m"] = rest_joints
        save_hand_motion(args.output/(clip+"_hand_motion.npz"), completed)
        if tiles:
            while len(tiles) % 4:
                tiles.append(np.full_like(tiles[0], 245))
            cv2.imwrite(str(args.output/(clip+".jpg")), np.vstack([np.hstack(tiles[i:i+4]) for i in range(0, len(tiles), 4)]))
        summary["clips"].append(result)
        (args.output/"summary.json").write_text(json.dumps(summary, indent=2)+"\n", encoding="utf-8")
        print(f"{clip}: WiLoR full {result['full_matches']}/{len(slots)}, crops {result['crop_matches']}/{len(slots)}; "
              f"MediaPipe {baseline['full_matches']}/{len(slots)}, {baseline['crop_matches']}/{len(slots)}", flush=True)


if __name__ == "__main__":
    main()
