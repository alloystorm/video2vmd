"""Fill missing WiLoR hand poses and save a numeric, inference-free motion cache."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from video2vmd.hand_motion import complete_wilor_observations, save_hand_motion, STATUS_NAMES


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="WiLoR per-clip JSON or probe directory")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--timeline", type=Path, help="NPZ with output timestamps, for one input clip")
    parser.add_argument("--candidate-source", choices=("crop", "full"), default="crop")
    args = parser.parse_args()
    files = sorted(args.input.glob("v[0-9]*.json")) if args.input.is_dir() else [args.input]
    if not files:
        parser.error("No per-clip observations found")
    if args.timeline and len(files) != 1:
        parser.error("--timeline requires one input clip")
    if args.output.exists():
        parser.error("Choose a fresh output directory to preserve observations")
    times = None
    if args.timeline:
        with np.load(args.timeline, allow_pickle=False) as data:
            times = data["timestamps"]
    args.output.mkdir(parents=True)
    report = {"policy": "Timestamp SLERP inside gaps of any duration; nearest pose at leading/trailing gaps; unavailable when never observed",
              "observed_poses_unchanged": True, "clips": []}
    for path in files:
        document = json.loads(path.read_text(encoding="utf-8"))
        payload = complete_wilor_observations(document, target_timestamps=times, candidate_source=args.candidate_source)
        clip = path.stem
        destination = args.output/clip
        destination.mkdir()
        save_hand_motion(destination/"hand_motion.npz", payload)
        row = {"clip": clip, "input_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
               "frames": len(payload["timestamps"]), "hands": {}}
        for hand_index, side in enumerate(("left", "right")):
            row["hands"][side] = {name: int((payload["status"][:, hand_index] == code).sum())
                                  for code, name in enumerate(STATUS_NAMES)}
        report["clips"].append(row)
        print(f"{clip}: {row['hands']}")
    (args.output/"completion.json").write_text(json.dumps(report, indent=2)+"\n", encoding="utf-8")


if __name__ == "__main__":
    main()
