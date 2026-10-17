"""Compare official WiLoR with saved MediaPipe samples; no animation is changed."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from video2vmd.hand_motion import complete_wilor_observations, save_hand_motion
CONNECTIONS = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8),
               (5, 9), (9, 10), (10, 11), (11, 12), (9, 13), (13, 14), (14, 15),
               (15, 16), (13, 17), (0, 17), (17, 18), (18, 19), (19, 20)]


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class WiLoRProbe:
    def __init__(self, repo, extras, weights, mano, device):
        sys.path[:0] = [str(extras.resolve()), str(repo.resolve()), str(ROOT)]
        if sys.platform == "win32":
            os.environ.setdefault("PYOPENGL_PLATFORM", "win32")
        os.environ["YOLO_AUTOINSTALL"] = "false"
        from gvhmr.utils._smpl_compat import apply
        apply()
        from wilor.models import WiLoR
        from wilor.configs import get_config
        from wilor.datasets.vitdet_dataset import ViTDetDataset
        from wilor.utils.renderer import cam_crop_to_full
        from ultralytics import YOLO
        cfg = get_config(str(repo/"pretrained_models/model_config.yaml"))
        cfg.defrost()
        cfg.MODEL.BACKBONE.pop("PRETRAINED_WEIGHTS", None)
        cfg.MODEL.BBOX_SHAPE = [192, 256]
        cfg.MANO.MODEL_PATH = str(mano.resolve().parent)
        cfg.MANO.DATA_DIR = cfg.MANO.MODEL_PATH
        cfg.MANO.MEAN_PARAMS = str((repo/"mano_data/mano_mean_params.npz").resolve())
        cfg.freeze()
        print("Loading official WiLoR reconstruction checkpoint", flush=True)
        self.model = WiLoR(cfg, init_renderer=False)
        # Official upstream checkpoint and user-provided MANO are trusted local inputs.
        checkpoint = torch.load(weights/"wilor_final.ckpt", map_location="cpu", weights_only=False)
        state = self.model.load_state_dict(checkpoint["state_dict"], strict=False)
        if state.missing_keys or state.unexpected_keys:
            raise ValueError(f"Checkpoint does not match official architecture: {state}")
        del checkpoint
        self.device = device
        self.model.to(device).eval()
        self.detector = YOLO(str(weights/"detector.pt"))
        self.cfg, self.dataset, self.camera = cfg, ViTDetDataset, cam_crop_to_full
        print(f"Detector classes: {self.detector.names}", flush=True)

    @torch.inference_mode()
    def detect(self, frame, origin=(0, 0)):
        detected = self.detector.predict(frame, conf=.3, imgsz=640, device=self.device, verbose=False)[0]
        boxes = detected.boxes.xyxy.cpu().numpy()
        right = detected.boxes.cls.cpu().numpy()
        confidence = detected.boxes.conf.cpu().numpy()
        if not len(boxes):
            return []
        dataset = self.dataset(self.cfg, frame, boxes, right, rescale_factor=2., fp16=False)
        hands = []
        for batch in torch.utils.data.DataLoader(dataset, batch_size=4, shuffle=False, num_workers=0):
            batch = {key: value.to(self.device) if isinstance(value, torch.Tensor) else value for key, value in batch.items()}
            output = self.model(batch)
            mirror = 2*batch["right"]-1
            camera = output["pred_cam"].clone()
            camera[:, 1] *= mirror
            focal = self.cfg.EXTRA.FOCAL_LENGTH/self.cfg.MODEL.IMAGE_SIZE*batch["img_size"].max()
            translation = self.camera(camera, batch["box_center"].float(), batch["box_size"].float(),
                                      batch["img_size"].float(), focal).cpu().numpy()
            joints = output["pred_keypoints_3d"].cpu().numpy().copy()
            joints[:, :, 0] *= mirror.cpu().numpy()[:, None]
            for n in range(len(joints)):
                camera_points = joints[n]+translation[n]
                xy = camera_points[:, :2]/camera_points[:, 2, None]*float(focal)+np.array(frame.shape[1::-1])/2
                xy += np.array(origin)
                index = int(batch["personid"][n])
                hands.append({"image_xy_px": xy.tolist(), "camera_relative_xyz_m": joints[n].tolist(),
                              "is_right": bool(batch["right"][n]), "detector_confidence": float(confidence[index]),
                              "detector_xyxy_px": (boxes[index]+np.tile(origin, 2)).tolist(),
                              "mano_pose_matrices": output["pred_mano_params"]["hand_pose"][n].cpu().numpy().tolist(),
                              "mano_global_orient_matrix": output["pred_mano_params"]["global_orient"][n].cpu().numpy().tolist(),
                              "mano_betas": output["pred_mano_params"]["betas"][n].cpu().numpy().tolist()})
        return hands


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
    with torch.no_grad():
        rest_joints = probe.model.mano().joints[0].cpu().numpy()
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
