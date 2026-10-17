"""Optional official WiLoR runtime; imported only during hand extraction.

No inference packages are imported when this module is inspected or configured.
"""
import os
import sys

import numpy as np


class WiLoRBackend:
    def __init__(self, repo, extras, weights, mano, device):
        import torch
        self.torch = torch
        sys.path[:0] = [str(extras.resolve()), str(repo.resolve())]
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

    def rest_joints(self):
        with self.torch.inference_mode():
            return self.model.mano().joints[0].cpu().numpy()

    def detect(self, frame, origin=(0, 0)):
        with self.torch.inference_mode():
            return self._detect(frame, origin)

    def _detect(self, frame, origin):
        torch = self.torch
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

