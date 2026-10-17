"""Hand extraction orchestration regressions; no licensed assets or GPU needed."""
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from verify_hand_retargeting import hand_fixture
from video2vmd.cli import build_parser, preflight
from video2vmd.hand_extract import (HandConfig, check_hand_runtime, load_hand_body, resolve_hand_source,
                                   body_hand_crops, associate_crops, extract_hand_motion, _source_frames)
from video2vmd.hand_motion import load_hand_motion
from video2vmd.mmd.pipeline import run_mmd_pipeline


class FakeBackend:
    def __init__(self, *args):
        self.calls = 0

    def rest_joints(self):
        return hand_fixture()[2]["mano_rest_joints_m"]

    def detect(self, frame, origin):
        self.calls += 1
        if self.calls == 5:
            return []
        xy = np.tile(np.array(origin)+np.array(frame.shape[1::-1])/2, (21, 1))
        points = np.zeros((21, 3)); points[5] = [-.03, -.06, 0]; points[9] = [0, -.08, 0]; points[17] = [.03, -.06, 0]
        return [{"image_xy_px": xy.tolist(), "camera_relative_xyz_m": points.tolist(), "is_right": self.calls % 2 == 0,
                 "detector_confidence": .9, "detector_xyxy_px": [*(xy[0]-12), *(xy[0]+12)],
                 "mano_pose_matrices": np.repeat(np.eye(3)[None], 15, axis=0).tolist(),
                 "mano_global_orient_matrix": np.eye(3)[None].tolist(), "mano_betas": [0.]*10}]


class HandExtractionTests(unittest.TestCase):
    def setUp(self):
        base = ROOT/"outputs/diagnostics"; base.mkdir(parents=True, exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(prefix="hand-extract-tests-", dir=base)
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.config = HandConfig(self.root/"repo", self.root/"extras", self.root/"weights", self.root/"MANO_RIGHT.pkl")
        for path in self.config.files().values():
            path.parent.mkdir(parents=True, exist_ok=True); path.write_bytes(b"synthetic test asset")
        stub = self.config.repo/"wilor/models/__init__.py"
        stub.parent.mkdir(parents=True); stub.write_text("")
        self.config.extras.mkdir()
        self.human, _, _ = hand_fixture(5)
        self.source = self.root/"source.webm"; self.source.write_bytes(b"synthetic media")
        self.skeleton = self.root/"skeleton.npz"
        camera = self.human.joints_3d*[1, -1, -1]+[0, .8, 4]
        self.body = dict(joints_3d=self.human.joints_3d, joint_names=self.human.joint_names, timestamps=self.human.timestamps,
                         fps=self.human.fps, video=str(self.source), camera_joints_3d=camera,
                         camera_intrinsics=np.array([[700., 0, 640], [0, 700, 360], [0, 0, 1]]),
                         camera_image_size=np.array([1280, 720]),
                         camera_world_to_camera=np.repeat(np.diag([1., -1, -1])[None], 5, axis=0))
        np.savez(self.skeleton, **self.body)

    def frames(self, source, times):
        for i, t in enumerate(times):
            yield np.zeros((720, 1280, 3), dtype=np.uint8), float(t), i

    def test_configuration_and_numeric_exports_do_not_import_training_packages(self):
        result = subprocess.run([sys.executable, "-c", "import sys; import video2vmd.hand_extract; import video2vmd.wilor_backend; assert not {'torch','wilor','mediapipe'} & set(sys.modules)"],
                                cwd=ROOT, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(check_hand_runtime(self.config), [])
        self.config.mano.unlink()
        self.assertTrue(any("mano" in problem for problem in check_hand_runtime(self.config)))

    def test_body_geometry_rejects_missing_camera_and_source_mismatch_before_inference(self):
        body = load_hand_body(self.skeleton)
        self.assertEqual(len(body["timestamps"]), 5)
        self.assertEqual(resolve_hand_source(self.skeleton), self.source.resolve())
        other = self.root/"other.webm"; other.write_bytes(b"different media")
        with self.assertRaisesRegex(ValueError, "differs"):
            resolve_hand_source(self.skeleton, other)
        self.source.unlink()
        with self.assertRaisesRegex(FileNotFoundError, "original source"):
            resolve_hand_source(self.skeleton)
        del self.body["camera_world_to_camera"]; np.savez(self.skeleton, **self.body)
        with self.assertRaisesRegex(ValueError, "camera geometry"):
            load_hand_body(self.skeleton)

    def test_crop_guidance_resizes_and_rejects_wrong_aspect_or_behind_camera(self):
        body = load_hand_body(self.skeleton)
        a, b = body_hand_crops(body, 0, [1280, 720]), body_hand_crops(body, 0, [2560, 1440])
        for original, doubled in zip(a, b):
            self.assertTrue(original["crop_valid"])
            np.testing.assert_allclose(np.array(original["body_wrist_xy"])*2, doubled["body_wrist_xy"])
        with self.assertRaisesRegex(ValueError, "aspect"):
            body_hand_crops(body, 0, [720, 1280])
        body["camera_joints_3d"][:, :, 2] = -1
        self.assertTrue(all(not c["crop_valid"] for c in body_hand_crops(body, 0, [1280, 720])))

    def test_duplicate_observation_cannot_animate_both_body_sides(self):
        candidate = {"image_xy_px": np.tile([100., 100.], (21, 1)).tolist(), "detector_xyxy_px": [80, 80, 120, 120]}
        crops = [{"crop_valid": True, "crop_match": None, "crop_candidates": [candidate], "body_wrist_xy": [100, 100],
                  "forearm_px": 80, "gate_px": 48} for _ in range(2)]
        associate_crops(crops)
        self.assertTrue(all(row["crop_match"] is None and row["rejection"] == "ambiguous_duplicate_hand" for row in crops))

    def test_completed_cache_reuse_and_changed_inputs_guard(self):
        events = []
        with patch("video2vmd.hand_extract._source_frames", side_effect=self.frames), \
             patch("video2vmd.wilor_backend.WiLoRBackend", side_effect=FakeBackend) as backend:
            cache, report = extract_hand_motion(self.skeleton, self.root/"cache", config=self.config, progress=lambda *e: events.append(e))
            self.assertEqual(backend.call_count, 1)
            self.assertEqual(events[-1], (5, 5))
            self.assertEqual(report["counts"]["left"]["interpolated"], 1)
            data = load_hand_motion(cache)
            self.assertEqual(data["camera_palm_quat_xyzw"].shape, (5, 2, 4))
        with patch("video2vmd.wilor_backend.WiLoRBackend", side_effect=AssertionError("Inference should be cached")):
            second, reused = extract_hand_motion(self.skeleton, self.root/"cache", config=self.config)
            self.assertEqual(cache, second); self.assertTrue(reused["reused"])
            self.body["camera_intrinsics"][0, 0] += 1; np.savez(self.skeleton, **self.body)
            with self.assertRaisesRegex(ValueError, "different media, camera geometry or models"):
                extract_hand_motion(self.skeleton, self.root/"cache", config=self.config)

    def test_still_is_observed_once_and_held_for_body_timeline(self):
        import cv2
        image = self.root/"source.png"
        cv2.imwrite(str(image), np.zeros((720, 1280, 3), dtype=np.uint8))
        self.body["image"] = str(image); np.savez(self.skeleton, **self.body)
        with patch("video2vmd.wilor_backend.WiLoRBackend", side_effect=FakeBackend) as backend:
            cache, report = extract_hand_motion(self.skeleton, self.root/"still-cache", config=self.config)
        self.assertEqual(backend.call_count, 1); self.assertEqual(report["observation_frames"], 1)
        data = load_hand_motion(cache)
        np.testing.assert_array_equal(data["status"][:, 0], [1, 3, 3, 3, 3])
        np.testing.assert_array_equal(data["palm_status"][:, 1], [1, 3, 3, 3, 3])

    def test_video_sampling_uses_presentation_times_without_relabeling_rate(self):
        import cv2
        video = self.root/"60fps.mp4"
        writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"mp4v"), 60., (64, 96))
        for i in range(60):
            writer.write(np.full((96, 64, 3), i*3, dtype=np.uint8))
        writer.release()
        samples = list(_source_frames(video, np.array([0., .25, .5, .75])))
        np.testing.assert_allclose([t for _, t, _ in samples], [0., .25, .5, .75], atol=1e-6)
        self.assertEqual([i for _, _, i in samples], [0, 15, 30, 45])

    def test_cli_exclusive_modes_preflight_and_pipeline_artifacts(self):
        parser = build_parser()
        with redirect_stderr(StringIO()), self.assertRaises(SystemExit):
            parser.parse_args(["run", "--hands", "--hand-motion", "cache.npz"])
        args = parser.parse_args(["run", "--skeleton", str(self.skeleton), "--hands"])
        with patch("video2vmd.hand_extract.check_hand_runtime", return_value=[]), redirect_stdout(StringIO()):
            self.assertEqual(preflight(args), (None, self.skeleton, None))
        with patch("video2vmd.hand_extract._source_frames", side_effect=self.frames), \
             patch("video2vmd.wilor_backend.WiLoRBackend", side_effect=FakeBackend):
            out = run_mmd_pipeline(skeleton_npz=self.skeleton, hands=True, hand_config=self.config,
                                   output_dir=self.root/"run", no_preview=True)
        run = json.loads((out/"run.json").read_text())
        self.assertTrue(run["hand_extraction"]["enabled"])
        self.assertIsNone(run["hand_motion_reused_from"])
        self.assertEqual(run["hands"]["wrist_policy"], "camera_palm_orientation_preserve_position")
        for key in ("hand_motion", "hand_extraction", "hand_observations", "target_motion_body"):
            self.assertTrue((out/run["outputs"][key]).is_file())


if __name__ == "__main__":
    unittest.main(verbosity=2)
