"""Hand rest-axis, export and pipeline regressions without inference or model files."""
from pathlib import Path
import json
import sys
import tempfile
import unittest

import numpy as np
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from verify_mmd import human_fixture
from video2vmd.hand_motion import fill_hand_gaps, save_hand_motion, load_hand_motion, resample_hand_motion, world_to_camera_rotations
from video2vmd.mmd.hands import FINGERS, finger_names, retarget_hands, _palm
from video2vmd.mmd.profile import MMDProfile, require_codec
from video2vmd.mmd.retarget import retarget, SOURCE_TO_MMD
from video2vmd.mmd.pipeline import run_mmd_pipeline
from video2vmd.inspection.server import load_target, prepare_inspection


def hand_fixture(count=8):
    human, profile = human_fixture(count)
    motion, _ = retarget(human, profile)
    points = np.zeros((21, 3)); points[0] = profile.position("RightWrist")
    for finger, _, indices, numbers in FINGERS:
        for point, number in zip(indices, numbers):
            points[point] = profile.position("Right"+finger+str(number))
        points[indices[-1]] = points[indices[-2]]*2-points[indices[-3]]
    q = np.zeros((count, 2, 16, 4)); q[..., 3] = 1.
    payload = fill_hand_gaps(np.arange(count)/30, q, np.ones((count, 2), dtype=bool), betas=np.zeros((count, 2, 10)))
    payload.update(version=np.int32(1), sides=np.array(["left", "right"]), mano_rest_joints_m=points,
                   metadata=np.array(json.dumps({"rotation_space": "canonical_right_MANO_local_axes_for_both_body_side_tracks"})))
    return human, motion, payload


class HandRetargetingTests(unittest.TestCase):
    def test_camera_palms_recover_known_world_wrist_rotation_on_both_sides(self):
        _, motion, payload = hand_fixture()
        n = motion.n_frames
        camera = Rotation.from_euler("xyz", np.column_stack((np.linspace(165, 185, n), np.linspace(-25, 15, n), np.linspace(10, -5, n))), degrees=True).as_matrix()
        expected = Rotation.from_euler("xyz", np.column_stack((np.linspace(-20, 50, n), np.linspace(15, 60, n), np.linspace(30, 95, n))), degrees=True).as_matrix()
        camera_palms = []
        for side in ("Left", "Right"):
            rest = _palm(*(motion.profile.position(side+name) for name in ("Wrist", "IndexFinger1", "MiddleFinger1", "Pinky1")))
            rays = camera @ SOURCE_TO_MMD @ expected @ rest
            palm = np.stack((rays[:, :, 0], rays[:, :, 1], np.cross(rays[:, :, 0], rays[:, :, 1])), axis=-1)
            camera_palms.append(Rotation.from_matrix(palm).as_quat())
        payload["camera_palm_quat_xyzw"] = np.stack(camera_palms, axis=1)
        payload["palm_status"] = np.ones((n, 2), dtype=np.uint8)
        payload["palm_status"][0, 0] = 0  # Explicit fallback to the body wrist.
        payload["palm_status"][3, 1] = 2
        fingers_only, _ = retarget_hands(motion, payload)
        result, report = retarget_hands(motion, payload, camera)
        np.testing.assert_array_equal(result.translations, motion.translations)
        np.testing.assert_array_equal(result.rotations[:, result.hand_channels], fingers_only.rotations[:, result.hand_channels])
        untouched = ~result.hand_channels.copy()
        for hand, side in enumerate(("Left", "Right")):
            wrist = motion.profile.index[side+"Wrist"]; untouched[wrist] = False
            active = payload["palm_status"][:, hand] != 0
            np.testing.assert_allclose(result.fk()[1][active, wrist], expected[active], atol=1e-10)
            np.testing.assert_allclose(result.fk()[0][:, wrist], motion.fk()[0][:, wrist], atol=1e-10)
            np.testing.assert_array_equal(result.rotations[~active, wrist], motion.rotations[~active, wrist])
        np.testing.assert_array_equal(result.rotations[:, untouched], motion.rotations[:, untouched])
        self.assertEqual(report["wrist_counts"]["right"]["interpolated"], 1)
        self.assertEqual(report["wrist_policy"], "camera_palm_orientation_preserve_position")
        broken = camera.copy(); broken[0, 0] *= -1
        with self.assertRaisesRegex(ValueError, "camera"):
            retarget_hands(motion, payload, broken)

    def test_camera_bridge_resamples_rotations_on_relative_source_clock(self):
        base = Path(__file__).resolve().parents[1]/"outputs/diagnostics"
        with tempfile.TemporaryDirectory(prefix="hand-camera-tests-", dir=base) as directory:
            path = Path(directory)/"body.npz"
            rotations = Rotation.from_euler("y", [[0.], [90.]], degrees=True).as_matrix()
            np.savez(path, timestamps=[5., 7.], camera_world_to_camera=rotations)
            out = world_to_camera_rotations(path, np.array([0., 1., 2., 3.]))
            np.testing.assert_allclose(out, Rotation.from_euler("y", [[0.], [45.], [90.], [90.]], degrees=True).as_matrix(), atol=1e-10)
            np.savez(path, timestamps=[0.])
            self.assertIsNone(world_to_camera_rotations(path, [0.]))

    def test_pipeline_persists_and_exports_camera_observed_wrist_orientation(self):
        human, motion, payload = hand_fixture()
        n = motion.n_frames
        camera = np.repeat(np.diag([1., -1., -1.])[None], n, axis=0)
        payload.update(camera_palm_quat_xyzw=np.broadcast_to([0., 0., 0., 1.], (n, 2, 4)).copy(),
                       palm_status=payload["status"].copy(), source_palm_observed=payload["source_observed"].copy(),
                       camera_palm_convention=np.array("opencv_xright_ydown_zforward; x=pinky_to_index; y=wrist_to_middle; z=cross(x,y)"))
        expected, _ = retarget_hands(motion, payload, camera)
        base = Path(__file__).resolve().parents[1]/"outputs/diagnostics"
        with tempfile.TemporaryDirectory(prefix="hand-wrist-pipeline-tests-", dir=base) as directory:
            root = Path(directory); skeleton = root/"body.npz"; cache = root/"hand.npz"
            np.savez(skeleton, joints_3d=human.joints_3d, joint_names=human.joint_names, fps=human.fps,
                     timestamps=human.timestamps, camera_world_to_camera=camera)
            save_hand_motion(cache, payload)
            out = run_mmd_pipeline(skeleton_npz=skeleton, output_dir=root/"vmd", hand_motion=cache,
                                   motion_encoder=True, no_preview=True)
            actual = load_target(out/"target_motion.npz")
            np.testing.assert_array_equal(actual.hand_wrist_status, payload["palm_status"])
            np.testing.assert_allclose(actual.rotations, expected.rotations, atol=1e-10)
            loaded = load_hand_motion(out/"hand_motion.npz")
            np.testing.assert_array_equal(loaded["camera_palm_quat_xyzw"], payload["camera_palm_quat_xyzw"])
            _, vmd = require_codec().parse_vmd(out/"motion.vmd")
            for frame in vmd["bone_frames"]:
                if frame["name"] in ("LeftWrist", "RightWrist"):
                    bone = motion.profile.index[frame["name"]]
                    np.testing.assert_allclose(Rotation.from_quat(frame["rotation"]).as_matrix(),
                                               Rotation.from_quat(expected.rotations[frame["frame_num"], bone]).as_matrix(), atol=1e-6)
            from vmd2png.bone import quat_to_6d
            encoded = np.load(out/"motion_encoder.npy")
            for side in ("Left", "Right"):
                bone = motion.profile.index[side+"Wrist"]
                np.testing.assert_allclose(encoded[:, 4+6*bone:10+6*bone],
                                           [quat_to_6d(q) for q in expected.rotations[:, bone]], atol=1e-7)
            bvh = run_mmd_pipeline(skeleton_npz=skeleton, output_dir=root/"bvh", hand_motion=cache,
                                   output_format="bvh", no_preview=True)
            info = json.loads((bvh/"motion.bvh.json").read_text())
            text = (bvh/"motion.bvh").read_text()
            frames = np.array([[float(v) for v in line.split()] for line in text.split("Frame Time:")[1].splitlines()[1:]])
            decoded = Rotation.from_euler("ZXY", frames[:, 3:].reshape(-1, 3), degrees=True).as_matrix().reshape(n, -1, 3, 3)
            np.testing.assert_allclose(decoded, Rotation.from_quat(expected.rotations[:, info["bone_indices"]].reshape(-1, 4)).as_matrix().reshape(decoded.shape), atol=1e-7)

    def test_neutral_pose_preserves_both_hands_and_body_exactly(self):
        _, motion, payload = hand_fixture()
        result, report = retarget_hands(motion, payload)
        self.assertEqual(report["finger_channels"], 30)
        np.testing.assert_allclose(result.rotations, motion.rotations, atol=1e-12)
        np.testing.assert_array_equal(result.translations, motion.translations)
        self.assertEqual(np.count_nonzero(np.linalg.norm(result.hand_tip_offsets, axis=1)), 10)

    def test_observed_neutral_wrists_export_dense_reset_keys(self):
        from video2vmd.mmd.export import write_vmd
        _, motion, payload = hand_fixture()
        result, _ = retarget_hands(motion, payload)
        result.hand_wrist_status[:] = 1
        base = Path(__file__).resolve().parents[1]/"outputs/diagnostics"
        with tempfile.TemporaryDirectory(prefix="hand-neutral-wrist-tests-", dir=base) as directory:
            path = Path(directory)/"neutral.vmd"
            write_vmd(path, result)
            _, vmd = require_codec().parse_vmd(path)
            for side in ("Left", "Right"):
                keys = [frame for frame in vmd["bone_frames"] if frame["name"] == side+"Wrist"]
                self.assertEqual(len(keys), motion.n_frames)
                np.testing.assert_allclose([frame["rotation"] for frame in keys], np.tile([0, 0, 0, 1], (motion.n_frames, 1)), atol=1e-10)

    def test_left_parity_and_finger_motion_preserve_wrist_and_bone_lengths(self):
        _, motion, payload = hand_fixture()
        payload["quat_xyzw"][:, :, 2] = Rotation.from_euler("z", 55., degrees=True).as_quat()
        result, _ = retarget_hands(motion, payload)
        reflection = np.diag([-1., 1., 1.])
        right, left = [motion.profile.index[s+"IndexFinger2"] for s in ("Right", "Left")]
        expected = Rotation.from_quat(payload["quat_xyzw"][0, 1, 2]).as_matrix()
        np.testing.assert_allclose(Rotation.from_quat(result.rotations[0, right]).as_matrix(), expected, atol=1e-10)
        np.testing.assert_allclose(Rotation.from_quat(result.rotations[0, left]).as_matrix(), reflection@expected@reflection, atol=1e-10)
        np.testing.assert_array_equal(result.rotations[:, ~result.hand_channels], motion.rotations[:, ~result.hand_channels])
        np.testing.assert_array_equal(result.translations, motion.translations)
        p, _ = result.fk()
        for side in ("Left", "Right"):
            for name in finger_names(side):
                i = motion.profile.index[name]; parent = motion.profile.parents[i]
                np.testing.assert_allclose(np.linalg.norm(p[:, i]-p[:, parent], axis=1),
                                           np.linalg.norm(motion.profile.offsets[i]), atol=1e-10)

    def test_thumb_and_terminal_rotations_move_tips_and_are_scale_invariant(self):
        _, motion, payload = hand_fixture()
        payload["quat_xyzw"][:, :, 15] = Rotation.from_euler("y", 40., degrees=True).as_quat()
        first, _ = retarget_hands(motion, payload)
        scaled_profile = MMDProfile("scaled", motion.profile.names, motion.profile.parents, motion.profile.rest*1.7, motion.profile.serialized)
        from video2vmd.mmd.motion import TargetMotion
        scaled = TargetMotion(scaled_profile, motion.rotations.copy(), motion.translations.copy(), motion.fps)
        second, _ = retarget_hands(scaled, payload)
        np.testing.assert_allclose(first.rotations, second.rotations, atol=1e-10)
        before_p, before_r = motion.fk(); after_p, after_r = first.fk()
        for side in ("Left", "Right"):
            i = motion.profile.index[side+"Thumb2"]; tip = first.hand_tip_offsets[i]
            np.testing.assert_allclose(after_p[:, i], before_p[:, i], atol=1e-10)
            difference = np.einsum("tij,j->ti", after_r[:, i]-before_r[:, i], tip)
            self.assertGreater(np.linalg.norm(difference[0]), .005)

    def test_cache_roundtrip_and_resampling_keep_real_observation_states(self):
        _, _, payload = hand_fixture()
        payload["source_observed"][3, 0] = False
        payload["status"][3, 0] = 2; payload["observed"][3, 0] = False
        base = Path(__file__).resolve().parents[1]/"outputs/diagnostics"
        with tempfile.TemporaryDirectory(prefix="hand-cache-tests-", dir=base) as directory:
            path = Path(directory)/"hand_motion.npz"
            save_hand_motion(path, payload)
            loaded = load_hand_motion(path)
            result = resample_hand_motion(loaded, payload["timestamps"])
            self.assertEqual(result["status"][3, 0], 2)
            np.testing.assert_array_equal(result["mano_rest_joints_m"], payload["mano_rest_joints_m"])
            del payload["mano_rest_joints_m"]
            save_hand_motion(path, payload)
            with self.assertRaisesRegex(ValueError, "rest calibration"):
                load_hand_motion(path)

    def test_unsupported_controls_are_rejected_for_active_hands(self):
        _, motion, payload = hand_fixture()
        motion.profile.unsupported_hand_controls = ["LeftIndexFinger1"]
        with self.assertRaisesRegex(ValueError, "Unsupported"):
            retarget_hands(motion, payload)

    def test_pipeline_exports_fingers_neutral_keys_encoder_and_inspector_tips(self):
        human, motion, payload = hand_fixture()
        payload["quat_xyzw"][:, :, 2] = Rotation.from_euler("z", 30., degrees=True).as_quat()
        base = Path(__file__).resolve().parents[1]/"outputs/diagnostics"
        with tempfile.TemporaryDirectory(prefix="hand-pipeline-tests-", dir=base) as directory:
            root = Path(directory); skeleton = root/"body.npz"; cache = root/"hand.npz"
            np.savez(skeleton, joints_3d=human.joints_3d, joint_names=human.joint_names, fps=human.fps, timestamps=human.timestamps)
            save_hand_motion(cache, payload)
            out = root/"vmd"
            run_mmd_pipeline(skeleton_npz=skeleton, output_dir=out, hand_motion=cache, motion_encoder=True, no_preview=True)
            target = load_target(out/"target_motion.npz")
            self.assertEqual(int(target.hand_channels.sum()), 30)
            with np.load(out/"target_motion.npz", allow_pickle=False) as data:
                self.assertFalse(json.loads(str(data["metadata"]))["unobserved_hands"])
                self.assertTrue(data["observed"][motion.profile.index["LeftThumb2"]])
            ok, vmd = require_codec().parse_vmd(out/"motion.vmd")
            self.assertTrue(ok)
            names = {frame["name"] for frame in vmd["bone_frames"]}
            self.assertTrue(set(finger_names("Left")+finger_names("Right")) <= names)
            for frame in vmd["bone_frames"]:
                if frame["name"] in finger_names("Left")+finger_names("Right"):
                    i = motion.profile.index[frame["name"]]
                    np.testing.assert_allclose(Rotation.from_quat(frame["rotation"]).as_matrix(),
                                               Rotation.from_quat(target.rotations[frame["frame_num"], i]).as_matrix(), atol=1e-6)
            encoded = np.load(out/"motion_encoder.npy")
            self.assertGreater(np.max(abs(encoded[:, 4+6*motion.profile.index["LeftIndexFinger2"]:
                                                       10+6*motion.profile.index["LeftIndexFinger2"]]-[1, 0, 0, 0, 1, 0])), .1)
            inspection = prepare_inspection(out, transcode=False)
            self.assertEqual(inspection.metadata["variants"], ["corrected", "raw", "body"])
            tips = np.frombuffer(inspection.hand_tips, dtype="<f4").reshape(3, 8, 10, 3)
            indices = inspection.metadata["hands"]["tip_bones"]
            p, r = target.fk()
            np.testing.assert_allclose(tips[0], p[:, indices]+np.einsum("tbij,bj->tbi", r[:, indices], target.hand_tip_offsets[indices]), atol=1e-7)
            run_mmd_pipeline(skeleton_npz=skeleton, output_dir=root/"bvh", hand_motion=cache, output_format="bvh", no_preview=True)
            text = (root/"bvh/motion.bvh").read_text()
            self.assertIn("JOINT LeftThumb2", text)
            self.assertIn("Frames: 8", text)
            info = json.loads((root/"bvh/motion.bvh.json").read_text())
            frames = np.array([[float(v) for v in line.split()] for line in text.split("Frame Time:")[1].splitlines()[1:]])
            decoded = Rotation.from_euler("ZXY", frames[:, 3:].reshape(-1, 3), degrees=True).as_matrix().reshape(8, -1, 3, 3)
            np.testing.assert_allclose(decoded, Rotation.from_quat(target.rotations[:, info["bone_indices"]].reshape(-1, 4)).as_matrix().reshape(decoded.shape), atol=1e-7)


if __name__ == "__main__":
    unittest.main(verbosity=2)
