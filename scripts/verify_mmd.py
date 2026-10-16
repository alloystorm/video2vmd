"""Deterministic MMD retarget/export/contact regressions; no licensed assets needed.

Run with the project's mmd extra installed: python scripts/verify_mmd.py
"""
import argparse
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from video2vmd import paths
from video2vmd.cli import build_parser, doctor, preflight
from video2vmd.extract import SMPL_JOINT_NAMES
from video2vmd.human_motion import HumanMotion
from video2vmd.mmd.profile import standard_profile
from video2vmd.mmd.motion import TargetMotion
from video2vmd.mmd.retarget import SOURCE_TO_MMD, retarget
from video2vmd.mmd.contacts import correct_feet, detect_contacts, foot_points, quality, constrain_ik_knees
from video2vmd.mmd.export import write_vmd, write_bvh, write_motion_encoder
from video2vmd.mmd.pipeline import run_mmd_pipeline


def human_fixture(count=60, fps=30.):
    profile = standard_profile()
    mapping = {"pelvis": "Waist", "spine1": "Torso1", "spine3": "Torso2", "neck": "Neck", "head": "Head"}
    for side, lower in (("Left", "left"), ("Right", "right")):
        mapping.update({lower+"_"+human: side+target for human, target in
                        (("hip", "Leg"), ("knee", "Knee"), ("ankle", "Ankle"), ("foot", "Toe"),
                         ("collar", "Shoulder"), ("shoulder", "Arm"), ("elbow", "Elbow"), ("wrist", "Wrist"), ("hand", "MiddleFinger1"))})
    joints = np.array([profile.position(mapping[name]) if name in mapping else profile.position("Torso1")
                       for name in SMPL_JOINT_NAMES]) @ SOURCE_TO_MMD.T
    values = np.repeat(joints[None], count, axis=0)
    return HumanMotion(values, SMPL_JOINT_NAMES, fps, np.arange(count)/fps, None, Path("fixture.npz")), profile


class MMDTests(unittest.TestCase):
    def setUp(self):
        base = (paths.OUTPUTS/"diagnostics").resolve()
        base.mkdir(parents=True, exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(prefix="mmd-tests-", dir=base)
        self.root = Path(self.temp.name).resolve()
        assert self.root.is_relative_to(base)
        self.addCleanup(self.temp.cleanup)

    def test_identity_rest_and_neutral_fingers(self):
        source, profile = human_fixture(1)
        target, meta = retarget(source, profile)
        position, _ = target.fk()
        self.assertLess(np.max(np.linalg.norm(position[0]-profile.rest, axis=1)), 1e-6)
        self.assertLess(np.max(np.abs(target.rotations[:, :, :3])), 1e-6)
        self.assertTrue(meta["unobserved_fingers"])

    def test_asymmetric_arm_preserves_anatomical_side(self):
        source, profile = human_fixture(1)
        points = source.joints_3d[0] @ SOURCE_TO_MMD.T
        shoulder = points[SMPL_JOINT_NAMES.index("left_shoulder")]
        rotation = Rotation.from_euler("z", 70, degrees=True)
        for name in ("left_elbow", "left_wrist", "left_hand"):
            i = SMPL_JOINT_NAMES.index(name)
            points[i] = rotation.apply(points[i]-shoulder)+shoulder
        source.joints_3d[0] = points @ SOURCE_TO_MMD.T
        target, _ = retarget(source, profile)
        position, _ = target.fk()
        for bone, joint in (("LeftElbow", "left_elbow"), ("LeftWrist", "left_wrist")):
            np.testing.assert_allclose(position[0, profile.index[bone]], points[SMPL_JOINT_NAMES.index(joint)], atol=1e-6)
        np.testing.assert_allclose(position[0, profile.index["RightWrist"]], profile.position("RightWrist"), atol=1e-6)

    def test_skull_center_offset_does_not_become_a_large_neck_bend(self):
        source, profile = human_fixture(1)
        points = source.joints_3d[0] @ SOURCE_TO_MMD.T
        # A SMPL-like skull-center offset: mostly anterior, little vertical
        # separation. This is not a 60-degree anatomical neck bend.
        points[SMPL_JOINT_NAMES.index("head")] = points[SMPL_JOINT_NAMES.index("neck")]+[0., .04, -.07]
        source.joints_3d[0] = points @ SOURCE_TO_MMD.T
        target, _ = retarget(source, profile)
        neck = target.rotations[0, profile.index["Neck"]]
        self.assertLess(np.degrees(Rotation.from_quat(neck).magnitude()), 20.)
        np.testing.assert_allclose(target.rotations[0, profile.index["Head"]], [0,0,0,1], atol=1e-7)

    def test_neck_and_head_follow_a_coherent_upper_body_lean(self):
        source, profile = human_fixture(1)
        points = source.joints_3d[0] @ SOURCE_TO_MMD.T
        pivot = points[SMPL_JOINT_NAMES.index("spine3")].copy()
        lean = Rotation.from_euler("x", 25, degrees=True)
        for name in ("neck", "head", "left_collar", "right_collar", "left_shoulder", "right_shoulder",
                     "left_elbow", "right_elbow", "left_wrist", "right_wrist", "left_hand", "right_hand"):
            i = SMPL_JOINT_NAMES.index(name)
            points[i] = lean.apply(points[i]-pivot)+pivot
        source.joints_3d[0] = points @ SOURCE_TO_MMD.T
        target, _ = retarget(source, profile)
        _, world = target.fk()
        for name in ("Neck", "Head"):
            np.testing.assert_allclose(target.rotations[0, profile.index[name]], [0,0,0,1], atol=1e-7)
            np.testing.assert_allclose(world[0, profile.index[name]], lean.as_matrix(), atol=1e-7)

    def test_timing_root_and_heading_are_not_doubled(self):
        source, profile = human_fixture(60, 60.)
        times = source.timestamps
        points = source.joints_3d @ SOURCE_TO_MMD.T
        heading = Rotation.from_euler("y", 90, degrees=True).as_matrix()
        points = points @ heading.T
        points[:, :, 0] += times[:, None] * .2
        source.joints_3d = points @ SOURCE_TO_MMD.T
        target, _ = retarget(source, profile)
        self.assertEqual(target.n_frames, 30)
        np.testing.assert_allclose(target.translations[:, 0, 0], np.arange(30)/30*.2, atol=1e-6)
        position, _ = target.fk()
        lateral = position[0, profile.index["LeftArm"]]-position[0, profile.index["RightArm"]]
        expected = heading @ (profile.position("LeftArm")-profile.position("RightArm"))
        np.testing.assert_allclose(lateral, expected, atol=1e-6)

    def test_foot_anchors_reduce_slip_and_preserve_lengths(self):
        source, profile = human_fixture(90)
        target, _ = retarget(source, profile)
        target.translations[:, 0, 0] += np.linspace(0, .09, 90)
        contact = detect_contacts(target, floor=0.)
        fixed, diagnostics = correct_feet(target, contact)
        before, after = quality(target, contact), quality(fixed, contact)
        self.assertLess(after["mean_contact_speed_m_s"], before["mean_contact_speed_m_s"]*.1)
        self.assertLess(after["max_floor_penetration_m"], .005)
        self.assertLess(diagnostics["max_reach_residual_m"], .005)
        p, _ = fixed.fk()
        for side in ("Left", "Right"):
            for a, b in (("Leg", "Knee"), ("Knee", "Ankle")):
                length = np.linalg.norm(p[:, profile.index[side+a]]-p[:, profile.index[side+b]], axis=1)
                np.testing.assert_allclose(length, np.linalg.norm(profile.position(side+a)-profile.position(side+b)), atol=1e-7)

    def test_vmd_playback_matches_target_fk_and_disables_ik(self):
        from vmd2png.skeleton import build_standard_skeleton
        from vmd2png.vmd import parse_vmd, load_vmd_to_skeleton, animate_skeleton
        source, profile = human_fixture(12)
        source.joints_3d[:, :, 0] += np.linspace(0., .025, 12)[:, None]
        target, _ = retarget(source, profile)
        target = correct_feet(target, detect_contacts(target))[0]
        path = write_vmd(self.root/"motion.vmd", target)
        ok, animation = parse_vmd(path)
        self.assertTrue(ok)
        self.assertFalse(any(animation["ik_frames"][0]["states"].values()))
        head_keys = animation["bones"]["Head"]
        self.assertEqual(len(head_keys), 12)
        np.testing.assert_allclose([key["rotation"] for key in head_keys], np.tile([0,0,0,1],(12,1)), atol=1e-7)
        root, bones = build_standard_skeleton()
        load_vmd_to_skeleton(animation, bones)
        expected, _ = target.fk()
        for frame in (0, 5, 11):
            animate_skeleton(root, frame, leg_ik=True)
            actual = np.array([bones[name].globalPos for name in profile.names])
            np.testing.assert_allclose(actual, expected[frame], atol=5e-7)

    def test_encoder_layout_and_bvh_timing(self):
        source, profile = human_fixture(10)
        motion, _ = retarget(source, profile)
        write_motion_encoder(self.root/"motion.npy", motion)
        data = np.load(self.root/"motion.npy")
        self.assertEqual(data.shape, (10, 404))
        np.testing.assert_allclose(data[:, 3], 1.)
        np.testing.assert_allclose(data[:, 4:10], np.tile([1,0,0,0,1,0], (10,1)), atol=1e-6)
        text = write_bvh(self.root/"motion.bvh", motion).read_text()
        self.assertIn("Frames: 10", text)
        self.assertIn("Frame Time: 0.033333", text)
        self.assertEqual(text.count("CHANNELS"), 65)

    def test_encoder_agrees_with_codec_on_turning_translating_motion(self):
        from vmd2png.vmd import vmd_to_motion_data
        source, profile = human_fixture(12)
        points = source.joints_3d @ SOURCE_TO_MMD.T
        points = points @ Rotation.from_euler("y", 135, degrees=True).as_matrix().T
        points[:, :, 0] += np.linspace(0., .1, 12)[:, None]
        source.joints_3d = points @ SOURCE_TO_MMD.T
        motion, _ = retarget(source, profile)
        write_motion_encoder(self.root/"motion.npy", motion)
        write_vmd(self.root/"motion.vmd", motion)
        decoded = vmd_to_motion_data(self.root/"motion.vmd", verbose=False)
        np.testing.assert_allclose(np.load(self.root/"motion.npy"), decoded, atol=2e-6)

    def test_hinge_constraints_preserve_ankle_and_foot_orientation(self):
        source, profile = human_fixture(1)
        profile.knee_limits = {side+"Knee": (-np.pi,-.0087) for side in ("Left", "Right")}
        motion, _ = retarget(source, profile)
        for side in ("Left", "Right"):
            motion.rotations[:, profile.index[side+"Knee"]] = Rotation.from_euler("xyz", [-.8,.12,.05]).as_quat()
        p, r = motion.fk()
        fixed = constrain_ik_knees(motion)
        actual, actual_r = fixed.fk()
        for side in ("Left", "Right"):
            ankle, knee = profile.index[side+"Ankle"], profile.index[side+"Knee"]
            np.testing.assert_allclose(actual[:, ankle], p[:, ankle], atol=1e-6)
            np.testing.assert_allclose(actual_r[:, ankle], r[:, ankle], atol=1e-6)
            vector = Rotation.from_quat(fixed.rotations[:, knee]).as_rotvec()
            np.testing.assert_allclose(vector[:, 1:], 0., atol=1e-8)
            self.assertLess(vector[0,0], 0.)

    def test_airborne_motion_is_not_foot_locked(self):
        source, profile = human_fixture(30)
        motion, _ = retarget(source, profile)
        motion.translations[:, 0, 1] += .5
        contacts = detect_contacts(motion, floor=0.)
        self.assertFalse(contacts["contact"].any())
        fixed, _ = correct_feet(motion, contacts)
        np.testing.assert_array_equal(fixed.rotations, motion.rotations)
        np.testing.assert_array_equal(fixed.translations, motion.translations)

    def test_fractional_fps_duration_and_invalid_source_contract(self):
        source, profile = human_fixture(300, 29.97)
        motion, _ = retarget(source, profile)
        self.assertLess(abs(motion.n_frames/30.-source.n_frames/source.fps), 1/30.)
        source.timestamps[12] = source.timestamps[11]
        with self.assertRaisesRegex(ValueError, "strictly increasing"):
            retarget(source, profile)
        source, profile = human_fixture(1)
        source.fps = np.nan
        with self.assertRaisesRegex(ValueError, "positive FPS"):
            retarget(source, profile)
        source, profile = human_fixture(1)
        motion, _ = retarget(source, profile)
        motion.save(self.root/"target.npz")
        fitted = np.load(self.root/"target.npz")["observed"]
        self.assertFalse(fitted[profile.index["LeftIndexFinger1"]])
        self.assertFalse(fitted[profile.index["Head"]])

    def test_skeleton_only_preflight_skips_all_licensed_assets(self):
        source, profile = human_fixture(1)
        path = self.root/"skeleton.npz"
        np.savez(path, joints_3d=source.joints_3d, joint_names=source.joint_names, fps=source.fps)
        args = build_parser().parse_args(["run", "--skeleton", str(path)])
        with patch.object(paths, "SMPLX_NEUTRAL", self.root/"absent.npz"), \
             patch("video2vmd.checks.check_video", side_effect=AssertionError("unexpected video check")):
            video, skeleton, image = preflight(args)
            with redirect_stdout(StringIO()) as output:
                self.assertEqual(doctor(str(path)), 0)
            self.assertNotIn("Mixamo rig", output.getvalue())
            self.assertNotIn("mixamo.com", output.getvalue())
        self.assertEqual(skeleton, path)
        self.assertIsNone(video)
        self.assertIsNone(image)

    def test_v2vmd_defaults_to_vmd_and_rejects_legacy_target_options(self):
        parser = build_parser()
        self.assertEqual(parser.prog, "v2vmd")
        self.assertEqual(parser.parse_args(["run"]).format, "vmd")
        for arguments in (["run", "--rig", "character.fbx"], ["run", "--format", "glb"]):
            with self.subTest(arguments=arguments), redirect_stderr(StringIO()), self.assertRaises(SystemExit) as error:
                parser.parse_args(arguments)
            self.assertEqual(error.exception.code, 2)

    def test_pipeline_writes_raw_corrected_and_compatible_vmd_files(self):
        source, profile = human_fixture(30)
        source.joints_3d[:, :, 0] += np.linspace(0., .08, 30)[:, None]
        path = self.root/"skeleton.npz"
        np.savez(path, joints_3d=source.joints_3d, joint_names=source.joint_names,
                 fps=source.fps, timestamps=source.timestamps)
        expected, _ = retarget(source, profile)
        for foot_lock in (False, True):
            with self.subTest(foot_lock=foot_lock):
                run = run_mmd_pipeline(skeleton_npz=path, output_dir=self.root/str(foot_lock),
                                       foot_lock=foot_lock, no_preview=True)
                report = json.loads((run/"run.json").read_text())
                outputs = report["outputs"]
                raw = run/outputs["motion_raw"]
                corrected = run/outputs["motion_corrected"]
                self.assertEqual(corrected.read_bytes(), (run/outputs["motion"]).read_bytes())
                self.assertEqual(raw.read_bytes() != corrected.read_bytes(), foot_lock)
                raw_target = np.load(run/outputs["target_motion_raw"])
                np.testing.assert_allclose(raw_target["quat_xyzw"], expected.rotations, atol=1e-10)
                np.testing.assert_allclose(raw_target["translations_m"], expected.translations, atol=1e-10)


if __name__ == "__main__":
    unittest.main()
