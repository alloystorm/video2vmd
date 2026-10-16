"""Heel/toe grounding regressions, independent of inference and licensed models."""
from pathlib import Path
import sys
import unittest

import numpy as np
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from verify_mmd import human_fixture
from video2vmd.mmd.contacts import correct_feet, detect_contacts, foot_points, _foot_target
from video2vmd.mmd.retarget import retarget


def fixture(count=12):
    source, profile = human_fixture(count)
    motion, _ = retarget(source, profile)
    return motion


def evidence(motion, support):
    points = foot_points(motion)
    contact = detect_contacts(motion, floor=0.)
    contact["contact"][:] = np.array(support, dtype=bool)
    contact["weights"][:] = np.array(support, dtype=float)
    contact["anchors_m"][:] = points[0]
    contact["anchors_m"][:, :, 1] = 0.
    return contact


class FeetTests(unittest.TestCase):
    def test_double_support_levels_sole_and_preserves_lengths(self):
        motion = fixture()
        for side in ("Left", "Right"):
            motion.rotations[:, motion.profile.index[side+"Ankle"]] = Rotation.from_euler("x", 10, degrees=True).as_quat()
        motion.translations[:, 0, 1] += .02
        fixed, report = correct_feet(motion, evidence(motion, [1, 1, 1, 1]))
        np.testing.assert_allclose(foot_points(fixed)[:, :, 1], 0., atol=1e-7)
        p, _ = fixed.fk()
        for side in ("Left", "Right"):
            for first, last in (("Leg", "Knee"), ("Knee", "Ankle")):
                a, b = [motion.profile.index[side+part] for part in (first, last)]
                np.testing.assert_allclose(np.linalg.norm(p[:, a]-p[:, b], axis=1),
                                           np.linalg.norm(motion.profile.rest[a]-motion.profile.rest[b]), atol=1e-8)
        self.assertLess(report["max_reach_residual_m"], 1e-7)
        self.assertLessEqual(report["max_foot_pitch_correction_rad"], .26)

    def test_partial_contact_preserves_heel_roll_and_toe_stand(self):
        for angle, endpoint in ((12., 0), (-12., 1)):
            with self.subTest(angle=angle):
                motion = fixture()
                ankle = motion.profile.index["LeftAnkle"]
                motion.rotations[:, ankle] = Rotation.from_euler("x", angle, degrees=True).as_quat()
                motion.translations[:, 0, 1] -= foot_points(motion)[0, endpoint, 1]
                support = [0, 0, 0, 0]; support[endpoint] = 1
                before, r = motion.fk()
                fixed, _ = correct_feet(motion, evidence(motion, support))
                points = foot_points(fixed)
                np.testing.assert_allclose(points[:, endpoint, 1], 0., atol=1e-7)
                self.assertGreater(points[:, 1-endpoint, 1].min(), .02)
                np.testing.assert_allclose(fixed.fk()[1][:, ankle], r[:, ankle], atol=1e-7)

    def test_swing_collision_is_up_only_and_preserves_toe_articulation(self):
        motion = fixture()
        motion.translations[:, 0, 1] -= .01
        ankle = motion.profile.index["LeftAnkle"]
        toe = motion.profile.index["LeftToe"]
        motion.rotations[:, ankle] = Rotation.from_euler("x", 8, degrees=True).as_quat()
        motion.rotations[:, toe] = Rotation.from_euler("x", 20, degrees=True).as_quat()
        p, _ = motion.fk()
        fixed, report = correct_feet(motion, evidence(motion, [0, 0, 0, 0]))
        after, _ = fixed.fk()
        self.assertGreaterEqual(foot_points(fixed)[:, :, 1].min(), -1e-7)
        np.testing.assert_allclose(after[:, ankle][:, [0, 2]], p[:, ankle][:, [0, 2]], atol=1e-7)
        self.assertGreaterEqual((after[:, ankle, 1]-p[:, ankle, 1]).min(), -1e-7)
        np.testing.assert_array_equal(fixed.rotations[:, toe], motion.rotations[:, toe])
        self.assertEqual(report["swing_clearance_frames"], motion.n_frames)

    def test_pelvis_reach_assist_keeps_other_foot_on_swing_path(self):
        motion = fixture()
        profile = motion.profile
        motion.rotations[:, profile.index["RightLeg"]] = Rotation.from_euler("x", -35, degrees=True).as_quat()
        contact = evidence(motion, [1, 1, 0, 0])
        contact["anchors_m"][:, :2, 0] += .10
        p, r = motion.fk()
        fixed, report = correct_feet(motion, contact)
        actual, world = fixed.fk()
        self.assertGreater(report["max_pelvis_lowering_m"], .001)
        self.assertLess(report["max_reach_residual_m"], 1e-7)
        ankle = profile.index["RightAnkle"]
        np.testing.assert_allclose(actual[:, ankle], p[:, ankle], atol=1e-7)
        np.testing.assert_allclose(world[:, ankle], r[:, ankle], atol=1e-7)

    def test_jumps_and_lifted_feet_are_unchanged(self):
        motion = fixture()
        motion.translations[:, 0, 1] += .25
        fixed, report = correct_feet(motion, evidence(motion, [0, 0, 0, 0]))
        np.testing.assert_array_equal(fixed.rotations, motion.rotations)
        np.testing.assert_array_equal(fixed.translations, motion.translations)
        self.assertEqual(report["corrected_frames"], 0)

    def test_degenerate_contacts_and_slide_safety_limit(self):
        points = np.array([[0., 0., 0.], [0., 0., 0.]])
        ankle = np.array([0., .1, 0.])
        target, tilt, pitch, limited = _foot_target(points, ankle, points+[1, 0, 0], np.ones(2), 0.)
        self.assertTrue(limited)
        np.testing.assert_allclose(target-ankle, [.12, 0., 0.], atol=1e-8)
        np.testing.assert_array_equal(tilt, np.eye(3))
        self.assertEqual(pitch, 0.)

    def test_world_heading_equivariance(self):
        points = np.array([[0., -.01, 0.], [0., .02, -.17]])
        ankle = np.array([0., .10, 0.])
        anchors = points+np.array([.02, 0., -.01])
        target, tilt, _, _ = _foot_target(points, ankle, anchors, np.ones(2), 0.)
        turn = Rotation.from_euler("y", 135, degrees=True).as_matrix()
        actual, actual_tilt, _, _ = _foot_target(points@turn.T, turn@ankle, anchors@turn.T, np.ones(2), 0.)
        np.testing.assert_allclose(actual, turn@target, atol=1e-9)
        np.testing.assert_allclose(actual_tilt, turn@tilt@turn.T, atol=1e-9)

    def test_unreachable_floor_reports_residual_and_caps_pelvis(self):
        motion = fixture()
        contact = evidence(motion, [1, 1, 1, 1])
        contact["floor_m"] = -1.
        fixed, report = correct_feet(motion, contact)
        self.assertAlmostEqual(report["max_pelvis_lowering_m"], .08)
        self.assertGreater(report["max_reach_residual_m"], .8)
        self.assertTrue(np.isfinite(fixed.rotations).all())
        np.testing.assert_allclose(np.linalg.norm(fixed.rotations, axis=2), 1., atol=1e-8)


if __name__ == "__main__":
    unittest.main()
