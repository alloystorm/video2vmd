"""Hand interpolation regressions, independent of inference and licensed assets."""
from pathlib import Path
import sys
import unittest

import numpy as np
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from video2vmd.hand_motion import fill_hand_gaps, complete_wilor_observations, OBSERVED, INTERPOLATED, HELD, UNAVAILABLE


def source(count=5, joints=2):
    q = np.zeros((count, 2, joints, 4)); q[..., 3] = 1.
    return q, np.zeros((count, 2), dtype=bool)


class HandMotionTests(unittest.TestCase):
    def test_gaps_follow_timestamps_and_preserve_observed_rotations(self):
        times = np.array([0., .1, .3, .8, 1.])
        q, mask = source()
        q[-1, 0] = Rotation.from_euler("z", 100., degrees=True).as_quat()
        mask[[0, -1], 0] = True
        q[1:-1, 0] = np.nan
        data = fill_hand_gaps(times, q, mask)
        angle = Rotation.from_quat(data["quat_xyzw"][:, 0, 0]).as_rotvec()[:, 2]
        np.testing.assert_allclose(angle, np.radians(times*100.), atol=1e-10)
        np.testing.assert_array_equal(data["status"][:, 0], [OBSERVED, INTERPOLATED, INTERPOLATED, INTERPOLATED, OBSERVED])
        for frame in (0, 4):
            np.testing.assert_allclose(Rotation.from_quat(data["quat_xyzw"][frame, 0]).as_matrix(),
                                       Rotation.from_quat(q[frame, 0]).as_matrix(), atol=1e-12)

    def test_shortest_arc_across_180_degrees_and_antipodal_quaternions(self):
        q, mask = source(3)
        q[0, 0] = Rotation.from_euler("y", 170., degrees=True).as_quat()
        q[-1, 0] = -Rotation.from_euler("y", -170., degrees=True).as_quat()
        mask[[0, 2], 0] = True
        data = fill_hand_gaps([0., 1., 2.], q, mask)
        angle = Rotation.from_quat(data["quat_xyzw"][1, 0, 0]).magnitude()
        self.assertAlmostEqual(angle, np.pi)
        q[-1, 0] = -q[0, 0]
        fixed = fill_hand_gaps([0., 1., 2.], q, mask)["quat_xyzw"]
        np.testing.assert_allclose(fixed[:, 0], np.broadcast_to(fixed[0, 0], fixed[:, 0].shape), atol=1e-12)

    def test_hands_are_independent_and_edges_hold_nearest(self):
        q, mask = source(6)
        q[1, 0] = Rotation.from_euler("x", 30., degrees=True).as_quat()
        q[4, 0] = Rotation.from_euler("x", 60., degrees=True).as_quat()
        q[3, 1] = Rotation.from_euler("z", 50., degrees=True).as_quat()
        mask[[1, 4], 0] = True; mask[3, 1] = True
        data = fill_hand_gaps(np.arange(6), q, mask)
        np.testing.assert_array_equal(data["status"][:, 0], [HELD, OBSERVED, INTERPOLATED, INTERPOLATED, OBSERVED, HELD])
        np.testing.assert_array_equal(data["status"][:, 1], [HELD, HELD, HELD, OBSERVED, HELD, HELD])
        np.testing.assert_allclose(data["quat_xyzw"][0, 0], q[1, 0], atol=1e-12)
        np.testing.assert_allclose(data["quat_xyzw"][-1, 0], q[4, 0], atol=1e-12)
        np.testing.assert_allclose(data["quat_xyzw"][:, 1], np.broadcast_to(q[3, 1], q[:, 1].shape), atol=1e-12)

    def test_long_gaps_interpolate_without_fading_and_missing_hand_stays_unavailable(self):
        q, mask = source(3)
        mask[[0, 2], 0] = True
        q[-1, 0] = Rotation.from_euler("x", 90., degrees=True).as_quat()
        q[:, 1] = np.nan
        data = fill_hand_gaps([0., 50., 100.], q, mask)
        self.assertEqual(data["status"][1, 0], INTERPOLATED)
        self.assertAlmostEqual(Rotation.from_quat(data["quat_xyzw"][1, 0, 0]).magnitude(), np.pi/4)
        np.testing.assert_array_equal(data["status"][:, 1], UNAVAILABLE)
        np.testing.assert_array_equal(data["quat_xyzw"][:, 1, :, :3], 0.)
        np.testing.assert_array_equal(data["quat_xyzw"][:, 1, :, 3], 1.)
        self.assertFalse(data["available"][:, 1].any())

    def test_output_timeline_and_shape_interpolation(self):
        q, mask = source(2)
        mask[:, 0] = True
        q[-1, 0] = Rotation.from_euler("z", 90., degrees=True).as_quat()
        betas = np.full((2, 2, 10), np.nan); betas[:, 0] = np.array([0., 4.])[:, None]
        data = fill_hand_gaps([10., 12.], q, mask, betas=betas, target_timestamps=[9., 10., 10.5, 11., 12., 13.])
        np.testing.assert_allclose(data["betas"][:, 0, 0], [0., 0., 1., 2., 4., 4.])
        np.testing.assert_array_equal(data["status"][:, 0], [HELD, OBSERVED, INTERPOLATED, INTERPOLATED, OBSERVED, HELD])
        np.testing.assert_array_equal(data["source_observed"], mask)
        np.testing.assert_array_equal(data["source_timestamps"], [10., 12.])

    def test_invalid_observations_and_timing_are_rejected(self):
        q, mask = source(3); mask[:, 0] = True
        for times in ([0., 0., 1.], [0., np.nan, 1.], [2., 1., 0.]):
            with self.assertRaises(ValueError):
                fill_hand_gaps(times, q, mask)
        for invalid in (np.nan, 0., 2.):
            broken = q.copy(); broken[0, 0] = invalid
            with self.assertRaises(ValueError):
                fill_hand_gaps([0., 1., 2.], broken, mask)

    def test_probe_conversion_keeps_body_identity_and_does_not_mutate_input(self):
        import copy
        candidate = {"mano_global_orient_matrix": np.eye(3)[None].tolist(),
                     "mano_pose_matrices": np.repeat(np.eye(3)[None], 15, axis=0).tolist(),
                     "mano_betas": [0.]*10, "is_right": False}
        frames = []
        for t in (0., 1., 2.):
            hand = copy.deepcopy(candidate)
            hand["is_right"] = t == 2.  # Raw detector labels cannot exchange tracks.
            frames.append({"time_s": t, "hands": [
                {"side": "left", "crop_match": 0 if t != 1. else None, "crop_candidates": [hand]},
                {"side": "right", "crop_match": None, "crop_candidates": []}]})
        document = {"clip": "fixture", "observations": frames}
        before = copy.deepcopy(document)
        data = complete_wilor_observations(document)
        self.assertEqual(document, before)
        np.testing.assert_array_equal(data["status"][:, 0], [OBSERVED, INTERPOLATED, OBSERVED])
        np.testing.assert_array_equal(data["status"][:, 1], UNAVAILABLE)
        np.testing.assert_array_equal(data["source_detector_right"][:, 0], [0, -1, 1])
        self.assertEqual(data["quat_xyzw"].shape, (3, 2, 16, 4))


if __name__ == "__main__":
    unittest.main()
