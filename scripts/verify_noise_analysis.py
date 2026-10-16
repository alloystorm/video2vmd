"""Numerical checks for noise-study units, filtering and common-motion analysis."""
import unittest

import numpy as np

from analyze_motion_noise import highpass, interior, trajectory_metrics, vector_rms, correlation


class NoiseAnalysisTests(unittest.TestCase):
    def test_highpass_rejects_slow_motion_and_retains_known_8hz_noise(self):
        fps=30.;t=np.arange(600)/fps
        slow=np.column_stack([.1*np.sin(2*np.pi*.5*t),np.zeros(len(t)),np.zeros(len(t))])
        noise=np.column_stack([np.zeros(len(t)),np.zeros(len(t)),.01*np.sin(2*np.pi*8*t)])
        hp=highpass(slow+noise,fps)[interior(len(t),fps)]
        self.assertLess(vector_rms(hp-noise[interior(len(t),fps)]),.00002)
        report=trajectory_metrics(slow+noise,fps)
        self.assertAlmostEqual(report["highpass_rms_mm"]["3"],10/np.sqrt(2),delta=.02)
        self.assertLess(report["highpass_axis_rms_mm"][0],.01)

    def test_shared_translation_cancels_in_hip_relative_foot(self):
        t=np.arange(300)/30
        shared=np.column_stack([0*t,0*t,.02*np.sin(2*np.pi*7*t)])
        hip=np.column_stack([.1*t,0*t+1.,0*t])
        foot=hip+np.column_stack([0*t+.1,-.8+.03*np.sin(2*np.pi*2*t),0*t])
        np.testing.assert_allclose((foot+shared)-(hip+shared),foot-hip,atol=1e-15)
        self.assertLess(trajectory_metrics(foot-hip,30)["highpass_rms_mm"]["3"],1.)
        self.assertGreater(trajectory_metrics(foot+shared,30)["highpass_rms_mm"]["3"],10.)

    def test_timing_and_undefined_correlations(self):
        self.assertEqual(interior(231,30),slice(30,201))
        self.assertEqual(interior(60,30),slice(12,48))
        self.assertIsNone(correlation(np.ones(20),np.arange(20)))
        self.assertAlmostEqual(correlation(np.arange(20),-np.arange(20)),-1.)


if __name__ == "__main__":
    unittest.main()
