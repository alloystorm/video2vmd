"""Numerical checks for noise-study units, filtering and common-motion analysis."""
import unittest
from pathlib import Path
import tempfile

import numpy as np

from analyze_motion_noise import highpass, interior, trajectory_metrics, vector_rms, correlation
from compare_reference_vmd import window_metrics,SampledWorldMotion


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

    def test_reference_windows_preserve_frequency_units_and_duration(self):
        t=np.arange(600)/30
        points=np.zeros((len(t),3,3))
        points[:,:,2]=.01*np.sin(2*np.pi*8*t)[:,None]*np.array([1.,2.,3.])
        windows=window_metrics(points)
        self.assertEqual(len(windows),4)
        self.assertEqual([w["start_s"] for w in windows],[0.,3.5,7.,10.5])
        for row in windows:
            self.assertEqual(row["end_s"]-row["start_s"],7.)
            np.testing.assert_allclose(row["highpass_rms_mm"],np.array([1.,2.,3.])*10/np.sqrt(2),atol=.1)

    def test_sampled_reference_uses_identical_contact_geometry(self):
        from verify_mmd import human_fixture
        from video2vmd.mmd.retarget import retarget
        from video2vmd.mmd.contacts import foot_points,detect_contacts,quality
        human,profile=human_fixture(90)
        target,_=retarget(human,profile)
        target.translations[:,0,0]+=np.linspace(0,.09,90)
        names=["HipMaster"]+[side+part for side in ("Left","Right") for part in ("Leg","Knee","Ankle","Toe")]
        ids=[profile.index[name] for name in names]
        p,r=target.fk()
        root=Path(__file__).resolve().parents[1]/"outputs/diagnostics"
        root.mkdir(parents=True,exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="reference-tests-",dir=root) as folder:
            path=Path(folder)/"world.npz"
            np.savez(path,positions_m=p[:,ids],world_rotations=r[:,ids],names=names,
                     rest_positions_m=profile.rest[ids],fps=30.)
            sampled=SampledWorldMotion(path)
            np.testing.assert_allclose(foot_points(sampled),foot_points(target),atol=1e-12)
            evidence=detect_contacts(target,floor=0.)
            actual=detect_contacts(sampled,floor=0.)
            np.testing.assert_array_equal(evidence["contact"],actual["contact"])
            self.assertAlmostEqual(quality(target,evidence)["mean_contact_speed_m_s"],
                                   quality(sampled,actual)["mean_contact_speed_m_s"],places=12)


if __name__ == "__main__":
    unittest.main()
