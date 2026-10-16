"""Depth-cleanup regressions: known motion, projection, contacts and exports."""
from dataclasses import replace
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
from scipy.signal import butter,sosfiltfilt

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from verify_mmd import human_fixture
from video2vmd.cli import build_parser
from video2vmd.depth_cleanup import regularize_depth,clean_skeleton,check_depth_input,_project
from video2vmd.mmd.pipeline import run_mmd_pipeline
from video2vmd.mmd.retarget import retarget


def fixture(count=300,noise=.012,travel=True):
    human,profile=human_fixture(count)
    times=human.timestamps
    truth=human.joints_3d.copy()
    if travel:
        truth[:,:,2]+=(.12*np.sin(2*np.pi*.6*times))[:,None]
    points=truth.copy();points[:,:,2]+=(noise*np.sin(2*np.pi*8*times))[:,None]
    human=replace(human,joints_3d=points,root_translation=points[:,0].copy())
    rotations=np.repeat(np.diag([1.,-1.,-1.])[None],count,axis=0)
    center=np.array([0.,1.2,4.])
    cam=np.einsum("tij,tbj->tbi",rotations,points-center)
    camera={"camera_joints_3d":cam,"camera_world_to_camera":rotations,
            "camera_intrinsics":np.array([[1300.,0,540],[0,1300,960],[0,0,1]]),
            "camera_image_size":np.array([1080,1920]),"camera":np.array("static")}
    return human,camera,truth


class DepthTests(unittest.TestCase):
    def setUp(self):
        root=Path(__file__).resolve().parents[1]/"outputs/diagnostics"
        root.mkdir(parents=True,exist_ok=True)
        self.temp=tempfile.TemporaryDirectory(prefix="depth-tests-",dir=root)
        self.root=Path(self.temp.name).resolve()
        assert self.root.is_relative_to(root.resolve())
        self.addCleanup(self.temp.cleanup)

    def save(self,human,camera):
        path=self.root/"skeleton.npz"
        np.savez_compressed(path,joints_3d=human.joints_3d,joint_names=human.joint_names,
                            timestamps=human.timestamps,fps=human.fps,
                            root_translation=human.root_translation,**camera)
        human.path=path
        return path

    def test_known_jitter_is_reduced_without_lag_or_losing_slow_travel(self):
        human,camera,truth=fixture()
        cleaned,cam,diag,report=regularize_depth(human,camera)
        trim=slice(30,-30)
        before=np.sqrt(np.mean((human.joints_3d[trim,0,2]-truth[trim,0,2])**2))
        after=np.sqrt(np.mean((cleaned.joints_3d[trim,0,2]-truth[trim,0,2])**2))
        self.assertLess(after,before*.55)
        self.assertLess(report["root_high_frequency_rms_after_m"],report["root_high_frequency_rms_before_m"]*.6)
        smooth=lambda a:sosfiltfilt(butter(3,1.5,fs=30,output="sos"),a)
        a=smooth(cleaned.joints_3d[:,0,2])[trim];b=smooth(truth[:,0,2])[trim]
        self.assertGreater(np.corrcoef(a,b)[0,1],.999)
        self.assertLess(np.max(np.abs(a-b)),.003)
        self.assertEqual(np.argmax(np.correlate(a-a.mean(),b-b.mean(),"full")),len(a)-1)

    def test_articulation_timing_and_endpoints_are_preserved(self):
        human,camera,_=fixture()
        cleaned,cam,diag,report=regularize_depth(human,camera)
        np.testing.assert_allclose(cleaned.joints_3d-cleaned.joints_3d[:,:1],human.joints_3d-human.joints_3d[:,:1],atol=1e-12)
        np.testing.assert_array_equal(cleaned.joints_3d[[0,-1]],human.joints_3d[[0,-1]])
        np.testing.assert_array_equal(cleaned.timestamps,human.timestamps)
        self.assertEqual(cleaned.fps,human.fps)
        np.testing.assert_allclose(cleaned.root_translation-human.root_translation,diag["delta_world_m"],atol=1e-12)

    def test_projection_and_edit_guards_hold_for_large_outlier(self):
        human,camera,_=fixture(noise=0,travel=False)
        points=human.joints_3d.copy();points[100,:,2]+=.5
        human=replace(human,joints_3d=points)
        camera["camera_joints_3d"]=np.einsum("tij,tbj->tbi",camera["camera_world_to_camera"],points-[0,1.2,4])
        cleaned,cam,diag,report=regularize_depth(human,camera)
        self.assertLessEqual(report["max_depth_correction_m"],.04)
        self.assertLessEqual(report["max_reprojection_rms_image_height"],.001+1e-10)
        self.assertLessEqual(report["max_reprojection_joint_image_height"],.002+1e-10)
        k=np.repeat(camera["camera_intrinsics"][None],human.n_frames,axis=0)
        np.testing.assert_allclose(_project(cam,k)[:,0],_project(camera["camera_joints_3d"],k)[:,0],atol=1e-9)
        old_center=human.joints_3d[:,0]-np.einsum("tji,tj->ti",camera["camera_world_to_camera"],camera["camera_joints_3d"][:,0])
        new_center=cleaned.joints_3d[:,0]-np.einsum("tji,tj->ti",camera["camera_world_to_camera"],cam[:,0])
        np.testing.assert_allclose(new_center,old_center,atol=1e-12)

    def test_constant_velocity_and_short_static_clips_are_unchanged(self):
        for count in (1,6,120):
            human,camera,_=fixture(count,noise=0,travel=False)
            p=human.joints_3d.copy();p[:,:,2]+=(human.timestamps*.2)[:,None]
            human=replace(human,joints_3d=p)
            camera["camera_joints_3d"]=np.einsum("tij,tbj->tbi",camera["camera_world_to_camera"],p-[0,1.2,4])
            cleaned,_,_,_=regularize_depth(human,camera)
            np.testing.assert_allclose(cleaned.joints_3d,p,atol=1e-10)

    def test_contact_evidence_reduces_supported_foot_motion(self):
        human,camera,_=fixture(travel=False)
        without,_,_,_=regularize_depth(human,camera)
        with_contact,_,_,report=regularize_depth(human,camera,contacts={"weights":np.ones((human.n_frames,4))})
        foot=human.index("left_foot")
        speed=lambda h:np.mean(np.diff(h.joints_3d[30:-30,foot,[0,2]],axis=0)**2)
        self.assertGreater(report["contact_pairs"],0)
        self.assertLess(speed(with_contact),speed(without))

    def test_nonuniform_timestamps_preserve_linear_depth_travel(self):
        human,camera,_=fixture(90,noise=0,travel=False)
        times=np.cumsum(np.resize([.03,.04,.03],90));times-=times[0]
        p=human.joints_3d.copy();p[:,:,2]+=(times*.15)[:,None]
        human=replace(human,joints_3d=p,timestamps=times)
        camera["camera_joints_3d"]=np.einsum("tij,tbj->tbi",camera["camera_world_to_camera"],p-[0,1.2,4])
        cleaned,_,_,_=regularize_depth(human,camera)
        np.testing.assert_allclose(cleaned.joints_3d,p,atol=1e-10)

    def test_fast_coherent_depth_change_is_protected(self):
        human,camera,_=fixture(180,noise=0,travel=False)
        step=.18/(1+np.exp(-(human.timestamps-2.5)/.04))
        p=human.joints_3d.copy();p[:,:,2]+=step[:,None]
        human=replace(human,joints_3d=p)
        camera["camera_joints_3d"]=np.einsum("tij,tbj->tbi",camera["camera_world_to_camera"],p-[0,1.2,4])
        cleaned,_,diag,_=regularize_depth(human,camera)
        self.assertGreater(diag["motion_protection"].max(),.7)
        self.assertLess(np.max(np.abs(cleaned.joints_3d[:,0,2]-p[:,0,2])),.012)
        expected=np.argmax(np.gradient(p[:,0,2]))
        actual=np.argmax(np.gradient(cleaned.joints_3d[:,0,2]))
        self.assertLessEqual(abs(actual-expected),1)

    def test_bad_camera_and_repeat_cleanup_fail_clearly(self):
        human,camera,_=fixture(60)
        path=self.save(human,camera)
        with self.assertRaisesRegex(ValueError,"separate"):
            clean_skeleton(path,path)
        destination=self.root/"corrected.npz"
        clean_skeleton(path,destination)
        self.assertIn("already",check_depth_input(destination))
        moving=dict(camera);moving["camera"]=np.array("simplevo")
        self.assertIn("static",check_depth_input(self.save(human,moving)))
        camera["camera_joints_3d"][0,0,2]=-1
        with self.assertRaisesRegex(ValueError,"in front"):
            regularize_depth(human,camera)
        np.savez(path,joints_3d=human.joints_3d)
        self.assertIn("observations",check_depth_input(path))

    def test_pipeline_keeps_raw_exports_and_lists_cleanup_artifacts(self):
        human,camera,_=fixture(60)
        source=self.save(human,camera);unchanged=source.read_bytes()
        expected,_=retarget(human,human_fixture(1)[1])
        run=run_mmd_pipeline(skeleton_npz=source,output_dir=self.root/"run",depth_cleanup=True,no_preview=True)
        self.assertEqual(source.read_bytes(),unchanged)
        self.assertEqual((run/"skeleton_motion.npz").read_bytes(),unchanged)
        report=json.loads((run/"run.json").read_text())
        for key in ("depth_cleanup","depth_cleanup_evidence","skeleton_motion_corrected","target_motion_depth"):
            self.assertTrue((run/report["outputs"][key]).is_file())
        with np.load(run/"target_motion_raw.npz") as raw:
            np.testing.assert_allclose(raw["quat_xyzw"],expected.rotations,atol=1e-10)
            np.testing.assert_allclose(raw["translations_m"],expected.translations,atol=1e-10)
        self.assertNotEqual((run/"motion_raw.vmd").read_bytes(),(run/"motion_corrected.vmd").read_bytes())
        self.assertEqual((run/"motion.vmd").read_bytes(),(run/"motion_corrected.vmd").read_bytes())
        self.assertTrue(report["depth_cleanup"]["enabled"])
        parser=build_parser()
        self.assertFalse(parser.parse_args(["run"]).depth_cleanup)
        self.assertTrue(parser.parse_args(["run","cached.npz","--depth-cleanup"]).depth_cleanup)


if __name__=="__main__":
    unittest.main(verbosity=2)
