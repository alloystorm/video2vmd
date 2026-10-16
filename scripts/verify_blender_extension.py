"""Extension command/report/package regressions; no Blender needed by default.

Background import/registration smoke check:
blender --background --factory-startup --python-exit-code 1
  --python scripts/verify_blender_extension.py -- --blender-smoke
  --bvh-run outputs/bvh_run --vmd-run outputs/vmd_run
  --mmd-tools outputs/dependencies/blender_mmd_tools --model character.pmx
"""
import argparse
import importlib
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
import zipfile

ROOT = Path(__file__).resolve().parents[1]
ADDON = ROOT/"blender_extension"/"video2vmd_blender"
sys.path.insert(0, str(ROOT))


def services():
    # Load pure services without running the Blender-only add-on entry point.
    package = types.ModuleType("extension_verification")
    package.__path__ = [str(ADDON)]
    sys.modules[package.__name__] = package
    return (importlib.import_module("extension_verification.services.backend"),
            importlib.import_module("extension_verification.services.result_reader"))


class ExtensionTests(unittest.TestCase):
    def setUp(self):
        base = ROOT/"outputs"/"diagnostics"
        base.mkdir(parents=True, exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(prefix="extension-tests-", dir=base)
        self.root = Path(self.temp.name).resolve()
        assert self.root.is_relative_to(base.resolve())
        self.addCleanup(self.temp.cleanup)
        self.backend, self.reader = services()

    def test_external_commands_match_the_renamed_cli(self):
        from video2vmd.cli import build_parser
        for source_type in ("video", "image", "skeleton"):
            for output_format in ("vmd", "bvh"):
                with self.subTest(source_type=source_type, output_format=output_format):
                    argv = self.backend.build_run_argv("python with spaces.exe", source_type=source_type,
                            source="source with spaces.npz", model="optional model.pmx", foot_lock=True,
                            output_format=output_format, device="cuda", output_dir=self.root,
                            progress_jsonl=self.root/"progress.jsonl", preview=False)
                    self.assertEqual(argv[1:3], ["-m", "video2vmd.cli"])
                    args = build_parser().parse_args(argv[3:])
                    self.assertEqual(args.format, output_format)
                    self.assertEqual(getattr(args, source_type), "source with spaces.npz")
                    self.assertTrue(args.foot_lock and args.no_preview)
                    self.assertNotIn("--rig", argv)
        args = self.backend.build_doctor_argv("python.exe", "cached.npz")
        self.assertEqual(args[-2:], ["--skeleton", "cached.npz"])
        with self.assertRaises(ValueError):
            self.backend.build_run_argv("python", source_type="video", source="clip.mp4", ik_mode="ik",
                                       device="cpu", output_dir=self.root, progress_jsonl="log", preview=False)

    def test_target_is_optional_and_codec_errors_block_readiness(self):
        python, source = self.root/"python.exe", self.root/"source.npz"
        python.touch(); source.touch()
        self.assertEqual(self.backend.validate_run_inputs(python=python, source=source, project_dir=self.root, busy=False), [])
        self.assertIn("err_model", self.backend.validate_run_inputs(python=python, source=source,
                      project_dir=self.root, model=self.root/"absent.pmx", busy=False))
        self.assertEqual(self.backend.interpret_doctor_output(1,"[missing] input source"), ("READY","env_ready_choose"))
        status, _ = self.backend.interpret_doctor_output(1,"[missing] input source\n[missing] VMD codec vmd2png")
        self.assertEqual(status,"ERROR")

    def test_result_reader_accepts_vmd_and_bvh_reports(self):
        for extension in ("vmd", "bvh"):
            motion = self.root/f"motion.{extension}"
            motion.touch()
            report = {"status":"completed", "fps":30., "n_frames":90, "outputs":{"motion":motion.name}}
            (self.root/"run.json").write_text(json.dumps(report))
            result = self.reader.read_run_json(self.root)
            self.assertEqual(result.status,"completed")
            self.assertEqual(result.motion_path,motion)
            motion.unlink()
            self.assertEqual(self.reader.read_run_json(self.root).error_code,"MOTION_MISSING")
        settings = self.reader.blender_frame_settings(29.97,90)
        self.assertAlmostEqual(settings["fps"]/settings["fps_base"],29.97)

    def test_package_contains_only_the_current_thin_extension(self):
        from scripts.package_blender_extension import build_zip, validate_zip
        archive = build_zip(dest=self.root/"extension.zip")
        self.assertEqual(validate_zip(archive), [])
        with zipfile.ZipFile(archive) as package:
            manifest = package.read("blender_manifest.toml").decode()
            self.assertIn('id = "video2vmd"',manifest)
            self.assertIn('version = "0.2.0"',manifest)
            for name in package.namelist():
                self.assertNotIn("motion2mixamorig",name)
                self.assertNotIn("__pycache__",name)
                if name.endswith(".py"):
                    source = package.read(name).decode()
                    self.assertNotIn("Mixamo",source)
                    self.assertNotIn("import torch",source)
                    self.assertNotIn("import gvhmr",source)
                    self.assertNotIn("import cv2",source)


def blender_smoke():
    import bpy
    parser = argparse.ArgumentParser()
    parser.add_argument("--blender-smoke", action="store_true")
    parser.add_argument("--bvh-run", type=Path, required=True)
    parser.add_argument("--vmd-run", type=Path)
    parser.add_argument("--mmd-tools", type=Path)
    parser.add_argument("--model", type=Path)
    args = parser.parse_args(sys.argv[sys.argv.index("--")+1:])
    sys.path.insert(0,str(ADDON.parent))
    import video2vmd_blender as addon
    from video2vmd_blender.operators.import_result import import_generated_motion
    from video2vmd_blender.services.result_reader import read_run_json
    addon.register()
    assert hasattr(bpy.types.Scene,"v2vmd")
    assert bpy.context.scene.v2vmd.output_format == "vmd"
    assert not hasattr(bpy.context.scene.v2vmd,"rig_path")
    # Exercise the actual external-process/report handoff with no target model.
    from video2vmd_blender.services.backend import build_run_argv, launch_process, subprocess_env
    from video2vmd_blender.operators.runtime import JobRuntime, _finish_run
    external = ROOT/".venv"/("Scripts/python.exe" if sys.platform == "win32" else "bin/python")
    job_dir = ROOT/"outputs/diagnostics/v2vmd_extension_subprocess"
    job_dir.mkdir(parents=True,exist_ok=True)
    progress, log = job_dir/"progress.jsonl", job_dir/"job.log"
    argv = build_run_argv(external,source_type="skeleton",source=(args.bvh_run/"skeleton_motion.npz").resolve(),
                          device="cpu",output_dir=job_dir,progress_jsonl=progress,preview=False)
    process, handle = launch_process(argv,cwd=ROOT,log_path=log,env=subprocess_env(ROOT))
    try:
        code = process.wait(timeout=60)
    finally:
        if process.poll() is None:
            process.kill()
        handle.close()
    assert code == 0,log.read_text(encoding="utf-8",errors="replace")
    runtime = JobRuntime(kind="run",process=process,owned_pid=process.pid,output_dir=job_dir,
                         log_path=log,progress_path=progress,auto_import=False)
    _finish_run(runtime,bpy.context.scene.v2vmd,code)
    assert bpy.context.scene.v2vmd.job_status == "completed"
    assert bpy.context.scene.v2vmd.job_motion_path.endswith("motion.vmd")
    result = read_run_json(args.bvh_run)
    import_generated_motion(bpy.context,result,source_name="smoke_bvh")
    assert bpy.context.view_layer.objects.active.type == "ARMATURE"
    bvh_arm = bpy.context.view_layer.objects.active
    bvh_action = bvh_arm.animation_data.action
    assert bpy.context.scene.frame_end == result.n_frames
    if args.vmd_run:
        result = read_run_json(args.vmd_run)
        try:
            import_generated_motion(bpy.context,result,source_name="needs_mmd_tools")
        except RuntimeError as exc:
            assert "MMD Tools" in str(exc),exc
        else:
            raise AssertionError("Missing MMD Tools should produce an actionable import error")
        if args.mmd_tools and args.model:
            vendor = (args.vmd_run/"validation_dependencies").resolve()
            vendor.mkdir(parents=True,exist_ok=True)
            for wheel in (args.mmd_tools/"mmd_tools"/"wheels").glob("*.whl"):
                with zipfile.ZipFile(wheel) as archive:
                    assert all((vendor/name).resolve().is_relative_to(vendor) for name in archive.namelist())
                    archive.extractall(vendor)
            sys.path[:0] = [str(args.mmd_tools.resolve()),str(vendor)]
            import mmd_tools
            mmd_tools.register()
            existing = set(bpy.data.objects)
            bpy.ops.mmd_tools.import_model(filepath=str(args.model.resolve()),types={"ARMATURE"},scale=.085,
                    rename_bones=False,fix_ik_links=False,apply_bone_fixed_axis=False,save_log=False)
            arm = next(o for o in bpy.data.objects if o not in existing and o.type == "ARMATURE")
            bpy.context.view_layer.objects.active = arm
            arm.select_set(True)
            bvh_arm.select_set(True)
            import_generated_motion(bpy.context,result,source_name="smoke_vmd")
            assert arm.animation_data and arm.animation_data.action
            assert bvh_arm.animation_data.action == bvh_action, "VMD import must not overwrite another selected armature"
            assert bpy.context.scene.frame_end == result.n_frames
    assert not any(name in sys.modules for name in ("torch","gvhmr","cv2"))
    addon.unregister()
    assert not hasattr(bpy.types.Scene,"v2vmd")
    addon.register(); addon.unregister()
    print("Video2VMD Blender subprocess, registration, BVH/VMD import and thin-runtime checks passed")


if __name__ == "__main__":
    if "--blender-smoke" in sys.argv:
        blender_smoke()
    else:
        unittest.main()
