"""Regression checks for the GVHMR video handoff (requires the project venv + ffmpeg).

Run: python scripts/verify_video_input.py
Uses generated test patterns; no user assets or inference weights are needed.
"""

from __future__ import annotations

import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from video2vmd.extract import _stage_gvhmr_video
from video2vmd.paths import OUTPUTS, export_gvhmr_env

export_gvhmr_env()

import av
from gvhmr.cli.demo import build_demo_cfg
from gvhmr.utils.video_io_utils import get_video_fps, get_video_lwh


@unittest.skipUnless(shutil.which("ffmpeg"), "ffmpeg is required")
class VideoInputChecks(unittest.TestCase):
    def setUp(self):
        test_root = (OUTPUTS / "diagnostics").resolve()
        test_root.mkdir(parents=True, exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(prefix="v2vmd-video-", dir=test_root)
        self.root = Path(self.temp.name).resolve()
        assert self.root.is_relative_to(test_root)
        self.addCleanup(self.temp.cleanup)
        self.source = self.root / "dance #日本 [test], wow!.webm"
        subprocess.run(
            [
                "ffmpeg", "-y", "-loglevel", "error", "-f", "lavfi",
                "-i", "testsrc2=size=64x96:rate=60:duration=1",
                "-c:v", "libvpx-vp9", str(self.source),
            ],
            check=True, capture_output=True,
        )
        self.work = self.root / "cache #日本 [test], wow!"

    def test_unknown_frame_count_and_hydra_special_characters(self):
        with av.open(str(self.source)) as container:
            self.assertEqual(container.streams.video[0].frames, 0)
        staged = _stage_gvhmr_video(self.source, self.work)
        self.assertEqual(get_video_lwh(staged), (30, 64, 96))
        self.assertEqual(get_video_fps(staged), 30.0)
        cfg = build_demo_cfg(
            staged, output_root=json.dumps(self.root.as_posix(), ensure_ascii=False),
            static_cam=True, use_dpvo=False, f_mm=26, verbose=False, render_scale=None,
            config_overrides=["output_dir=" + json.dumps(self.work.as_posix(), ensure_ascii=False)],
        )
        self.assertEqual(Path(cfg.output_dir), self.work)
        self.assertEqual(Path(cfg.video_path), staged)
        # Preparing the same input again must preserve expensive cached passes.
        timestamp = staged.stat().st_mtime_ns
        self.assertEqual(_stage_gvhmr_video(self.source, self.work).stat().st_mtime_ns, timestamp)

    def test_compatible_mp4_without_ffmpeg(self):
        compatible = _stage_gvhmr_video(self.source, self.work)
        with patch("shutil.which", return_value=None):
            staged = _stage_gvhmr_video(compatible, self.root / "fallback")
        self.assertEqual(get_video_lwh(staged), (30, 64, 96))

    def test_unknown_count_without_ffmpeg_has_actionable_error(self):
        with patch("shutil.which", return_value=None):
            with self.assertRaisesRegex(RuntimeError, "Install ffmpeg"):
                _stage_gvhmr_video(self.source, self.work)

    def test_failed_transcode_does_not_leave_a_staged_video(self):
        invalid = self.root / "invalid.mp4"
        invalid.write_bytes(b"not a video")
        with self.assertRaises((RuntimeError, av.error.FFmpegError)):
            _stage_gvhmr_video(invalid, self.work)
        self.assertFalse((self.work / "0_input_video.mp4").exists())
        self.assertFalse((self.work / "0_input_video.pending.mp4").exists())
        self.assertFalse((self.work / "input_source.json").exists())


if __name__ == "__main__":
    unittest.main()
