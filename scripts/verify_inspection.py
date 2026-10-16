"""Inspector data/HTTP/CLI integration tests; no licensed model or inference required."""
from contextlib import redirect_stdout
import http.client
from io import StringIO
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import threading
import unittest
from unittest.mock import patch

import numpy as np
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from video2vmd.cli import build_parser, preflight, run, inspect_run
from video2vmd.inspection.server import prepare_inspection, prepare_media, make_server, resolve_source
from video2vmd.mmd.motion import TargetMotion
from video2vmd.mmd.profile import standard_profile


class InspectionTests(unittest.TestCase):
    def setUp(self):
        base = Path(__file__).resolve().parents[1] / "outputs" / "diagnostics"
        base.mkdir(parents=True, exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(prefix="inspector-tests-", dir=base)
        self.root = Path(self.temp.name).resolve()
        assert self.root.is_relative_to(base.resolve())
        self.addCleanup(self.temp.cleanup)
        profile = standard_profile()
        rotations = np.zeros((10, len(profile.names), 4)); rotations[:, :, 3] = 1
        translations = np.zeros((10, len(profile.names), 3))
        self.raw = TargetMotion(profile, rotations.copy(), translations.copy(), 30)
        rotations[:, profile.index["LeftArm"]] = Rotation.from_euler("z", (np.arange(10)*3)[:, None], degrees=True).as_quat()
        translations[:, profile.index[profile.root], 0] = np.arange(10) * .01
        self.corrected = TargetMotion(profile, rotations, translations, 30)
        self.raw.save(self.root / "target_motion_raw.npz")
        self.corrected.save(self.root / "target_motion.npz")

    def test_browser_buffer_matches_export_fk_and_preserves_variants(self):
        inspection = prepare_inspection(self.root, transcode=False)
        meta = inspection.metadata
        self.assertEqual(meta["variants"], ["corrected", "raw"])
        self.assertEqual(meta["fps"], 30)
        self.assertEqual(meta["frames"], 10)
        data = np.frombuffer(inspection.poses, dtype="<f4").reshape(2, 10, len(meta["names"]), 3)
        np.testing.assert_allclose(data[0], self.corrected.fk()[0], atol=1e-7)
        np.testing.assert_allclose(data[1], self.raw.fk()[0], atol=1e-7)
        self.assertGreater(np.max(np.abs(data[0]-data[1])), .05)
        self.assertIsNone(inspection.media)

    def test_old_run_without_raw_and_mismatched_comparison(self):
        path = self.root / "target_motion_raw.npz"
        path.unlink()
        self.assertEqual(prepare_inspection(self.root, transcode=False).metadata["variants"], ["corrected"])
        self.raw.fps = 60
        self.raw.save(path)
        with self.assertRaisesRegex(ValueError, "incompatible"):
            prepare_inspection(self.root, transcode=False)

    def test_source_resolution_for_cached_video_still_and_override(self):
        video = self.root / "original clip.webm"; video.write_bytes(b"test")
        image = self.root / "original still.png"; image.write_bytes(b"test")
        np.savez(self.root / "skeleton_motion.npz", video=str(video))
        (self.root / "run.json").write_text(json.dumps({"source":str(self.root / "skeleton_motion.npz")}), encoding="utf-8")
        self.assertEqual(resolve_source(self.root), video)
        np.savez(self.root / "skeleton_motion.npz", video=str(video), image=str(image))
        self.assertEqual(resolve_source(self.root), image)
        self.assertEqual(resolve_source(self.root, video), video)
        with self.assertRaises(FileNotFoundError):
            resolve_source(self.root, self.root / "missing.mp4")

    def test_http_media_seeking_and_closed_file_routes(self):
        media = self.root / "test.webm"; media.write_bytes(bytes(range(256))*8)
        server = make_server(prepare_inspection(self.root, media, transcode=False))
        worker = threading.Thread(target=server.serve_forever, daemon=True); worker.start()
        def cleanup():
            server.shutdown(); server.server_close(); worker.join(timeout=2)
        self.addCleanup(cleanup)
        connection = http.client.HTTPConnection("127.0.0.1", server.server_port, timeout=5)
        self.addCleanup(connection.close)
        def request(route, method="GET", headers=None):
            connection.request(method, route, headers=headers or {})
            response = connection.getresponse()
            return response.status, dict(response.getheaders()), response.read()
        status, headers, data = request("/api/meta")
        self.assertEqual(status, 200); self.assertEqual(json.loads(data)["frames"], 10)
        self.assertEqual(server.server_address[0], "127.0.0.1")
        for asset in ("/", "/viewer.js", "/style.css", "/api/poses.bin"):
            self.assertEqual(request(asset)[0], 200)
        for value, start, end in (("bytes=17-37",17,37),("bytes=2040-",2040,2047),
                                  ("bytes=-8",2040,2047),("bytes=2040-9999",2040,2047)):
            status, headers, data = request("/media", headers={"Range":value})
            self.assertEqual(status, 206)
            self.assertEqual(headers["Content-Range"], f"bytes {start}-{end}/2048")
            self.assertEqual(data, media.read_bytes()[start:end+1])
        status, headers, data = request("/media", "HEAD", {"Range":"bytes=100-199"})
        self.assertEqual((status,headers["Content-Length"],data),(206,"100",b""))
        for value in ("bytes=2048-", "bytes=10-2", "bytes=-0", "bytes=0-1,4-5", "bad"):
            self.assertEqual(request("/media", headers={"Range":value})[0], 416)
        self.assertEqual(request("/../target_motion.npz")[0], 404)
        self.assertEqual(request("/media?path=elsewhere")[2], media.read_bytes())

    @unittest.skipUnless(shutil.which("ffmpeg"), "ffmpeg not installed")
    def test_seekable_proxy_timing_and_cache(self):
        source = self.root / "unicode 测试.avi"
        subprocess.run([shutil.which("ffmpeg"), "-v", "error", "-f", "lavfi", "-i",
                        "testsrc2=size=96x160:rate=24", "-t", "1", "-c:v", "mpeg4", str(source)], check=True)
        proxy, kind = prepare_media(source, self.root)
        self.assertEqual(kind, "video"); self.assertEqual(proxy.suffix, ".mp4")
        import cv2
        cap = cv2.VideoCapture(str(proxy))
        try:
            self.assertAlmostEqual(cap.get(cv2.CAP_PROP_FPS),24,places=2)
            self.assertEqual(cap.get(cv2.CAP_PROP_FRAME_COUNT),24)
        finally:
            cap.release()
        modified = proxy.stat().st_mtime_ns
        self.assertEqual(prepare_media(source, self.root)[0], proxy)
        self.assertEqual(proxy.stat().st_mtime_ns, modified)

    def test_cli_source_alias_and_inspect_dispatch(self):
        parser = build_parser()
        args = parser.parse_args(["run", "clip.mp4", "--inspect", "--skeleton", "cached.npz"])
        self.assertTrue(args.inspect)
        with patch("video2vmd.cli.preflight",return_value=(Path("clip.mp4"),Path("cached.npz"),None)), \
             patch("video2vmd.mmd.pipeline.run_mmd_pipeline",return_value=self.root), \
             patch("video2vmd.cli.inspect_run", return_value=0) as inspect, redirect_stdout(StringIO()):
            self.assertEqual(run(args),0)
            inspect.assert_called_once_with(self.root,Path("clip.mp4"))
        args = parser.parse_args(["run","cached.npz","--video","other.mp4"])
        with redirect_stdout(StringIO()), self.assertRaises(SystemExit):
            preflight(args)
        args = parser.parse_args(["run", "cached.npz"])
        # Stop after input resolution: no licensed assets are needed for the alias.
        with patch("video2vmd.cli._check",return_value=True), \
             patch("video2vmd.cli._check_content",return_value=True), redirect_stdout(StringIO()):
            video, skeleton, image = preflight(args)
        self.assertEqual((video,skeleton,image),(None,Path("cached.npz"),None))
        with patch("video2vmd.inspection.server.serve_inspection") as serve:
            self.assertEqual(inspect_run(self.root,"clip.mp4",port=8765,open_browser=False),0)
            serve.assert_called_once_with(self.root,Path("clip.mp4"),port=8765,open_browser=False)


if __name__ == "__main__":
    unittest.main(verbosity=2)
