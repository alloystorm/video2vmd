"""Offline inspection server; canvas interaction adapted from motion_encoder/inspect_data.py.

Use the saved target transforms and this project's FK, rather than the training
viewer's approximate MotionFK. Only explicitly selected media is served.
"""
from __future__ import annotations

import json
import mimetypes
import re
import shutil
import subprocess
import webbrowser
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit

import numpy as np

from .. import paths
from ..mmd.motion import TargetMotion
from ..mmd.profile import MMDProfile

WEB = Path(__file__).with_name("web")


def load_target(path: Path) -> TargetMotion:
    """Reconstruct the exported hierarchy without needing the original PMX."""
    with np.load(path, allow_pickle=False) as data:
        meta = json.loads(str(data["metadata"]))
        profile = MMDProfile(meta["profile"], tuple(str(n) for n in data["bone_names"]),
                             data["parents"], data["rest_positions_m"], {},
                             float(meta["meters_per_vmd_unit"]))
        return TargetMotion(profile, data["quat_xyzw"], data["translations_m"], float(data["fps"]))


def resolve_source(run_dir: Path, override: Path | None = None) -> Path | None:
    if override is not None:
        source = Path(override).resolve()
        if not source.is_file():
            raise FileNotFoundError(f"Inspection source not found: {source}")
        return source
    candidates = []
    report = run_dir / "run.json"
    if report.is_file():
        candidates.append(json.loads(report.read_text(encoding="utf-8")).get("source", ""))
    skeleton = run_dir / "skeleton_motion.npz"
    if skeleton.is_file():
        with np.load(skeleton, allow_pickle=False) as data:
            # Stills may also have a temporary, now-deleted hold video.
            candidates.extend(str(data[key]) for key in ("image", "video") if key in data)
    for candidate in candidates:
        if not candidate:
            continue
        source = Path(candidate)
        for resolved in (source, run_dir / source, paths.PROJECT_ROOT / source):
            if resolved.is_file() and resolved.suffix.lower() != ".npz":
                return resolved.resolve()
    return None


def prepare_media(source: Path | None, run_dir: Path) -> tuple[Path | None, str | None]:
    if source is None:
        return None, None
    if paths.is_image(source):
        return source, "image"
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        print("Inspection: ffmpeg unavailable; serving the original video (browser codec support required).")
        return source, "video"
    cache = run_dir / "inspection"
    cache.mkdir(exist_ok=True)
    proxy, stamp = cache / "source.mp4", cache / "source.json"
    stat = source.stat()
    identity = {"source": str(source), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns, "version": 1}
    if proxy.is_file() and stamp.is_file() and json.loads(stamp.read_text(encoding="utf-8")) == identity:
        return proxy, "video"
    print("Inspection: preparing a seekable H.264 video preview…", flush=True)
    pending = cache / "source.pending.mp4"
    result = subprocess.run([
        ffmpeg, "-hide_banner", "-loglevel", "error", "-y", "-i", str(source),
        "-map", "0:v:0", "-map", "0:a:0?", "-vf",
        "scale=w='min(1280,iw)':h='min(1280,ih)':force_original_aspect_ratio=decrease:force_divisible_by=2,setsar=1",
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "23", "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-b:a", "128k", "-movflags", "+faststart", str(pending),
    ], capture_output=True)
    if result.returncode:
        pending.unlink(missing_ok=True)
        print("Inspection: preview encoding failed; serving the original video.")
        return source, "video"
    pending.replace(proxy)
    stamp.write_text(json.dumps(identity, ensure_ascii=False), encoding="utf-8")
    return proxy, "video"


@dataclass
class Inspection:
    metadata: dict
    poses: bytes
    media: Path | None
    camera_poses: bytes | None = None
    reconstruction: bytes | None = None
    world_reconstruction: bytes | None = None


def prepare_inspection(run_dir: Path, source: Path | None = None, *, transcode=True) -> Inspection:
    run_dir = Path(run_dir).resolve()
    corrected = load_target(run_dir / "target_motion.npz")
    motions = {"corrected": corrected}
    raw_path = run_dir / "target_motion_raw.npz"
    if raw_path.is_file():
        raw = load_target(raw_path)
        if (raw.profile.names != corrected.profile.names or raw.fps != corrected.fps
                or raw.n_frames != corrected.n_frames
                or not np.array_equal(raw.profile.parents, corrected.profile.parents)):
            raise ValueError("Raw and corrected target motion have incompatible timelines/hierarchies")
        motions["raw"] = raw
    source = resolve_source(run_dir, source)
    media, kind = (prepare_media(source, run_dir) if transcode else
                   (source, "image" if source and paths.is_image(source) else "video" if source else None))
    profile = corrected.profile
    body = {profile.root, "Center", "HipMaster", "Torso", "Torso1", "Torso2", "Neck", "Head"}
    body.update(side + part for side in ("Left", "Right") for part in
                ("Shoulder", "Arm", "Elbow", "Wrist", "Leg", "Knee", "Ankle", "Toe"))
    visible = set()
    for name in body & profile.index.keys():
        index = profile.index[name]
        while index >= 0:
            visible.add(index)
            index = int(profile.parents[index])
    positions = [motion.fk()[0].astype("<f4") for motion in motions.values()]
    floor = float(np.percentile(positions[0][:, [profile.index[s+"Toe"] for s in ("Left", "Right")], 1], 5))
    # Contiguous little-endian float32: variant, frame, bone, xyz.
    metadata = {
        "title": run_dir.name, "frames": corrected.n_frames, "fps": corrected.fps,
        "duration": corrected.n_frames / corrected.fps,
        "names": profile.names, "parents": profile.parents.tolist(),
        "root": profile.index[profile.root], "body": sorted(visible), "floor": floor,
        # Hair/physics helpers can extend far beyond the character's body.
        "height": max(.5, float(np.ptp(profile.rest[[profile.index[n] for n in body & profile.index.keys()], 1])) + .18),
        "variants": list(motions), "source": source.name if source else None, "mediaKind": kind,
        "format": "float32-le / variant,frame,bone,xyz", "up": "+Y", "units": "meters",
    }
    from .camera import camera_inspection, world_reconstruction
    skeleton = run_dir / "skeleton_motion.npz"
    camera = None
    reconstruction = None
    if skeleton.is_file():
        try:
            reconstruction = world_reconstruction(skeleton,corrected.n_frames,corrected.fps)
        except (ValueError,KeyError) as exc:
            print(f"Inspection: source reconstruction unavailable ({exc}).")
        try:
            camera = camera_inspection(skeleton, motions)
        except (ValueError, KeyError) as exc:
            print(f"Inspection: estimated camera unavailable ({exc}); manual alignment is available.")
    metadata["camera"] = camera.metadata if camera else None
    metadata["reconstruction"] = reconstruction[0] if reconstruction else None
    return Inspection(metadata, b"".join(p.tobytes() for p in positions), media,
                      camera.targets if camera else None, camera.reconstruction if camera else None,
                      reconstruction[1] if reconstruction else None)


def byte_range(value: str, size: int) -> tuple[int, int]:
    """One HTTP byte range, including open-ended and suffix forms."""
    match = re.fullmatch(r"bytes=(\d*)-(\d*)", value)
    if match is None or size <= 0 or not any(match.groups()):
        raise ValueError("Invalid range")
    first, last = match.groups()
    if not first:
        suffix = int(last)
        if suffix <= 0:
            raise ValueError("Invalid suffix range")
        return max(0, size - suffix), size - 1
    start, end = int(first), min(int(last), size - 1) if last else size - 1
    if start >= size or start > end:
        raise ValueError("Unsatisfiable range")
    return start, end


def make_server(inspection: Inspection, port=0) -> ThreadingHTTPServer:
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def send_headers(self, status, mime, length, extra=()):
            self.send_response(status)
            self.send_header("Content-Type", mime)
            self.send_header("Content-Length", str(length))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Content-Security-Policy", "default-src 'self'; style-src 'self'; script-src 'self'; connect-src 'self'; media-src 'self'; img-src 'self'; frame-ancestors 'none'")
            for key, value in extra:
                self.send_header(key, value)
            self.end_headers()

        def do_HEAD(self):
            self.do_GET()

        def do_GET(self):
            route = urlsplit(self.path).path
            if route == "/media" and inspection.media is not None:
                media = inspection.media
                size = media.stat().st_size
                status, start, end = 200, 0, size-1
                extra = [("Accept-Ranges", "bytes")]
                if "Range" in self.headers:
                    try:
                        start, end = byte_range(self.headers["Range"], size)
                    except ValueError:
                        self.send_headers(416, "text/plain", 0, [("Content-Range", f"bytes */{size}")])
                        return
                    status = 206
                    extra.append(("Content-Range", f"bytes {start}-{end}/{size}"))
                self.send_headers(status, mimetypes.guess_type(media.name)[0] or "application/octet-stream", end-start+1, extra)
                if self.command == "HEAD":
                    return
                try:
                    with media.open("rb") as stream:
                        stream.seek(start)
                        remaining = end-start+1
                        while remaining > 0:
                            chunk = stream.read(min(256*1024, remaining))
                            if not chunk:
                                break
                            self.wfile.write(chunk)
                            remaining -= len(chunk)
                except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
                    pass  # Normal when the user scrubs or closes the tab.
                return
            assets = {"/": ("index.html", "text/html; charset=utf-8"),
                      "/viewer.js": ("viewer.js", "text/javascript; charset=utf-8"),
                      "/projection.js": ("projection.js", "text/javascript; charset=utf-8"),
                      "/style.css": ("style.css", "text/css; charset=utf-8")}
            if route in assets:
                name, mime = assets[route]
                data = (WEB / name).read_bytes()
            elif route == "/api/meta":
                data, mime = json.dumps(inspection.metadata, ensure_ascii=False).encode(), "application/json; charset=utf-8"
            elif route == "/api/poses.bin":
                data, mime = inspection.poses, "application/octet-stream"
            elif route == "/api/camera-poses.bin" and inspection.camera_poses is not None:
                data, mime = inspection.camera_poses, "application/octet-stream"
            elif route == "/api/reconstruction.bin" and inspection.reconstruction is not None:
                data, mime = inspection.reconstruction, "application/octet-stream"
            elif route == "/api/world-reconstruction.bin" and inspection.world_reconstruction is not None:
                data, mime = inspection.world_reconstruction, "application/octet-stream"
            else:
                self.send_error(404)
                return
            self.send_headers(200, mime, len(data))
            if self.command != "HEAD":
                self.wfile.write(data)

    # No external binding: source files stay on the user's own computer.
    return ThreadingHTTPServer(("127.0.0.1", port), Handler)


def serve_inspection(run_dir: Path, source: Path | None = None, *, port=0, open_browser=True):
    inspection = prepare_inspection(run_dir, source)
    server = make_server(inspection, port)
    url = f"http://127.0.0.1:{server.server_port}/"
    print(f"\nMotion inspector: {url}\nKeep this command running; press Ctrl+C to close the inspector.", flush=True)
    if open_browser:
        webbrowser.open(url)
    try:
        server.serve_forever(poll_interval=.25)
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
