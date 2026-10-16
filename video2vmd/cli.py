"""The `v2vmd` CLI: video, still or cached human skeleton to VMD / BVH."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from . import checks, paths
from .progress import ProgressWriter
from .run_report import write_failed_report

OK = "[ok]     "
MISS = "[missing]"
BAD = "[invalid]"

SMPLX_HINT = (
    "register at https://smpl-x.is.tue.mpg.de/ , download SMPLX_NEUTRAL.npz, "
    "place it at assets/body_models/smplx/SMPLX_NEUTRAL.npz"
)
VIDEO_HINT = (
    "put a *single-person* motion video (.mp4/.mov/...) into assets/video/ "
    "— group shots are rejected before extraction"
)
IMAGE_HINT = (
    "put a *single-person* still (.jpg/.png/...) into assets/image/ "
    "— group photos are rejected before extraction"
)
SOURCE_HINT = VIDEO_HINT + "; or " + IMAGE_HINT


def _check(label: str, present: bool, hint: str) -> bool:
    print(f"{OK if present else MISS} {label}")
    if not present:
        print(f"          -> {hint}")
    return present


def _check_content(label: str, path: Path, checker) -> bool:
    """Existence is assumed; `checker` inspects the file and returns a hint or None."""
    problem = checker(path)
    print(f"{OK if problem is None else BAD} {label}")
    if problem is not None:
        print(f"          -> {path.name} {problem}")
    return problem is None


def _module_available(name: str) -> bool:
    import importlib.util

    return importlib.util.find_spec(name) is not None


def doctor(skeleton: str | None = None) -> int:
    """Check every user-supplied asset and dependency; say where to get what's missing."""
    print(f"project root: {paths.PROJECT_ROOT}\n")

    ok = True
    if skeleton:
        cached = Path(skeleton)
        ok &= _check(f"skeleton npz       {cached}", cached.is_file(), "check the --skeleton path")
        if cached.is_file():
            ok &= _check_content("skeleton loads", cached, checks.check_skeleton_npz)
    else:
        smplx_label = f"SMPL-X body model  {paths.SMPLX_NEUTRAL}"
        if paths.SMPLX_NEUTRAL.exists():
            ok &= _check_content(smplx_label, paths.SMPLX_NEUTRAL, checks.check_smplx)
        else:
            ok &= _check(smplx_label, False, SMPLX_HINT)
        videos, images = paths.list_videos(), paths.list_images()
        ok &= _check(f"input source       assets/video/ + assets/image/  ({len(videos)} videos; {len(images)} images)",
                     bool(videos) or bool(images), SOURCE_HINT)
        for video in videos:
            ok &= _check_content(f"  - {video.name}", video, checks.check_video)
        for image in images:
            ok &= _check_content(f"  - {image.name}", image, checks.check_image)

    print()
    deps_ok = True
    modules = ("numpy", "scipy", "cv2") if skeleton else ("numpy", "scipy", "cv2", "torch", "gvhmr")
    for module in modules:
        deps_ok &= _check(f"python module      {module}", _module_available(module), "pip install -e .")
    try:
        from .mmd.profile import require_codec
        require_codec()
        deps_ok &= _check("VMD codec          vmd2png", True, "")
    except ImportError as exc:
        deps_ok &= _check("VMD codec          vmd2png", False, str(exc))
    import shutil as _shutil

    _check(
        "ffmpeg (optional)  H.264 skeleton previews and input normalization",
        _shutil.which("ffmpeg") is not None,
        "add ffmpeg to PATH — needed to normalize some inputs; otherwise previews are silent mp4v",
    )

    weights_note = (
        "downloaded automatically into weights/ on the first `v2vmd run` (~5 GB)"
    )
    print(f"\ninference weights: {weights_note}")

    if ok and deps_ok:
        print("\neverything in place — try: v2vmd run")
        return 0
    print("\nfix the [missing] items above, then re-run: v2vmd doctor")
    return 1


def preflight(args: argparse.Namespace) -> tuple[Path | None, Path | None, Path | None]:
    """Resolve run inputs, checking everything doctor checks for this run.

    All problems are collected and reported together (same format as doctor)
    instead of failing on the first one. Exits with code 1 if anything is
    missing. Returns (video, skeleton, image).
    """
    problems = 0

    def check(label: str, present: bool, hint: str) -> bool:
        nonlocal problems
        if not _check(label, present, hint):
            problems += 1
        return present

    def check_content(label: str, path: Path | None, checker) -> None:
        """Inspect the file's contents, but only when it exists at all."""
        nonlocal problems
        if path is not None and path.exists() and not _check_content(label, path, checker):
            problems += 1

    video: Path | None = None
    image: Path | None = None
    output_format = args.format
    if getattr(args, "floor", None) is not None:
        import math
        check("floor height", math.isfinite(args.floor), "--floor must be a finite number in meters")
    skeleton_only = bool(args.skeleton) and not args.video and args.image is None
    want_image = getattr(args, "image", None) is not None
    want_video = bool(getattr(args, "video", None))

    if skeleton_only:
        pass  # Motion-only export needs no source video or SMPL-X model.
    elif want_image and want_video:
        check(
            "input source       --video / --image",
            False,
            "pass only one of --video or --image",
        )
    elif want_image:
        if args.image:
            image = Path(args.image)
            check(f"input image        {image}", image.exists(), "check the --image path")
        else:
            image = paths.latest_image()
            check(
                "input image        assets/image/"
                + (f"  (latest: {image.name})" if image else "  (0 found)"),
                image is not None,
                IMAGE_HINT + ", or pass --image PATH",
            )
        if image is not None and image.exists() and not paths.is_image(image):
            check(
                f"image suffix       {image.name}",
                False,
                "use a still (.jpg/.png/webp/bmp), or pass a clip with --video",
            )
        check_content(f"image is usable    {image.name if image else ''}", image, checks.check_image)
    elif want_video:
        video = Path(args.video)
        if video.exists() and paths.is_image(video):
            image = video
            video = None
            print(f"{OK} input image        {image}  (detected from --video path)")
            check_content(f"image is usable    {image.name}", image, checks.check_image)
        else:
            check(f"input video        {video}", video.exists(), "check the --video path")
            check_content(f"video is usable    {video.name}", video, checks.check_video)
    else:
        video = paths.latest_video()
        if video is None:
            image = paths.latest_image()
            check(
                "input image        assets/image/"
                + (f"  (latest: {image.name})" if image else "  (0 found)"),
                image is not None,
                SOURCE_HINT + ", or pass --video / --image",
            )
            check_content(
                f"image is usable    {image.name if image else ''}", image, checks.check_image
            )
        else:
            check(
                "input video        assets/video/"
                + f"  (latest: {video.name})",
                True,
                VIDEO_HINT + ", or pass --video",
            )
            check_content(f"video is usable    {video.name}", video, checks.check_video)

    try:
        from .mmd.profile import require_codec
        require_codec()
    except ImportError as exc:
        check("MMD dependencies", False, str(exc))
    if args.model:
        check(f"PMX model          {args.model}", Path(args.model).is_file() and Path(args.model).suffix.lower() == ".pmx",
              "use a readable .pmx file (extract ZIP archives first)")
    if args.ik_mode == "ik":
        check("IK target profile", output_format == "vmd" and bool(args.model),
              "--ik-mode ik requires --format vmd and --model character.pmx")

    # Skeleton reuse skips extraction; only then is SMPL-X not needed.
    skeleton = Path(args.skeleton) if args.skeleton else None
    if skeleton is not None:
        check(
            f"skeleton npz       {skeleton}",
            skeleton.exists(),
            "check the --skeleton path (a previous run's skeleton_motion.npz)",
        )
        check_content(f"skeleton loads     {skeleton.name}", skeleton, checks.check_skeleton_npz)
    else:
        check(
            f"SMPL-X body model  {paths.SMPLX_NEUTRAL.relative_to(paths.PROJECT_ROOT)}",
            paths.SMPLX_NEUTRAL.exists(),
            SMPLX_HINT,
        )
        check_content(
            "SMPL-X loads       SMPLX_NEUTRAL.npz", paths.SMPLX_NEUTRAL, checks.check_smplx
        )

    if problems:
        print(f"\n{problems} problem(s) — nothing was run. `v2vmd doctor` checks everything.")
        sys.exit(1)
    return video, skeleton, image


def run(args: argparse.Namespace) -> int:
    progress = ProgressWriter(getattr(args, "progress_jsonl", None))
    output_dir = Path(args.output_dir).resolve() if getattr(args, "output_dir", None) else None
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)

    progress.emit("preflight", 0.05, "Checking inputs")
    try:
        video, skeleton, image = preflight(args)
    except SystemExit as exc:
        code = exc.code if isinstance(exc.code, int) else (0 if exc.code in (0, None) else 1)
        if code != 0:
            progress.emit("failed", 0.05, "Input check failed")
            write_failed_report(
                output_dir,
                stage="preflight",
                error_code="PREFLIGHT_FAILED",
                message="Input check failed",
            )
        raise

    print()

    try:
        from .mmd.pipeline import run_mmd_pipeline
        run_dir = run_mmd_pipeline(video, image=image, skeleton_npz=skeleton, output_dir=output_dir,
                                  device=args.device, output_format=args.format, model=args.model,
                                  foot_lock=args.foot_lock, floor=args.floor, motion_encoder=args.motion_encoder,
                                  no_preview=args.no_preview, progress=progress, ik_mode=args.ik_mode)
    except Exception:
        import traceback

        traceback.print_exc()
        return 1
    print(f"\nresults: {run_dir}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="v2vmd",
        description=(
            "Extract human body motion from video or still images and export VMD or BVH animation."
        ),
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_doctor = sub.add_parser("doctor", help="check assets, weights and dependencies")
    p_doctor.add_argument("--skeleton", help="check cached-skeleton export without source video or SMPL-X")

    p_run = sub.add_parser("run", help="video, still or cached skeleton -> VMD/BVH, results in outputs/")
    p_run.add_argument("--format", choices=["vmd", "bvh"], default="vmd", help="output format (default: vmd)")
    p_run.add_argument("--mmd-profile", choices=["standard"], default="standard", help="MMD reference skeleton")
    p_run.add_argument("--model", help="PMX target skeleton/proportions for VMD/BVH (default: standard profile)")
    p_run.add_argument("--foot-lock", action="store_true", help="bake temporal foot-contact correction into VMD/BVH")
    p_run.add_argument("--ik-mode", choices=["fk", "ik"], default="fk", help="VMD baked FK or PMX foot-IK controls (default: fk)")
    p_run.add_argument("--floor", type=float, help="target floor height in meters (default: estimated)")
    p_run.add_argument("--motion-encoder", action="store_true", help="also write canonical 404-column motion_encoder.npy")
    p_run.add_argument(
        "--video",
        help="input video (default: the file most recently added to assets/video/)",
    )
    p_run.add_argument(
        "--image",
        nargs="?",
        const="",
        default=None,
        help=(
            "still photo of one person (default: the file most recently added "
            "to assets/image/). Recovers a static 3D pose; do not combine with --video"
        ),
    )
    p_run.add_argument(
        "--skeleton",
        help="reuse a previous run's skeleton_motion.npz and skip extraction",
    )
    p_run.add_argument(
        "--device",
        default="cpu",
        choices=["cpu", "cuda", "mps"],
        help="inference device for the extraction step (default: cpu)",
    )
    p_run.add_argument(
        "--output-dir",
        help="write this run's files into PATH instead of a timestamped outputs/ folder",
    )
    p_run.add_argument(
        "--progress-jsonl",
        help="append machine-readable JSONL progress events to PATH",
    )
    p_run.add_argument(
        "--no-preview",
        action="store_true",
        help="skip preview videos/images; still write motion files, NPZ and run.json",
    )
    return parser


def main() -> None:
    # Windows pipes can default to cp1252 even when the input paths contain
    # Unicode. The Blender job log and progress files use UTF-8 as well.
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    parser = build_parser()
    args = parser.parse_args()
    if args.command == "doctor":
        sys.exit(doctor(args.skeleton))
    sys.exit(run(args))


if __name__ == "__main__":
    main()
