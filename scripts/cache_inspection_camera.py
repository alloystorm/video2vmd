"""Add camera observations to an old run from its trusted local GVHMR cache.

No detector or inference rerun. Requires the extraction environment and SMPL-X.
Example: python scripts/cache_inspection_camera.py outputs/RUN --prediction
outputs/ORIGINAL/cache/CLIP/hmr4d_results.pt --device cuda
"""
import argparse
from pathlib import Path
from types import SimpleNamespace
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from video2vmd.paths import export_gvhmr_env
from video2vmd.extract import inspection_camera_payload


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--prediction", type=Path, required=True, help="trusted local hmr4d_results.pt")
    parser.add_argument("--device", choices=["cpu","cuda","mps"], default="cpu")
    args = parser.parse_args()
    export_gvhmr_env()
    import torch
    prediction = torch.load(args.prediction, map_location="cpu", weights_only=True)
    skeleton = args.run_dir / "skeleton_motion.npz"
    with np.load(skeleton, allow_pickle=False) as data:
        payload = {key: data[key] for key in data.files}
    expected = prediction["smpl_params_global"]
    for key, prediction_key in (("root_translation","transl"),("root_orientation","global_orient")):
        actual = expected[prediction_key].detach().cpu().numpy()
        if payload[key].shape != actual.shape or not np.allclose(payload[key],actual,atol=1e-5):
            parser.error("prediction does not match this skeleton's root transforms")
    result = SimpleNamespace(smpl_params_world=expected, smpl_params_camera=prediction["smpl_params_incam"], intrinsics=prediction["K_fullimg"])
    payload.update(inspection_camera_payload(result, args.prediction.parent/"0_input_video.mp4",args.device))
    pending = skeleton.with_name("skeleton_motion.camera.pending.npz")
    np.savez_compressed(pending, **payload)
    pending.replace(skeleton)
    print(f"Saved camera observations: {skeleton}")


if __name__ == "__main__":
    main()
