# User assets

User-supplied models and media are gitignored. Run `v2vmd doctor` to check the
extraction environment, or `v2vmd doctor --skeleton previous/skeleton_motion.npz`
for cached-skeleton export.

| Path | Contents |
| --- | --- |
| `body_models/smplx/SMPLX_NEUTRAL.npz` | Licensed SMPL-X model from https://smpl-x.is.tue.mpg.de/; needed only for extraction |
| `body_models/mano/MANO_RIGHT.pkl` | Optional licensed MANO model for `--hands`; see `docs/hand-pose.md` for the WiLoR runtime and weights |
| `mmd/<character>.pmx` | Optional MMD target proportions; extract archives first and respect the model's license |
| `video/<clip>.mp4` | Single-person video; `.mov`, `.avi`, `.mkv`, `.webm` also accepted |
| `image/<pose>.jpg` | Single-person still; PNG/WebP/BMP also accepted |

Without an explicit input, `v2vmd run` picks the newest video, falling back to the
newest image. The default target is the shared standard MMD skeleton. Reuse
`--skeleton` to skip extraction; no source video or SMPL-X model is then needed.
