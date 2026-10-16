# Release checklist

- Keep package and Blender manifest versions aligned (current fork: 0.2.0).
- Run the MMD, video-input and Blender-extension verification scripts.
- Export VMD/BVH from a cached skeleton through `v2vmd`; inspect raw/corrected
  files, quality reports and preview. Validate independently in Blender.
- Build a Python wheel and confirm it contains only the `video2vmd` package and
  the `v2vmd` console entry point; no legacy exporter or gated assets.
- Build `dist/video2vmd-0.2.0.zip` and run Blender's extension validator.
- Verify install/upgrade in an external Python environment and the Blender UI.
- Confirm the pinned vmd2png commit is published and installable from GitHub.
- Review README, migration notes, CHANGELOG, LICENSE and NOTICE. Older release
  notes / translated upstream READMEs are explicitly historical.
- Ensure models, weights, videos, temporary worktrees and generated outputs are
  excluded from commits and distribution packages.
