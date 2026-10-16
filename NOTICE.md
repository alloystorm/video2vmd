# NOTICE

This project's own code is released under the license in `LICENSE`. It builds
on the following upstream components, each governed by its own terms. None of
their gated assets are redistributed here; users obtain them directly from the
sources below.

## GVHMR

World-Grounded Human Motion Recovery via Gravity-View Coordinates
(Shen et al., SIGGRAPH Asia 2024). Installed as the `gvhmr` Python package.
GVHMR is released for **non-commercial research purposes only** — see its
license before any commercial use.
https://github.com/zju3dv/GVHMR

## SMPL-X body model

The SMPL-X body model is registration-gated and licensed by the Max Planck
Institute for Intelligent Systems for non-commercial scientific research.
Users must register and download `SMPLX_NEUTRAL.npz` themselves.
https://smpl-x.is.tue.mpg.de/

## Project origins and shared VMD codec

Video2VMD derives from Max Liu's Motion2MixamoRig; original copyright and license
notices are retained. The legacy character-export implementation has been removed.
https://github.com/MaxLiu22/Motion2MixamoRig

The shared vmd2png codec/skeleton package is MIT-licensed. It is installed as a
pinned dependency, not copied into this repository.
https://github.com/alloystorm/vmd2png

The local motion inspector's canvas projection, orbit/zoom interactions and
precomputed pose-buffer approach are adapted from the user's motion_encoder
project (`inspect_data.py`). This viewer uses Video2VMD's saved target hierarchy
and FK, with a shared video clock, rather than importing its training stack.

Foot cleanup draws on the user's DanceXR implementation in
`dxr_native/rig/src/FeetSolver.cpp`: separate heel/toe clearance, bounded residual
pitch, up-only swing correction and a horizontal anti-slide limit. Video2VMD
implements these in its offline target-space solver with inferred contacts and
exact rigid rotation/reach calculations; it does not import the native runtime.

## Inference weights

The extraction step auto-downloads model checkpoints (GVHMR, HMR2, RTMPose /
ViTPose 2D pose, YOLOv8 person detector) into `weights/`. Note that YOLOv8
(Ultralytics) is licensed under **AGPL-3.0**, and the pose/HMR checkpoints
carry their respective research licenses.

## Summary

The end-to-end pipeline is effectively restricted to non-commercial /
research use by its upstream components, regardless of this repository's own
code license.
