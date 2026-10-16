"""Per-scene UI and job state."""

from __future__ import annotations

import bpy
from bpy.props import BoolProperty, EnumProperty, FloatProperty, StringProperty
from bpy.types import PropertyGroup, Scene

from .registration import safe_register_class, safe_unregister_class


_SOURCE_TYPE_ITEMS = ()
_DEVICE_ITEMS = ()


def _source_type_items(self, context):
    from .i18n import t

    global _SOURCE_TYPE_ITEMS
    _SOURCE_TYPE_ITEMS = (
        ("VIDEO", t("source_video"), t("source_video_desc")),
        ("IMAGE", t("source_image"), t("source_image_desc")),
        ("SKELETON", t("source_skeleton"), t("source_skeleton_desc")),
    )
    return _SOURCE_TYPE_ITEMS


def _device_items(self, context):
    from .i18n import t

    global _DEVICE_ITEMS
    _DEVICE_ITEMS = (
        ("cpu", "CPU", t("device_cpu_desc")),
        ("cuda", "CUDA", t("device_cuda_desc")),
        ("mps", "MPS", t("device_mps_desc")),
    )
    return _DEVICE_ITEMS


class V2VMDProperties(PropertyGroup):
    source_type: EnumProperty(
        name="Source Type",
        items=_source_type_items,
    )
    source_path: StringProperty(
        name="Source File",
        subtype="FILE_PATH",
        maxlen=2048,
        default="",
    )
    model_path: StringProperty(
        name="PMX Target (optional)",
        subtype="FILE_PATH",
        maxlen=2048,
        default="",
    )
    output_format: EnumProperty(name="Output Format", items=(("vmd", "VMD", ""), ("bvh", "BVH", "")), default="vmd")
    foot_lock: BoolProperty(name="Foot Contact Cleanup", default=False)
    depth_cleanup: BoolProperty(name="Root Depth Cleanup",default=False,
        description="Reduce root-depth jitter using saved static-camera observations; preserves raw exports")
    ik_mode: EnumProperty(name="VMD Leg Mode", items=(("fk", "Baked FK", ""), ("ik", "PMX Foot IK", "")), default="fk")
    device: EnumProperty(
        name="Device",
        items=_device_items,
    )
    generate_preview: BoolProperty(
        name="Generate Preview Videos",
        description="Also render preview videos/images (slower)",
        default=False,
    )

    env_status: EnumProperty(
        name="Environment",
        items=(
            ("NOT_CHECKED", "Not checked", ""),
            ("CHECKING", "Checking", ""),
            ("READY", "Ready", ""),
            ("ERROR", "Error", ""),
        ),
        default="NOT_CHECKED",
    )
    env_message: StringProperty(name="Environment message", default="", maxlen=512)

    job_status: StringProperty(name="Job status", default="idle", maxlen=64)
    job_stage: StringProperty(name="Job stage", default="", maxlen=64)
    job_progress: FloatProperty(name="Progress", default=0.0, min=0.0, max=1.0)
    job_error: StringProperty(name="Job error", default="", maxlen=512)
    job_motion_path: StringProperty(name="Motion path", default="", maxlen=2048, subtype="FILE_PATH")
    job_output_dir: StringProperty(name="Output folder", default="", maxlen=2048, subtype="DIR_PATH")
    job_log_path: StringProperty(name="Log path", default="", maxlen=2048, subtype="FILE_PATH")
    job_id: StringProperty(name="Job id", default="", maxlen=128)
    job_source_name: StringProperty(name="Source name", default="", maxlen=128)
    import_note: StringProperty(name="Import note", default="", maxlen=512)


def register() -> None:
    safe_register_class(V2VMDProperties)
    Scene.v2vmd = bpy.props.PointerProperty(type=V2VMDProperties)


def unregister() -> None:
    if hasattr(Scene, "v2vmd"):
        del Scene.v2vmd
    safe_unregister_class(V2VMDProperties)
