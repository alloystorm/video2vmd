"""Video2VMD Blender Extension: UI + external `v2vmd` process control."""

from __future__ import annotations

bl_info = {
    "name": "Video2VMD",
    "author": "Max Liu",
    "version": (0, 2, 0),
    "blender": (4, 2, 0),
    "location": "View3D > Sidebar > Video2VMD",
    "description": "Extract and export VMD or BVH body motion",
    "category": "Animation",
}

from . import operators, panels, preferences, properties


def register() -> None:
    try:
        unregister()
    except Exception:
        pass
    preferences.register()
    properties.register()
    operators.register()
    panels.register()


def unregister() -> None:
    try:
        panels.unregister()
    except Exception:
        pass
    try:
        operators.unregister()
    except Exception:
        pass
    try:
        properties.unregister()
    except Exception:
        pass
    try:
        preferences.unregister()
    except Exception:
        pass
