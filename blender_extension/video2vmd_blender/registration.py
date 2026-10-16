"""Reload-safe class registration.

Installing a zip over an enabled extension can import a new class object
while Blender still holds the previous one under the same type name.
"""

from __future__ import annotations

import bpy


def registered_class(cls):
    # AddonPreferences are not always exposed on bpy.types by Python name.
    # Checking the class itself is essential for disable/re-enable cycles.
    if getattr(cls, "is_registered", False):
        return cls
    for base in cls.__mro__[1:]:
        lookup = getattr(base, "bl_rna_get_subclass_py", None)
        if lookup is not None:
            existing = lookup(cls.__name__, None)
            if existing is not None:
                return existing
    return getattr(bpy.types, cls.__name__, None)


def safe_register_class(cls) -> None:
    existing = registered_class(cls)
    if existing is not None:
        try:
            bpy.utils.unregister_class(existing)
        except Exception:
            pass
    bpy.utils.register_class(cls)


def safe_unregister_class(cls) -> None:
    existing = registered_class(cls)
    if existing is None:
        return
    try:
        bpy.utils.unregister_class(existing)
    except Exception:
        pass
