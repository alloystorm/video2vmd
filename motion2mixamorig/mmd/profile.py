"""MMD reference/PMX skeletons in meters, with explicit VMD name ownership."""
from dataclasses import dataclass
from pathlib import Path
import hashlib
import json

import numpy as np

TESTED_CODEC_REVISION = "3c8837e005456b7081bd1a1366aaa755eb0d0c5d"


def require_codec():
    try:
        from vmd2png import vmd
        from vmd2png.pmx import read_pmx_skeleton
    except ImportError as exc:
        raise ImportError("VMD support requires the mmd extra: pip install -e '.[mmd]'") from exc
    if not hasattr(vmd, "ik_enabled"):
        raise ImportError("Update vmd2png to the revision pinned by the mmd extra (IK-state support required)")
    return vmd


@dataclass
class MMDProfile:
    name: str
    names: tuple[str, ...]
    parents: np.ndarray
    rest: np.ndarray
    serialized: dict[str, str]
    unit: float = 0.085
    source: str | None = None

    def __post_init__(self):
        if len(set(self.names)) != len(self.names):
            raise ValueError("Target bone names are not unique")
        self.index = {name: i for i, name in enumerate(self.names)}
        self.parents = np.asarray(self.parents, dtype=np.int64)
        self.rest = np.asarray(self.rest, dtype=np.float64)
        if self.parents.shape != (len(self.names),) or not np.isfinite(self.unit) or self.unit <= 0:
            raise ValueError("Invalid target hierarchy or meters-per-VMD-unit scale")
        if self.rest.shape != (len(self.names), 3) or not np.isfinite(self.rest).all():
            raise ValueError("Invalid target rest positions")
        for i, parent in enumerate(self.parents):
            if parent < -1 or parent >= i:
                raise ValueError("Target hierarchy must be parent-before-child")
        self.root = "Master" if "Master" in self.index else "Center"
        required = [self.root, "HipMaster", "Torso", "Torso2", "Neck", "Head"]
        required += [side + part for side in ("Left", "Right")
                     for part in ("Shoulder", "Arm", "Elbow", "Wrist", "Leg", "Knee", "Ankle", "Toe")]
        missing = set(required) - self.index.keys()
        if missing:
            raise ValueError(f"Unsupported MMD skeleton; missing semantic bones: {sorted(missing)}")
        for name in required:
            i = self.index[name]
            while i >= 0 and i != self.index[self.root]:
                i = self.parents[i]
            if i < 0:
                raise ValueError(f"Unsupported MMD hierarchy: {name} is outside the {self.root} root tree")
        self.offsets = self.rest.copy()
        for i, parent in enumerate(self.parents):
            if parent >= 0:
                self.offsets[i] -= self.rest[parent]
        details = json.dumps({"names": self.names, "serialized": self.serialized, "unit": self.unit}, sort_keys=True).encode()
        self.identity = hashlib.sha256(self.rest.tobytes() + self.parents.tobytes() + details).hexdigest()

    def position(self, name):
        return self.rest[self.index[name]]


def standard_profile():
    vmd = require_codec()
    from vmd2png.skeleton import build_standard_skeleton
    root, _ = build_standard_skeleton()
    bones = root.export_bones()
    index = {b.name: i for i, b in enumerate(bones)}
    # Optional synthetic controls are not part of the 65-bone learning contract.
    mapping = {b.name: vmd.bone_name_reverse_translation[b.name] for b in bones
               if b.name in vmd.bone_name_reverse_translation}
    return MMDProfile("standard", tuple(b.name for b in bones),
                      np.array([index.get(b.parent.name, -1) if b.parent else -1 for b in bones]),
                      np.array([b.zeroPos for b in bones]), mapping)


def pmx_profile(path: Path, unit=0.085):
    vmd = require_codec()
    from vmd2png.pmx import read_pmx_skeleton
    path = Path(path).resolve()
    data = read_pmx_skeleton(path)
    raw = data["bones"]
    order, pending = [], set(range(len(raw)))
    while pending:
        ready = sorted(i for i in pending if raw[i]["parent"] == -1 or raw[i]["parent"] in order)
        if not ready:
            raise ValueError("PMX skeleton has a parenting cycle")
        order.extend(ready)
        pending.difference_update(ready)
    translated = [vmd.bone_name_translation.get(raw[i]["name"], raw[i]["name"]) for i in order]
    old_to_new = {old: new for new, old in enumerate(order)}
    mapping = {name: raw[i]["name"] for name, i in zip(translated, order)}
    profile = MMDProfile(path.stem, tuple(translated),
                      np.array([old_to_new.get(raw[i]["parent"], -1) for i in order]),
                      np.array([raw[i]["position"] for i in order]) * unit,
                      mapping, unit, str(path))
    profile.knee_limits = {}
    profile.ik_chains = {}
    body = {"HipMaster", "Torso", "Torso2", "Neck", "Head"}
    body.update(side+part for side in ("Left", "Right") for part in
                ("Shoulder", "Arm", "Elbow", "Wrist", "Leg", "Knee", "Ankle"))
    ancestors = set()
    for name in body:
        i = profile.index[name]
        while i >= 0:
            ancestors.add(i)
            i = profile.parents[i]
    inherited = [translated[new] for new, old in enumerate(order) if new in ancestors and "inherit" in raw[old]]
    if inherited:
        raise ValueError(f"Unsupported PMX inherited body transforms: {inherited}; use a standard profile or simplify the rig")
    fixed = [translated[new] for new, old in enumerate(order) if translated[new] in body and "fixed_axis" in raw[old]]
    if fixed:
        raise ValueError(f"Unsupported PMX fixed-axis body controls: {fixed}; use a standard profile or simplify the rig")
    for bone in raw:
        if "ik" in bone:
            canonical = vmd.bone_name_translation.get(bone["name"], bone["name"])
            chain = bone["ik"]
            profile.ik_chains[canonical] = {
                "target": vmd.bone_name_translation.get(raw[chain["target"]]["name"], raw[chain["target"]]["name"]),
                "links": [vmd.bone_name_translation.get(raw[link["bone"]]["name"], raw[link["bone"]]["name"])
                          for link in chain["links"]]}
        for link in bone.get("ik", {}).get("links", []):
            if "limits" in link:
                name = vmd.bone_name_translation.get(raw[link["bone"]]["name"], raw[link["bone"]]["name"])
                limits = link["limits"]
                if name in ("LeftKnee", "RightKnee") and np.allclose([limits[1],limits[2],limits[4],limits[5]], 0):
                    profile.knee_limits[name] = (limits[0], limits[3])
    return profile
