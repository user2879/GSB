"""In-memory animation model plus the self-owned JSON exchange format.

JSON schema
-----------
v2 (current)::

    {
      "schema_version": "2.0",
      "quat_order": "wxyz",          // wxyz | xyzw
      "interp": "slerp",             // slerp | nlerp | step
      "joint_names": ["root", ...],
      "joint_index": {"root": 0},    // joint name -> index mapping
      "frames": [
        {"t": 0.0,
         "rotations": [[w,x,y,z], ...J],   // local rotations, joint order
         "root_translation": [x,y,z]}
      ]
    }

v1 (legacy) stores ``frames[].quats`` without ``quat_order``,
``interp``, ``joint_index`` or ``root_translation``.  The fixed migration
table in :data:`MIGRATIONS` fills those fields without changing existing
numeric values.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

import numpy as np

from .errors import E_SCHEMA_VERSION, E_TIME_NONMONOTONIC
from .hashutil import animation_hash, canonical_json_bytes, sha256_hex
from .quat import QUAT_ORDERS, align_sequence, check_quat, reorder

CURRENT_SCHEMA = "2.0"
LEGACY_SCHEMA = "1.0"
INTERP_MODES = ("slerp", "nlerp", "step")

MIGRATIONS: dict[str, dict[str, Any]] = {
    "1.0->2.0": {
        "add_quat_order": "wxyz",
        "add_interp": "slerp",
        "add_root_translation": [0.0, 0.0, 0.0],
        "rename_frames_field": ("quats", "rotations"),
    }
}


@dataclass
class Animation:
    joint_names: tuple[str, ...]
    times: np.ndarray  # (F,)
    rotations: np.ndarray  # (F, J, 4) wxyz
    root_translations: np.ndarray  # (F, 3)
    interp: str = "slerp"
    quat_order: str = "wxyz"
    schema_version: str = CURRENT_SCHEMA
    parameter_version: str = "baseline"

    @property
    def frame_count(self) -> int:
        return int(self.times.shape[0])

    @property
    def joint_count(self) -> int:
        return len(self.joint_names)

    def joint_index(self) -> dict[str, int]:
        return {name: index for index, name in enumerate(self.joint_names)}

    def content_hash(self) -> str:
        return animation_hash(
            self.schema_version,
            self.interp,
            self.quat_order,
            list(self.joint_names),
            self.times,
            self.rotations,
            self.root_translations,
            self.parameter_version,
        )


def check_time_monotonic(times: np.ndarray) -> None:
    arr = np.asarray(times, dtype=np.float64)
    if arr.size >= 2 and np.any(np.diff(arr) <= 0.0):
        bad = int(np.argmax(np.diff(arr) <= 0.0))
        raise E_TIME_NONMONOTONIC(
            "frame timestamps must be strictly increasing",
            {"frame": bad + 1, "t_prev": float(arr[bad]), "t": float(arr[bad + 1])},
        )


def _as_wxyz(frames_rotations: list[list[list[float]]], quat_order: str) -> np.ndarray:
    arr = np.asarray(frames_rotations, dtype=np.float64)
    return reorder(arr, quat_order, "wxyz")


def migrate_schema(document: dict[str, Any]) -> dict[str, Any]:
    """Apply the fixed migration table until the document is current.

    Idempotent: migrating an already-current document returns an equal
    document with an identical content hash.  Unknown versions raise
    ``E_SCHEMA_VERSION`` naming the offending field.
    """

    version = document.get("schema_version")
    if version is None:
        raise E_SCHEMA_VERSION(
            "document is missing schema_version",
            {"field": "schema_version", "found": None},
        )
    if version == CURRENT_SCHEMA:
        return dict(document)

    current = json.loads(json.dumps(document))  # deep copy, no mutation
    seen: set[str] = set()
    while current.get("schema_version") != CURRENT_SCHEMA:
        v = current.get("schema_version")
        key = f"{v}->{CURRENT_SCHEMA}"
        if key not in MIGRATIONS or v in seen:
            missing = [
                field
                for field in ("quat_order", "interp", "joint_index", "root_translation")
                if field not in current
            ]
            raise E_SCHEMA_VERSION(
                f"no migration path for schema version {v!r}",
                {"field": "schema_version", "found": v, "missing_fields": missing},
            )
        seen.add(v)
        rule = MIGRATIONS[key]
        current["quat_order"] = rule["add_quat_order"]
        current["interp"] = rule["add_interp"]
        old_name, new_name = rule["rename_frames_field"]
        for frame in current.get("frames", []):
            if old_name in frame and new_name not in frame:
                frame[new_name] = frame.pop(old_name)
            frame.setdefault(
                "root_translation", list(rule["add_root_translation"])
            )
        names = current.get("joint_names")
        if names is not None and "joint_index" not in current:
            current["joint_index"] = {
                name: index for index, name in enumerate(names)
            }
        current["schema_version"] = CURRENT_SCHEMA
    return current


def migration_hash(document: dict[str, Any]) -> str:
    """Hash of the migration-relevant document content."""

    return sha256_hex(canonical_json_bytes(document))


def animation_from_document(document: dict[str, Any]) -> Animation:
    doc = migrate_schema(document)
    names = tuple(doc["joint_names"])
    index_map = doc.get("joint_index") or {n: i for i, n in enumerate(names)}
    if index_map != {n: i for i, n in enumerate(names)}:
        raise E_SCHEMA_VERSION(
            "joint_index mapping disagrees with joint_names ordering",
            {"field": "joint_index", "found": index_map},
        )
    quat_order = doc["quat_order"]
    if quat_order not in QUAT_ORDERS:
        raise E_SCHEMA_VERSION(
            "unsupported quaternion component order",
            {"field": "quat_order", "found": quat_order},
        )
    interp = doc["interp"]
    if interp not in INTERP_MODES:
        raise E_SCHEMA_VERSION(
            "unsupported interpolation mode",
            {"field": "interp", "found": interp},
        )

    frames = doc["frames"]
    times = np.asarray([frame["t"] for frame in frames], dtype=np.float64)
    check_time_monotonic(times)
    rotations = _as_wxyz(
        [frame["rotations"] for frame in frames], quat_order
    )
    for frame_index, frame_rot in enumerate(rotations):
        for joint_index, quat in enumerate(frame_rot):
            check_quat(quat, joint=joint_index, frame=frame_index)
    rotations = align_sequence(rotations)
    roots = np.asarray(
        [frame.get("root_translation", [0.0, 0.0, 0.0]) for frame in frames],
        dtype=np.float64,
    )
    return Animation(
        joint_names=names,
        times=times,
        rotations=rotations,
        root_translations=roots,
        interp=interp,
        quat_order=quat_order,
        schema_version=CURRENT_SCHEMA,
    )


def load_json_animation(path: str) -> Animation:
    with open(path, "r", encoding="utf-8") as handle:
        document = json.load(handle)
    return animation_from_document(document)


def animation_to_document(anim: Animation) -> dict[str, Any]:
    """Serialize with the declared quaternion order and aligned signs."""

    ordered = reorder(anim.rotations, "wxyz", anim.quat_order)
    frames: list[dict[str, Any]] = []
    for frame_index, time_value in enumerate(anim.times):
        frames.append(
            {
                "t": float(time_value),
                "rotations": [
                    [float(component) for component in quat]
                    for quat in ordered[frame_index]
                ],
                "root_translation": [
                    float(value) for value in anim.root_translations[frame_index]
                ],
            }
        )
    return {
        "schema_version": CURRENT_SCHEMA,
        "quat_order": anim.quat_order,
        "interp": anim.interp,
        "joint_names": list(anim.joint_names),
        "joint_index": anim.joint_index(),
        "frames": frames,
    }


def dump_json_animation(anim: Animation, path: str) -> None:
    document = animation_to_document(anim)
    with open(path, "w", encoding="utf-8", newline="\n") as handle:
        json.dump(document, handle, sort_keys=True, indent=2)
        handle.write("\n")


def build_legacy_document(anim: Animation) -> dict[str, Any]:
    """Emit a v1 document so the migration path is testable end to end."""

    return {
        "schema_version": LEGACY_SCHEMA,
        "joint_names": list(anim.joint_names),
        "frames": [
            {"t": float(t), "quats": anim.rotations[i].tolist()}
            for i, t in enumerate(anim.times)
        ],
    }
