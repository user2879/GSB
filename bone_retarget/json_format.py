"""Versioned JSON animation format, migration, and parsing."""

from __future__ import annotations

import copy
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .errors import ErrorCode, PipelineError
from .hashing import sha256_bytes, sha256_json
from .math3d import QUAT_COMPONENT_ORDERS, format_quaternion, reorder_quaternion
from .models import Animation

LATEST_SCHEMA_VERSION = 3
SUPPORTED_SCHEMA_VERSIONS = (1, 2, 3)
MIGRATION_TABLE: dict[int, dict[str, Any]] = {
    1: {
        "quat_order": "xyzw",
        "interpolation": "slerp",
        "joint_map": "from joints order",
        "root_translation": "zero",
    },
    2: {
        "quat_order": "xyzw",
        "interpolation": "slerp",
        "joint_map": "from joints order when absent",
        "root_translation": "zero",
    },
}


def document_json_bytes(document: Any, *, pretty: bool = True, allow_nan: bool = False) -> bytes:
    kwargs: dict[str, Any] = {
        "ensure_ascii": False,
        "sort_keys": True,
        "allow_nan": allow_nan,
    }
    if pretty:
        kwargs["indent"] = 2
        text = json.dumps(document, **kwargs) + "\n"
    else:
        kwargs["separators"] = (",", ":")
        text = json.dumps(document, **kwargs)
    return text.encode("utf-8")


@dataclass
class MigrationResult:
    document: dict[str, Any]
    input_version: int
    output_version: int
    added_fields: tuple[str, ...]
    before_sha256: str
    after_sha256: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "input_schema_version": self.input_version,
            "output_schema_version": self.output_version,
            "added_fields": list(self.added_fields),
            "before_sha256": self.before_sha256,
            "after_sha256": self.after_sha256,
            "changed": self.before_sha256 != self.after_sha256,
        }


def _schema_version(document: Any) -> int:
    if not isinstance(document, dict):
        raise PipelineError(
            ErrorCode.E_FIELD_MISSING,
            "JSON animation document must be an object",
            field="document",
        )
    if "schema_version" not in document:
        raise PipelineError(
            ErrorCode.E_FIELD_MISSING,
            "schema_version is required",
            field="schema_version",
        )
    version = document["schema_version"]
    if isinstance(version, bool) or not isinstance(version, int):
        raise PipelineError(
            ErrorCode.E_SCHEMA_VERSION,
            "schema_version must be an integer",
            field="schema_version",
            details={"supported_versions": list(SUPPORTED_SCHEMA_VERSIONS)},
        )
    if version not in SUPPORTED_SCHEMA_VERSIONS:
        raise PipelineError(
            ErrorCode.E_SCHEMA_VERSION,
            f"unsupported JSON schema version {version}",
            field="schema_version",
            details={
                "unsupported_version": version,
                "supported_versions": list(SUPPORTED_SCHEMA_VERSIONS),
                "missing_fields": [],
            },
        )
    return version


def _require_object(document: dict[str, Any], key: str) -> dict[str, Any]:
    value = document.get(key)
    if not isinstance(value, dict) or not value:
        raise PipelineError(
            ErrorCode.E_FIELD_MISSING,
            f"{key} must be a non-empty object",
            field=key,
        )
    return value


def _require_frames(document: dict[str, Any]) -> list[dict[str, Any]]:
    frames = document.get("frames")
    if not isinstance(frames, list) or not frames:
        frames = document.get("keyframes")
    if not isinstance(frames, list) or not frames:
        raise PipelineError(
            ErrorCode.E_FIELD_MISSING,
            "frames must be a non-empty array",
            field="frames",
        )
    for index, frame in enumerate(frames):
        if not isinstance(frame, dict):
            raise PipelineError(
                ErrorCode.E_FIELD_MISSING,
                "each frame must be an object",
                frame=index,
                field="frames",
            )
    return frames


def _joint_map_from_joints(document: dict[str, Any]) -> dict[str, int]:
    joints = document.get("joints")
    if not isinstance(joints, list) or not joints:
        raise PipelineError(
            ErrorCode.E_FIELD_MISSING,
            "cannot infer joint_map because joints is missing",
            field="joints",
        )
    mapping: dict[str, int] = {}
    for index, name in enumerate(joints):
        if not isinstance(name, str) or not name:
            raise PipelineError(
                ErrorCode.E_FIELD_MISSING,
                "joints must contain non-empty names",
                field=f"joints[{index}]",
            )
        if name in mapping:
            raise PipelineError(
                ErrorCode.E_INDEX_OVERFLOW,
                "duplicate joint name in joints",
                joint=name,
                field="joints",
            )
        mapping[name] = index
    return mapping


def _normalize_frame_aliases(frames: list[dict[str, Any]], added: list[str]) -> None:
    for index, frame in enumerate(frames):
        if "time" not in frame:
            if "t" not in frame:
                raise PipelineError(
                    ErrorCode.E_FIELD_MISSING,
                    "frame timestamp is missing",
                    frame=index,
                    field=f"frames[{index}].time",
                )
            frame["time"] = frame["t"]
            added.append(f"frames[{index}].time")
        if "rotations" not in frame:
            if "r" not in frame:
                raise PipelineError(
                    ErrorCode.E_FIELD_MISSING,
                    "frame rotations are missing",
                    frame=index,
                    field=f"frames[{index}].rotations",
                )
            frame["rotations"] = frame["r"]
            added.append(f"frames[{index}].rotations")


def migrate_json_document(document: Any) -> MigrationResult:
    if not isinstance(document, dict):
        raise PipelineError(
            ErrorCode.E_FIELD_MISSING,
            "JSON animation document must be an object",
            field="document",
        )
    input_version = _schema_version(document)
    before_sha256 = sha256_json(document)
    output = copy.deepcopy(document)
    added: list[str] = []

    if input_version == LATEST_SCHEMA_VERSION:
        return MigrationResult(
            document=output,
            input_version=input_version,
            output_version=input_version,
            added_fields=(),
            before_sha256=before_sha256,
            after_sha256=before_sha256,
        )

    if input_version == 1:
        frames = _require_frames(output)
        if "joint_map" not in output:
            output["joint_map"] = _joint_map_from_joints(output)
            added.append("joint_map")
        if "quat_order" not in output:
            output["quat_order"] = MIGRATION_TABLE[1]["quat_order"]
            added.append("quat_order")
        if "interpolation" not in output:
            output["interpolation"] = MIGRATION_TABLE[1]["interpolation"]
            added.append("interpolation")
        _normalize_frame_aliases(frames, added)
        output["schema_version"] = 2

    _require_frames(output)
    if "joint_map" not in output:
        output["joint_map"] = _joint_map_from_joints(output)
        added.append("joint_map")
    if "quat_order" not in output:
        output["quat_order"] = MIGRATION_TABLE[2]["quat_order"]
        added.append("quat_order")
    if "interpolation" not in output:
        output["interpolation"] = MIGRATION_TABLE[2]["interpolation"]
        added.append("interpolation")

    frames = output.get("frames")
    if not isinstance(frames, list) or not frames:
        frames = output.get("keyframes")
    if not isinstance(frames, list) or not frames:
        raise PipelineError(
            ErrorCode.E_FIELD_MISSING,
            "frames must be a non-empty array",
            field="frames",
        )
    _normalize_frame_aliases(frames, added)
    keyframes = copy.deepcopy(frames)
    joint_count = len(output.get("joint_map", {}))
    for index, frame in enumerate(keyframes):
        if "root_translation" not in frame:
            frame["root_translation"] = [0.0, 0.0, 0.0]
            added.append(f"keyframes[{index}].root_translation")
        rotations = frame.get("rotations")
        if not isinstance(rotations, list) or len(rotations) != joint_count:
            raise PipelineError(
                ErrorCode.E_QUAT_INVALID,
                "rotation count does not match joint_map size during migration",
                frame=index,
                field=f"keyframes[{index}].rotations",
                details={"joint_count": joint_count, "rotation_count": len(rotations or [])},
            )
    output["keyframes"] = keyframes
    output["schema_version"] = LATEST_SCHEMA_VERSION
    after_sha256 = sha256_json(output)
    return MigrationResult(
        document=output,
        input_version=input_version,
        output_version=LATEST_SCHEMA_VERSION,
        added_fields=tuple(added),
        before_sha256=before_sha256,
        after_sha256=after_sha256,
    )


def _parse_joint_map(value: Any) -> tuple[tuple[str, ...], dict[str, int]]:
    if not isinstance(value, dict) or not value:
        raise PipelineError(
            ErrorCode.E_FIELD_MISSING,
            "joint_map must be a non-empty object",
            field="joint_map",
        )
    by_index: dict[int, str] = {}
    mapping: dict[str, int] = {}
    for name, raw_index in value.items():
        if not isinstance(name, str) or not name:
            raise PipelineError(
                ErrorCode.E_FIELD_MISSING,
                "joint_map names must be non-empty strings",
                field="joint_map",
            )
        if isinstance(raw_index, bool) or not isinstance(raw_index, int) or raw_index < 0:
            raise PipelineError(
                ErrorCode.E_INDEX_OVERFLOW,
                "joint_map index must be a non-negative integer",
                joint=name,
                field="joint_map",
                details={"index": raw_index},
            )
        if raw_index in by_index:
            raise PipelineError(
                ErrorCode.E_INDEX_OVERFLOW,
                "joint_map contains duplicate indices",
                joint=name,
                field="joint_map",
                details={"index": raw_index, "other_joint": by_index[raw_index]},
            )
        by_index[raw_index] = name
        mapping[name] = raw_index
    expected = set(range(len(value)))
    actual = set(by_index)
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        raise PipelineError(
            ErrorCode.E_INDEX_OVERFLOW,
            "joint_map indices must be contiguous from zero",
            field="joint_map",
            details={"missing_indices": missing, "extra_indices": extra},
        )
    joint_names = tuple(by_index[index] for index in range(len(value)))
    return joint_names, mapping


def _numeric(value: Any, field: str, *, frame: int, joint: str | None = None) -> float:
    if isinstance(value, bool):
        raise PipelineError(ErrorCode.E_FIELD_MISSING, "value must be numeric", frame=frame, joint=joint, field=field)
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise PipelineError(
            ErrorCode.E_FIELD_MISSING,
            "value must be numeric",
            frame=frame,
            joint=joint,
            field=field,
        ) from exc
    if not np.isfinite(number):
        raise PipelineError(
            ErrorCode.E_FIELD_MISSING,
            "value must be finite",
            frame=frame,
            joint=joint,
            field=field,
        )
    return number


def animation_from_document(document: Any, *, migrate: bool = True) -> Animation:
    if not isinstance(document, dict):
        raise PipelineError(
            ErrorCode.E_FIELD_MISSING,
            "JSON animation document must be an object",
            field="document",
        )
    version = _schema_version(document)
    working = migrate_json_document(document).document if migrate and version != LATEST_SCHEMA_VERSION else document
    if "joint_map" not in working:
        raise PipelineError(ErrorCode.E_FIELD_MISSING, "joint_map is required", field="joint_map")
    joint_names, mapping = _parse_joint_map(working["joint_map"])
    quat_order = working.get("quat_order")
    if quat_order is None:
        raise PipelineError(ErrorCode.E_FIELD_MISSING, "quat_order is required", field="quat_order")
    if not isinstance(quat_order, str) or quat_order.lower() not in QUAT_COMPONENT_ORDERS:
        raise PipelineError(
            ErrorCode.E_FIELD_MISSING,
            f"unsupported quat_order: {quat_order}",
            field="quat_order",
        )
    quat_order = quat_order.lower()
    interpolation = working.get("interpolation")
    if interpolation is None:
        raise PipelineError(ErrorCode.E_FIELD_MISSING, "interpolation is required", field="interpolation")
    if interpolation not in {"slerp", "linear", "step"}:
        raise PipelineError(
            ErrorCode.E_FIELD_MISSING,
            f"unsupported interpolation method: {interpolation}",
            field="interpolation",
        )
    frames = working.get("keyframes")
    if not isinstance(frames, list) or not frames:
        raise PipelineError(
            ErrorCode.E_FIELD_MISSING,
            "keyframes must be a non-empty array",
            field="keyframes",
        )
    times = np.empty((len(frames),), dtype=np.float64)
    rotations = np.empty((len(frames), len(joint_names), 4), dtype=np.float64)
    roots = np.empty((len(frames), 3), dtype=np.float64)
    for frame_index, frame in enumerate(frames):
        if not isinstance(frame, dict):
            raise PipelineError(
                ErrorCode.E_FIELD_MISSING,
                "each keyframe must be an object",
                frame=frame_index,
                field="keyframes",
            )
        if "time" not in frame:
            raise PipelineError(
                ErrorCode.E_FIELD_MISSING,
                "keyframe time is missing",
                frame=frame_index,
                field=f"keyframes[{frame_index}].time",
            )
        times[frame_index] = _numeric(
            frame["time"], f"keyframes[{frame_index}].time", frame=frame_index
        )
        raw_rotations = frame.get("rotations")
        if not isinstance(raw_rotations, list):
            raise PipelineError(
                ErrorCode.E_FIELD_MISSING,
                "keyframe rotations are missing",
                frame=frame_index,
                field=f"keyframes[{frame_index}].rotations",
            )
        if len(raw_rotations) != len(joint_names):
            raise PipelineError(
                ErrorCode.E_QUAT_INVALID,
                "rotation count does not match joint_map size",
                frame=frame_index,
                field=f"keyframes[{frame_index}].rotations",
                details={"joint_count": len(joint_names), "rotation_count": len(raw_rotations)},
            )
        for joint_index, joint_name in enumerate(joint_names):
            raw_quat = raw_rotations[joint_index]
            try:
                quat = reorder_quaternion(
                    raw_quat,
                    quat_order,
                    context=f"keyframes[{frame_index}].rotations[{joint_index}].",
                )
            except PipelineError as exc:
                exc.joint = joint_name
                exc.frame = frame_index
                raise
            if quat.shape != (4,):
                raise PipelineError(
                    ErrorCode.E_QUAT_INVALID,
                    "quaternion must have exactly 4 components",
                    joint=joint_name,
                    frame=frame_index,
                    field=f"keyframes[{frame_index}].rotations[{joint_index}]",
                )
            rotations[frame_index, joint_index] = quat
        raw_root = frame.get("root_translation")
        if not isinstance(raw_root, list) or len(raw_root) != 3:
            raise PipelineError(
                ErrorCode.E_FIELD_MISSING,
                "root_translation must contain exactly 3 values",
                frame=frame_index,
                field=f"keyframes[{frame_index}].root_translation",
            )
        roots[frame_index] = [
            _numeric(value, f"keyframes[{frame_index}].root_translation[{axis}]", frame=frame_index)
            for axis, value in enumerate(raw_root)
        ]
    fps = working.get("fps")
    if fps is not None:
        fps = _numeric(fps, "fps", frame=0)
    animation = Animation(
        joint_names=joint_names,
        times=times,
        rotations=rotations,
        root_translation=roots,
        interpolation=str(interpolation),
        fps=fps,
        source_schema_version=LATEST_SCHEMA_VERSION,
    )
    if "joints" in working and isinstance(working["joints"], list):
        if len(working["joints"]) != len(joint_names):
            raise PipelineError(
                ErrorCode.E_INDEX_OVERFLOW,
                "joints and joint_map lengths differ",
                field="joints",
            )
        for index, name in enumerate(working["joints"]):
            if mapping.get(name) != index:
                raise PipelineError(
                    ErrorCode.E_INDEX_OVERFLOW,
                    "joints order conflicts with joint_map",
                    joint=name,
                    field="joints",
                )
    return animation


def parse_json_bytes(data: bytes, *, migrate: bool = True) -> Animation:
    try:
        document = json.loads(data.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise PipelineError(
            ErrorCode.E_FIELD_MISSING,
            "input is not valid UTF-8 JSON",
            field="document",
            details={"error": str(exc)},
        ) from exc
    return animation_from_document(document, migrate=migrate)


def animation_to_document(
    animation: Animation,
    *,
    schema_version: int = LATEST_SCHEMA_VERSION,
    quat_order: str = "wxyz",
) -> dict[str, Any]:
    if schema_version not in SUPPORTED_SCHEMA_VERSIONS:
        raise PipelineError(
            ErrorCode.E_SCHEMA_VERSION,
            f"unsupported output schema version {schema_version}",
            field="schema_version",
            details={"supported_versions": list(SUPPORTED_SCHEMA_VERSIONS)},
        )
    order = quat_order.lower()
    if order not in QUAT_COMPONENT_ORDERS:
        raise PipelineError(
            ErrorCode.E_FIELD_MISSING,
            f"unsupported quat_order: {quat_order}",
            field="quat_order",
        )
    if schema_version == 1:
        order = "xyzw"
    mapping = {name: index for index, name in enumerate(animation.joint_names)}
    frames: list[dict[str, Any]] = []
    for frame_index in range(animation.frame_count):
        frame: dict[str, Any] = {
            "time": float(animation.times[frame_index]),
            "rotations": [
                format_quaternion(animation.rotations[frame_index, joint_index], order)
                for joint_index in range(animation.joint_count)
            ],
        }
        if schema_version == LATEST_SCHEMA_VERSION:
            frame["root_translation"] = [
                float(value) for value in animation.root_translation[frame_index]
            ]
        frames.append(frame)
    document: dict[str, Any] = {"schema_version": schema_version}
    if schema_version == 1:
        document["joints"] = list(animation.joint_names)
        document["frames"] = frames
    else:
        document["joint_map"] = mapping
        document["quat_order"] = order
        document["interpolation"] = animation.interpolation
        if schema_version == 2:
            document["frames"] = frames
        else:
            document["keyframes"] = frames
    if animation.fps is not None:
        document["fps"] = animation.fps
    return document


def load_json_animation(path: str | Path, *, migrate: bool = True) -> Animation:
    return parse_json_bytes(Path(path).read_bytes(), migrate=migrate)


def save_json_document(path: str | Path, document: Any, *, pretty: bool = True) -> str:
    data = document_json_bytes(document, pretty=pretty)
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(data)
    return sha256_bytes(data)
