"""Carrier writers/parsers and normalization into one internal model."""

from __future__ import annotations

import copy
import hashlib
import struct
import zlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from .constants import (
    BINARY_SCHEMA,
    KNOWN_FIELD_BITMAP,
    LEGACY_PROJECT_SCHEMA,
    METRIC_ENCODING_VERSION,
    METRIC_FIELD_BITS,
    METRIC_POLICY,
    PROJECT_SCHEMA,
    REQUIRED_FIELD_BITMAP,
    V1_FIELD_BITMAP,
    V2_FIELD_BITMAP,
)
from .errors import BusinessError, as_error
from .quality import summarize_segments, validate_shots
from .util import (
    atomic_write_bytes,
    canonical_json_bytes,
    hash_object,
    json_load,
    resolve_workspace_path,
)


@dataclass
class CarrierModel:
    carrier_type: str
    schema_version: str
    content: dict[str, Any]
    encoding_params: dict[str, Any]
    shots: list[dict[str, Any]]
    metric_policy: dict[str, Any]
    declared_frames: list[dict[str, Any]]
    segments: list[dict[str, Any]]
    assumptions: list[dict[str, Any]] = field(default_factory=list)
    provenance: dict[str, Any] = field(default_factory=dict)

    def summary(self) -> dict[str, Any]:
        return {
            "carrier_type": self.carrier_type,
            "schema_version": self.schema_version,
            "content_hash": self.content.get("content_hash"),
            "frame_count": len(self.declared_frames),
            "shot_count": len(self.shots),
            "policy_version": self.metric_policy.get("policy_version"),
            "policy_hash": hash_object(self.metric_policy),
            "encoding_hash": hash_object(self.encoding_params),
            "shots_hash": hash_object(self.shots),
            "assumptions": self.assumptions,
            "provenance": self.provenance,
        }


def _require_mapping(value: Any, path: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise BusinessError("E_SCHEMA", f"{path} must be an object", {"path": path})
    return value


def _require_list(value: Any, path: str) -> list[Any]:
    if not isinstance(value, list):
        raise BusinessError("E_SCHEMA", f"{path} must be an array", {"path": path})
    return value


def _require_keys(document: dict[str, Any], keys: tuple[str, ...], context: str) -> None:
    missing = [key for key in keys if key not in document]
    if missing:
        raise BusinessError("E_SCHEMA", f"missing required fields in {context}", {"missing": missing})


def _normalize_policy(raw: Any) -> dict[str, Any]:
    policy = _require_mapping(raw, "metric_policy")
    result = copy.deepcopy(METRIC_POLICY)
    result.update(copy.deepcopy(policy))
    _require_keys(
        result,
        ("policy_version", "pixel_range", "channel_weights", "downsample", "ssim_window"),
        "metric_policy",
    )
    if list(result["pixel_range"]) != [0, 255]:
        raise BusinessError("E_SCHEMA_VERSION", "unsupported metric pixel range", {"pixel_range": result["pixel_range"]})
    if result.get("downsample") != "none":
        raise BusinessError("E_SCHEMA_VERSION", "unsupported metric downsampling mode", {"downsample": result.get("downsample")})
    return result


def _normalize_frame(raw: Any, index: int, shots: list[dict], flags_default: int = 0) -> dict[str, Any]:
    frame = _require_mapping(raw, f"frames[{index}]")
    _require_keys(frame, ("frame_index", "psnr_db", "ssim"), f"frames[{index}]")
    try:
        frame_index = int(frame["frame_index"])
        timestamp_ms = float(frame.get("timestamp_ms", frame_index * 1000.0 / 30.0))
        shot_id = int(frame["shot_id"]) if "shot_id" in frame else None
        psnr = float(frame["psnr_db"]) if frame["psnr_db"] is not None else None
        ssim = float(frame["ssim"]) if frame["ssim"] is not None else None
    except (TypeError, ValueError) as exc:
        raise BusinessError("E_SCHEMA", f"invalid metric record at frames[{index}]", {"index": index}) from exc
    if shot_id is None:
        shot_id = next(
            int(shot["shot_id"])
            for shot in shots
            if int(shot["start_frame"]) <= frame_index <= int(shot["end_frame"])
        )
    record = {
        "frame_index": frame_index,
        "shot_id": shot_id,
        "timestamp_ms": round(timestamp_ms, 6),
        "psnr_db": round(psnr, 9) if psnr is not None else None,
        "ssim": round(ssim, 9) if ssim is not None else None,
        "chroma_psnr_db": round(float(frame["chroma_psnr_db"]), 9) if frame.get("chroma_psnr_db") is not None else None,
        "flags": int(frame.get("flags", flags_default)),
        "missing": bool(frame.get("missing", psnr is None)),
        "reason": frame.get("reason"),
        "retries": int(frame.get("retries", 0)),
        "metric_encoding_version": int(frame.get("metric_encoding_version", METRIC_ENCODING_VERSION)),
    }
    return record


def _normalize_frames(raw_frames: list[Any], shots: list[dict], frame_count: int, flags_default: int = 0) -> list[dict]:
    frames = [_normalize_frame(raw, index, shots, flags_default) for index, raw in enumerate(raw_frames)]
    frames.sort(key=lambda frame: frame["frame_index"])
    indices = [int(frame["frame_index"]) for frame in frames]
    if indices != list(range(frame_count)):
        duplicate_count = len(indices) - len(set(indices))
        raise BusinessError(
            "E_SCHEMA",
            "metric frames must be contiguous and unique",
            {"expected_count": frame_count, "actual_count": len(indices), "duplicates": duplicate_count},
        )
    return frames


def _model_from_parts(
    carrier_type: str,
    schema_version: str,
    content: dict[str, Any],
    encoding_params: dict[str, Any],
    shots_raw: list[Any],
    policy_raw: dict[str, Any],
    frames_raw: list[Any],
    assumptions: list[dict[str, Any]] | None = None,
    provenance: dict[str, Any] | None = None,
    flags_default: int = 0,
) -> CarrierModel:
    content = copy.deepcopy(_require_mapping(content, "content"))
    encoding_params = copy.deepcopy(_require_mapping(encoding_params, "encoding_params"))
    _require_keys(content, ("content_hash", "video_path", "audio_path", "truth_path"), "content")
    frame_count = len(frames_raw)
    shots = validate_shots(shots_raw, frame_count)
    policy = _normalize_policy(policy_raw)
    frames = _normalize_frames(frames_raw, shots, frame_count, flags_default)
    segments = summarize_segments(frames, shots, policy)
    return CarrierModel(
        carrier_type=carrier_type,
        schema_version=schema_version,
        content=content,
        encoding_params=encoding_params,
        shots=shots,
        metric_policy=policy,
        declared_frames=frames,
        segments=segments,
        assumptions=list(assumptions or []),
        provenance=dict(provenance or {}),
    )


def parse_project_json(path: Path) -> CarrierModel:
    document = _require_mapping(json_load(path), "project")
    schema_version = document.get("schema_version")
    if schema_version != PROJECT_SCHEMA:
        raise BusinessError(
            "E_SCHEMA_VERSION",
            "unsupported project schema version",
            {"expected": PROJECT_SCHEMA, "actual": schema_version, "path": str(path)},
        )
    _require_keys(document, ("content", "encoding_params", "shots", "metric_policy", "frames"), "project")
    return _model_from_parts(
        "project_json",
        schema_version,
        document["content"],
        document["encoding_params"],
        _require_list(document["shots"], "shots"),
        document["metric_policy"],
        _require_list(document["frames"], "frames"),
        provenance={"path": path.name},
    )


LEGACY_MIGRATION_TABLE = {
    "shots": {"source": "content.truth.shots", "semantic_uncertainty": True},
    "metric_policy": {"source": "canonical/quality/2.0", "semantic_uncertainty": True},
    "frame.shot_id": {"source": "content.truth.frame_shot_map", "semantic_uncertainty": True},
    "frame.flags": {"source": "default/0", "semantic_uncertainty": False},
    "frame.chroma_psnr_db": {"source": "default/null", "semantic_uncertainty": False},
}


def _load_truth_for_content(workspace: Path, content: dict[str, Any]) -> dict[str, Any]:
    truth_path = resolve_workspace_path(workspace, str(content["truth_path"]))
    truth = _require_mapping(json_load(truth_path), "truth")
    if truth.get("content_hash") != content.get("content_hash"):
        raise BusinessError(
            "E_CONTENT_HASH",
            "truth manifest content hash differs from legacy carrier",
            {"declared": content.get("content_hash"), "truth": truth.get("content_hash")},
        )
    return truth


def parse_legacy_json(path: Path, workspace: Path, truth: dict[str, Any] | None = None) -> CarrierModel:
    document = _require_mapping(json_load(path), "legacy_project")
    schema_version = document.get("schema_version")
    if schema_version != LEGACY_PROJECT_SCHEMA:
        raise BusinessError(
            "E_SCHEMA_VERSION",
            "unsupported legacy schema version",
            {"expected": LEGACY_PROJECT_SCHEMA, "actual": schema_version, "path": str(path)},
        )
    _require_keys(document, ("content", "encoding_params", "frames"), "legacy_project")
    content = _require_mapping(document["content"], "content")
    truth_document = truth or _load_truth_for_content(workspace, content)
    if truth_document.get("content_hash") != content.get("content_hash"):
        raise BusinessError("E_CONTENT_HASH", "legacy content hash does not match truth")
    shots = copy.deepcopy(truth_document["shots"])
    assumptions = [
        {"kind": "missing_field", "field": field_name, **migration, "migrated_value_source": migration["source"]}
        for field_name, migration in LEGACY_MIGRATION_TABLE.items()
    ]
    frames = _require_list(document["frames"], "frames")
    migrated_frames: list[Any] = []
    truth_shot_map = {int(frame["frame_index"]): int(frame["shot_id"]) for frame in truth_document["frames"]}
    for index, raw in enumerate(frames):
        frame = copy.deepcopy(_require_mapping(raw, f"frames[{index}]"))
        frame_index = int(frame["frame_index"])
        frame["shot_id"] = truth_shot_map[frame_index]
        frame.setdefault("flags", 0)
        frame.setdefault("chroma_psnr_db", None)
        frame.setdefault("metric_encoding_version", METRIC_ENCODING_VERSION)
        migrated_frames.append(frame)
    return _model_from_parts(
        "legacy_json",
        schema_version,
        content,
        document["encoding_params"],
        shots,
        METRIC_POLICY,
        migrated_frames,
        assumptions=assumptions,
        provenance={"path": path.name, "migration_table": LEGACY_MIGRATION_TABLE},
    )


METRIC_MIGRATION_TABLE = {
    "1->2": {
        "fill": {"chroma_psnr_db": None, "flags": 0, "metric_encoding_version": METRIC_ENCODING_VERSION},
        "semantic_uncertainty": False,
    }
}


def migrate_metric_record(record: dict[str, Any], source_version: int) -> tuple[dict[str, Any], bool]:
    if source_version >= METRIC_ENCODING_VERSION:
        return copy.deepcopy(record), False
    migrated = copy.deepcopy(record)
    fill = METRIC_MIGRATION_TABLE["1->2"]["fill"]
    for key, value in fill.items():
        migrated.setdefault(key, value)
    migrated["metric_encoding_version"] = METRIC_ENCODING_VERSION
    return migrated, True


def _binary_header(
    endian: str,
    index_width: int,
    metric_version: int,
    field_bitmap: int,
    frame_count: int,
    segment_count: int,
    meta_len: int,
    payload_crc: int,
) -> bytes:
    prefix = "<" if endian == "little" else ">"
    marker = b"\xff\xfe" if endian == "little" else b"\xfe\xff"
    return (
        b"EMF1"
        + marker
        + struct.pack("B", index_width)
        + struct.pack(prefix + "H", metric_version)
        + struct.pack(prefix + "I", field_bitmap)
        + struct.pack(prefix + "I", frame_count)
        + struct.pack(prefix + "I", segment_count)
        + struct.pack(prefix + "I", meta_len)
        + struct.pack(prefix + "I", payload_crc)
    )


def write_metric_binary(
    path: Path,
    meta: dict[str, Any],
    frames: list[dict[str, Any]],
    metric_version: int = METRIC_ENCODING_VERSION,
    endian: str = "little",
    index_width: int = 2,
    field_bitmap: int | None = None,
) -> dict[str, Any]:
    if endian not in {"little", "big"}:
        raise ValueError("endian must be little or big")
    if index_width not in {1, 2, 4}:
        raise ValueError("index_width must be 1, 2, or 4")
    if field_bitmap is None:
        field_bitmap = V1_FIELD_BITMAP if metric_version == 1 else V2_FIELD_BITMAP
    cap = (1 << (8 * index_width)) - 1
    if len(frames) > cap:
        raise BusinessError("E_INDEX_OVERFLOW", "frame count exceeds index width", {"count": len(frames), "cap": cap})
    if len(meta.get("shots", [])) > cap:
        raise BusinessError(
            "E_INDEX_OVERFLOW",
            "segment count exceeds index width",
            {"count": len(meta.get("shots", [])), "cap": cap},
        )
    if any(int(frame["frame_index"]) > cap for frame in frames):
        raise BusinessError("E_INDEX_OVERFLOW", "frame index exceeds index width", {"cap": cap})
    meta_with_contract = copy.deepcopy(meta)
    meta_with_contract["schema_version"] = PROJECT_SCHEMA
    meta_with_contract["binary_contract"] = {
        "endian": endian,
        "index_width": index_width,
        "metric_encoding_version": metric_version,
        "field_bitmap": field_bitmap,
    }
    meta_bytes = canonical_json_bytes(meta_with_contract)
    prefix = "<" if endian == "little" else ">"
    payload = bytearray()
    for frame in frames:
        if field_bitmap & METRIC_FIELD_BITS["frame_index"]:
            payload += int(frame["frame_index"]).to_bytes(index_width, "little" if endian == "little" else "big", signed=False)
        if field_bitmap & METRIC_FIELD_BITS["shot_id"]:
            payload += struct.pack(prefix + "H", int(frame.get("shot_id", 0)))
        if field_bitmap & METRIC_FIELD_BITS["psnr_db"]:
            payload += struct.pack(prefix + "d", float(frame["psnr_db"]))
        if field_bitmap & METRIC_FIELD_BITS["ssim"]:
            payload += struct.pack(prefix + "d", float(frame["ssim"]))
        if field_bitmap & METRIC_FIELD_BITS["timestamp_ms"]:
            payload += struct.pack(prefix + "d", float(frame.get("timestamp_ms", 0.0)))
        if field_bitmap & METRIC_FIELD_BITS["flags"]:
            payload += struct.pack(prefix + "H", int(frame.get("flags", 0)))
        if field_bitmap & METRIC_FIELD_BITS["chroma_psnr_db"]:
            payload += struct.pack(prefix + "d", float(frame.get("chroma_psnr_db", frame["psnr_db"])))
    payload_crc = zlib.crc32(meta_bytes + bytes(payload)) & 0xFFFFFFFF
    header = _binary_header(
        endian,
        index_width,
        metric_version,
        field_bitmap,
        len(frames),
        len(meta_with_contract.get("shots", [])),
        len(meta_bytes),
        payload_crc,
    )
    data = header + meta_bytes + bytes(payload)
    atomic_write_bytes(path, data)
    return {
        "path": path.name,
        "bytes": len(data),
        "sha256": hashlib.sha256(data).hexdigest(),
        "endian": endian,
        "index_width": index_width,
        "metric_encoding_version": metric_version,
        "field_bitmap": field_bitmap,
    }


def apply_binary_fault(raw: bytes, fault: str | None) -> bytes:
    if not fault:
        return raw
    data = bytearray(raw)
    if len(data) < 29 or data[:4] != b"EMF1":
        raise BusinessError("E_CARRIER_PARSE", "binary fault injection requires an EMF1 carrier")
    if fault == "endian":
        data[4:6] = b"\xfe\xff" if data[4:6] == b"\xff\xfe" else b"\xff\xfe"
    elif fault == "index_width":
        data[6] = 1
    elif fault == "metric_version":
        endian = "<" if data[4:6] == b"\xff\xfe" else ">"
        data[7:9] = struct.pack(endian + "H", 99)
    elif fault == "field_bitmap":
        endian = "<" if data[4:6] == b"\xff\xfe" else ">"
        data[9:13] = struct.pack(endian + "I", 0xFFFFFFFF)
    else:
        raise BusinessError("E_INJECTION", "unknown binary fault", {"fault": fault})
    return bytes(data)


def parse_metric_binary(path: Path, binary_fault: str | None = None) -> CarrierModel:
    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise BusinessError("E_IO", f"cannot read binary carrier {path.name}", {"path": str(path)}) from exc
    raw = apply_binary_fault(raw, binary_fault)
    if len(raw) < 29 or raw[:4] != b"EMF1":
        raise BusinessError("E_CARRIER_PARSE", "binary magic is missing", {"path": path.name})
    marker = raw[4:6]
    if marker == b"\xff\xfe":
        endian = "little"
    elif marker == b"\xfe\xff":
        endian = "big"
    else:
        raise BusinessError("E_ENDIAN_MISMATCH", "unknown byte-order marker", {"marker": marker.hex()})
    prefix = "<" if endian == "little" else ">"
    index_width = raw[6]
    try:
        metric_version = struct.unpack(prefix + "H", raw[7:9])[0]
        field_bitmap = struct.unpack(prefix + "I", raw[9:13])[0]
        frame_count = struct.unpack(prefix + "I", raw[13:17])[0]
        segment_count = struct.unpack(prefix + "I", raw[17:21])[0]
        meta_len = struct.unpack(prefix + "I", raw[21:25])[0]
        payload_crc = struct.unpack(prefix + "I", raw[25:29])[0]
    except struct.error as exc:
        raise BusinessError("E_CARRIER_PARSE", "truncated binary header") from exc
    if index_width not in {1, 2, 4}:
        raise BusinessError("E_INDEX_OVERFLOW", "unsupported index width", {"index_width": index_width})
    if meta_len > len(raw) - 29:
        raise BusinessError(
            "E_ENDIAN_MISMATCH",
            "metadata length is implausible under the declared byte order",
            {"declared_endian": endian, "meta_len": meta_len, "file_size": len(raw)},
        )
    if frame_count > (1 << (8 * index_width)) - 1:
        raise BusinessError(
            "E_INDEX_OVERFLOW",
            "declared frame count does not fit index width",
            {"frame_count": frame_count, "index_width": index_width},
        )
    meta_bytes = raw[29 : 29 + meta_len]
    try:
        import json

        meta = json.loads(meta_bytes.decode("utf-8"))
    except Exception as exc:
        raise BusinessError("E_SCHEMA", "binary metadata is not valid JSON", {"path": path.name}) from exc
    if not isinstance(meta, dict):
        raise BusinessError("E_SCHEMA", "binary metadata must be an object")
    if meta.get("schema_version") != PROJECT_SCHEMA:
        raise BusinessError(
            "E_SCHEMA_VERSION",
            "binary metadata project schema is unsupported",
            {"expected": PROJECT_SCHEMA, "actual": meta.get("schema_version")},
        )
    contract = _require_mapping(meta.get("binary_contract"), "binary_contract")
    if contract.get("endian") != endian:
        raise BusinessError(
            "E_ENDIAN_MISMATCH",
            "marker and declared endian disagree",
            {"marker": endian, "declared": contract.get("endian")},
        )
    if int(contract.get("index_width", -1)) != index_width:
        raise BusinessError(
            "E_INDEX_OVERFLOW",
            "header index width and format declaration disagree",
            {"header": index_width, "declared": contract.get("index_width")},
        )
    if metric_version not in REQUIRED_FIELD_BITMAP:
        raise BusinessError("E_METRIC_VERSION", "unsupported metric encoding version", {"metric_version": metric_version})
    if int(contract.get("metric_encoding_version", -1)) != metric_version:
        raise BusinessError(
            "E_METRIC_VERSION",
            "metric version label and header disagree",
            {"header": metric_version, "declared": contract.get("metric_encoding_version")},
        )
    declared_bitmap = int(contract.get("field_bitmap", -1))
    if declared_bitmap != field_bitmap:
        raise BusinessError(
            "E_FIELD_BITMAP",
            "field bitmap declaration differs from header",
            {"header": field_bitmap, "declared": declared_bitmap},
        )
    if field_bitmap & ~KNOWN_FIELD_BITMAP:
        raise BusinessError("E_FIELD_BITMAP", "field bitmap contains unknown bits", {"field_bitmap": field_bitmap})
    required = REQUIRED_FIELD_BITMAP[metric_version]
    if field_bitmap & required != required:
        raise BusinessError(
            "E_FIELD_BITMAP",
            "field bitmap misses required fields for metric version",
            {"field_bitmap": field_bitmap, "required": required, "metric_version": metric_version},
        )
    if metric_version == 1 and field_bitmap & METRIC_FIELD_BITS["chroma_psnr_db"]:
        raise BusinessError("E_FIELD_BITMAP", "version 1 cannot declare the v2 chroma field")
    record_size = 0
    for field_name, bit in METRIC_FIELD_BITS.items():
        if not field_bitmap & bit:
            continue
        if field_name == "frame_index":
            record_size += index_width
        elif field_name in {"shot_id", "flags"}:
            record_size += 2
        else:
            record_size += 8
    payload_start = 29 + meta_len
    expected_size = payload_start + frame_count * record_size
    if len(raw) != expected_size:
        raise BusinessError(
            "E_SCHEMA",
            "binary payload length does not match field bitmap",
            {"expected": expected_size, "actual": len(raw), "record_size": record_size},
        )
    payload = raw[payload_start:]
    if zlib.crc32(meta_bytes + payload) & 0xFFFFFFFF != payload_crc:
        raise BusinessError("E_CARRIER_PARSE", "binary payload CRC mismatch", {"path": path.name})
    frames: list[dict[str, Any]] = []
    offset = 0
    try:
        for _ in range(frame_count):
            frame: dict[str, Any] = {}
            if field_bitmap & METRIC_FIELD_BITS["frame_index"]:
                frame["frame_index"] = int.from_bytes(
                    payload[offset : offset + index_width],
                    "little" if endian == "little" else "big",
                    signed=False,
                )
                offset += index_width
            if field_bitmap & METRIC_FIELD_BITS["shot_id"]:
                frame["shot_id"] = struct.unpack(prefix + "H", payload[offset : offset + 2])[0]
                offset += 2
            if field_bitmap & METRIC_FIELD_BITS["psnr_db"]:
                frame["psnr_db"] = struct.unpack(prefix + "d", payload[offset : offset + 8])[0]
                offset += 8
            if field_bitmap & METRIC_FIELD_BITS["ssim"]:
                frame["ssim"] = struct.unpack(prefix + "d", payload[offset : offset + 8])[0]
                offset += 8
            if field_bitmap & METRIC_FIELD_BITS["timestamp_ms"]:
                frame["timestamp_ms"] = struct.unpack(prefix + "d", payload[offset : offset + 8])[0]
                offset += 8
            if field_bitmap & METRIC_FIELD_BITS["flags"]:
                frame["flags"] = struct.unpack(prefix + "H", payload[offset : offset + 2])[0]
                offset += 2
            if field_bitmap & METRIC_FIELD_BITS["chroma_psnr_db"]:
                frame["chroma_psnr_db"] = struct.unpack(prefix + "d", payload[offset : offset + 8])[0]
                offset += 8
            frame, _changed = migrate_metric_record(frame, metric_version)
            if not np.isfinite(frame["psnr_db"]) or not np.isfinite(frame["ssim"]):
                raise ValueError("non-finite metric")
            frames.append(frame)
    except (struct.error, KeyError, ValueError) as exc:
        raise BusinessError("E_CARRIER_PARSE", "binary metric record could not be interpreted", {"offset": offset}) from exc
    if len(meta.get("shots", [])) != segment_count:
        raise BusinessError(
            "E_SCHEMA",
            "segment count differs from shot declaration",
            {"header": segment_count, "meta": len(meta.get("shots", []))},
        )
    assumptions: list[dict[str, Any]] = []
    if metric_version < METRIC_ENCODING_VERSION:
        assumptions.append(
            {
                "kind": "metric_version_migration",
                "field": "metric_encoding_version",
                "from": metric_version,
                "to": METRIC_ENCODING_VERSION,
                "migration_table": METRIC_MIGRATION_TABLE,
                "semantic_uncertainty": False,
            }
        )
    model = _model_from_parts(
        "metric_binary",
        PROJECT_SCHEMA,
        _require_mapping(meta.get("content"), "content"),
        _require_mapping(meta.get("encoding_params"), "encoding_params"),
        _require_list(meta.get("shots"), "shots"),
        _require_mapping(meta.get("metric_policy"), "metric_policy"),
        frames,
        assumptions=assumptions,
        provenance={
            "path": path.name,
            "format": BINARY_SCHEMA,
            "endian": endian,
            "index_width": index_width,
            "metric_encoding_version": metric_version,
            "field_bitmap": field_bitmap,
            "record_size": record_size,
        },
    )
    return model
