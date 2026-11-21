"""v1 fixed-width text, v2 JSON and v3 partitioned binary exchange codecs."""

from __future__ import annotations

import json
import struct
from decimal import Decimal
from pathlib import Path
from typing import Any, Iterable

from .constants import (
    ERROR_JSON_INVALID,
    ERROR_SIDECAR_STALE,
    ERROR_VERSION_LOSSY,
    V3_FEATURES,
    V3_MAGIC,
    V3_VERSION,
)
from .errors import VersionValidationError
from .model import ArbitrationConfig
from .numeric import (
    atomic_write_bytes,
    atomic_write_text,
    decimal_text,
    fingerprint,
    quantize_fixed,
    read_json,
    sha256_bytes,
)


V1_TICKET_WIDTH = 16
V1_UPGRADE_WIDTH = 1
V1_PRIORITY_WIDTH = 2
V1_SCORE_WIDTH = 9
V1_MISSING_WIDTH = 1
V1_LINE_WIDTH = (
    V1_TICKET_WIDTH
    + 1
    + V1_UPGRADE_WIDTH
    + 1
    + V1_PRIORITY_WIDTH
    + 1
    + V1_SCORE_WIDTH
    + 1
    + V1_MISSING_WIDTH
)
V1_SIDECAR_SCHEMA = "arbitration-v1-sidecar/v1"
V2_SCHEMA = "arbitration-v2"
V3_OFFSET_STRUCT = struct.Struct("<QI")
V3_SCORE_STRUCT = struct.Struct("<q")


def normalize_ir_record(record: dict, config: ArbitrationConfig) -> dict:
    if not isinstance(record, dict):
        raise VersionValidationError(ERROR_JSON_INVALID, "IR record must be an object")
    ticket_id = record.get("ticket_id")
    if not isinstance(ticket_id, str) or not ticket_id:
        raise VersionValidationError(ERROR_JSON_INVALID, "IR ticket_id must be non-empty")
    upgrade_code = record.get("upgrade_code")
    if isinstance(upgrade_code, bool):
        upgrade_code = int(upgrade_code)
    if upgrade_code not in (0, 1):
        raise VersionValidationError(ERROR_JSON_INVALID, f"invalid upgrade_code for {ticket_id}")
    priority_level = record.get("priority_level")
    if isinstance(priority_level, bool) or not isinstance(priority_level, int) or priority_level < 0:
        raise VersionValidationError(ERROR_JSON_INVALID, f"invalid priority_level for {ticket_id}")
    priority_name = record.get("priority_name", f"P{priority_level}")
    if not isinstance(priority_name, str) or not priority_name:
        raise VersionValidationError(ERROR_JSON_INVALID, f"invalid priority_name for {ticket_id}")
    score_fixed = record.get("score_fixed")
    if isinstance(score_fixed, bool) or not isinstance(score_fixed, int):
        if record.get("score") is not None:
            score_fixed = quantize_fixed(record["score"], config.scale, config.quantization_units)
        else:
            raise VersionValidationError(ERROR_JSON_INVALID, f"invalid score_fixed for {ticket_id}")
    if score_fixed < 0:
        raise VersionValidationError(ERROR_JSON_INVALID, f"negative score_fixed for {ticket_id}")
    missing = record.get("missing", 0)
    if missing not in (0, 1, False, True):
        raise VersionValidationError(ERROR_JSON_INVALID, f"invalid missing bit for {ticket_id}")
    reason_codes = record.get("reason_codes", [])
    if not isinstance(reason_codes, (list, tuple)) or not all(isinstance(code, str) for code in reason_codes):
        raise VersionValidationError(ERROR_JSON_INVALID, f"invalid reason_codes for {ticket_id}")
    precision = record.get("precision")
    if not isinstance(precision, str) or not precision:
        precision = f"fixed_{config.fixed_decimal_places}"
    boundary = bool(record.get("boundary", False))
    return {
        "ticket_id": ticket_id,
        "upgrade_code": int(upgrade_code),
        "priority_level": priority_level,
        "priority_name": priority_name,
        "score_fixed": score_fixed,
        "score": decimal_text(Decimal(score_fixed) / Decimal(config.scale)),
        "precision": precision,
        "reason_codes": list(reason_codes),
        "missing": int(missing),
        "boundary": boundary,
    }


def decision_to_ir(decision: dict, config: ArbitrationConfig) -> dict:
    return normalize_ir_record(
        {
            "ticket_id": decision["ticket_id"],
            "upgrade_code": decision["upgrade_code"],
            "priority_level": decision["priority_level"],
            "priority_name": decision["priority_name"],
            "score_fixed": decision["score_fixed"],
            "precision": decision["precision"],
            "reason_codes": decision["reason_codes"],
            "missing": decision.get("missing", 0),
            "boundary": decision.get("boundary", False),
        },
        config,
    )


def _v1_fields(record: dict, config: ArbitrationConfig) -> dict:
    ticket_id = record["ticket_id"]
    if len(ticket_id) > V1_TICKET_WIDTH or any(char in ticket_id for char in "|\r\n"):
        raise VersionValidationError(ERROR_VERSION_LOSSY, f"ticket id {ticket_id!r} cannot fit v1 fixed width")
    if not ticket_id.isascii():
        raise VersionValidationError(ERROR_VERSION_LOSSY, f"ticket id {ticket_id!r} is not ASCII")
    missing = int(record["missing"])
    score_fixed = int(record["score_fixed"])
    if score_fixed < 0:
        raise VersionValidationError(ERROR_JSON_INVALID, "v1 cannot represent a negative score")
    # The task specifies truncation residual, so non-negative scores use floor.
    score_milli = 0 if missing else (score_fixed * 1000) // config.scale
    if score_milli >= 10 ** V1_SCORE_WIDTH:
        raise VersionValidationError(ERROR_VERSION_LOSSY, f"score for {ticket_id} exceeds v1 field width")
    represented_fixed = quantize_fixed(
        Decimal(score_milli) / Decimal(1000), config.scale, config.quantization_units
    )
    residual_fixed = score_fixed - represented_fixed
    unrepresentable: list[str] = []
    if residual_fixed:
        unrepresentable.append("score_truncation_residual")
    if record["reason_codes"]:
        unrepresentable.append("reason_codes")
    if record["precision"] != "v1_milli":
        unrepresentable.append("precision_label")
    if record["priority_name"] != f"P{record['priority_level']}":
        unrepresentable.append("priority_name")
    if record["boundary"]:
        unrepresentable.append("boundary_marker")
    return {
        "ticket_id": ticket_id,
        "upgrade_code": int(record["upgrade_code"]),
        "priority_level": int(record["priority_level"]),
        "score_milli": int(score_milli),
        "missing": missing,
        "represented_fixed": represented_fixed,
        "residual_fixed": residual_fixed,
        "residual": decimal_text(Decimal(residual_fixed) / Decimal(config.scale)),
        "unrepresentable": unrepresentable,
    }


def v1_degradation_detail(record: dict, config: ArbitrationConfig, *, source_version: str) -> dict | None:
    fields = _v1_fields(record, config)
    fields_lost = list(fields["unrepresentable"])
    if source_version == "v3" and record["score_fixed"] % max(1, config.scale // 1000) != 0:
        fields_lost.append("v3_fixed_block_to_v1_milli")
    if not fields_lost:
        return None
    return {
        "ticket_id": record["ticket_id"],
        "code": ERROR_VERSION_LOSSY,
        "source_version": source_version,
        "target_version": "v1",
        "fields": sorted(set(fields_lost)),
        "score_fixed": record["score_fixed"],
        "score_milli": fields["score_milli"],
        "truncation_residual_fixed": fields["residual_fixed"],
        "truncation_residual": fields["residual"],
        "precision": record["precision"],
        "reason_codes": list(record["reason_codes"]),
        "priority_name": record["priority_name"],
    }


def _v1_line(fields: dict) -> str:
    return (
        f"{fields['ticket_id']:<{V1_TICKET_WIDTH}}|"
        f"{fields['upgrade_code']:0{V1_UPGRADE_WIDTH}d}|"
        f"{fields['priority_level']:0{V1_PRIORITY_WIDTH}d}|"
        f"{fields['score_milli']:0{V1_SCORE_WIDTH}d}|"
        f"{fields['missing']:0{V1_MISSING_WIDTH}d}"
    )


def write_v1(
    records: Iterable[dict],
    path: str | Path,
    config: ArbitrationConfig,
    *,
    sidecar_path: str | Path | None = None,
    temp_dir: str | Path | None = None,
) -> dict:
    normalized = [normalize_ir_record(record, config) for record in records]
    normalized.sort(key=lambda record: record["ticket_id"])
    lines: list[str] = []
    sidecar_records: dict[str, dict] = {}
    degradations: list[dict] = []
    for record in normalized:
        fields = _v1_fields(record, config)
        lines.append(_v1_line(fields))
        detail = v1_degradation_detail(record, config, source_version="ir")
        if detail is not None:
            degradations.append(detail)
        sidecar_records[record["ticket_id"]] = {
            "upgrade_code": record["upgrade_code"],
            "priority_level": record["priority_level"],
            "priority_name": record["priority_name"],
            "score_milli": fields["score_milli"],
            "score_fixed": record["score_fixed"],
            "precision": record["precision"],
            "reason_codes": list(record["reason_codes"]),
            "missing": int(record["missing"]),
            "boundary": bool(record["boundary"]),
            "truncation_residual_fixed": fields["residual_fixed"],
            "truncation_residual": fields["residual"],
            "unrepresentable": list(fields["unrepresentable"]),
        }
    text = "".join(line + "\n" for line in lines)
    payload = text.encode("utf-8")
    atomic_write_bytes(path, payload, temp_dir=temp_dir)
    actual_sidecar = Path(sidecar_path) if sidecar_path is not None else Path(str(path) + ".meta.json")
    sidecar = {
        "schema": V1_SIDECAR_SCHEMA,
        "v1_sha256": sha256_bytes(payload),
        "v1_line_width": V1_LINE_WIDTH,
        "fixed_scale": config.scale,
        "records": {key: sidecar_records[key] for key in sorted(sidecar_records)},
        "degradations": degradations,
    }
    atomic_write_text(
        actual_sidecar,
        json.dumps(sidecar, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        temp_dir=temp_dir,
    )
    return {
        "path": str(path),
        "sidecar_path": str(actual_sidecar),
        "record_count": len(normalized),
        "v1_line_width": V1_LINE_WIDTH,
        "v1_sha256": sidecar["v1_sha256"],
        "degradations": degradations,
    }


def _parse_v1_line(line: str, line_no: int) -> dict:
    if len(line) != V1_LINE_WIDTH:
        raise VersionValidationError(
            ERROR_JSON_INVALID,
            f"v1 line {line_no} has width {len(line)}, expected {V1_LINE_WIDTH}",
        )
    parts = line.split("|")
    if len(parts) != 5:
        raise VersionValidationError(ERROR_JSON_INVALID, f"v1 line {line_no} has invalid delimiters")
    ticket_raw, upgrade_raw, priority_raw, score_raw, missing_raw = parts
    if not ticket_raw.strip() or ticket_raw != ticket_raw.rstrip().ljust(V1_TICKET_WIDTH):
        raise VersionValidationError(ERROR_JSON_INVALID, f"v1 line {line_no} ticket padding invalid")
    if not all(value.isdigit() for value in (upgrade_raw, priority_raw, score_raw, missing_raw)):
        raise VersionValidationError(ERROR_JSON_INVALID, f"v1 line {line_no} contains non-integer fields")
    ticket_id = ticket_raw.rstrip()
    upgrade_code = int(upgrade_raw)
    priority_level = int(priority_raw)
    score_milli = int(score_raw)
    missing = int(missing_raw)
    if upgrade_code not in (0, 1) or missing not in (0, 1):
        raise VersionValidationError(ERROR_JSON_INVALID, f"v1 line {line_no} has invalid 0/1 field")
    return {
        "ticket_id": ticket_id,
        "upgrade_code": upgrade_code,
        "priority_level": priority_level,
        "score_milli": score_milli,
        "missing": missing,
    }


def read_v1(
    path: str | Path,
    config: ArbitrationConfig,
    *,
    sidecar_path: str | Path | None = None,
) -> list[dict]:
    payload = Path(path).read_bytes()
    text = payload.decode("utf-8")
    raw_lines = text.splitlines()
    parsed: list[dict] = []
    seen: set[str] = set()
    for line_no, line in enumerate(raw_lines, start=1):
        record = _parse_v1_line(line, line_no)
        if record["ticket_id"] in seen:
            raise VersionValidationError(ERROR_JSON_INVALID, f"duplicate v1 ticket {record['ticket_id']}")
        seen.add(record["ticket_id"])
        parsed.append(record)

    actual_sidecar = Path(sidecar_path) if sidecar_path is not None else Path(str(path) + ".meta.json")
    sidecar: dict | None = None
    if actual_sidecar.exists():
        sidecar = read_json(actual_sidecar)
        if not isinstance(sidecar, dict) or sidecar.get("schema") != V1_SIDECAR_SCHEMA:
            raise VersionValidationError(ERROR_SIDECAR_STALE, "v1 sidecar schema is invalid")
        if sidecar.get("v1_sha256") != sha256_bytes(payload):
            raise VersionValidationError(ERROR_SIDECAR_STALE, "v1 sidecar does not match v1 payload")
        if int(sidecar.get("fixed_scale", 0)) != config.scale:
            raise VersionValidationError(ERROR_SIDECAR_STALE, "v1 sidecar fixed_scale is stale")

    restored: list[dict] = []
    for record in parsed:
        ticket_id = record["ticket_id"]
        side_record = sidecar.get("records", {}).get(ticket_id, {}) if sidecar else {}
        if side_record:
            if (
                int(side_record.get("upgrade_code", -1)) != record["upgrade_code"]
                or int(side_record.get("priority_level", -1)) != record["priority_level"]
                or int(side_record.get("score_milli", -1)) != record["score_milli"]
                or int(side_record.get("missing", -1)) != record["missing"]
            ):
                raise VersionValidationError(ERROR_SIDECAR_STALE, f"v1 sidecar mismatch for {ticket_id}")
            score_fixed = int(side_record["score_fixed"])
            precision = str(side_record["precision"])
            reason_codes = list(side_record.get("reason_codes", []))
            priority_name = str(side_record.get("priority_name", f"P{record['priority_level']}"))
            boundary = bool(side_record.get("boundary", False))
        else:
            score_fixed = quantize_fixed(
                Decimal(record["score_milli"]) / Decimal(1000), config.scale, config.quantization_units
            )
            precision = "v1_milli"
            reason_codes = []
            priority_name = f"P{record['priority_level']}"
            boundary = False
        restored.append(
            {
                "ticket_id": ticket_id,
                "upgrade_code": record["upgrade_code"],
                "priority_level": record["priority_level"],
                "priority_name": priority_name,
                "score_fixed": score_fixed,
                "score": decimal_text(Decimal(score_fixed) / Decimal(config.scale)),
                "precision": precision,
                "reason_codes": reason_codes,
                "missing": record["missing"],
                "boundary": boundary,
            }
        )
    restored.sort(key=lambda record: record["ticket_id"])
    return restored


def write_v2(
    records: Iterable[dict],
    path: str | Path,
    config: ArbitrationConfig,
    *,
    temp_dir: str | Path | None = None,
) -> dict:
    normalized = [normalize_ir_record(record, config) for record in records]
    normalized.sort(key=lambda record: record["ticket_id"])
    output_records = []
    for record in normalized:
        output_records.append(
            {
                "ticket_id": record["ticket_id"],
                "upgrade_code": record["upgrade_code"],
                "priority": record["priority_name"],
                "priority_level": record["priority_level"],
                "score": None if record["missing"] else float(Decimal(record["score_fixed"]) / Decimal(config.scale)),
                "precision": record["precision"],
                "reason_codes": list(record["reason_codes"]),
                "missing": record["missing"],
                "boundary": record["boundary"],
            }
        )
    document = {
        "schema": V2_SCHEMA,
        "fixed_decimal_places": config.fixed_decimal_places,
        "fixed_scale": config.scale,
        "quantization_units": config.quantization_units,
        "records": output_records,
    }
    text = json.dumps(document, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n"
    atomic_write_text(path, text, temp_dir=temp_dir)
    return {"path": str(path), "record_count": len(normalized), "sha256": sha256_bytes(text.encode("utf-8"))}


def read_v2(path: str | Path, config: ArbitrationConfig) -> list[dict]:
    try:
        document = read_json(path)
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise VersionValidationError(ERROR_JSON_INVALID, f"invalid v2 JSON: {exc}") from exc
    if not isinstance(document, dict) or document.get("schema") != V2_SCHEMA:
        raise VersionValidationError(ERROR_JSON_INVALID, "v2 schema is invalid")
    scale = document.get("fixed_scale", config.scale)
    quantum = document.get("quantization_units", config.quantization_units)
    if isinstance(scale, bool) or not isinstance(scale, int) or scale <= 0:
        raise VersionValidationError(ERROR_JSON_INVALID, "v2 fixed_scale is invalid")
    if isinstance(quantum, bool) or not isinstance(quantum, int) or quantum <= 0:
        raise VersionValidationError(ERROR_JSON_INVALID, "v2 quantization_units is invalid")
    raw_records = document.get("records")
    if not isinstance(raw_records, list):
        raise VersionValidationError(ERROR_JSON_INVALID, "v2 records must be an array")
    output: list[dict] = []
    seen: set[str] = set()
    for raw in raw_records:
        if not isinstance(raw, dict):
            raise VersionValidationError(ERROR_JSON_INVALID, "v2 record must be an object")
        ticket_id = raw.get("ticket_id")
        if not isinstance(ticket_id, str) or not ticket_id or ticket_id in seen:
            raise VersionValidationError(ERROR_JSON_INVALID, "v2 ticket_id is invalid or duplicated")
        seen.add(ticket_id)
        missing = int(raw.get("missing", 1 if raw.get("score") is None else 0))
        if missing not in (0, 1):
            raise VersionValidationError(ERROR_JSON_INVALID, f"v2 missing flag invalid for {ticket_id}")
        if missing:
            score_fixed = 0
        else:
            if raw.get("score") is None:
                raise VersionValidationError(ERROR_JSON_INVALID, f"v2 score missing for {ticket_id}")
            try:
                score_fixed = quantize_fixed(raw["score"], scale, quantum)
            except (ValueError, TypeError) as exc:
                raise VersionValidationError(ERROR_JSON_INVALID, f"v2 score invalid for {ticket_id}") from exc
        priority_level = raw.get("priority_level", 0)
        if isinstance(priority_level, bool) or not isinstance(priority_level, int) or priority_level < 0:
            raise VersionValidationError(ERROR_JSON_INVALID, f"v2 priority_level invalid for {ticket_id}")
        priority_name = raw.get("priority", f"P{priority_level}")
        reason_codes = raw.get("reason_codes", [])
        if not isinstance(reason_codes, list) or not all(isinstance(code, str) for code in reason_codes):
            raise VersionValidationError(ERROR_JSON_INVALID, f"v2 reason_codes invalid for {ticket_id}")
        upgrade_code = raw.get("upgrade_code")
        if upgrade_code not in (0, 1):
            raise VersionValidationError(ERROR_JSON_INVALID, f"v2 upgrade_code invalid for {ticket_id}")
        output.append(
            normalize_ir_record(
                {
                    "ticket_id": ticket_id,
                    "upgrade_code": upgrade_code,
                    "priority_level": priority_level,
                    "priority_name": priority_name,
                    "score_fixed": score_fixed,
                    "precision": raw.get("precision", "v2_float"),
                    "reason_codes": reason_codes,
                    "missing": missing,
                    "boundary": bool(raw.get("boundary", False)),
                },
                config,
            )
        )
    output.sort(key=lambda record: record["ticket_id"])
    return output


def _encode_feature_name(name: str) -> bytes:
    encoded = name.encode("utf-8")
    if len(encoded) > 65535:
        raise VersionValidationError(ERROR_VERSION_LOSSY, f"v3 feature name too long: {name}")
    return struct.pack("<H", len(encoded)) + encoded


def write_v3(
    records: Iterable[dict],
    path: str | Path,
    config: ArbitrationConfig,
    *,
    temp_dir: str | Path | None = None,
) -> dict:
    normalized = [normalize_ir_record(record, config) for record in records]
    normalized.sort(key=lambda record: record["ticket_id"])
    feature_table = b"".join(_encode_feature_name(name) for name in V3_FEATURES)
    header_prefix = V3_MAGIC + struct.pack(
        "<HHIII", V3_VERSION, 0, len(V3_FEATURES), config.scale, len(normalized)
    )
    header = header_prefix + feature_table
    offset_table_size = V3_OFFSET_STRUCT.size * len(normalized)
    score_block_size = V3_SCORE_STRUCT.size * len(normalized)
    payload_base = len(header) + offset_table_size + score_block_size
    metadata_payloads: list[bytes] = []
    for record in normalized:
        metadata = {name: record[name] for name in V3_FEATURES if name != "score_fixed"}
        metadata_payloads.append(
            json.dumps(metadata, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
        )
    offsets = bytearray()
    cursor = payload_base
    for payload in metadata_payloads:
        offsets.extend(V3_OFFSET_STRUCT.pack(cursor, len(payload)))
        cursor += len(payload)
    score_block = b"".join(V3_SCORE_STRUCT.pack(record["score_fixed"]) for record in normalized)
    payload = header + bytes(offsets) + score_block + b"".join(metadata_payloads)
    atomic_write_bytes(path, payload, temp_dir=temp_dir)
    return {
        "path": str(path),
        "record_count": len(normalized),
        "sha256": sha256_bytes(payload),
        "feature_names": list(V3_FEATURES),
        "offset_table_offset": len(header),
        "score_block_offset": len(header) + offset_table_size,
        "score_block_size": score_block_size,
    }


def read_v3(path: str | Path, config: ArbitrationConfig) -> list[dict]:
    payload = Path(path).read_bytes()
    if len(payload) < 24 or payload[:8] != V3_MAGIC:
        raise VersionValidationError(ERROR_JSON_INVALID, "v3 magic is invalid")
    version, flags, feature_count, scale, record_count = struct.unpack_from("<HHIII", payload, 8)
    if version != V3_VERSION or flags != 0:
        raise VersionValidationError(ERROR_JSON_INVALID, "v3 header version/flags are unsupported")
    if feature_count > 256 or record_count > 10_000_000:
        raise VersionValidationError(ERROR_JSON_INVALID, "v3 header count is unreasonable")
    cursor = 24
    features: list[str] = []
    for _ in range(feature_count):
        if cursor + 2 > len(payload):
            raise VersionValidationError(ERROR_JSON_INVALID, "v3 feature table truncated")
        (length,) = struct.unpack_from("<H", payload, cursor)
        cursor += 2
        if cursor + length > len(payload):
            raise VersionValidationError(ERROR_JSON_INVALID, "v3 feature name truncated")
        try:
            features.append(payload[cursor : cursor + length].decode("utf-8"))
        except UnicodeDecodeError as exc:
            raise VersionValidationError(ERROR_JSON_INVALID, "v3 feature name is not UTF-8") from exc
        cursor += length
    if len(set(features)) != len(features) or "score_fixed" not in features:
        raise VersionValidationError(ERROR_JSON_INVALID, "v3 feature table is invalid")
    offset_table_pos = cursor
    score_block_pos = offset_table_pos + V3_OFFSET_STRUCT.size * record_count
    payload_start = score_block_pos + V3_SCORE_STRUCT.size * record_count
    if payload_start > len(payload):
        raise VersionValidationError(ERROR_JSON_INVALID, "v3 blocks truncated")
    offsets: list[tuple[int, int]] = []
    ranges: list[tuple[int, int, int]] = []
    for index in range(record_count):
        offset, length = V3_OFFSET_STRUCT.unpack_from(payload, offset_table_pos + index * V3_OFFSET_STRUCT.size)
        if offset < payload_start or offset + length > len(payload):
            raise VersionValidationError(ERROR_JSON_INVALID, "v3 record offset is out of bounds")
        offsets.append((offset, length))
        ranges.append((offset, offset + length, index))
    ranges.sort()
    for previous, current in zip(ranges, ranges[1:]):
        if current[0] < previous[1]:
            raise VersionValidationError(ERROR_JSON_INVALID, "v3 record ranges overlap")
    scores = [
        V3_SCORE_STRUCT.unpack_from(payload, score_block_pos + index * V3_SCORE_STRUCT.size)[0]
        for index in range(record_count)
    ]
    output: list[dict] = []
    seen: set[str] = set()
    for index, (offset, length) in enumerate(offsets):
        try:
            metadata = json.loads(payload[offset : offset + length].decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise VersionValidationError(ERROR_JSON_INVALID, "v3 record metadata is invalid JSON") from exc
        if not isinstance(metadata, dict):
            raise VersionValidationError(ERROR_JSON_INVALID, "v3 record metadata must be an object")
        metadata["score_fixed"] = scores[index]
        ticket_id = metadata.get("ticket_id")
        if not isinstance(ticket_id, str) or not ticket_id or ticket_id in seen:
            raise VersionValidationError(ERROR_JSON_INVALID, "v3 ticket_id is invalid or duplicated")
        seen.add(ticket_id)
        output.append(normalize_ir_record(metadata, config))
    output.sort(key=lambda record: record["ticket_id"])
    return output


def read_version(
    path: str | Path,
    version: str,
    config: ArbitrationConfig,
    *,
    sidecar_path: str | Path | None = None,
) -> list[dict]:
    version = version.lower()
    if version == "v1":
        return read_v1(path, config, sidecar_path=sidecar_path)
    if version == "v2":
        return read_v2(path, config)
    if version == "v3":
        return read_v3(path, config)
    raise VersionValidationError(ERROR_JSON_INVALID, f"unknown version {version!r}")


def write_version(
    records: Iterable[dict],
    version: str,
    path: str | Path,
    config: ArbitrationConfig,
    *,
    sidecar_path: str | Path | None = None,
    temp_dir: str | Path | None = None,
) -> dict:
    version = version.lower()
    if version == "v1":
        return write_v1(records, path, config, sidecar_path=sidecar_path, temp_dir=temp_dir)
    if version == "v2":
        return write_v2(records, path, config, temp_dir=temp_dir)
    if version == "v3":
        return write_v3(records, path, config, temp_dir=temp_dir)
    raise VersionValidationError(ERROR_JSON_INVALID, f"unknown version {version!r}")


def conversion_report(
    *,
    source_version: str,
    target_version: str,
    source_records: list[dict],
    target_records: list[dict],
    output_path: str,
    degradations: list[dict] | None = None,
    sidecar_path: str | None = None,
) -> dict:
    degradation_details = degradations or []
    return {
        "schema": "version-conversion-report/v1",
        "source_version": source_version,
        "target_version": target_version,
        "source_count": len(source_records),
        "target_count": len(target_records),
        "output_path": output_path,
        "sidecar_path": sidecar_path,
        "codes": [ERROR_VERSION_LOSSY] if degradation_details else [],
        "degradation_details": degradation_details,
    }


def convert_file(
    source_path: str | Path,
    source_version: str,
    target_path: str | Path,
    target_version: str,
    config: ArbitrationConfig,
    *,
    source_sidecar_path: str | Path | None = None,
    target_sidecar_path: str | Path | None = None,
    temp_dir: str | Path | None = None,
) -> dict:
    source_records = read_version(
        source_path, source_version, config, sidecar_path=source_sidecar_path
    )
    degradation_details: list[dict] = []
    if target_version.lower() == "v1":
        for record in source_records:
            detail = v1_degradation_detail(record, config, source_version=source_version.lower())
            if detail is not None:
                degradation_details.append(detail)
    write_info = write_version(
        source_records,
        target_version,
        target_path,
        config,
        sidecar_path=target_sidecar_path,
        temp_dir=temp_dir,
    )
    target_records = read_version(
        target_path,
        target_version,
        config,
        sidecar_path=write_info.get("sidecar_path"),
    )
    return conversion_report(
        source_version=source_version.lower(),
        target_version=target_version.lower(),
        source_records=source_records,
        target_records=target_records,
        output_path=str(target_path),
        degradations=degradation_details,
        sidecar_path=write_info.get("sidecar_path"),
    )


def roundtrip_consistency(left: list[dict], right: list[dict]) -> dict:
    left_index = {record["ticket_id"]: record for record in left}
    right_index = {record["ticket_id"]: record for record in right}
    missing_left = sorted(ticket_id for ticket_id, record in left_index.items() if int(record["missing"]))
    missing_right = sorted(ticket_id for ticket_id, record in right_index.items() if int(record["missing"]))
    upgrade_mismatches = sorted(
        ticket_id
        for ticket_id in set(left_index) & set(right_index)
        if int(left_index[ticket_id]["upgrade_code"]) != int(right_index[ticket_id]["upgrade_code"])
    )
    return {
        "count_equal": len(left) == len(right),
        "missing_positions_equal": missing_left == missing_right,
        "upgrade_codes_equal": not upgrade_mismatches,
        "ticket_ids_equal": sorted(left_index) == sorted(right_index),
        "missing_positions": missing_left,
        "upgrade_mismatches": upgrade_mismatches,
        "equal": (
            len(left) == len(right)
            and missing_left == missing_right
            and not upgrade_mismatches
            and sorted(left_index) == sorted(right_index)
        ),
    }
