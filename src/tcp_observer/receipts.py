from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from .canonical import canonical_text
from .errors import FormatError

O1_WIDTH = 42
O1_PREFIX = "O1"
DEFAULT_SCALE = 1024
DEFAULT_SSTHRESH = 16
DEFAULT_SRTT_MS = 24
DEFAULT_PEER_ID = "legacy-o1"


@dataclass(frozen=True)
class NormalizationDefaults:
    scale: int = DEFAULT_SCALE
    ssthresh: int = DEFAULT_SSTHRESH
    srtt_ms: int = DEFAULT_SRTT_MS
    peer_id: str = DEFAULT_PEER_ID
    duration_ms: int = 0
    source_basis: str = "built-in normalization rule v1"


def _format_error(message: str, line_number: int, field_name: str | None = None, **details: Any) -> FormatError:
    return FormatError(message, field_name=field_name, line=line_number, details=details)


def encode_o1(start_seq: int, end_seq: int, ack_count: int, cwnd: int) -> str:
    return f"O1{start_seq:012d}{end_seq:012d}{ack_count:08d}{cwnd:08d}"


def decode_o1_line(line: str, line_number: int = 1) -> dict[str, Any]:
    raw = line.rstrip("\r\n")
    if not raw.startswith(O1_PREFIX):
        raise _format_error("unknown receipt version", line_number, "version", prefix=raw[:2])
    if len(raw) < O1_WIDTH:
        raise _format_error(
            "o1 fixed-width field is shorter than required width",
            line_number,
            "fixed_width",
            expected_width=O1_WIDTH,
            actual_width=len(raw),
        )
    if len(raw) != O1_WIDTH:
        raise _format_error(
            "o1 fixed-width line has invalid trailing width",
            line_number,
            "fixed_width",
            expected_width=O1_WIDTH,
            actual_width=len(raw),
        )
    fields = {
        "start_seq": raw[2:14],
        "end_seq": raw[14:26],
        "ack_count": raw[26:34],
        "cwnd": raw[34:42],
    }
    for name, value in fields.items():
        if not value.isdigit():
            raise _format_error(f"o1 field {name} must contain digits", line_number, name, value=value)
    return {
        "format_version": 1,
        "start_seq": int(fields["start_seq"]),
        "end_seq": int(fields["end_seq"]),
        "ack_count": int(fields["ack_count"]),
        "cwnd": int(fields["cwnd"]),
    }


def _require_fields(obj: dict[str, Any], required: Iterable[str], line_number: int, version: int) -> None:
    for name in required:
        if name not in obj:
            raise _format_error(
                f"o{version} receipt is missing required field {name}",
                line_number,
                name,
                version=version,
            )


def _require_int(obj: dict[str, Any], name: str, line_number: int, *, minimum: int | None = None) -> int:
    value = obj[name]
    if isinstance(value, bool) or not isinstance(value, int):
        raise _format_error(f"field {name} must be an integer", line_number, name, actual_type=type(value).__name__)
    if minimum is not None and value < minimum:
        raise _format_error(f"field {name} must be >= {minimum}", line_number, name, value=value)
    return value


def decode_json_line(line: str, line_number: int = 1) -> dict[str, Any]:
    raw = line.strip()
    try:
        obj = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise _format_error(f"invalid JSON receipt: {exc.msg}", exc.lineno, "$", column=exc.colno) from exc
    if not isinstance(obj, dict):
        raise _format_error("receipt must be a JSON object", line_number, "$")
    version = obj.get("format_version", obj.get("v"))
    if version == 2:
        _require_fields(
            obj,
            ["start_seq", "end_seq", "ack_count", "cwnd", "ssthresh", "srtt_ms", "peer_id", "duration_ms"],
            line_number,
            2,
        )
        for name in ["start_seq", "end_seq", "ack_count", "cwnd", "ssthresh", "srtt_ms", "duration_ms"]:
            _require_int(obj, name, line_number)
        if obj["duration_ms"] < 0:
            raise _format_error("duration_ms must not be negative", line_number, "duration_ms", value=obj["duration_ms"])
        if not isinstance(obj["peer_id"], str) or not obj["peer_id"]:
            raise _format_error("peer_id must be a non-empty string", line_number, "peer_id")
        return {"format_version": 2, **{k: obj[k] for k in [
            "start_seq", "end_seq", "ack_count", "cwnd", "ssthresh", "srtt_ms", "peer_id", "duration_ms"
        ]}}
    if version == 3:
        _require_fields(
            obj,
            [
                "start_seq", "end_seq", "ack_count", "cwnd", "ssthresh", "srtt_ms",
                "peer_id", "duration_ms", "scale", "error_bound_scaled", "affected",
            ],
            line_number,
            3,
        )
        for name in [
            "start_seq", "end_seq", "ack_count", "cwnd", "ssthresh", "srtt_ms",
            "duration_ms", "scale", "error_bound_scaled",
        ]:
            minimum = 1 if name == "scale" else 0
            _require_int(obj, name, line_number, minimum=minimum)
        if obj["duration_ms"] < 0:
            raise _format_error("duration_ms must not be negative", line_number, "duration_ms", value=obj["duration_ms"])
        affected = obj["affected"]
        if not isinstance(affected, list) or not all(isinstance(item, bool) for item in affected):
            raise _format_error("affected must be an array of booleans", line_number, "affected")
        expected_items = obj["end_seq"] - obj["start_seq"] + 1
        if len(affected) != expected_items:
            raise _format_error(
                "affected length must equal event sequence interval length",
                line_number,
                "affected",
                expected_length=expected_items,
                actual_length=len(affected),
            )
        if not isinstance(obj["peer_id"], str) or not obj["peer_id"]:
            raise _format_error("peer_id must be a non-empty string", line_number, "peer_id")
        return {
            "format_version": 3,
            **{k: obj[k] for k in [
                "start_seq", "end_seq", "ack_count", "cwnd", "ssthresh", "srtt_ms",
                "peer_id", "duration_ms", "scale", "error_bound_scaled", "affected",
            ]},
        }
    raise _format_error("unknown receipt version", line_number, "format_version", version=version)


def decode_receipt_line(line: str, line_number: int = 1) -> dict[str, Any]:
    if line.startswith(O1_PREFIX):
        return decode_o1_line(line, line_number)
    if len(line) >= 2 and line[0] == "O" and line[1].isdigit():
        raise _format_error(
            "unknown receipt version",
            line_number,
            "format_version",
            prefix=line[:2],
        )
    return decode_json_line(line, line_number)


def default_error_bound_scaled(item_count: int, scale: int) -> int:
    # Conservative pre-run bound: one fixed-point unit per event plus one RTT
    # recurrence unit and one threshold halving unit.
    return max(1, item_count + 2) * max(1, scale // DEFAULT_SCALE)


def normalize_receipt(raw: dict[str, Any], line_number: int = 1, defaults: NormalizationDefaults | None = None) -> dict[str, Any]:
    defaults = defaults or NormalizationDefaults()
    version = f"o{raw['format_version']}"
    start = int(raw["start_seq"])
    end = int(raw["end_seq"])
    item_count = max(1, end - start + 1)
    filled: set[str] = set()
    basis: dict[str, str] = {}

    scale = int(raw.get("scale", defaults.scale))
    if "scale" not in raw:
        filled.add("scale")
        basis["scale"] = f"{defaults.source_basis}:default_scale={defaults.scale}"

    ssthresh = int(raw.get("ssthresh", defaults.ssthresh))
    if "ssthresh" not in raw:
        filled.add("ssthresh")
        basis["ssthresh"] = f"{defaults.source_basis}:default_ssthresh={defaults.ssthresh}_packets"

    srtt_ms = int(raw.get("srtt_ms", defaults.srtt_ms))
    if "srtt_ms" not in raw:
        filled.add("srtt_ms")
        basis["srtt_ms"] = f"{defaults.source_basis}:default_srtt_ms={defaults.srtt_ms}"

    peer_id = raw.get("peer_id", defaults.peer_id)
    if "peer_id" not in raw:
        filled.add("peer_id")
        basis["peer_id"] = f"{defaults.source_basis}:legacy_peer={defaults.peer_id}"

    duration_ms = int(raw.get("duration_ms", defaults.duration_ms))
    if "duration_ms" not in raw:
        filled.add("duration_ms")
        basis["duration_ms"] = f"{defaults.source_basis}:zero_for_legacy_fixed_width"

    error_bound = int(raw.get("error_bound_scaled", default_error_bound_scaled(item_count, scale)))
    if "error_bound_scaled" not in raw:
        filled.add("error_bound_scaled")
        basis["error_bound_scaled"] = "conservative_default:item_count_plus_two_fixed_point_units"

    affected = raw.get("affected")
    if affected is None:
        affected = [False for _ in range(item_count)]
        filled.add("affected")
        basis["affected"] = "default:all_false_because_legacy_receipt_has_no_item_marks"

    normalized = {
        "source_version": version,
        "source_line": line_number,
        "start_seq": start,
        "end_seq": end,
        "item_count": item_count,
        "ack_count": int(raw["ack_count"]),
        "cwnd": int(raw["cwnd"]),
        "cwnd_scaled": int(raw["cwnd"]) * scale,
        "ssthresh": ssthresh,
        "ssthresh_scaled": ssthresh * scale,
        "srtt_ms": srtt_ms,
        "srtt_scaled": srtt_ms * scale,
        "peer_id": peer_id,
        "duration_ms": duration_ms,
        "scale": scale,
        "error_bound_scaled": error_bound,
        "affected": affected,
        "filled_fields": sorted(filled),
        "fill_basis": {key: basis[key] for key in sorted(basis)},
    }
    return normalized


def decode_and_normalize_text(text: str, defaults: NormalizationDefaults | None = None) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for line_number, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        raw = decode_receipt_line(line, line_number)
        results.append(normalize_receipt(raw, line_number, defaults))
    return results


def decode_and_normalize_file(path: str | Path, defaults: NormalizationDefaults | None = None) -> list[dict[str, Any]]:
    return decode_and_normalize_text(Path(path).read_text(encoding="utf-8"), defaults)


def encode_receipt(record: dict[str, Any], version: int) -> str:
    if version == 1:
        return encode_o1(record["start_seq"], record["end_seq"], record["ack_count"], record["cwnd"])
    if version == 2:
        obj = {
            "format_version": 2,
            "start_seq": record["start_seq"],
            "end_seq": record["end_seq"],
            "ack_count": record["ack_count"],
            "cwnd": record["cwnd"],
            "ssthresh": record["ssthresh"],
            "srtt_ms": record["srtt_ms"],
            "peer_id": record["peer_id"],
            "duration_ms": record["duration_ms"],
        }
        return canonical_text(obj)
    if version == 3:
        obj = {
            "format_version": 3,
            "start_seq": record["start_seq"],
            "end_seq": record["end_seq"],
            "ack_count": record["ack_count"],
            "cwnd": record["cwnd"],
            "ssthresh": record["ssthresh"],
            "srtt_ms": record["srtt_ms"],
            "peer_id": record["peer_id"],
            "duration_ms": record["duration_ms"],
            "scale": record["scale"],
            "error_bound_scaled": record["error_bound_scaled"],
            "affected": record["affected"],
        }
        return canonical_text(obj)
    raise ValueError(f"unknown receipt version {version}")


def write_receipt_file(path: str | Path, records: list[dict[str, Any]], version: int) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [encode_receipt(record, version) for record in records]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path
