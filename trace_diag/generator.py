from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any, Iterable

from .canonical import canonical_bytes, sha256_bytes
from .errors import RecordError
from .spec import REQUIRED_FIELDS, ServiceSpec


def stable_u64(*parts: object) -> int:
    payload = "\x1f".join(str(part) for part in parts).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def _trace_id(service_id: str, seed: int, index: int) -> str:
    payload = f"trace\x1f{service_id}\x1f{seed}\x1f{index}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:32]


@dataclass(frozen=True)
class RecordStream:
    service_id: str
    records: tuple[dict[str, Any], ...]
    stream_bytes: bytes
    stream_digest: str


def generate_records(service: ServiceSpec) -> RecordStream:
    timestamp = int(service.fields["timestamp_ms"]["start"])
    max_step = int(service.fields["timestamp_ms"]["max_step_ms"])
    magnitude_min = int(service.fields["magnitude"]["min"])
    magnitude_max = int(service.fields["magnitude"]["max"])
    key_space = int(service.fields["client_key"]["key_space"])
    duration_min = int(service.fields["duration_us"]["min_us"])
    duration_max = int(service.fields["duration_us"]["max_us"])
    sampled_rate = int(service.fields["sampled"]["rate_bp"])

    magnitude_span = magnitude_max - magnitude_min + 1
    duration_span = duration_max - duration_min + 1
    records: list[dict[str, Any]] = []
    for index in range(service.record_count):
        timestamp_hash = stable_u64(service.seed, index, "timestamp_ms")
        if index == 0:
            timestamp = int(service.fields["timestamp_ms"]["start"])
        else:
            timestamp += timestamp_hash % (max_step + 1) if max_step else 0
        magnitude_hash = stable_u64(service.seed, index, "magnitude")
        key_hash = stable_u64(service.seed, index, "client_key")
        duration_hash = stable_u64(service.seed, index, "duration_us")
        sampled_hash = stable_u64(service.seed, index, "sampled")
        record = {
            "timestamp_ms": timestamp,
            "magnitude": magnitude_min + magnitude_hash % magnitude_span,
            "client_key": f"{service.id}:{key_hash % key_space:06d}",
            "duration_us": duration_min + duration_hash % duration_span,
            "trace_id": _trace_id(service.id, service.seed, index),
            "sampled": sampled_hash % 10_000 < sampled_rate,
        }
        records.append(record)

    stream_bytes = record_stream_bytes(records)
    return RecordStream(
        service_id=service.id,
        records=tuple(records),
        stream_bytes=stream_bytes,
        stream_digest=sha256_bytes(stream_bytes),
    )


def record_stream_bytes(records: Iterable[dict[str, Any]]) -> bytes:
    return b"\n".join(canonical_bytes(record) for record in records) + b"\n"


def validate_record_stream(service: ServiceSpec, records: Iterable[dict[str, Any]]) -> None:
    records_list = list(records)
    if len(records_list) != service.record_count:
        raise RecordError(
            f"$.services[{service.id}].record_count",
            f"expected {service.record_count} records, got {len(records_list)}",
            {"record_index": 0},
        )
    previous_timestamp: int | None = None
    magnitude_min = int(service.fields["magnitude"]["min"])
    magnitude_max = int(service.fields["magnitude"]["max"])
    duration_min = int(service.fields["duration_us"]["min_us"])
    duration_max = int(service.fields["duration_us"]["max_us"])
    required = set(REQUIRED_FIELDS)

    for index, record in enumerate(records_list):
        base = f"$.services[{service.id}].records[{index}]"
        if not isinstance(record, dict):
            raise RecordError(base, "record must be an object", {"record_index": index})
        missing = sorted(required - set(record))
        if missing:
            raise RecordError(
                f"{base}.{missing[0]}",
                "required record field is missing or null",
                {"record_index": index},
            )
        extra = sorted(set(record) - required)
        if extra:
            raise RecordError(
                f"{base}.{extra[0]}",
                "unknown record field",
                {"record_index": index},
            )
        timestamp = record["timestamp_ms"]
        if type(timestamp) is not int:
            raise RecordError(
                f"{base}.timestamp_ms",
                "timestamp_ms must be an integer",
                {"record_index": index},
            )
        if timestamp < 0:
            raise RecordError(
                f"{base}.timestamp_ms",
                "timestamp_ms must be nonnegative",
                {"record_index": index},
            )
        if previous_timestamp is not None and timestamp < previous_timestamp:
            raise RecordError(
                f"{base}.timestamp_ms",
                f"timestamp regressed from {previous_timestamp} to {timestamp}",
                {"record_index": index},
            )
        previous_timestamp = timestamp

        magnitude = record["magnitude"]
        if type(magnitude) is not int or not magnitude_min <= magnitude <= magnitude_max:
            raise RecordError(
                f"{base}.magnitude",
                f"magnitude must be an integer in [{magnitude_min}, {magnitude_max}]",
                {"record_index": index},
            )
        client_key = record["client_key"]
        if not isinstance(client_key, str) or not client_key:
            raise RecordError(
                f"{base}.client_key",
                "client_key must be a nonempty string",
                {"record_index": index},
            )
        duration = record["duration_us"]
        if type(duration) is not int or not duration_min <= duration <= duration_max:
            raise RecordError(
                f"{base}.duration_us",
                f"duration_us must be an integer in [{duration_min}, {duration_max}]",
                {"record_index": index},
            )
        trace_id = record["trace_id"]
        if not isinstance(trace_id, str) or not trace_id:
            raise RecordError(
                f"{base}.trace_id",
                "trace_id must be a nonempty string",
                {"record_index": index},
            )
        sampled = record["sampled"]
        if type(sampled) is not bool:
            raise RecordError(
                f"{base}.sampled",
                "sampled must be a boolean",
                {"record_index": index},
            )


def summarize_records(service: ServiceSpec, records: Iterable[dict[str, Any]]) -> dict[str, Any]:
    records_tuple = tuple(records)
    validate_record_stream(service, records_tuple)
    durations = [record["duration_us"] for record in records_tuple]
    stream = record_stream_bytes(records_tuple)
    return {
        "record_count": len(records_tuple),
        "duration_min_us": min(durations),
        "duration_max_us": max(durations),
        "null_or_invalid_field_count": 0,
        "timestamp_regression_count": 0,
        "sampled_hint_count": sum(1 for record in records_tuple if record["sampled"]),
        "first_timestamp_ms": records_tuple[0]["timestamp_ms"],
        "last_timestamp_ms": records_tuple[-1]["timestamp_ms"],
        "stream_digest": sha256_bytes(stream),
        "field_types": {
            name: service.fields[name]["type"] for name in REQUIRED_FIELDS
        },
    }
