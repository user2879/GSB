from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

from .errors import ConfigFormatError, OrderError
from .models import AccessEvent
from .utils import atomic_write_text


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _pick(record: dict[str, Any], *names: str) -> tuple[str | None, Any]:
    for name in names:
        if name in record:
            return name, record[name]
    return None, None


def parse_event(record: Any, *, line_number: int, source: str) -> AccessEvent:
    if not isinstance(record, dict):
        raise ConfigFormatError(
            "event must be a JSON object",
            field_path=f"$[{line_number}]",
            details={"source": source},
        )

    seq_key, seq_value = _pick(record, "seq", "sequence", "event_id")
    if seq_key is None:
        raise ConfigFormatError(
            "missing event seq",
            field_path=f"$[{line_number}].seq",
            details={"source": source},
        )
    if not _is_int(seq_value) or seq_value < 0:
        raise ConfigFormatError(
            "event seq must be a non-negative integer",
            field_path=f"$[{line_number}].{seq_key}",
            details={"source": source},
        )

    ts_key, ts_value = _pick(record, "timestamp_ms", "ts_ms", "timestamp")
    if ts_key is None:
        raise ConfigFormatError(
            "missing event timestamp_ms",
            field_path=f"$[{line_number}].timestamp_ms",
            details={"source": source},
        )
    if not _is_int(ts_value) or ts_value < 0:
        raise ConfigFormatError(
            "event timestamp_ms must be a non-negative integer",
            field_path=f"$[{line_number}].{ts_key}",
            details={"source": source},
        )

    volume_key, volume_value = _pick(record, "volume_id", "volume")
    if volume_key is None:
        raise ConfigFormatError(
            "missing event volume_id",
            field_path=f"$[{line_number}].volume_id",
            details={"source": source},
        )
    if not isinstance(volume_value, str) or not volume_value.strip():
        raise ConfigFormatError(
            "event volume_id must be a non-empty string",
            field_path=f"$[{line_number}].{volume_key}",
            details={"source": source},
        )

    ops_key, ops_value = _pick(record, "operation_count", "ops", "operations")
    if ops_key is None:
        raise ConfigFormatError(
            "missing event operation_count",
            volume_id=volume_value.strip(),
            field_path=f"$[{line_number}].operation_count",
            details={"source": source},
        )
    if not _is_int(ops_value) or ops_value < 0:
        raise ConfigFormatError(
            "event operation_count must be a non-negative integer",
            volume_id=volume_value.strip(),
            field_path=f"$[{line_number}].{ops_key}",
            details={"source": source},
        )

    bytes_key, bytes_value = _pick(record, "write_bytes", "bytes", "write_size")
    if bytes_key is None:
        raise ConfigFormatError(
            "missing event write_bytes",
            volume_id=volume_value.strip(),
            field_path=f"$[{line_number}].write_bytes",
            details={"source": source},
        )
    if not _is_int(bytes_value) or bytes_value < 0:
        raise ConfigFormatError(
            "event write_bytes must be a non-negative integer",
            volume_id=volume_value.strip(),
            field_path=f"$[{line_number}].{bytes_key}",
            details={"source": source},
        )

    known = {seq_key, ts_key, volume_key, ops_key, bytes_key}
    extra = {key: value for key, value in record.items() if key not in known}
    return AccessEvent(
        seq=seq_value,
        timestamp_ms=ts_value,
        volume_id=volume_value.strip(),
        operation_count=ops_value,
        write_bytes=bytes_value,
        extra=extra,
    )


def load_events(path: str | Path) -> list[AccessEvent]:
    source = Path(path)
    try:
        lines = source.read_text(encoding="utf-8-sig").splitlines()
    except OSError as exc:
        raise ConfigFormatError(
            f"cannot read event stream: {exc}",
            field_path="$",
            details={"source": str(source)},
        ) from exc

    events: list[AccessEvent] = []
    previous_seq: int | None = None
    previous_timestamp: int | None = None
    previous_volume: str | None = None
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            raw = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ConfigFormatError(
                f"invalid NDJSON: {exc.msg}",
                field_path=f"$[{line_number}] (column {exc.colno})",
                details={"source": str(source)},
            ) from exc
        event = parse_event(raw, line_number=line_number, source=str(source))
        if previous_seq is not None and event.seq <= previous_seq:
            raise OrderError(
                "event sequence must be strictly increasing",
                volume_id=event.volume_id,
                field_path=f"$[{line_number}].seq",
                details={"previous_seq": previous_seq, "seq": event.seq},
            )
        if previous_timestamp is not None and event.timestamp_ms < previous_timestamp:
            raise OrderError(
                "event timestamp moved backwards",
                volume_id=event.volume_id,
                field_path=f"$[{line_number}].timestamp_ms",
                details={
                    "previous_timestamp_ms": previous_timestamp,
                    "timestamp_ms": event.timestamp_ms,
                    "previous_volume_id": previous_volume,
                    "seq": event.seq,
                },
            )
        previous_seq = event.seq
        previous_timestamp = event.timestamp_ms
        previous_volume = event.volume_id
        events.append(event)
    return events


def events_as_ndjson(events: Iterable[AccessEvent]) -> str:
    return "".join(
        json.dumps(event.to_json(), ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        + "\n"
        for event in events
    )


def write_events(path: str | Path, events: Iterable[AccessEvent]) -> None:
    atomic_write_text(path, events_as_ndjson(events))
