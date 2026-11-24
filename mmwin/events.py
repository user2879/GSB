from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

from .canonical import content_hash, sha256_bytes, sha256_text
from .config import WindowConfig
from .png import read_png


_FRAME_NAME = re.compile(r"^frame[_-](\d+)(?:[_-]t?(\d+))?\.png$", re.IGNORECASE)


@dataclass(frozen=True)
class Event:
    event_id: str
    timestamp_ms: int
    kind: str
    fingerprint: str
    data: dict[str, Any] = field(compare=False, hash=False)

    @property
    def canonical_payload(self) -> dict[str, Any]:
        return {
            "event_id": self.event_id,
            "timestamp_ms": self.timestamp_ms,
            "kind": self.kind,
            "data": self.data,
            "fingerprint": self.fingerprint,
        }

    def reference(self) -> dict[str, Any]:
        return {
            "event_id": self.event_id,
            "timestamp_ms": self.timestamp_ms,
            "kind": self.kind,
            "fingerprint": self.fingerprint,
        }


@dataclass(frozen=True)
class ValidationFailure:
    code: str
    event_id: str | None
    message: str
    arrival_seq: int | None = None
    window_start_ms: int | None = None
    window_end_ms: int | None = None
    detail: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "event_id": self.event_id,
            "message": self.message,
            "arrival_seq": self.arrival_seq,
            "window_start_ms": self.window_start_ms,
            "window_end_ms": self.window_end_ms,
            "detail": self.detail,
        }


@dataclass
class EventLoadResult:
    events: dict[str, Event]
    failures: list[ValidationFailure]
    source_hash: str
    counts: dict[str, int]
    malformed_lines: int
    source_ids: set[str]

    def referenced_ids(self) -> set[str]:
        return set(self.events)


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _record_source(
    records: list[tuple[Path, int, Any]], path: Path, raw_line: str, line_no: int, failures: list[ValidationFailure]
) -> None:
    try:
        value = json.loads(raw_line)
    except json.JSONDecodeError as exc:
        failures.append(
            ValidationFailure(
                code="E_EVENT_INVALID",
                event_id=f"__malformed__:{path.name}:{line_no}",
                message=f"invalid JSON: {exc.msg}",
            )
        )
        records.append((path, line_no, {"__raw__": raw_line}))
        return
    if not isinstance(value, dict):
        failures.append(
            ValidationFailure(
                code="E_EVENT_INVALID",
                event_id=f"__malformed__:{path.name}:{line_no}",
                message="event must be a JSON object",
            )
        )
        records.append((path, line_no, {"__raw__": raw_line}))
        return
    records.append((path, line_no, value))


def _read_jsonl(
    path: Path, failures: list[ValidationFailure]
) -> list[tuple[Path, int, dict[str, Any]]]:
    if not path.exists():
        return []
    records: list[tuple[Path, int, Any]] = []
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line_no, raw in enumerate(handle, start=1):
                if raw.strip():
                    _record_source(records, path, raw, line_no, failures)
    except OSError as exc:
        failures.append(
            ValidationFailure(
                code="E_EVENT_INVALID",
                event_id=f"__file__:{path.name}",
                message=f"cannot read JSONL file: {exc}",
            )
        )
    return records


def _get_timestamp(record: dict[str, Any]) -> int | None:
    value = record.get(
        "timestamp_ms",
        record.get(
            "event_time_ms",
            record.get("event_timestamp_ms", record.get("timestamp")),
        ),
    )
    return value if _is_int(value) and value >= 0 else None


def _entity_fields(record: dict[str, Any]) -> tuple[str, str] | None:
    entity = record.get("entity")
    if isinstance(entity, dict):
        entity_id = entity.get("id")
        entity_type = entity.get("type")
    else:
        entity_id = record.get("entity_id")
        entity_type = record.get("entity_type")
    if isinstance(entity_id, str) and entity_id and isinstance(entity_type, str) and entity_type:
        return entity_id, entity_type
    return None


def _discover_frames(sample_dir: Path, failures: list[ValidationFailure]) -> list[dict[str, Any]]:
    discovered: list[dict[str, Any]] = []
    for path in sorted(sample_dir.rglob("*.png")):
        match = _FRAME_NAME.match(path.name)
        if not match:
            continue
        index = int(match.group(1))
        timestamp = match.group(2)
        if timestamp is None:
            failures.append(
                ValidationFailure(
                    code="E_EVENT_INVALID",
                    event_id=path.stem,
                    message="discovered PNG filename has no timestamp; provide frames.jsonl",
                )
            )
            continue
        discovered.append(
            {
                "event_id": path.stem,
                "timestamp_ms": int(timestamp),
                "frame_index": index,
                "path": path.relative_to(sample_dir).as_posix(),
            }
        )
    return discovered


def _normalise_frame(
    record: dict[str, Any],
    sample_dir: Path,
    config: WindowConfig,
    seen_frame_indices: set[int],
    expected_size: list[int | None],
) -> tuple[dict[str, Any] | None, str | None]:
    timestamp = _get_timestamp(record)
    if timestamp is None:
        return None, "timestamp_ms must be a non-negative integer"
    frame_index = record.get("frame_index", record.get("frame_no", record.get("index")))
    if not _is_int(frame_index) or frame_index < 0:
        return None, "frame_index must be a non-negative integer"
    if frame_index in seen_frame_indices:
        return None, f"duplicate frame_index {frame_index}"
    path_value = record.get("path", record.get("file", record.get("file_path")))
    if not isinstance(path_value, str) or not path_value:
        return None, "image event requires a non-empty path"
    try:
        path = (sample_dir / path_value).resolve()
        sample_root = sample_dir.resolve()
        path.relative_to(sample_root)
    except (OSError, ValueError):
        return None, "image path escapes the sample directory"
    if not path.is_file():
        return None, f"image file not found: {path_value}"
    try:
        image = read_png(path)
    except (OSError, ValueError) as exc:
        return None, f"cannot decode PNG: {exc}"
    height, width = image.shape[:2]
    if config.frame_width is not None and width != config.frame_width:
        return None, f"frame width {width} != configured {config.frame_width}"
    if config.frame_height is not None and height != config.frame_height:
        return None, f"frame height {height} != configured {config.frame_height}"
    if expected_size[0] is None:
        expected_size[0], expected_size[1] = width, height
    elif (width, height) != (expected_size[0], expected_size[1]):
        return None, "all frames must have the same fixed resolution"
    seen_frame_indices.add(frame_index)
    image_hash = sha256_bytes(path.read_bytes())
    normalised = {
        "path": path_value,
        "frame_index": frame_index,
        "width": width,
        "height": height,
        "image_sha256": image_hash,
    }
    return normalised, None


def load_events(
    sample_dir: str | Path,
    config: WindowConfig,
    *,
    frames_jsonl: str | Path | None = None,
    text_jsonl: str | Path | None = None,
    sensor_jsonl: str | Path | None = None,
) -> EventLoadResult:
    """Load and validate all event sources without sorting by event time."""
    sample_dir = Path(sample_dir).resolve()
    failures: list[ValidationFailure] = []
    frames_path = Path(frames_jsonl) if frames_jsonl else sample_dir / "frames.jsonl"
    text_path = Path(text_jsonl) if text_jsonl else sample_dir / "text_assertions.jsonl"
    sensor_path = Path(sensor_jsonl) if sensor_jsonl else sample_dir / "sensor_readings.jsonl"

    frame_records = _read_jsonl(frames_path, failures)
    text_records = _read_jsonl(text_path, failures)
    sensor_records = _read_jsonl(sensor_path, failures)
    if not frame_records:
        for record in _discover_frames(sample_dir, failures):
            frame_records.append((frames_path, 0, record))
    sources: list[tuple[str, list[tuple[Path, int, dict[str, Any]]]]] = [
        ("image", frame_records),
        ("text", text_records),
        ("sensor", sensor_records),
    ]

    events: dict[str, Event] = {}
    seen_ids: set[str] = set()
    seen_frame_indices: set[int] = set()
    expected_size: list[int | None] = [None, None]
    counts = {"image": 0, "text": 0, "sensor": 0}
    source_fingerprints: list[dict[str, Any]] = []
    source_ids: set[str] = set()
    malformed_lines = 0

    for kind, records in sources:
        for source_path, line_no, record in records:
            if "__raw__" in record:
                malformed_lines += 1
                source_fingerprints.append(
                    {"kind": kind, "source": source_path.name, "line": line_no, "raw": record["__raw__"]}
                )
                continue
            event_id = record.get("event_id", record.get("id"))
            if not isinstance(event_id, str) or not event_id:
                malformed_lines += 1
                failures.append(
                    ValidationFailure(
                        code="E_EVENT_INVALID",
                        event_id=f"__missing_id__:{source_path.name}:{line_no}",
                        message="event_id must be a non-empty string",
                    )
                )
                source_fingerprints.append(
                    {"kind": kind, "source": source_path.name, "line": line_no, "record": record}
                )
                continue
            source_ids.add(event_id)
            if event_id in seen_ids:
                failures.append(
                    ValidationFailure(
                        code="E_EVENT_INVALID",
                        event_id=event_id,
                        message=f"duplicate event_id at {source_path.name}:{line_no}",
                    )
                )
                source_fingerprints.append(
                    {
                        "kind": kind,
                        "source": source_path.name,
                        "line": line_no,
                        "event_id": event_id,
                        "duplicate": True,
                        "record": record,
                    }
                )
                continue
            timestamp = _get_timestamp(record)
            if timestamp is None:
                failures.append(
                    ValidationFailure(
                        code="E_EVENT_INVALID",
                        event_id=event_id,
                        message="timestamp_ms must be a non-negative integer",
                    )
                )
                source_fingerprints.append(
                    {"kind": kind, "source": source_path.name, "line": line_no, "record": record}
                )
                continue

            normalised: dict[str, Any] | None = None
            error: str | None = None
            if kind == "image":
                normalised, error = _normalise_frame(
                    record, sample_dir, config, seen_frame_indices, expected_size
                )
            elif kind == "text":
                entity = _entity_fields(record)
                attribute = record.get("attribute", record.get("property"))
                value = record.get("value")
                if entity is None:
                    error = "entity must contain non-empty id and type fields"
                elif not isinstance(attribute, str) or not attribute:
                    error = "attribute must be a non-empty string"
                elif value is None or isinstance(value, (dict, list)):
                    error = "value must be a JSON scalar"
                elif (
                    isinstance(value, (int, float))
                    and not isinstance(value, bool)
                    and not math.isfinite(float(value))
                ):
                    error = "numeric value must be finite"
                else:
                    confidence = record.get("confidence", 1.0)
                    if isinstance(confidence, bool) or not isinstance(confidence, (int, float)) or not math.isfinite(float(confidence)):
                        error = "confidence must be a finite number"
                    else:
                        polarity_raw = record.get("polarity")
                        if polarity_raw is not None and (
                            isinstance(polarity_raw, bool)
                            or not isinstance(polarity_raw, (int, float))
                            or not math.isfinite(float(polarity_raw))
                        ):
                            error = "polarity must be a finite number"
                        else:
                            normalised = {
                                "entity": {"id": entity[0], "type": entity[1]},
                                "attribute": attribute,
                                "value": value,
                                "confidence": float(confidence),
                                "polarity": None if polarity_raw is None else float(polarity_raw),
                            }
            else:
                channel = record.get("channel")
                reading = record.get("value", record.get("reading"))
                if not isinstance(channel, str) or channel not in config.sensor_channels:
                    error = f"channel must be one of {sorted(config.sensor_channels)}"
                elif (
                    isinstance(reading, bool)
                    or not isinstance(reading, (int, float))
                    or not math.isfinite(float(reading))
                ):
                    error = "sensor reading must be a finite number"
                else:
                    normalised = {"channel": channel, "value": float(reading)}

            if error is not None or normalised is None:
                failures.append(
                    ValidationFailure(
                        code="E_EVENT_INVALID",
                        event_id=event_id,
                        message=error or "invalid event",
                    )
                )
                source_fingerprints.append(
                    {"kind": kind, "source": source_path.name, "line": line_no, "record": record}
                )
                continue

            payload = {
                "event_id": event_id,
                "timestamp_ms": timestamp,
                "kind": kind,
                "data": normalised,
            }
            fingerprint = content_hash(payload)
            event = Event(
                event_id=event_id,
                timestamp_ms=timestamp,
                kind=kind,
                fingerprint=fingerprint,
                data=normalised,
            )
            events[event_id] = event
            seen_ids.add(event_id)
            counts[kind] += 1
            source_fingerprints.append(
                {
                    "kind": kind,
                    "source": source_path.name,
                    "line": line_no,
                    "event_id": event_id,
                    "fingerprint": fingerprint,
                }
            )

    # Invalid source records may contain non-finite JSON numbers.  They must
    # still participate in the source fingerprint, so use a deterministic
    # permissive encoder for this audit-only digest.
    source_hash = sha256_text(
        json.dumps(
            source_fingerprints,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=True,
        )
    )
    return EventLoadResult(
        events=events,
        failures=failures,
        source_hash=source_hash,
        counts=counts,
        malformed_lines=malformed_lines,
        source_ids=source_ids,
    )


def event_payloads(events: Iterable[Event]) -> list[dict[str, Any]]:
    return [event.canonical_payload for event in sorted(events, key=lambda item: (item.timestamp_ms, item.event_id))]
