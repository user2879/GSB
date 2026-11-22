"""Strict v1/v2/v3 JSONL decoding into one provenance-preserving model."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from .canonical import sha256_obj
from .config import Declarations, TopicConfig, VERDICTS
from .errors import format_error, io_error


ENVELOPE_FIELDS = {"format_version", "consumer_group", "record"}
V1_FIELDS = {"message_id", "topic", "acknowledged"}
V2_FIELDS = V1_FIELDS | {"delivery_time_ms", "retry_count", "consumer_group"}
V3_FIELDS = V2_FIELDS | {"idempotency_key", "delivery_sequence_range", "semantic_verdict"}
FORMAT_FIELDS = {"v1": V1_FIELDS, "v2": V2_FIELDS, "v3": V3_FIELDS}
REQUIRED_FIELDS = {
    "v1": V1_FIELDS,
    "v2": V2_FIELDS,
    "v3": V3_FIELDS,
}


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


@dataclass(frozen=True)
class NormalizedRecord:
    line_no: int
    source_format: str
    source_group: str
    message_id: str
    topic: str
    message_seq: int
    acknowledged: bool
    delivery_time_ms: int | None
    retry_count: int
    consumer_group: str
    idempotency_key: str | None
    range_start: int | None
    range_end: int | None
    semantic_verdict: str | None

    @property
    def attempts(self) -> int:
        return max(1, self.retry_count + 1)

    @property
    def range_present(self) -> bool:
        return self.range_start is not None and self.range_end is not None

    def model_dict(self) -> dict[str, Any]:
        return {
            "acknowledged": self.acknowledged,
            "consumer_group": self.consumer_group,
            "delivery_sequence_range": (
                [self.range_start, self.range_end] if self.range_present else None
            ),
            "delivery_time_ms": self.delivery_time_ms,
            "idempotency_key": self.idempotency_key,
            "message_id": self.message_id,
            "message_seq": self.message_seq,
            "retry_count": self.retry_count,
            "semantic_verdict": self.semantic_verdict,
            "source_format": self.source_format,
            "source_group": self.source_group,
            "topic": self.topic,
        }

    def provenance_dict(self) -> dict[str, Any]:
        value = self.model_dict()
        value["source_line"] = self.line_no
        return value

    @classmethod
    def from_dict(cls, value: dict[str, Any], *, include_line: bool = True) -> "NormalizedRecord":
        range_value = value.get("delivery_sequence_range")
        return cls(
            line_no=int(value.get("source_line", 0)) if include_line else 0,
            source_format=str(value["source_format"]),
            source_group=str(value["source_group"]),
            message_id=str(value["message_id"]),
            topic=str(value["topic"]),
            message_seq=int(value["message_seq"]),
            acknowledged=bool(value["acknowledged"]),
            delivery_time_ms=value.get("delivery_time_ms"),
            retry_count=int(value.get("retry_count", 0)),
            consumer_group=str(value["consumer_group"]),
            idempotency_key=value.get("idempotency_key"),
            range_start=(int(range_value[0]) if range_value is not None else None),
            range_end=(int(range_value[1]) if range_value is not None else None),
            semantic_verdict=value.get("semantic_verdict"),
        )


def _check_keys(record: dict[str, Any], allowed: set[str], line_no: int, version: str) -> None:
    unknown = sorted(set(record) - allowed)
    if unknown:
        raise format_error(
            line_no,
            f"record.{unknown[0]}",
            "field is not allowed in this format version",
            format_version=version,
            extra_fields=unknown,
        )
    for field in sorted(allowed):
        if field not in record:
            raise format_error(line_no, f"record.{field}", "required field is missing")


def _expect_bool(record: dict[str, Any], field: str, line_no: int) -> bool:
    value = record[field]
    if not isinstance(value, bool):
        raise format_error(line_no, f"record.{field}", "must be a boolean", value=value)
    return value


def _expect_int(record: dict[str, Any], field: str, line_no: int, *, minimum: int | None = None) -> int:
    value = record[field]
    if not _is_int(value):
        raise format_error(line_no, f"record.{field}", "must be an integer", value=value)
    if minimum is not None and value < minimum:
        raise format_error(line_no, f"record.{field}", f"must be >= {minimum}", value=value)
    return value


def _expect_str(record: dict[str, Any], field: str, line_no: int) -> str:
    value = record[field]
    if not isinstance(value, str) or not value:
        raise format_error(line_no, f"record.{field}", "must be a non-empty string", value=value)
    return value


def _topic_for_message(declarations: Declarations, topic_name: Any, message_id: Any, line_no: int) -> tuple[TopicConfig, int]:
    if not isinstance(topic_name, str) or not topic_name:
        raise format_error(line_no, "record.topic", "must be a non-empty string", value=topic_name)
    topic = declarations.topics.get(topic_name)
    if topic is None:
        raise format_error(line_no, "record.topic", "topic is not declared", value=topic_name)
    if not isinstance(message_id, str) or not message_id:
        raise format_error(line_no, "record.message_id", "must be a non-empty string", value=message_id)
    seq = topic.sequence_for_id(message_id)
    if seq is None:
        raise format_error(
            line_no,
            "record.message_id",
            "message id does not satisfy the topic sequence rule",
            value=message_id,
            topic=topic_name,
            rule={
                "start": topic.start,
                "end": topic.end,
                "step": topic.step,
                "id_template": topic.id_template,
            },
        )
    return topic, seq


def _decode_one(
    raw: Any,
    line_no: int,
    declarations: Declarations,
    *,
    inject_format: bool,
) -> NormalizedRecord:
    if not isinstance(raw, dict):
        raise format_error(line_no, "$", "record line must be a JSON object")
    unknown_envelope = sorted(set(raw) - ENVELOPE_FIELDS)
    if unknown_envelope:
        raise format_error(line_no, unknown_envelope[0], "unknown envelope field")
    for field in sorted(ENVELOPE_FIELDS):
        if field not in raw:
            raise format_error(line_no, field, "required envelope field is missing")

    version = raw["format_version"]
    if version not in FORMAT_FIELDS:
        raise format_error(
            line_no,
            "format_version",
            "unknown format version",
            value=version,
            supported=list(FORMAT_FIELDS),
        )

    group_name = raw["consumer_group"]
    if not isinstance(group_name, str) or not group_name:
        raise format_error(line_no, "consumer_group", "must be a non-empty string", value=group_name)
    group = declarations.groups.get(group_name)
    if group is None:
        raise format_error(line_no, "consumer_group", "consumer group is not declared", value=group_name)
    if group.format_version != version:
        raise format_error(
            line_no,
            "format_version",
            "record version does not match the consumer group declaration",
            value=version,
            expected=group.format_version,
            consumer_group=group_name,
        )

    record = raw["record"]
    if not isinstance(record, dict):
        raise format_error(line_no, "record", "must be an object")

    if inject_format and version in {"v2", "v3"}:
        record = dict(record)
        record["retry_count"] = -1

    _check_keys(record, FORMAT_FIELDS[version], line_no, version)
    acknowledged = _expect_bool(record, "acknowledged", line_no)
    topic, seq = _topic_for_message(declarations, record["topic"], record["message_id"], line_no)
    message_id = record["message_id"]

    delivery_time_ms: int | None = None
    retry_count = 0
    consumer_group = group_name
    idempotency_key: str | None = None
    range_start: int | None = None
    range_end: int | None = None
    semantic_verdict: str | None = None

    if version in {"v2", "v3"}:
        delivery_time_ms = _expect_int(record, "delivery_time_ms", line_no, minimum=0)
        retry_count = _expect_int(record, "retry_count", line_no, minimum=0)
        payload_group = _expect_str(record, "consumer_group", line_no)
        if payload_group != group_name:
            raise format_error(
                line_no,
                "record.consumer_group",
                "payload consumer_group must match the envelope",
                value=payload_group,
                expected=group_name,
            )

    if version == "v3":
        idempotency_key = _expect_str(record, "idempotency_key", line_no)
        range_value = record["delivery_sequence_range"]
        if not isinstance(range_value, dict):
            raise format_error(
                line_no,
                "record.delivery_sequence_range",
                "must be an object with start and end",
            )
        if set(range_value) != {"start", "end"}:
            raise format_error(
                line_no,
                "record.delivery_sequence_range",
                "must contain exactly start and end",
                value=range_value,
            )
        if not _is_int(range_value["start"]):
            raise format_error(
                line_no,
                "record.delivery_sequence_range.start",
                "must be an integer",
                value=range_value["start"],
            )
        if not _is_int(range_value["end"]):
            raise format_error(
                line_no,
                "record.delivery_sequence_range.end",
                "must be an integer",
                value=range_value["end"],
            )
        range_start = range_value["start"]
        range_end = range_value["end"]
        if range_start < 0:
            raise format_error(
                line_no,
                "record.delivery_sequence_range.start",
                "must be >= 0",
                value=range_start,
            )
        if range_start > range_end:
            raise format_error(
                line_no,
                "record.delivery_sequence_range",
                "start must be <= end",
                start=range_start,
                end=range_end,
            )
        semantic_verdict = record["semantic_verdict"]
        if semantic_verdict not in VERDICTS:
            raise format_error(
                line_no,
                "record.semantic_verdict",
                f"must be one of {list(VERDICTS)}",
                value=semantic_verdict,
            )

    return NormalizedRecord(
        line_no=line_no,
        source_format=version,
        source_group=group_name,
        message_id=message_id,
        topic=topic.name,
        message_seq=seq,
        acknowledged=acknowledged,
        delivery_time_ms=delivery_time_ms,
        retry_count=retry_count,
        consumer_group=consumer_group,
        idempotency_key=idempotency_key,
        range_start=range_start,
        range_end=range_end,
        semantic_verdict=semantic_verdict,
    )


def load_normalized_records(
    path: str | Path,
    declarations: Declarations,
    *,
    inject_format: bool = False,
) -> list[NormalizedRecord]:
    source = Path(path)
    try:
        text = source.read_text(encoding="utf-8-sig")
    except FileNotFoundError:
        raise io_error("record file not found", path=str(source))
    except OSError as exc:
        raise io_error("could not read record file", path=str(source), reason=str(exc))

    records: list[NormalizedRecord] = []
    for line_no, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            raw = json.loads(line)
        except json.JSONDecodeError as exc:
            raise format_error(
                line_no,
                "$",
                "invalid JSON",
                reason=exc.msg,
                column=exc.colno,
            )
        records.append(_decode_one(raw, line_no, declarations, inject_format=inject_format))
    if not records:
        raise format_error(1, "$", "record file contains no records", path=str(source))
    return records


def sorted_records(records: Iterable[NormalizedRecord]) -> list[NormalizedRecord]:
    return sorted(
        records,
        key=lambda record: (
            record.topic,
            record.message_seq,
            record.message_id,
            record.consumer_group,
            record.source_format,
            record.line_no,
            canonical_record_key(record),
        ),
    )


def canonical_record_key(record: NormalizedRecord) -> str:
    return json.dumps(record.model_dict(), ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def normalized_model(records: Iterable[NormalizedRecord]) -> dict[str, Any]:
    rows = [record.model_dict() for record in records]
    rows.sort(key=lambda row: json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")))
    return {"schema_version": 1, "records": rows}


def model_digest(records: Iterable[NormalizedRecord]) -> str:
    return sha256_obj(normalized_model(records))


def provenance_rows(records: Iterable[NormalizedRecord]) -> list[dict[str, Any]]:
    rows = [record.provenance_dict() for record in records]
    rows.sort(
        key=lambda row: (
            row["topic"],
            row["message_seq"],
            row["consumer_group"],
            row["source_format"],
            row["source_line"],
            json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")),
        )
    )
    return rows


def duplicate_idempotency_details(records: Iterable[NormalizedRecord]) -> dict[str, Any] | None:
    occurrences: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        if record.idempotency_key is None:
            continue
        occurrences.setdefault(record.idempotency_key, []).append(
            {
                "topic": record.topic,
                "message_seq": record.message_seq,
                "message_id": record.message_id,
                "consumer_group": record.consumer_group,
                "source_format": record.source_format,
                "source_line": record.line_no,
            }
        )
    for key in sorted(occurrences):
        rows = sorted(
            occurrences[key],
            key=lambda row: (
                row["consumer_group"],
                row["message_id"],
                row["source_format"],
                row["source_line"],
            ),
        )
        groups = sorted({row["consumer_group"] for row in rows})
        if len(groups) > 1:
            return {"idempotency_key": key, "consumer_groups": groups, "occurrences": rows}
    return None

