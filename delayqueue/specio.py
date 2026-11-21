"""Loaders and strong validation for topic declarations and NDJSON streams.

Any single violation rejects the *whole* load with :class:`SpecError`
(``E_SPEC``) pointing at the first offending line number and field.
"""

from __future__ import annotations

import json
import os

from .errors import SpecError
from .models import MessageInput, QueueSpec, TopicSpec, payload_digest

_REQUIRED_TOPIC_FIELDS = (
    "visibility_timeout_ms",
    "max_deliveries",
    "backoff_ms",
    "concurrency",
)
_REQUIRED_MESSAGE_FIELDS = (
    "message_id",
    "topic",
    "deliver_at",
    "payload",
    "idempotency_key",
)


def _is_plain_int(value) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _require_int(value, field: str, line: int, *, minimum: int) -> int:
    if not _is_plain_int(value):
        raise SpecError(f"field '{field}' must be an integer", line, field)
    if value < minimum:
        raise SpecError(
            f"field '{field}' must be >= {minimum}", line, field)
    return value


def load_queue_spec(path: str) -> QueueSpec:
    """Read and fully validate a topic declaration JSON file."""
    if not os.path.exists(path):
        raise SpecError(f"declaration file not found: {path}", 0, "file")
    try:
        with open(path, "r", encoding="utf-8") as handle:
            data = json.load(handle)
    except json.JSONDecodeError as exc:
        raise SpecError(f"invalid JSON: {exc.msg}", 0, "json") from exc

    if not isinstance(data, dict):
        raise SpecError("declaration root must be an object", 0, "root")

    spec_version = data.get("spec_version", "v1")
    if not isinstance(spec_version, str) or not spec_version:
        raise SpecError("spec_version must be a non-empty string", 0,
                        "spec_version")

    global_concurrency = _require_int(
        data.get("global_concurrency"), "global_concurrency", 0, minimum=1)

    raw_topics = data.get("topics")
    if not isinstance(raw_topics, (dict, list)) or not raw_topics:
        raise SpecError("topics must be a non-empty object or list", 0,
                        "topics")

    topics: dict[str, TopicSpec] = {}
    if isinstance(raw_topics, dict):
        iterator = ((name, body) for name, body in raw_topics.items())
    else:
        iterator = ((body.get("topic"), body) for body in raw_topics)

    for raw_name, body in iterator:
        if not isinstance(raw_name, str) or not raw_name:
            raise SpecError("topic id must be a non-empty string", 0,
                            "topic")
        if raw_name in topics:  # explicit uniqueness check
            raise SpecError(f"duplicate topic id '{raw_name}'", 0,
                            f"topics.{raw_name}")
        if not isinstance(body, dict):
            raise SpecError(f"topic '{raw_name}' must be an object", 0,
                            f"topics.{raw_name}")
        for required in _REQUIRED_TOPIC_FIELDS:
            if required not in body:
                raise SpecError(
                    f"topic '{raw_name}' missing field '{required}'", 0,
                    f"topics.{raw_name}.{required}")

        visibility_timeout_ms = _require_int(
            body["visibility_timeout_ms"],
            f"topics.{raw_name}.visibility_timeout_ms", 0, minimum=1)
        max_deliveries = _require_int(
            body["max_deliveries"],
            f"topics.{raw_name}.max_deliveries", 0, minimum=1)
        concurrency = _require_int(
            body["concurrency"],
            f"topics.{raw_name}.concurrency", 0, minimum=1)

        backoff = body["backoff_ms"]
        if not isinstance(backoff, list) or not backoff:
            raise SpecError(
                f"topic '{raw_name}' backoff_ms must be a non-empty list",
                0, f"topics.{raw_name}.backoff_ms")
        for index, element in enumerate(backoff):
            if not _is_plain_int(element) or element < 0:
                raise SpecError(
                    f"backoff_ms[{index}] must be a non-negative integer",
                    0, f"topics.{raw_name}.backoff_ms")
        if len(backoff) < max_deliveries - 1:
            raise SpecError(
                f"backoff_ms length {len(backoff)} must be >= "
                f"max_deliveries - 1 ({max_deliveries - 1})",
                0, f"topics.{raw_name}.backoff_ms")

        topics[raw_name] = TopicSpec(
            topic=raw_name,
            visibility_timeout_ms=visibility_timeout_ms,
            max_deliveries=max_deliveries,
            backoff_ms=tuple(backoff),
            concurrency=concurrency,
        )

    return QueueSpec(
        global_concurrency=global_concurrency,
        topics=topics,
        spec_version=spec_version,
    )


def _encode_integer_payload(raw, line: int) -> bytes:
    """An integer byte payload: non-negative int stored as minimal bytes."""
    if not _is_plain_int(raw):
        raise SpecError("payload must be a non-negative integer", line,
                        "payload")
    if raw < 0:
        raise SpecError("payload must be a non-negative integer", line,
                        "payload")
    length = max(1, (raw.bit_length() + 7) // 8)
    return raw.to_bytes(length, "big")


def load_messages(path: str, spec: QueueSpec) -> list[MessageInput]:
    """Read and fully validate an NDJSON message stream against a spec."""
    if not os.path.exists(path):
        raise SpecError(f"message stream not found: {path}", 0, "file")

    messages: list[MessageInput] = []
    seen_ids: set[str] = set()
    idem_payloads: dict[str, str] = {}

    with open(path, "r", encoding="utf-8") as handle:
        for line_no, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise SpecError(f"invalid NDJSON: {exc.msg}", line_no,
                                "json") from exc
            if not isinstance(record, dict):
                raise SpecError("message must be a JSON object", line_no,
                                "root")
            for required in _REQUIRED_MESSAGE_FIELDS:
                if required not in record:
                    raise SpecError(f"missing field '{required}'", line_no,
                                    required)

            message_id = record["message_id"]
            if not isinstance(message_id, str) or not message_id:
                raise SpecError("message_id must be a non-empty string",
                                line_no, "message_id")
            if message_id in seen_ids:
                raise SpecError(f"duplicate message_id '{message_id}'",
                                line_no, "message_id")

            topic = record["topic"]
            if not isinstance(topic, str) or not topic:
                raise SpecError("topic must be a non-empty string", line_no,
                                "topic")
            if topic not in spec.topics:
                raise SpecError(f"unknown topic '{topic}'", line_no, "topic")

            deliver_at = _require_int(
                record["deliver_at"], "deliver_at", line_no, minimum=0)

            idem = record["idempotency_key"]
            if not isinstance(idem, str) or not idem:
                raise SpecError("idempotency_key must be a non-empty string",
                                line_no, "idempotency_key")

            payload = _encode_integer_payload(record["payload"], line_no)
            summary = payload_digest(payload)
            previous = idem_payloads.get(idem)
            if previous is not None and previous != summary:
                raise SpecError(
                    f"idempotency_key '{idem}' maps to different payload "
                    "summaries", line_no, "idempotency_key")
            idem_payloads.setdefault(idem, summary)

            seen_ids.add(message_id)
            messages.append(MessageInput(
                message_id=message_id,
                topic=topic,
                deliver_at=deliver_at,
                payload=payload,
                idempotency_key=idem,
                line_no=line_no,
            ))

    if not messages:
        raise SpecError("message stream contains no records", 0, "file")
    return messages
