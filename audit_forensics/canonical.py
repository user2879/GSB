"""Canonical event and timestamp normalization."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from typing import Any

from .errors import AuditConfigError


CORE_EVENT_FIELDS = (
    "event_id",
    "trace_id",
    "actor",
    "object",
    "action",
    "ts",
    "seq",
    "parent_event_id",
)

CONFLICT_FIELDS = CORE_EVENT_FIELDS[1:]

_EPOCH = datetime(1970, 1, 1, tzinfo=timezone.utc)


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _datetime_to_ms(value: datetime) -> int:
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    value = value.astimezone(timezone.utc)
    delta = value - _EPOCH
    return (
        delta.days * 86_400_000
        + delta.seconds * 1_000
        + delta.microseconds // 1_000
    )


def normalize_timestamp(value: Any, *, allow_iso: bool = True) -> int:
    """Normalize integer milliseconds or ISO-8601 to integer milliseconds.

    Naive ISO-8601 values are interpreted as UTC so normalization never depends
    on the host timezone.
    """

    if _is_int(value):
        return int(value)
    if allow_iso and isinstance(value, str) and value.strip():
        text = value.strip()
        if text.endswith(("Z", "z")):
            text = text[:-1] + "+00:00"
        try:
            parsed = datetime.fromisoformat(text)
        except ValueError as exc:
            raise ValueError(f"invalid ISO-8601 timestamp: {value!r}") from exc
        return _datetime_to_ms(parsed)
    kind = "null" if value is None else type(value).__name__
    raise ValueError(f"unsupported timestamp value ({kind}): {value!r}")


def canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def event_core(event: dict[str, Any]) -> dict[str, Any]:
    """Return the eight-field cross-format canonical record."""

    return {field: event.get(field) for field in CORE_EVENT_FIELDS}


def event_fingerprint(event: dict[str, Any]) -> str:
    return sha256_text(canonical_json(event_core(event)))


def make_sync_marker(
    *,
    event_id: str,
    source_format: str,
    position: str,
    original_ts: Any,
    source: str | None = None,
) -> str:
    payload = {
        "event_id": event_id,
        "source_format": source_format,
        "position": position,
        "original_ts": original_ts,
    }
    if source is not None:
        payload["source"] = source
    material = canonical_json(payload)
    return "SYNC-" + sha256_text(material)[:24]


def normalize_event(
    raw: Any,
    *,
    source_format: str,
    source: str,
    position: str,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any] | None]:
    """Validate an input event.

    Returns ``(core, extension_metadata, error)``. Extension metadata is kept
    outside the normalized core so an optional v2 ``severity``/``source`` does
    not make an otherwise identical v1/v2 event unequal.
    """

    from .errors import E_SCHEMA, E_TIMESTAMP
    from .errors import business_error

    if not isinstance(raw, dict):
        return {}, {}, business_error(
            E_SCHEMA,
            "event element must be a JSON object",
            source_format=source_format,
            source=source,
            position=position,
        )

    for field in ("event_id", "trace_id", "actor", "object", "action"):
        value = raw.get(field)
        if not isinstance(value, str) or not value:
            return {}, {}, business_error(
                E_SCHEMA,
                f"{field} must be a non-empty string",
                source_format=source_format,
                source=source,
                position=position,
                event_id=(
                    raw.get("event_id")
                    if isinstance(raw.get("event_id"), str)
                    else None
                ),
            )

    if not _is_int(raw.get("ts")):
        if source_format != "v2" or not isinstance(raw.get("ts"), str):
            return {}, {}, business_error(
                E_TIMESTAMP,
                "ts must be integer milliseconds"
                + (" or an ISO-8601 string for v2" if source_format == "v2" else ""),
                source_format=source_format,
                source=source,
                position=position,
                event_id=raw["event_id"],
                trace_id=raw["trace_id"],
            )
    try:
        normalized_ts = normalize_timestamp(
            raw.get("ts"), allow_iso=(source_format == "v2")
        )
    except ValueError as exc:
        return {}, {}, business_error(
            E_TIMESTAMP,
            str(exc),
            source_format=source_format,
            source=source,
            position=position,
            event_id=raw["event_id"],
            trace_id=raw["trace_id"],
        )

    if not _is_int(raw.get("seq")):
        return {}, {}, business_error(
            E_SCHEMA,
            "seq must be an integer",
            source_format=source_format,
            source=source,
            position=position,
            event_id=raw["event_id"],
            trace_id=raw["trace_id"],
        )

    parent_event_id = raw.get("parent_event_id")
    if parent_event_id is not None and (
        not isinstance(parent_event_id, str) or not parent_event_id
    ):
        return {}, {}, business_error(
            E_SCHEMA,
            "parent_event_id must be null, absent, or a non-empty string",
            source_format=source_format,
            source=source,
            position=position,
            event_id=raw["event_id"],
            trace_id=raw["trace_id"],
        )

    severity = raw.get("severity")
    if severity is not None and not isinstance(severity, str):
        return {}, {}, business_error(
            E_SCHEMA,
            "severity must be null, absent, or a string",
            source_format=source_format,
            source=source,
            position=position,
            event_id=raw["event_id"],
            trace_id=raw["trace_id"],
        )

    source_name = raw.get("source")
    if source_name is not None and not isinstance(source_name, str):
        return {}, {}, business_error(
            E_SCHEMA,
            "source must be null, absent, or a string",
            source_format=source_format,
            source=source,
            position=position,
            event_id=raw["event_id"],
            trace_id=raw["trace_id"],
        )

    core = {
        "event_id": raw["event_id"],
        "trace_id": raw["trace_id"],
        "actor": raw["actor"],
        "object": raw["object"],
        "action": raw["action"],
        "ts": normalized_ts,
        "seq": int(raw["seq"]),
        "parent_event_id": parent_event_id,
    }
    extension = {
        "severity": severity,
        "source": source_name,
        "original_ts": raw.get("ts"),
    }
    return core, extension, None


def parse_rules_timestamp(value: Any) -> int:
    try:
        return normalize_timestamp(value, allow_iso=True)
    except ValueError as exc:
        raise AuditConfigError(f"invalid rule time value: {exc}") from exc
