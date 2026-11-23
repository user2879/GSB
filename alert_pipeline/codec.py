"""v1 / v2 / v3 alert record codec and normalization.

Schema history
--------------
v1: {version, timestamp_ms, service_id, severity}
v2: v1 + fingerprint, silence_window_ms
v3: v2 + root_cause_refs, strategy_id  (+ optional labels)

Normalization always yields :class:`NormalizedAlert`. Fill rules are explicit
and traceable through ``FILL_RULES``; every synthesized field name is listed in
``NormalizedAlert.filled``. Normalization output depends solely on logical
record content, never on row order.
"""
from __future__ import annotations

import json
import re
from typing import Any

from .errors import FieldError
from .models import NormalizedAlert, SCHEMA_VERSIONS, SEVERITIES
from .util import digest_parts, stable_id

# Explicit, traceable fill rules, keyed by (source version, filled field).
FILL_RULES: dict[tuple[str, str], str] = {
    ("v1", "fingerprint"):
        "derived: sha256(service_id|severity|timestamp_ms)[0:16] prefixed fp_",
    ("v1", "silence_window_ms"):
        "default: pipeline default silence window from rule spec",
    ("v1", "labels"):
        "default: empty label set",
    ("v1", "root_cause_refs"):
        "default: empty root-cause reference set",
    ("v1", "strategy_id"):
        "default: strategy_default",
    ("v2", "labels"):
        "default: empty label set",
    ("v2", "root_cause_refs"):
        "default: empty root-cause reference set",
    ("v2", "strategy_id"):
        "default: strategy_default",
    ("v3", "labels"):
        "default: empty label set",
}

FINGERPRINT_RE = re.compile(r"^fp_[0-9a-f]{8,32}$")
DEFAULT_STRATEGY = "strategy_default"
DEFAULT_LABELS: tuple = ()
DEFAULT_REFS: tuple = ()


def alert_labels_kv(labels):
    return [list(kv) for kv in labels]



def detect_version(raw: dict[str, Any], line: int) -> str:
    if not isinstance(raw, dict):
        raise FieldError("alert record must be a JSON object", line, "record")
    version = raw.get("version")
    if version is None:
        raise FieldError("missing required field 'version'", line, "version")
    if version not in SCHEMA_VERSIONS:
        raise FieldError(f"unknown schema version {version!r}", line, "version")
    return version


def _require_int(raw: dict, field: str, line: int, *, allow_none=False):
    if field not in raw or raw[field] is None:
        if allow_none:
            return None
        raise FieldError(f"missing required field {field!r}", line, field)
    value = raw[field]
    # bool is a subclass of int: reject explicitly.
    if isinstance(value, bool) or not isinstance(value, int):
        raise FieldError(f"field {field!r} must be an integer", line, field,
                         repr(value))
    return value


def _require_str(raw: dict, field: str, line: int):
    if field not in raw or raw[field] is None:
        raise FieldError(f"missing required field {field!r}", line, field)
    value = raw[field]
    if not isinstance(value, str) or not value:
        raise FieldError(f"field {field!r} must be a non-empty string", line,
                         field, repr(value))
    return value


def _validate_fingerprint(fp: str, line: int) -> None:
    if not isinstance(fp, str) or not FINGERPRINT_RE.match(fp):
        raise FieldError(
            "fingerprint must match ^fp_[0-9a-f]{8,32}$", line, "fingerprint",
            repr(fp))


def _validate_labels(raw_labels: Any, line: int) -> tuple[tuple[str, str], ...]:
    if not isinstance(raw_labels, dict):
        raise FieldError("labels must be an object of string key/value pairs",
                         line, "labels")
    labels = []
    for key in sorted(raw_labels):
        val = raw_labels[key]
        if not isinstance(key, str) or not isinstance(val, str):
            raise FieldError("label keys and values must be strings", line,
                             f"labels.{key}")
        labels.append((key, val))
    return tuple(labels)


def _validate_refs(refs: Any, line: int, known_services: set[str]
                   ) -> tuple[str, ...]:
    if not isinstance(refs, list) or not all(isinstance(r, str) for r in refs):
        raise FieldError("root_cause_refs must be a list of strings", line,
                         "root_cause_refs")
    for ref in refs:
        if ref not in known_services:
            raise FieldError("dangling root-cause reference", line,
                             "root_cause_refs", ref)
    return tuple(sorted(set(refs)))


def _derived_fingerprint(service_id: str, severity: str, ts: int) -> str:
    return "fp_" + digest_parts([service_id, severity, ts])[:16]


def decode_record(text: str, line: int, default_silence_ms: int,
                  known_services: set[str]) -> NormalizedAlert:
    """Parse one raw NDJSON line and normalize it."""
    try:
        raw = json.loads(text)
    except json.JSONDecodeError as exc:
        raise FieldError("invalid JSON", line, "record", exc.msg) from exc

    version = detect_version(raw, line)
    ts = _require_int(raw, "timestamp_ms", line)
    if ts < 0:
        raise FieldError("timestamp_ms must be a non-negative integer", line,
                         "timestamp_ms")
    service_id = _require_str(raw, "service_id", line)
    if known_services and service_id not in known_services:
        raise FieldError("service_id is not declared in dependency graph",
                         line, "service_id", service_id)
    severity = _require_str(raw, "severity", line)
    if severity not in SEVERITIES:
        raise FieldError(f"severity must be one of {SEVERITIES}", line,
                         "severity", severity)

    filled: list[str] = []

    if version == "v1":
        fingerprint = _derived_fingerprint(service_id, severity, ts)
        silence_window_ms = int(default_silence_ms)
        labels = DEFAULT_LABELS
        refs = DEFAULT_REFS
        strategy_id = DEFAULT_STRATEGY
        filled.extend(["fingerprint", "silence_window_ms", "labels",
                       "root_cause_refs", "strategy_id"])
    else:
        fingerprint = _require_str(raw, "fingerprint", line)
        _validate_fingerprint(fingerprint, line)
        silence_window_ms = _require_int(raw, "silence_window_ms", line)
        if silence_window_ms < 0:
            raise FieldError("silence_window_ms must be a non-negative integer",
                             line, "silence_window_ms")
        labels = (_validate_labels(raw["labels"], line)
                  if "labels" in raw else DEFAULT_LABELS)
        if version == "v2":
            refs = DEFAULT_REFS
            strategy_id = DEFAULT_STRATEGY
            filled.extend(["labels", "root_cause_refs", "strategy_id"])
        else:
            refs = _validate_refs(raw.get("root_cause_refs", []), line,
                                  known_services)
            strategy_id = _require_str(raw, "strategy_id", line)
            if "labels" not in raw:
                filled.append("labels")

    # Every fill rule must be explicitly declared and traceable.
    for fname in filled:
        if (version, fname) not in FILL_RULES:
            raise FieldError(f"undeclared fill rule for field {fname!r}", line,
                             fname)

    # Identity is based on logical content only; ``line`` stays as
    # provenance and must not affect normalized bytes / alert identity.
    alert_id = stable_id("al", [version, service_id, severity, fingerprint,
                                ts, sorted(alert_labels_kv(labels)),
                                sorted(refs), strategy_id,
                                silence_window_ms])
    return NormalizedAlert(
        alert_id=alert_id,
        source_version=version,
        timestamp_ms=ts,
        service_id=service_id,
        severity=severity,
        fingerprint=fingerprint,
        silence_window_ms=silence_window_ms,
        labels=labels,
        root_cause_refs=refs,
        strategy_id=strategy_id,
        filled=tuple(filled),
        line=line,
    )


def encode_record(alert: NormalizedAlert) -> dict[str, Any]:
    """Inverse mapping: normalize back to the originating wire schema version."""
    base = {
        "version": alert.source_version,
        "timestamp_ms": alert.timestamp_ms,
        "service_id": alert.service_id,
        "severity": alert.severity,
    }
    if alert.source_version == "v1":
        return base
    base["fingerprint"] = alert.fingerprint
    base["silence_window_ms"] = alert.silence_window_ms
    if alert.source_version == "v2":
        return base
    base["root_cause_refs"] = list(alert.root_cause_refs)
    base["strategy_id"] = alert.strategy_id
    if alert.labels:
        base["labels"] = {k: v for k, v in alert.labels}
    return base
