"""JSON-lines protocol with per-session field whitelists.

Every message carries a ``wl`` field equal to the current session whitelist
version. Validation rejects missing required fields, unexpected extra fields
and wrong field types with E_FIELD, recording the field path and the
whitelist version used for the check.
"""
from __future__ import annotations

import json

from .errors import E_FIELD, PlatformError

WL_SCHEMA_VERSION = 1


def wl_version_for_session(session_seq: int) -> str:
    return f"wl{WL_SCHEMA_VERSION}.{session_seq}"


def _type_ok(value, expected) -> bool:
    if expected is bool:
        return isinstance(value, bool)
    if expected is int:
        return isinstance(value, int) and not isinstance(value, bool)
    if expected is float:
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if expected is str:
        return isinstance(value, str)
    if expected is list:
        return isinstance(value, list)
    if expected is dict:
        return isinstance(value, dict)
    return False


MESSAGE_SPEC = {
    "session": {"type": str, "wl": str, "session": str},
    "hello": {"type": str, "wl": str, "node_id": str},
    "op": {
        "type": str, "wl": str, "lease_id": str, "node_id": str,
        "op": str, "agg": str, "sectors": list,
        "seq_lo": int, "seq_hi": int, "delta": float,
    },
    "op_result": {
        "type": str, "wl": str, "lease_id": str, "node_id": str,
        "status": str, "summary": dict, "progress": int,
    },
    "error": {"type": str, "wl": str, "code": str, "path": str},
    "shutdown": {"type": str, "wl": str},
}

SUMMARY_SPEC = {
    "lease_id": str, "sectors": list, "seq_lo": int, "seq_hi": int,
    "versions": dict, "value": float,
}


def _check_fields(obj, spec, wl, path="$"):
    if not isinstance(obj, dict):
        raise PlatformError(E_FIELD, path=path, wl=wl, reason="expected object")
    for key, typ in spec.items():
        if key not in obj:
            raise PlatformError(E_FIELD, path=f"{path}.{key}", wl=wl,
                                reason="missing required field")
        if not _type_ok(obj[key], typ):
            raise PlatformError(E_FIELD, path=f"{path}.{key}", wl=wl,
                                reason=f"type mismatch, expected {typ.__name__}")
    for key in obj:
        if key not in spec:
            raise PlatformError(E_FIELD, path=f"{path}.{key}", wl=wl,
                                reason="unexpected extra field")


def validate_summary(summary, wl):
    _check_fields(summary, SUMMARY_SPEC, wl, path="$.summary")
    for key, val in summary["versions"].items():
        if not isinstance(key, str) or not _type_ok(val, int):
            raise PlatformError(E_FIELD, path=f"$.summary.versions.{key}",
                                wl=wl, reason="versions must map str->int")


def validate_message(msg, wl):
    """Validate a decoded message against the current session whitelist."""
    if not isinstance(msg, dict):
        raise PlatformError(E_FIELD, path="$", wl=wl, reason="expected object")
    mtype = msg.get("type")
    if not isinstance(mtype, str) or mtype not in MESSAGE_SPEC:
        raise PlatformError(E_FIELD, path="$.type", wl=wl,
                            reason="unknown message type")
    if msg.get("wl") != wl:
        raise PlatformError(E_FIELD, path="$.wl", wl=wl,
                            reason="whitelist version mismatch",
                            expected=wl, got=msg.get("wl"))
    _check_fields(msg, MESSAGE_SPEC[mtype], wl)
    if mtype == "op_result" and isinstance(msg.get("summary"), dict):
        validate_summary(msg["summary"], wl)
    return msg


def encode(msg) -> bytes:
    return (json.dumps(msg, separators=(",", ":")) + "\n").encode("utf-8")


def decode(line, wl):
    try:
        msg = json.loads(line)
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise PlatformError(E_FIELD, path="$", wl=wl,
                            reason=f"invalid json: {exc}") from exc
    return validate_message(msg, wl)
