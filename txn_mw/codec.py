"""Binary framing for transaction event-log records.

Wire layout (network byte order)::

    magic (4 bytes, b"TMW1")
    payload_length (uint32)
    canonical JSON payload (payload_length bytes)
    sha256(payload) (32 bytes)

The JSON payload carries the five mandatory transaction fields plus an event
type and an arbitrary data object.  The trailing checksum protects the whole
payload.  ``half`` events additionally carry the message body, which lets the
decoder verify the mandatory ``body_digest`` field without consulting SQLite.
"""

from __future__ import annotations

import json
import hashlib
import struct
from typing import Any, Iterable

from .canonical import body_digest, canonical_bytes
from .errors import ProtocolError

MAGIC = b"TMW1"
HEADER = struct.Struct(">4sI")
CHECKSUM_SIZE = 32
MAX_RECORD_BYTES = 8 * 1024 * 1024
MANDATORY_FIELDS = ("txn_id", "version", "state", "event_seq", "body_digest", "type")
VALID_STATES = frozenset({"half", "committable", "consumed"})


def _require_type(payload: dict[str, Any], key: str, expected: type, path: str) -> Any:
    if key not in payload:
        raise ProtocolError("missing mandatory field", f"{path}.{key}")
    value = payload[key]
    if expected is int:
        if isinstance(value, bool) or not isinstance(value, int):
            raise ProtocolError("field must be an integer", f"{path}.{key}")
    elif not isinstance(value, expected):
        raise ProtocolError(f"field must be {expected.__name__}", f"{path}.{key}")
    return value


def validate_record_payload(payload: Any) -> dict[str, Any]:
    path = "$.record"
    if not isinstance(payload, dict):
        raise ProtocolError("record payload must be a JSON object", path)
    for key in MANDATORY_FIELDS:
        if key not in payload:
            raise ProtocolError("missing mandatory field", f"{path}.{key}")

    txn_id = _require_type(payload, "txn_id", str, path)
    if not txn_id or len(txn_id) > 128:
        raise ProtocolError("txn_id must be a non-empty string of at most 128 chars", f"{path}.txn_id")
    version = _require_type(payload, "version", int, path)
    if version <= 0:
        raise ProtocolError("version must be positive", f"{path}.version")
    state = _require_type(payload, "state", str, path)
    if state not in VALID_STATES:
        raise ProtocolError("unknown transaction state", f"{path}.state")
    seq = _require_type(payload, "event_seq", int, path)
    if seq <= 0:
        raise ProtocolError("event_seq must be positive", f"{path}.event_seq")
    digest = _require_type(payload, "body_digest", str, path)
    if len(digest) != 64 or any(ch not in "0123456789abcdef" for ch in digest.lower()):
        raise ProtocolError("body_digest must be a hex SHA-256", f"{path}.body_digest")
    event_type = _require_type(payload, "type", str, path)
    if not event_type:
        raise ProtocolError("type must be non-empty", f"{path}.type")

    data = payload.get("data", {})
    if not isinstance(data, dict):
        raise ProtocolError("data must be an object", f"{path}.data")
    if event_type == "half" and "body" in data:
        actual = body_digest(data["body"])
        if actual.lower() != digest.lower():
            raise ProtocolError("message body digest does not match", f"{path}.body_digest")
    return payload


def encode_record(payload: dict[str, Any]) -> bytes:
    validate_record_payload(payload)
    encoded = canonical_bytes(payload)
    if len(encoded) > MAX_RECORD_BYTES:
        raise ProtocolError("record payload exceeds size limit", "$.record.length")
    return HEADER.pack(MAGIC, len(encoded)) + encoded + hashlib.sha256(encoded).digest()


def decode_record(data: bytes, expected_seq: int | None = None, after_seq: int | None = None) -> dict[str, Any]:
    if not isinstance(data, (bytes, bytearray, memoryview)):
        raise ProtocolError("record must be bytes", "$.record")
    raw = bytes(data)
    if len(raw) < HEADER.size + CHECKSUM_SIZE:
        raise ProtocolError("record is shorter than its framing", "$.record.length")
    magic, payload_len = HEADER.unpack(raw[: HEADER.size])
    if magic != MAGIC:
        raise ProtocolError("invalid record magic", "$.record.magic")
    expected_len = HEADER.size + payload_len + CHECKSUM_SIZE
    if len(raw) != expected_len:
        raise ProtocolError(
            "length field does not match record size",
            "$.record.length",
            {"declared": expected_len, "actual": len(raw)},
        )
    if payload_len > MAX_RECORD_BYTES:
        raise ProtocolError("declared payload exceeds size limit", "$.record.length")
    payload_bytes = raw[HEADER.size : HEADER.size + payload_len]
    checksum = raw[-CHECKSUM_SIZE:]
    if hashlib.sha256(payload_bytes).digest() != checksum:
        raise ProtocolError("record payload checksum mismatch", "$.record.digest")
    try:
        payload = json.loads(payload_bytes.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ProtocolError(f"invalid canonical JSON payload: {exc}", "$.record.payload") from exc
    validate_record_payload(payload)
    seq = payload["event_seq"]
    if expected_seq is not None and seq != expected_seq:
        raise ProtocolError("event_seq does not match ledger position", "$.record.event_seq")
    if after_seq is not None and seq <= after_seq:
        raise ProtocolError("event_seq is not strictly increasing", "$.record.event_seq")
    return payload


def validate_sequence(payloads: Iterable[dict[str, Any]]) -> None:
    per_txn: dict[str, int] = {}
    for item in payloads:
        txn_id = item["txn_id"]
        seq = item["event_seq"]
        previous = per_txn.get(txn_id)
        if previous is not None and seq <= previous:
            raise ProtocolError(
                "event_seq must strictly increase for a transaction",
                "$.record.event_seq",
                {"txn_id": txn_id, "previous": previous, "current": seq},
            )
        per_txn[txn_id] = seq


def decode_records(records: Iterable[bytes]) -> list[dict[str, Any]]:
    """Decode a record stream and enforce per-transaction sequence monotonicity."""
    decoded = [decode_record(record) for record in records]
    validate_sequence(decoded)
    return decoded
