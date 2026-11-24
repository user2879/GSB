"""Deterministic loopback UDP probe protocol."""

from __future__ import annotations

import hashlib
import json
from typing import Any, Mapping

import numpy as np

PROTOCOL_VERSION = 1
MAX_DATAGRAM_BYTES = 65_507


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def make_request(
    *,
    session_id: str,
    probe_id: str,
    cycle: int,
    lease_version: int,
    attempt: int,
    tx_id: str,
) -> dict[str, Any]:
    return {
        "attempt": attempt,
        "cycle": cycle,
        "kind": "probe.request",
        "lease_version": lease_version,
        "probe_id": probe_id,
        "protocol_version": PROTOCOL_VERSION,
        "session_id": session_id,
        "tx_id": tx_id,
    }


def encode_request(request: Mapping[str, Any]) -> bytes:
    return canonical_json(dict(request)).encode("utf-8")


def decode_json_datagram(data: bytes) -> dict[str, Any]:
    if len(data) > MAX_DATAGRAM_BYTES:
        raise ValueError("datagram_too_large")
    try:
        value = json.loads(data.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid_json:{exc}") from exc
    if not isinstance(value, dict):
        raise ValueError("payload_not_object")
    return value


def validate_request(value: dict[str, Any]) -> None:
    required = {
        "attempt": int,
        "cycle": int,
        "kind": str,
        "lease_version": int,
        "probe_id": str,
        "protocol_version": int,
        "session_id": str,
        "tx_id": str,
    }
    for key, expected in required.items():
        actual = value.get(key)
        if isinstance(actual, bool) or not isinstance(actual, expected):
            raise ValueError(f"invalid_field:{key}")
    if value["kind"] != "probe.request":
        raise ValueError("invalid_kind")
    if value["protocol_version"] != PROTOCOL_VERSION:
        raise ValueError("unsupported_protocol_version")
    if value["cycle"] < 1 or value["attempt"] < 1:
        raise ValueError("invalid_cycle_or_attempt")


def response_sequence(cycle: int, sequence_offset: int) -> int:
    return cycle + sequence_offset


def response_token(seed: int, session_id: str, probe_id: str, cycle: int) -> str:
    mixed_seed = ((seed & 0xFFFFFFFFFFFFFFFF) ^ ((cycle * 0x9E3779B1) & 0xFFFFFFFFFFFFFFFF))
    rng = np.random.default_rng(mixed_seed)
    sample = int(rng.integers(0, 2**63 - 1, dtype=np.int64))
    material = f"{seed}:{session_id}:{probe_id}:{cycle}:{sample}".encode("utf-8")
    return hashlib.sha256(material).hexdigest()


def make_response(
    request: dict[str, Any], seed: int, sequence_offset: int
) -> dict[str, Any]:
    cycle = int(request["cycle"])
    return {
        "cycle": cycle,
        "kind": "probe.response",
        "lease_version": int(request["lease_version"]),
        "probe_id": request["probe_id"],
        "protocol_version": PROTOCOL_VERSION,
        "seq": response_sequence(cycle, sequence_offset),
        "session_id": request["session_id"],
        "token": response_token(seed, request["session_id"], request["probe_id"], cycle),
        "tx_id": request["tx_id"],
    }
