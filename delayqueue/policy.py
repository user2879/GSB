"""Deterministic consumer policy shared by the demo and the replay engine.

Integer payload semantics (kept deliberately tiny and fully deterministic):

* ``payload > 0`` -> the consumer fails exactly that many times, then acks.
* ``payload == 0`` -> ack immediately.

The value is encoded as minimal big-endian bytes by the loaders, so both the
live workers and the replay simulator derive the exact same failure sequence.
"""

from __future__ import annotations


def integer_value(payload: bytes) -> int:
    return int.from_bytes(payload, "big")


def planned_failures(payload: bytes) -> int:
    return integer_value(payload)


def should_fail(attempt_index: int, payload: bytes) -> bool:
    """``attempt_index`` is zero-based within one message's delivery history."""
    return attempt_index < planned_failures(payload)
