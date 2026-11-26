"""Deterministic NumPy integer fixed-point aggregation.

Weights are derived per (seed, event_seq) through an explicit integer hash so
the same event always produces the same weight on every machine and regardless
of batch or arrival order. All arithmetic stays in signed 64-bit integers:
there are no floating-point values anywhere in the aggregate.
"""

from __future__ import annotations

import hashlib

import numpy as np

CALIBER_VERSION = "v1"
FIXED_SCALE = 1000
_WEIGHT_MOD = 1000


def _weight(seed: int, seq: int) -> int:
    key = f"gsb.weight|{CALIBER_VERSION}|{seed}|{seq}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(key).digest()[:8], "big") % _WEIGHT_MOD


def weight_vector(seed: int, event_seqs) -> np.ndarray:
    return np.array([_weight(seed, seq) for seq in event_seqs], dtype=np.int64)


def aggregate(seed: int, event_seqs, payloads) -> dict:
    """Return the bound integer fixed-point aggregate for one batch."""
    seqs = np.asarray(list(event_seqs), dtype=np.int64)
    values = np.asarray(list(payloads), dtype=np.int64)
    if seqs.shape != values.shape or seqs.size == 0:
        raise ValueError("event_seqs and payloads must be non-empty and aligned")
    weights = weight_vector(seed, seqs.tolist())
    weighted = np.multiply(weights, values, dtype=np.int64)
    weighted_sum = int(weighted.sum(dtype=np.int64))
    return {
        "caliber_version": CALIBER_VERSION,
        "event_count": int(seqs.size),
        "weight_scale": FIXED_SCALE,
        "weight_sum": int(weights.sum(dtype=np.int64)),
        "payload_sum": int(values.sum(dtype=np.int64)),
        "fixed_point_sum": weighted_sum,
        "fixed_point_q1000": int(weighted_sum // FIXED_SCALE),
    }


def summary_digest(
    batch_id: str, event_seqs, aggregate_doc: dict, caliber_version: str
) -> str:
    from .digest import sha256_hex

    document = {
        "kind": "gsb.summary_digest.v1",
        "batch_id": batch_id,
        "caliber_version": caliber_version,
        "event_seqs": list(event_seqs),
        "aggregate": aggregate_doc,
    }
    return sha256_hex(document)
