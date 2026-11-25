"""Deterministic inference computation and output summaries.

The same pure function runs inside the hosted subprocess backend and inside
the gateway's verification path, so an output summary can be re-derived
byte-for-byte without trusting the network response.
"""
from __future__ import annotations

import hashlib

import numpy as np

from .canonical import canonical_bytes, content_hash


def _seed(param_summary: str) -> int:
    return int(hashlib.sha256(
        f"inference|{param_summary}".encode("utf-8")).hexdigest()[:16], 16)


def deterministic_output(vector: tuple[float, ...] | list[float],
                         model_dim: int, hidden_dim: int,
                         param_summary: str) -> list[float]:
    """Two-layer fixed-seed transform; identical inputs always give same floats."""
    if len(vector) != model_dim:
        raise ValueError("vector length does not match model_dim")
    rng = np.random.default_rng(_seed(param_summary))
    w1 = rng.standard_normal((hidden_dim, model_dim))
    w2 = rng.standard_normal((model_dim, hidden_dim))
    x = np.asarray(vector, dtype=np.float64)
    hidden = np.tanh(w1 @ x)
    out = w2 @ hidden
    return np.round(out, 9).tolist()


def output_summary(vector: tuple[float, ...] | list[float],
                   model_dim: int, hidden_dim: int,
                   param_summary: str) -> str:
    values = deterministic_output(vector, model_dim, hidden_dim, param_summary)
    return content_hash(values)[:32]


def output_summary_from_values(values: list[float]) -> str:
    return content_hash(np.round(np.asarray(values, dtype=np.float64), 9)
                        .tolist())[:32]
