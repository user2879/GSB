"""Canonical derived-state summary shared by nodes and the platform.

Both the node subprocess (what it puts in its receipt) and the platform (the
authoritative expected value) call :func:`node_summary`, so agreement is
byte-for-byte by construction.  NumPy performs the sample aggregation; all
floating rendering uses a fixed precision before canonical encoding.
"""
from __future__ import annotations

import math
from fractions import Fraction
from typing import Any

import numpy as np

from ..util.canonical import canonical_bytes, canonical_text, digest

_RENDER_DIGITS = 10


def _render(number: float) -> str:
    if not math.isfinite(number):
        raise ValueError("non-finite aggregate produced")
    return f"{number:.{_RENDER_DIGITS}f}"


def _aggregate(sample: list[float], aggregate: str) -> float:
    vector = np.asarray(sample, dtype=np.float64)
    if aggregate == "sum":
        return float(np.sum(vector))
    if aggregate == "mean":
        return float(np.mean(vector))
    if aggregate == "l2norm":
        return float(np.sqrt(np.sum(vector * vector)))
    if aggregate == "max":
        return float(np.max(vector))
    raise ValueError(f"unsupported aggregate: {aggregate}")


def _item_state(base: float, aggregate: str, item_real: Fraction) -> float:
    real = float(item_real)
    if aggregate in ("sum", "mean"):
        return base + real
    if aggregate == "l2norm":
        return math.sqrt(base * base + real * real)
    if aggregate == "max":
        return max(base, real)
    raise ValueError(f"unsupported aggregate: {aggregate}")


def node_summary(
    node_id: str,
    version: int,
    aggregate: str,
    sample: list[float],
    item_reals: dict[str, Fraction],
    ordered_keys: list[str],
) -> dict[str, Any]:
    """Build the deterministic summary object for one node receipt."""
    base = _aggregate(sample, aggregate)
    states: dict[str, str] = {}
    for key in ordered_keys:
        states[key] = _render(_item_state(base, aggregate, item_reals[key]))
    summary = {
        "node_id": node_id,
        "version": version,
        "aggregate": aggregate,
        "base": _render(base),
        "states": states,
    }
    summary["fingerprint"] = digest(
        {
            "node_id": node_id,
            "version": version,
            "aggregate": aggregate,
            "base": summary["base"],
            "states": states,
        }
    )
    return summary


def summary_bytes(summary: dict[str, Any]) -> bytes:
    return canonical_bytes(summary)


def summary_text(summary: dict[str, Any]) -> str:
    return canonical_text(summary)
