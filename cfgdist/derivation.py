"""Deterministic derived-state digest computation.

The digest must be identical for the same configuration version on every node
and every recovery path, independent of receipt arrival order. Inputs are
therefore sorted by key and all numeric values go through float64 NumPy
aggregation, whose results are rendered through a stable repr.
"""

from __future__ import annotations

import hashlib
from typing import Any

import numpy as np

from .manifest import SUPPORTED_OPS, canonical_json


def _as_float(value: Any) -> float:
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError as exc:
            raise ValueError(f"non-numeric string cannot participate in derivation: {value!r}") from exc
    raise ValueError(f"unsupported derivation input type: {type(value).__name__}")


def aggregate(values: list[Any], op: str) -> float:
    if op not in SUPPORTED_OPS:
        raise ValueError(f"unsupported op {op!r}")
    if op == "count":
        return round(float(len(values)), 10)
    array = np.asarray([_as_float(value) for value in values], dtype=np.float64)
    if op == "sum":
        result = float(np.sum(array))
    elif op == "mean":
        result = float(np.mean(array))
    elif op == "min":
        result = float(np.min(array))
    elif op == "max":
        result = float(np.max(array))
    return round(result, 10)


def _stable_number(value: float) -> str:
    return np.format_float_scientific(np.float64(value), precision=15, trim="k", unique=True)


def compute_derived_value(derivation: dict, keyed_values: dict[str, Any]) -> dict:
    inputs = sorted(derivation["inputs"])
    missing = [key for key in inputs if key not in keyed_values]
    if missing:
        raise ValueError(
            f"derivation '{derivation['name']}' missing inputs on node "
            f"{derivation.get('target_node')}: {missing}"
        )
    raw = aggregate([keyed_values[key] for key in inputs], derivation["op"])
    return {
        "name": derivation["name"],
        "op": derivation["op"],
        "inputs": inputs,
        "value": _stable_number(raw),
    }


def node_digest(
    node_id: str,
    entries: list[dict],
    derivations: list[dict],
) -> str:
    """Expected digest of the derived state held by one node."""
    keyed = {entry["key"]: entry["value"] for entry in entries}
    local_derivations = sorted(
        (item for item in derivations if item["target_node"] == node_id),
        key=lambda item: item["name"],
    )
    derived = [compute_derived_value(item, keyed) for item in local_derivations]
    payload = {
        "node": node_id,
        "entries": sorted(
            ({"key": entry["key"], "value": entry["value"]} for entry in entries),
            key=lambda item: item["key"],
        ),
        "derived": derived,
    }
    return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()
