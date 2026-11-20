"""Deterministic synthetic tabular data generation.

All randomness flows from a single seeded RandomState consumed in a fixed
column order, so the same description always yields byte-identical data.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .config_loader import DataDescription
from .fingerprint import fingerprint


def resolve_rows(desc: DataDescription, scale: str) -> int:
    factor = desc.scales.get(scale)
    if factor is None:
        factor = 1
    return int(desc.base_rows) * int(factor)


def generate(desc: DataDescription, scale: str) -> dict[str, Any]:
    """Generate raw columns. Missing values are None inside object arrays."""
    n_rows = resolve_rows(desc, scale)
    rng = np.random.RandomState(desc.seed)
    columns: dict[str, np.ndarray] = {}
    for spec in desc.columns:  # fixed description order
        p = spec.params
        missing_rate = float(p.get("missing_rate", 0.0))
        if spec.type == "numeric":
            dist = p.get("distribution", "normal")
            if dist == "normal":
                vals = rng.normal(loc=float(p.get("mean", 0.0)),
                                  scale=float(p.get("std", 1.0)), size=n_rows)
            elif dist == "uniform":
                vals = rng.uniform(low=float(p.get("low", 0.0)),
                                   high=float(p.get("high", 1.0)), size=n_rows)
            else:  # int
                vals = rng.randint(low=int(p.get("low", 0)),
                                   high=int(p.get("high", 10)),
                                   size=n_rows).astype(np.float64)
            col = vals.astype(object)
        elif spec.type == "categorical":
            cats = [str(c) for c in p["categories"]]
            weights = p.get("weights")
            if weights is not None:
                w = np.asarray(weights, dtype=np.float64)
                w = w / w.sum()
            else:
                w = None
            col = rng.choice(cats, size=n_rows, p=w).astype(object)
        elif spec.type == "high_cardinality":
            card = int(p["cardinality"])
            cats = [f"hc_{i:05d}" for i in range(card)]
            ranks = np.arange(1, card + 1, dtype=np.float64)
            w = 1.0 / ranks
            w = w / w.sum()
            col = rng.choice(cats, size=n_rows, p=w).astype(object)
        elif spec.type == "redundant":
            src = columns[str(p["source"])]
            col = src.copy()
        else:  # pragma: no cover - validated earlier
            continue
        if spec.type != "redundant" and missing_rate > 0.0:
            mask = rng.rand(n_rows) < missing_rate
            col = col.copy()
            col[mask] = None
        columns[spec.name] = col
    return {"columns": columns, "n_rows": n_rows, "scale": scale}


def apply_sample_replacements(data: dict[str, Any],
                              injections: list[dict]) -> list[dict]:
    """Apply replace_sample injections in manifest order (deterministic)."""
    applied = []
    cols = data["columns"]
    for inj in injections:
        if inj.get("type") != "replace_sample":
            continue
        name = str(inj.get("column"))
        row = int(inj.get("row", -1))
        if name not in cols or not (0 <= row < data["n_rows"]):
            continue
        cols[name][row] = inj.get("value")
        applied.append({"id": inj.get("id"), "column": name, "row": row,
                        "value": inj.get("value")})
    return applied


def _is_numeric_col(col: np.ndarray) -> bool:
    for v in col:
        if v is None:
            continue
        return isinstance(v, (int, float, np.integer, np.floating))
    return True


def data_fingerprint(desc: DataDescription, data: dict[str, Any]) -> str:
    """Content fingerprint of the raw generated data (post-injection)."""
    parts: list[Any] = ["raw-data", desc.desc_fingerprint, data["scale"],
                        data["n_rows"]]
    for spec in desc.columns:  # fixed order
        col = data["columns"][spec.name]
        if spec.type == "numeric" or (
                spec.type == "redundant" and _is_numeric_col(col)):
            arr = np.array([np.nan if v is None else float(v) for v in col],
                           dtype=np.float64)
            parts.append(arr.tobytes())
        else:
            parts.append(["" if v is None else str(v) for v in col])
    return fingerprint(*parts)
