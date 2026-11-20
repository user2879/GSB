"""Dot-product attention scoring with in-train / extrapolation stratification."""
from __future__ import annotations

import numpy as np


def compute_scores(
    rotated_query: np.ndarray,
    rotated_keys: np.ndarray,
    dim_range: tuple[int, int] | None = None,
    score_chunk: int | None = None,
) -> np.ndarray:
    """Dot-product scores q.k for each key.

    ``dim_range`` restricts the feature dimensions that participate, and
    ``score_chunk`` forces a chunked reduction -- both are driven by the
    selected pruning policy and therefore genuinely change the compute path.
    """
    q = np.asarray(rotated_query, dtype=np.float64)
    ks = np.asarray(rotated_keys, dtype=np.float64)
    if dim_range is not None:
        lo, hi = dim_range
        q = q[lo:hi]
        ks = ks[:, lo:hi]
    if score_chunk is None or score_chunk <= 0 or score_chunk >= q.shape[0]:
        return ks @ q
    scores = np.zeros(ks.shape[0], dtype=np.float64)
    for start in range(0, q.shape[0], score_chunk):
        stop = min(start + score_chunk, q.shape[0])
        scores += ks[:, start:stop] @ q[start:stop]
    return scores


def stratified_stats(
    scores: np.ndarray,
    query_pos: int,
    key_positions: list[int],
    train_len: int,
) -> dict:
    """Aggregate scores per relative distance, split by train-length strata.

    Returns ``{"in_train": {dist: {...}}, "extrapolation": {dist: {...}}}``
    where each entry carries mean/min/max/count. Distances are sorted so the
    output is deterministic.
    """
    buckets: dict[str, dict[int, list[float]]] = {"in_train": {}, "extrapolation": {}}
    for score, kpos in zip(scores, key_positions):
        dist = abs(int(query_pos) - int(kpos))
        stratum = "in_train" if dist <= train_len else "extrapolation"
        buckets[stratum].setdefault(dist, []).append(float(score))
    out: dict[str, dict] = {}
    for stratum in ("in_train", "extrapolation"):
        out[stratum] = {}
        for dist in sorted(buckets[stratum]):
            vals = np.asarray(buckets[stratum][dist], dtype=np.float64)
            out[stratum][str(dist)] = {
                "mean": float(vals.mean()),
                "min": float(vals.min()),
                "max": float(vals.max()),
                "count": int(vals.size),
            }
    return out


def merge_stratified(parts: list[dict]) -> dict:
    """Merge per-sample stratified stats into a scheme-level summary."""
    merged: dict[str, dict[int, dict]] = {"in_train": {}, "extrapolation": {}}
    for part in parts:
        for stratum in ("in_train", "extrapolation"):
            for dist, stats in part.get(stratum, {}).items():
                entry = merged[stratum].setdefault(
                    int(dist), {"sum": 0.0, "count": 0, "min": None, "max": None}
                )
                entry["sum"] += stats["mean"] * stats["count"]
                entry["count"] += stats["count"]
                entry["min"] = (
                    stats["min"] if entry["min"] is None else min(entry["min"], stats["min"])
                )
                entry["max"] = (
                    stats["max"] if entry["max"] is None else max(entry["max"], stats["max"])
                )
    out: dict[str, dict] = {}
    for stratum in ("in_train", "extrapolation"):
        out[stratum] = {}
        for dist in sorted(merged[stratum]):
            entry = merged[stratum][dist]
            out[stratum][str(dist)] = {
                "mean": entry["sum"] / entry["count"],
                "min": entry["min"],
                "max": entry["max"],
                "count": entry["count"],
            }
    return out
