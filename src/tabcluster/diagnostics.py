"""Diagnostics summary: ranking, budget ledger, failures, cache stats."""

from __future__ import annotations

import math
from typing import Any

from .fingerprint import canonical_json


def _metric_sort_value(v: Any) -> float:
    if isinstance(v, dict):  # NaN marker from normalization
        return math.nan
    if v is None:
        return math.nan
    return float(v)


def rank_candidates(candidate_entries: list[dict],
                    primary_metric: str) -> list[str]:
    """Rank by primary metric desc; NaN last; ties by method name then the
    hyperparameter dictionary in canonical (sorted-key) order."""

    def key(entry: dict):
        v = _metric_sort_value(entry.get("metrics", {}).get(primary_metric))
        is_nan = 1 if (v != v) else 0
        neg = 0.0 if is_nan else -v
        return (is_nan, neg, entry["method"], canonical_json(entry["params"]))

    ordered = sorted(candidate_entries, key=key)
    return [e["short_key"] for e in ordered]


def build_diagnostics(*, experiment: str, seed: int, sample_scale: str,
                      n_rows: int, n_features: int, config_hash: str,
                      injection_hash: str, data_fingerprint: str,
                      preprocess_fingerprint: str, preprocess_meta: dict,
                      candidate_entries: list[dict], primary_metric: str,
                      budget: dict, failures: list[dict],
                      cache_stats: dict, manifest_version: int,
                      diagnostics_fingerprint: str) -> dict:
    return {
        "experiment": experiment,
        "seed": seed,
        "sample_scale": sample_scale,
        "n_rows": n_rows,
        "n_features": n_features,
        "config_hash": config_hash,
        "injection_hash": injection_hash,
        "data_fingerprint": data_fingerprint,
        "preprocess_fingerprint": preprocess_fingerprint,
        "preprocess": preprocess_meta,
        "candidates": candidate_entries,
        "ranking": rank_candidates(candidate_entries, primary_metric),
        "budget": budget,
        "failures": failures,
        "cache": cache_stats,
        "manifest_version": manifest_version,
        "diagnostics_fingerprint": diagnostics_fingerprint,
    }
