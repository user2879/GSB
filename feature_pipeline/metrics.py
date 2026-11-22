"""Downstream decision metrics and boundary attribution helpers."""

from __future__ import annotations

from typing import Any

import numpy as np

from .config import ENGINE_REVISION, PipelineConfig
from .standardize import dtype_for_tier
from .util import fingerprint


def _safe_accuracy(decisions: np.ndarray, target: np.ndarray) -> float:
    if target.size == 0:
        return 0.0
    return float(np.mean(decisions.astype(np.int8) == target.astype(np.int8), dtype=np.float64))


def compute_metrics(
    projected: list[list[float]],
    projected_raw: list[list[float]],
    target: list[int],
    *,
    tier: str,
    path: str,
    threshold: float,
) -> dict[str, Any]:
    dtype = dtype_for_tier(tier)
    matrix = np.asarray(projected, dtype=dtype)
    raw_matrix = np.asarray(projected_raw, dtype=dtype)
    labels = np.asarray(target, dtype=np.int8)
    if matrix.ndim != 2 or matrix.shape[1] < 1:
        raise ValueError("metrics require at least one reduced dimension")
    positive = labels == 1
    negative = labels == 0
    if positive.any():
        positive_center = matrix[positive].mean(axis=0, dtype=dtype)
    else:
        positive_center = matrix.mean(axis=0, dtype=dtype)
    if negative.any():
        negative_center = matrix[negative].mean(axis=0, dtype=dtype)
    else:
        negative_center = matrix.mean(axis=0, dtype=dtype)
    direction = np.asarray(positive_center - negative_center, dtype=dtype)
    center = matrix.mean(axis=0, dtype=dtype)
    scores = (matrix - center) @ direction
    raw_scores = (raw_matrix - center) @ direction
    # Compare the fixed threshold in double precision so float32 score rounding
    # remains observable at an audited boundary instead of being rounded again.
    decision_threshold = np.asarray(threshold, dtype=np.float64)
    decisions = scores.astype(np.float64) >= decision_threshold
    raw_decisions = raw_scores.astype(np.float64) >= decision_threshold
    accuracy = _safe_accuracy(decisions, labels)
    raw_accuracy = _safe_accuracy(raw_decisions, labels)
    decision_count = int(decisions.sum())
    return {
        "content": {
            "path": path,
            "tier": tier,
            "threshold": float(threshold),
            "scores": [float(value) for value in scores],
            "raw_scores": [float(value) for value in raw_scores],
            "decisions": [int(value) for value in decisions],
            "raw_decisions": [int(value) for value in raw_decisions],
            "metrics": {
                "accuracy": accuracy,
                "raw_accuracy": raw_accuracy,
                "positive_rate": float(np.mean(decisions, dtype=np.float64)),
                "mean_score": float(np.mean(scores, dtype=dtype)),
                "min_score": float(np.min(scores)),
                "max_score": float(np.max(scores)),
                "decision_count": decision_count,
                "sample_count": int(labels.size),
            },
            "classifier": {
                "type": "centroid_direction",
                "direction_norm": float(np.linalg.norm(direction)),
                "direction_fingerprint": fingerprint([float(value) for value in direction]),
                "threshold": float(threshold),
            },
        }
    }


def metrics_cache_key(
    config: PipelineConfig,
    tier: str,
    path: str,
    upstream_reduction_content_fingerprint: str,
    target_fingerprint: str,
) -> str:
    return fingerprint(
        {
            "stage": "metrics",
            "engine_revision": ENGINE_REVISION,
            "precision_fingerprint": config.precision_fingerprint(tier),
            "path": path,
            "upstream_reduction_content_fingerprint": upstream_reduction_content_fingerprint,
            "target_fingerprint": target_fingerprint,
            "params": config.raw["metrics"],
        }
    )


def attribute_boundary_cases(
    *,
    path: str,
    single: dict[str, Any],
    double: dict[str, Any],
) -> list[dict[str, Any]]:
    """Explain each row-level decision disagreement.

    If the two tiers already disagree before fixed-point projection rounding,
    the flip is attributed to dimensional numerical approximation.  Otherwise,
    fixed-point quantization is the deciding contributor.
    """

    single_decisions = single["decisions"]
    double_decisions = double["decisions"]
    single_raw = single["raw_decisions"]
    double_raw = double["raw_decisions"]
    rows = min(len(single_decisions), len(double_decisions))
    cases: list[dict[str, Any]] = []
    for row in range(rows):
        if single_decisions[row] == double_decisions[row]:
            continue
        approximation_changed = single_raw[row] != double_raw[row]
        quantization_changed = (
            single_decisions[row] != single_raw[row]
            or double_decisions[row] != double_raw[row]
        )
        if approximation_changed:
            source = "dimensional_approximation"
        else:
            source = "quantization"
        cases.append(
            {
                "path": path,
                "row": row,
                "source": source,
                "single_decision": int(single_decisions[row]),
                "double_decision": int(double_decisions[row]),
                "single_raw_decision": int(single_raw[row]),
                "double_raw_decision": int(double_raw[row]),
                "single_score": float(single["scores"][row]),
                "double_score": float(double["scores"][row]),
                "score_delta": float(single["scores"][row]) - float(double["scores"][row]),
                "quantization_changed": bool(quantization_changed),
                "approximation_changed": bool(approximation_changed),
                "sort_key": f"boundary/{path}/row={row:08d}",
            }
        )
    return cases
