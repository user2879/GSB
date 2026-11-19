"""Downstream metrics, four-stage precision deltas, and error attribution."""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from .config import STAGE_ORDER
from .errors import E_PRECISION_SENSITIVE
from .fingerprint import array_fingerprint, fingerprint


def _dtype(precision: str) -> np.dtype:
    return np.dtype(precision)


def compute_metric(reduction: dict[str, Any], metric_name: str, precision: str) -> dict[str, Any]:
    dtype = _dtype(precision)
    scores = np.asarray(reduction["scores"], dtype=dtype)
    if metric_name == "first_coordinate":
        values = np.asarray(scores[:, 0], dtype=dtype)
    elif metric_name == "l2_norm":
        values = np.asarray(np.sqrt(np.sum(scores * scores, axis=1, dtype=dtype)), dtype=dtype)
    elif metric_name == "mean_absolute":
        values = np.asarray(np.mean(np.abs(scores), axis=1, dtype=dtype), dtype=dtype)
    else:
        raise ValueError(f"unsupported metric: {metric_name}")
    parameter_fingerprint = fingerprint(
        {
            "kind": "metrics",
            "metric": metric_name,
            "precision": precision,
            "upstream": reduction["fingerprint"],
        }
    )
    result_fingerprint = fingerprint(
        {
            "upstream": reduction["fingerprint"],
            "parameters": parameter_fingerprint,
            "precision": precision,
            "values": array_fingerprint(values),
        }
    )
    finite = values[np.isfinite(values)]
    return {
        "scenario_id": reduction["scenario_id"],
        "sample_size": reduction["sample_size"],
        "order_limit": reduction["order_limit"],
        "path": reduction["path"],
        "precision": precision,
        "sample_ids": np.asarray(reduction["sample_ids"], dtype=object).copy(),
        "metric_name": metric_name,
        "values": values,
        "aggregate": {
            "mean": float(np.mean(finite)) if finite.size else None,
            "median": float(np.median(finite)) if finite.size else None,
            "std": float(np.std(finite, ddof=0)) if finite.size else None,
            "minimum": float(np.min(finite)) if finite.size else None,
            "maximum": float(np.max(finite)) if finite.size else None,
        },
        "upstream_fingerprint": reduction["fingerprint"],
        "parameter_fingerprint": parameter_fingerprint,
        "fingerprint": result_fingerprint,
    }


def threshold_from_reference(metric: dict[str, Any], decision: dict[str, Any]) -> float:
    mode = decision.get("threshold_mode", "fixed")
    if mode == "fixed":
        return float(decision.get("threshold", 0.0))
    if mode == "reference_median":
        values = np.asarray(metric["values"], dtype=np.float64)
        finite = values[np.isfinite(values)]
        if finite.size == 0:
            return 0.0
        return float(np.median(finite))
    raise ValueError(f"unsupported threshold mode: {mode}")


def _row_max_abs(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    left64 = np.asarray(left, dtype=np.float64)
    right64 = np.asarray(right, dtype=np.float64)
    if left64.shape != right64.shape:
        raise ValueError(f"array shape mismatch: {left64.shape} vs {right64.shape}")
    if left64.ndim == 1:
        return np.abs(left64 - right64)
    return np.max(np.abs(left64 - right64), axis=1)


def _feature_max_abs(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    left64 = np.asarray(left, dtype=np.float64)
    right64 = np.asarray(right, dtype=np.float64)
    if left64.ndim == 1:
        return np.abs(left64 - right64)
    return np.max(np.abs(left64 - right64), axis=0)


def _stage_stat(name: str, per_sample: np.ndarray, tolerance: float) -> dict[str, Any]:
    array = np.asarray(per_sample, dtype=np.float64)
    finite = array[np.isfinite(array)]
    return {
        "stage": name,
        "tolerance": float(tolerance),
        "max_error": float(np.max(finite)) if finite.size else 0.0,
        "mean_error": float(np.mean(finite)) if finite.size else 0.0,
        "p95_error": float(np.quantile(finite, 0.95)) if finite.size else 0.0,
        "exceeded_sample_count": int(np.count_nonzero(finite > tolerance)),
    }


def _origin_for_stage(stage: str | None) -> str:
    if stage in {"cross_encoding", "standardization"}:
        return "encoding_rounding"
    if stage in {"projection", "final_metric"}:
        return "projection_accumulation"
    return "below_tolerance"


def compare_precision(
    *,
    scenario_id: str,
    sample_size: int,
    order_limit: int,
    path: str,
    sample_ids: np.ndarray,
    design32: dict[str, Any],
    design64: dict[str, Any],
    standard32: dict[str, Any],
    standard64: dict[str, Any],
    reduction32: dict[str, Any],
    reduction64: dict[str, Any],
    metric32: dict[str, Any],
    metric64: dict[str, Any],
    decision_config: dict[str, Any],
    tolerances: dict[str, float],
) -> dict[str, Any]:
    threshold = threshold_from_reference(metric64, decision_config)
    values32 = np.asarray(metric32["values"], dtype=np.float64)
    values64 = np.asarray(metric64["values"], dtype=np.float64)
    decision32 = values32 >= threshold
    decision64 = values64 >= threshold

    errors = {
        "cross_encoding": _row_max_abs(design32["matrix"], design64["matrix"]),
        "standardization": _row_max_abs(standard32["matrix"], standard64["matrix"]),
        "projection": _row_max_abs(reduction32["scores"], reduction64["scores"]),
        "final_metric": _row_max_abs(values32, values64),
    }
    feature_errors = {
        "cross_encoding": _feature_max_abs(design32["matrix"], design64["matrix"]),
        "standardization": _feature_max_abs(standard32["matrix"], standard64["matrix"]),
        "projection": _feature_max_abs(reduction32["scores"], reduction64["scores"]),
        "final_metric": _feature_max_abs(values32, values64),
    }

    per_sample: list[dict[str, Any]] = []
    precision_sensitive: list[dict[str, Any]] = []
    ids = [str(value) for value in sample_ids]
    for index, sample_id in enumerate(ids):
        stage_values = {stage: float(errors[stage][index]) for stage in STAGE_ORDER}
        first_exceeded: str | None = None
        for stage in STAGE_ORDER:
            if stage_values[stage] > float(tolerances[stage]):
                first_exceeded = stage
                break
        normalized = {
            stage: stage_values[stage] / float(tolerances[stage]) if float(tolerances[stage]) > 0.0 else 0.0
            for stage in STAGE_ORDER
        }
        if first_exceeded is not None:
            primary_stage: str | None = first_exceeded
        else:
            best_stage = max(STAGE_ORDER, key=lambda stage: normalized[stage])
            primary_stage = best_stage if normalized[best_stage] > 0.0 else None
        record = {
            "sample_id": sample_id,
            "stage_errors": stage_values,
            "first_exceeded_stage": first_exceeded,
            "primary_error_stage": primary_stage,
            "metric_float32": float(values32[index]),
            "metric_float64": float(values64[index]),
            "metric_delta": float(abs(values32[index] - values64[index])),
            "decision_float32": bool(decision32[index]),
            "decision_float64": bool(decision64[index]),
        }
        per_sample.append(record)
        if bool(decision32[index]) != bool(decision64[index]):
            precision_sensitive.append(
                {
                    "code": E_PRECISION_SENSITIVE,
                    "scenario_id": scenario_id,
                    "sample_size": sample_size,
                    "order_limit": order_limit,
                    "path": path,
                    "sample_id": sample_id,
                    "threshold": threshold,
                    "metric_float32": float(values32[index]),
                    "metric_float64": float(values64[index]),
                    "metric_delta": float(abs(values32[index] - values64[index])),
                    "decision_float32": bool(decision32[index]),
                    "decision_float64": bool(decision64[index]),
                    "first_exceeded_stage": first_exceeded,
                    "primary_error_stage": primary_stage,
                    "error_origin": _origin_for_stage(primary_stage),
                }
            )

    stage_summary = [
        _stage_stat(stage, errors[stage], float(tolerances[stage]))
        for stage in STAGE_ORDER
    ]
    feature_error_summary = {
        stage: {
            "names": (
                list(design32["feature_names"])
                if stage in {"cross_encoding", "standardization"}
                else [f"component_{index + 1}" for index in range(len(feature_errors[stage]))]
            ),
            "max_abs": [float(value) for value in feature_errors[stage]],
        }
        for stage in STAGE_ORDER
    }
    precision_sensitive.sort(key=lambda item: (item["scenario_id"], item["sample_size"], item["order_limit"], item["path"], item["sample_id"]))
    result = {
        "scenario_id": scenario_id,
        "sample_size": sample_size,
        "order_limit": order_limit,
        "path": path,
        "sample_ids": list(ids),
        "threshold": threshold,
        "decision_metric": metric32["metric_name"],
        "metrics_float32": {
            "aggregate": metric32["aggregate"],
            "fingerprint": metric32["fingerprint"],
        },
        "metrics_float64": {
            "aggregate": metric64["aggregate"],
            "fingerprint": metric64["fingerprint"],
        },
        "decision_true_count_float32": int(np.count_nonzero(decision32)),
        "decision_true_count_float64": int(np.count_nonzero(decision64)),
        "stage_summary": stage_summary,
        "feature_error_summary": feature_error_summary,
        "per_sample": per_sample,
        "precision_sensitive": precision_sensitive,
        "max_stage_error": {
            stage: max(
                (float(errors[stage][index]) for index in range(len(ids))),
                default=0.0,
            )
            for stage in STAGE_ORDER
        },
    }
    result["comparison_fingerprint"] = fingerprint(
        {
            "scenario_id": scenario_id,
            "sample_size": sample_size,
            "order_limit": order_limit,
            "path": path,
            "threshold": threshold,
            "stage_errors": {stage: [float(value) for value in errors[stage]] for stage in STAGE_ORDER},
            "metrics32": metric32["fingerprint"],
            "metrics64": metric64["fingerprint"],
        }
    )
    return result
