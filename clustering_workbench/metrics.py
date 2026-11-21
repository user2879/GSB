"""Deterministic clustering quality, density likelihood, ranking, and cache probes."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .cache import ResidentBlockCache
from .methods import CandidateSpec, MethodExecution
from .utils import hash_array, hash_object, json_safe


def evaluation_identity(
    cluster_identity_fingerprint: str,
    evaluation_config: dict[str, Any],
) -> str:
    requested_metrics = evaluation_config.get(
        "metrics", ["silhouette", "calinski_harabasz", "density_log_likelihood"]
    )
    return hash_object(
        "evalid",
        {
            "cluster_identity_fingerprint": cluster_identity_fingerprint,
            "metrics": requested_metrics,
            "density_reg_covar": float(evaluation_config.get("density_reg_covar", 1e-6)),
            "primary_metric": evaluation_config.get("primary_metric", "silhouette"),
        },
    )


@dataclass
class EvaluationResult:
    metrics: dict[str, float | None]
    metric_flags: list[str]
    diagnostic_flags: list[str]
    cluster_sizes: list[int]
    noise_count: int
    non_noise_count: int
    boundary_check: dict[str, Any] | None
    label_fingerprint: str
    evaluation_identity_fingerprint: str
    artifact_fingerprint: str
    cache_snapshot: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return json_safe(
            {
                "metrics": self.metrics,
                "metric_flags": self.metric_flags,
                "diagnostic_flags": self.diagnostic_flags,
                "cluster_sizes": self.cluster_sizes,
                "noise_count": self.noise_count,
                "non_noise_count": self.non_noise_count,
                "boundary_check": self.boundary_check,
                "label_fingerprint": self.label_fingerprint,
                "evaluation_identity_fingerprint": self.evaluation_identity_fingerprint,
                "artifact_fingerprint": self.artifact_fingerprint,
            }
        )


def _feature_blocks(
    X: np.ndarray,
    block_size: int,
    cache: ResidentBlockCache,
) -> list[np.ndarray]:
    blocks: list[np.ndarray] = []
    for start in range(0, X.shape[0], block_size):
        end = min(X.shape[0], start + block_size)
        key = f"feature:{start:07d}:{end:07d}"
        blocks.append(cache.get(key, lambda start=start, end=end: X[start:end].copy()))
    # Re-reference every block so evicted blocks are deterministically recomputed.
    for start in range(0, X.shape[0], block_size):
        end = min(X.shape[0], start + block_size)
        key = f"feature:{start:07d}:{end:07d}"
        cache.get(key, lambda start=start, end=end: X[start:end].copy())
    return blocks


def _silhouette(
    X: np.ndarray,
    labels: np.ndarray,
    cache: ResidentBlockCache,
    stable_key: str,
    block_size: int,
) -> float | None:
    valid = labels >= 0
    valid_labels = labels[valid]
    unique = np.unique(valid_labels)
    if unique.size < 2 or valid_labels.size <= unique.size:
        return None
    valid_indices = np.flatnonzero(valid)
    block_size = max(1, int(block_size))
    scores = np.zeros(valid_indices.size, dtype=np.float64)
    for position, point_index in enumerate(valid_indices):
        point = X[point_index]
        distance_sums = {int(cluster): 0.0 for cluster in unique}
        distance_counts = {int(cluster): 0 for cluster in unique}
        for start in range(0, X.shape[0], block_size):
            end = min(X.shape[0], start + block_size)
            key = f"feature:{start:07d}:{end:07d}"
            block = cache.get(key, lambda start=start, end=end: X[start:end].copy())
            difference = block - point
            distances = np.sqrt(np.sum(difference * difference, axis=1, dtype=np.float64))
            block_labels = labels[start:end]
            for row_offset in range(distances.size):
                global_index = start + row_offset
                label = int(block_labels[row_offset])
                if label < 0 or global_index == point_index:
                    continue
                distance_sums[label] += float(distances[row_offset])
                distance_counts[label] += 1
        own_label = int(labels[point_index])
        own_count = distance_counts.get(own_label, 0)
        if own_count == 0:
            scores[position] = 0.0
            continue
        a = distance_sums[own_label] / own_count
        b = float("inf")
        for cluster in unique:
            cluster_int = int(cluster)
            if cluster_int == own_label or distance_counts.get(cluster_int, 0) == 0:
                continue
            mean_distance = distance_sums[cluster_int] / distance_counts[cluster_int]
            b = min(b, mean_distance)
        if not math.isfinite(b):
            scores[position] = 0.0
        else:
            denominator = max(a, b)
            scores[position] = 0.0 if denominator == 0 else (b - a) / denominator

    # Materialize score blocks through the same budgeted cache.
    total = 0.0
    count = 0
    score_block_size = block_size
    for start in range(0, scores.size, score_block_size):
        end = min(scores.size, start + score_block_size)
        key = f"score:{stable_key}:silhouette:{start:07d}"
        local = scores[start:end]
        values = cache.get(key, lambda local=local: local.copy())
        values = cache.get(key, lambda local=local: local.copy())
        total += float(np.sum(values, dtype=np.float64))
        count += int(values.size)
    return total / count if count else None


def _calinski_harabasz(X: np.ndarray, labels: np.ndarray) -> float | None:
    valid = labels >= 0
    valid_X = X[valid]
    valid_labels = labels[valid]
    unique = np.unique(valid_labels)
    k = int(unique.size)
    n = int(valid_X.shape[0])
    if k < 2 or n <= k:
        return None
    global_mean = np.mean(valid_X, axis=0, dtype=np.float64)
    between = 0.0
    within = 0.0
    for cluster in unique:
        members = valid_X[valid_labels == cluster]
        if members.shape[0] == 0:
            continue
        mean = np.mean(members, axis=0, dtype=np.float64)
        difference = mean - global_mean
        between += float(members.shape[0] * np.sum(difference * difference, dtype=np.float64))
        residual = members - mean
        within += float(np.sum(residual * residual, dtype=np.float64))
    if within <= 1e-15:
        return None
    return float((between / (k - 1)) / (within / (n - k)))


def _logdet_and_quadratic(difference: np.ndarray, covariance: np.ndarray, reg_covar: float) -> tuple[float, np.ndarray]:
    covariance = covariance + np.eye(covariance.shape[0], dtype=np.float64) * reg_covar
    sign, logdet = np.linalg.slogdet(covariance)
    if sign <= 0 or not np.isfinite(logdet):
        covariance = covariance + np.eye(covariance.shape[0], dtype=np.float64) * 1e-8
        sign, logdet = np.linalg.slogdet(covariance)
    try:
        inverse = np.linalg.inv(covariance)
    except np.linalg.LinAlgError:
        inverse = np.linalg.pinv(covariance)
    quadratic = np.einsum("ij,jk,ik->i", difference, inverse, difference, optimize=False)
    return float(logdet), quadratic


def _density_log_likelihood(
    X: np.ndarray,
    labels: np.ndarray,
    reg_covar: float,
    execution: MethodExecution,
) -> float | None:
    if execution.loglik_per_sample is not None:
        values = np.asarray(execution.loglik_per_sample, dtype=np.float64)
        values = values[np.isfinite(values)]
        return float(np.mean(values, dtype=np.float64)) if values.size else None

    n_features = X.shape[1]
    per_sample = np.empty(X.shape[0], dtype=np.float64)
    valid = labels >= 0
    unique = np.unique(labels[valid])
    if unique.size == 0:
        return None
    global_mean = np.mean(X[valid], axis=0, dtype=np.float64)
    global_difference = X - global_mean
    global_covariance = (
        np.einsum("ij,ik->jk", global_difference, global_difference, optimize=False)
        / max(1, X.shape[0])
    )
    global_logdet, global_quadratic = _logdet_and_quadratic(
        global_difference, global_covariance, reg_covar
    )
    global_score = -0.5 * (
        n_features * math.log(2.0 * math.pi) + global_logdet + global_quadratic
    )
    for cluster in unique:
        members_mask = labels == cluster
        members = X[members_mask]
        mean = np.mean(members, axis=0, dtype=np.float64)
        difference = members - mean
        covariance = (
            np.einsum("ij,ik->jk", difference, difference, optimize=False)
            / max(1, members.shape[0])
        )
        logdet, quadratic = _logdet_and_quadratic(difference, covariance, reg_covar)
        per_sample[members_mask] = -0.5 * (
            n_features * math.log(2.0 * math.pi) + logdet + quadratic
        )
    noise_mask = labels < 0
    per_sample[noise_mask] = global_score[noise_mask]
    finite = per_sample[np.isfinite(per_sample)]
    return float(np.mean(finite, dtype=np.float64)) if finite.size else None


def evaluate_candidate(
    X: np.ndarray,
    spec: CandidateSpec,
    execution: MethodExecution,
    cluster_identity_fingerprint: str,
    evaluation_config: dict[str, Any],
    cache: ResidentBlockCache,
    block_size: int = 64,
) -> EvaluationResult:
    labels = np.asarray(execution.labels, dtype=np.int64)
    unique_labels = np.unique(labels)
    cluster_sizes = [int(value) for value in np.bincount(labels[labels >= 0]).tolist()]
    noise_count = int(np.count_nonzero(labels < 0))
    non_noise_count = int(labels.size - noise_count)
    flags = list(execution.diagnostic_flags)
    metric_flags: list[str] = []
    if any(size == 0 for size in cluster_sizes):
        if "EMPTY_CLUSTER" not in flags:
            flags.append("EMPTY_CLUSTER")
    if labels.size == noise_count:
        if "ALL_NOISE" not in flags:
            flags.append("ALL_NOISE")

    requested_metrics = evaluation_config.get(
        "metrics", ["silhouette", "calinski_harabasz", "density_log_likelihood"]
    )
    metrics: dict[str, float | None] = {}
    if "silhouette" in requested_metrics:
        metrics["silhouette"] = _silhouette(X, labels, cache, spec.stable_key, block_size)
        if metrics["silhouette"] is None:
            metric_flags.append("silhouette")
    if "calinski_harabasz" in requested_metrics:
        metrics["calinski_harabasz"] = _calinski_harabasz(X, labels)
        if metrics["calinski_harabasz"] is None:
            metric_flags.append("calinski_harabasz")
    if "density_log_likelihood" in requested_metrics:
        metrics["density_log_likelihood"] = _density_log_likelihood(
            X,
            labels,
            float(evaluation_config.get("density_reg_covar", 1e-6)),
            execution,
        )
        if metrics["density_log_likelihood"] is None:
            metric_flags.append("density_log_likelihood")
    if metric_flags:
        flags.append("METRIC_UNDEFINED")

    threshold = evaluation_config.get("boundary_threshold")
    boundary_check = None
    if threshold is not None and metrics.get("silhouette") is not None:
        silhouette = float(metrics["silhouette"])
        boundary_check = {
            "metric": "silhouette",
            "threshold": float(threshold),
            "score": silhouette,
            "operator": ">=",
            "decision": "accept" if silhouette >= float(threshold) else "reject",
            "boundary_hit": abs(silhouette - float(threshold)) <= 1e-15,
        }

    identity = evaluation_identity(cluster_identity_fingerprint, evaluation_config)
    artifact = hash_object(
        "evalart",
        {
            "identity": identity,
            "labels": hash_array(labels.astype(np.float64)),
            "metrics": metrics,
            "flags": flags,
            "boundary_check": boundary_check,
        },
    )
    return EvaluationResult(
        metrics=metrics,
        metric_flags=metric_flags,
        diagnostic_flags=flags,
        cluster_sizes=cluster_sizes,
        noise_count=noise_count,
        non_noise_count=non_noise_count,
        boundary_check=boundary_check,
        label_fingerprint=execution.label_fingerprint,
        evaluation_identity_fingerprint=identity,
        artifact_fingerprint=artifact,
    )


def metric_sort_value(value: float | None) -> float:
    return float("-inf") if value is None else float(value)


def rank_candidates(candidate_rows: list[dict[str, Any]], primary_metric: str) -> list[dict[str, Any]]:
    rankable = [
        row
        for row in candidate_rows
        if row.get("status") in {"completed", "converged", "truncated", "reused"}
        and row.get("metrics")
    ]

    def key(row: dict[str, Any]) -> tuple[Any, ...]:
        metrics = row.get("metrics", {})
        return (
            -metric_sort_value(metrics.get(primary_metric)),
            row["method"],
            row["canonical_params"],
            row.get("occurrence", "000"),
        )

    ordered = sorted(rankable, key=key)
    for index, row in enumerate(ordered, start=1):
        row["rank"] = index
    return ordered


def ranking_table(candidate_rows: list[dict[str, Any]], primary_metric: str) -> list[dict[str, Any]]:
    table = []
    for row in candidate_rows:
        table.append(
            {
                "rank": row.get("rank"),
                "display_id": row.get("display_id"),
                "stable_key": row.get("stable_key"),
                "method": row.get("method"),
                "params": row.get("params"),
                "status": row.get("status"),
                "metrics": row.get("metrics"),
                "diagnostic_flags": row.get("diagnostic_flags", []),
                "budget": row.get("budget"),
            }
        )
    return table
