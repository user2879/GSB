from __future__ import annotations

import math
from typing import Any

import numpy as np

from .errors import (
    E_ALL_NOISE,
    E_EMPTY_CLUSTER,
    E_METRIC_UNAVAILABLE,
    Issue,
)
from .util import canonical_json


def _pairwise_distances(matrix: np.ndarray) -> np.ndarray:
    differences = matrix[:, None, :] - matrix[None, :, :]
    return np.sqrt(np.sum(differences * differences, axis=2, dtype=np.float64))


def silhouette_score(matrix: np.ndarray, labels: np.ndarray) -> tuple[float | None, int, dict[str, Any]]:
    clustered = np.flatnonzero(labels >= 0)
    if clustered.size < 2:
        return None, 0, {"reason": "fewer_than_two_clustered_samples"}
    unique = sorted(int(value) for value in np.unique(labels[clustered]))
    if len(unique) < 2:
        return None, int(clustered.size), {"reason": "fewer_than_two_clusters"}
    distances = _pairwise_distances(matrix)
    scores: list[float] = []
    for index in clustered:
        point_label = int(labels[index])
        same_mask = (labels == point_label)
        same_count = int(same_mask.sum())
        if same_count <= 1:
            scores.append(0.0)
            continue
        same_distance_sum = 0.0
        for other in np.flatnonzero(same_mask):
            if int(other) != int(index):
                same_distance_sum += float(distances[index, other])
        a = same_distance_sum / float(same_count - 1)
        b = math.inf
        for cluster in unique:
            if cluster == point_label:
                continue
            members = np.flatnonzero(labels == cluster)
            if members.size == 0:
                continue
            total = 0.0
            for other in members:
                total += float(distances[index, other])
            b = min(b, total / float(members.size))
        denominator = max(a, b)
        scores.append(0.0 if denominator == 0 else (b - a) / denominator)
    return float(sum(scores) / len(scores)), int(clustered.size), {}


def calinski_harabasz_score(matrix: np.ndarray, labels: np.ndarray) -> tuple[float | None, int, dict[str, Any]]:
    clustered = np.flatnonzero(labels >= 0)
    if clustered.size < 2:
        return None, 0, {"reason": "fewer_than_two_clustered_samples"}
    unique = sorted(int(value) for value in np.unique(labels[clustered]))
    n_samples = int(clustered.size)
    n_clusters = len(unique)
    if n_clusters < 2:
        return None, n_samples, {"reason": "fewer_than_two_clusters"}
    if n_samples <= n_clusters:
        return None, n_samples, {"reason": "not_enough_samples_for_degrees_of_freedom"}
    global_mean = np.zeros(matrix.shape[1], dtype=np.float64)
    for index in clustered:
        global_mean += matrix[index]
    global_mean /= float(n_samples)
    between = 0.0
    within = 0.0
    for cluster in unique:
        members = np.flatnonzero(labels == cluster)
        mean = np.zeros(matrix.shape[1], dtype=np.float64)
        for index in members:
            mean += matrix[index]
        mean /= float(members.size)
        difference = mean - global_mean
        between += float(members.size) * float(np.sum(difference * difference, dtype=np.float64))
        for index in members:
            delta = matrix[index] - mean
            within += float(np.sum(delta * delta, dtype=np.float64))
    if within <= 1e-15:
        return None, n_samples, {"reason": "within_cluster_dispersion_is_zero"}
    value = (between / within) * (float(n_samples - n_clusters) / float(n_clusters - 1))
    return float(value), n_samples, {}


def density_log_likelihood(
    matrix: np.ndarray, labels: np.ndarray, *, regularization: float
) -> tuple[float | None, int, dict[str, Any]]:
    """Mean log likelihood of a hard-label diagonal Gaussian mixture.

    DBSCAN noise is retained as an explicit component (label -1), so the metric
    still covers every sample while all-noise candidates remain diagnosable.
    """

    n_samples, dimension = matrix.shape
    if n_samples == 0:
        return None, 0, {"reason": "empty_sample_set"}
    clusters = sorted(int(value) for value in np.unique(labels))
    means: list[np.ndarray] = []
    variances: list[np.ndarray] = []
    weights: list[float] = []
    for cluster in clusters:
        members = np.flatnonzero(labels == cluster)
        mean = np.zeros(dimension, dtype=np.float64)
        for index in members:
            mean += matrix[index]
        mean /= float(members.size)
        variance = np.zeros(dimension, dtype=np.float64)
        for index in members:
            delta = matrix[index] - mean
            variance += delta * delta
        variance /= float(members.size)
        variance = np.maximum(variance + regularization, regularization)
        means.append(mean)
        variances.append(variance)
        weights.append(float(members.size) / float(n_samples))
    normalizer = dimension * math.log(2.0 * math.pi)
    total_log_likelihood = 0.0
    for index in range(n_samples):
        component_values: list[float] = []
        for mean, variance, weight in zip(means, variances, weights):
            delta = matrix[index] - mean
            log_determinant = float(np.sum(np.log(variance), dtype=np.float64))
            mahalanobis = float(np.sum((delta * delta) / variance, dtype=np.float64))
            component_values.append(
                math.log(max(weight, 1e-300))
                - 0.5 * (normalizer + log_determinant + mahalanobis)
            )
        maximum = max(component_values)
        log_sum = maximum + math.log(sum(math.exp(value - maximum) for value in component_values))
        total_log_likelihood += log_sum
    return float(total_log_likelihood / float(n_samples)), n_samples, {
        "components_including_noise": len(clusters)
    }


def evaluate_labels(
    matrix: np.ndarray,
    labels: np.ndarray,
    *,
    method: str,
    expected_clusters: int | None,
    metrics: list[str],
    density_regularization: float,
    candidate_id: str,
    params: dict[str, Any],
) -> tuple[dict[str, Any], list[Issue], list[int]]:
    issues: list[Issue] = []
    sample_count = int(labels.size)
    counts: dict[str, int] = {}
    noncomparable_reasons: list[str] = []

    if expected_clusters is not None:
        for cluster in range(expected_clusters):
            count = int(np.sum(labels == cluster))
            counts[str(cluster)] = count
            if count == 0:
                issues.append(
                    Issue(
                        E_EMPTY_CLUSTER,
                        f"cluster {cluster} is empty",
                        severity="warning",
                        context={
                            "scope": "evaluation",
                            "candidate": candidate_id,
                            "method": method,
                            "params": params,
                            "cluster": cluster,
                        },
                    )
                )
                noncomparable_reasons.append(f"empty_cluster:{cluster}")
    else:
        for cluster in sorted(int(value) for value in np.unique(labels) if int(value) >= 0):
            counts[str(cluster)] = int(np.sum(labels == cluster))

    if sample_count and np.all(labels == -1):
        issues.append(
            Issue(
                E_ALL_NOISE,
                "all samples were classified as noise",
                severity="warning",
                context={
                    "scope": "evaluation",
                    "candidate": candidate_id,
                    "method": method,
                    "params": params,
                },
            )
        )
        noncomparable_reasons.append("all_noise")

    comparable = not noncomparable_reasons
    result: dict[str, Any] = {}
    requested = metrics if metrics else ["silhouette", "calinski_harabasz", "density_log_likelihood"]
    coverage: dict[str, int] = {}
    for metric in requested:
        if metric == "silhouette":
            value, covered, extra = silhouette_score(matrix, labels)
        elif metric == "calinski_harabasz":
            value, covered, extra = calinski_harabasz_score(matrix, labels)
        elif metric == "density_log_likelihood":
            value, covered, extra = density_log_likelihood(
                matrix, labels, regularization=density_regularization
            )
        else:
            value, covered, extra = None, 0, {"reason": "unknown_metric"}
        metric_comparable = comparable and value is not None
        reason = None
        if value is None:
            reason = str(extra.get("reason", E_METRIC_UNAVAILABLE))
        elif not comparable:
            reason = noncomparable_reasons[0]
        result[metric] = {
            "value": value,
            "comparable": metric_comparable,
            "reason_code": None if metric_comparable else (reason or E_METRIC_UNAVAILABLE),
            "covered_samples": covered,
            "details": extra,
        }
        coverage[metric] = covered
    return result, issues, sorted(noncomparable_reasons)


def ranking_key(candidate: dict[str, Any]) -> tuple[Any, ...]:
    metrics = candidate.get("metrics", {})
    comparable = bool(candidate.get("metrics_comparable", False))
    silhouette = metrics.get("silhouette", {}).get("value")
    ch = metrics.get("calinski_harabasz", {}).get("value")
    density = metrics.get("density_log_likelihood", {}).get("value")
    return (
        0 if comparable else 1,
        -(float(silhouette) if silhouette is not None else -math.inf),
        -(float(ch) if ch is not None else -math.inf),
        -(float(density) if density is not None else -math.inf),
        str(candidate.get("method", "")),
        canonical_json(candidate.get("params", {})),
        str(candidate.get("id", "")),
    )


def rank_candidates(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    ranked = sorted(candidates, key=ranking_key)
    for position, candidate in enumerate(ranked, start=1):
        candidate["rank"] = position
        candidate["rank_tiebreak"] = {
            "metrics_first": True,
            "then": ["method_name", "canonical_params", "candidate_id"],
        }
    return ranked
