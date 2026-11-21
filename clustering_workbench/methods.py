"""Deterministic implementations of KMeans, DBSCAN, and Gaussian mixture."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np

from .utils import canonical_json, hash_array, sha256_text


METHOD_DEFAULTS: dict[str, dict[str, Any]] = {
    "KMeans": {"n_clusters": 3, "max_iter": 30, "tol": 1e-6},
    "DBSCAN": {"eps": 0.6, "min_samples": 5},
    "GaussianMixture": {
        "n_components": 3,
        "covariance_type": "full",
        "max_iter": 40,
        "tol": 1e-5,
        "reg_covar": 1e-6,
    },
}


@dataclass(frozen=True)
class CandidateSpec:
    method: str
    params: dict[str, Any]
    stable_key: str
    lexical_key: tuple[str, str, str]
    configured_index: int
    display_id: str


@dataclass
class MethodExecution:
    labels: np.ndarray
    status: str
    iterations_executed: int
    iterations_requested: int
    diagnostics: dict[str, Any]
    trajectory: list[float]
    loglik_per_sample: np.ndarray | None = None
    diagnostic_flags: list[str] = field(default_factory=list)
    error: str | None = None
    error_code: str | None = None

    @property
    def label_fingerprint(self) -> str:
        return hash_array(self.labels.astype(np.float64))


def normalize_method_params(method: str, params: dict[str, Any] | None) -> dict[str, Any]:
    normalized = dict(METHOD_DEFAULTS.get(method, {}))
    normalized.update(params or {})
    for key in ("n_clusters", "n_components", "max_iter", "min_samples", "quantile_points"):
        if key in normalized:
            try:
                normalized[key] = int(normalized[key])
            except (TypeError, ValueError):
                pass
    for key in ("eps", "tol", "reg_covar"):
        if key in normalized:
            try:
                normalized[key] = float(normalized[key])
            except (TypeError, ValueError):
                pass
    if "covariance_type" in normalized:
        normalized["covariance_type"] = str(normalized["covariance_type"]).lower()
    return normalized


def build_candidate_specs(config: dict[str, Any]) -> list[CandidateSpec]:
    raw_candidates: list[tuple[str, dict[str, Any], int]] = []
    configured_index = 0
    methods = config.get("methods", {})
    for method in ("KMeans", "DBSCAN", "GaussianMixture"):
        method_config = methods.get(method, {})
        if not isinstance(method_config, dict) or not method_config.get("enabled", True):
            continue
        candidates = method_config.get("candidates", [])
        if not isinstance(candidates, list):
            continue
        for raw in candidates:
            params = normalize_method_params(method, raw if isinstance(raw, dict) else {})
            raw_candidates.append((method, params, configured_index))
            configured_index += 1

    raw_candidates.sort(
        key=lambda item: (item[0], canonical_json(item[1]), item[2])
    )
    occurrences: dict[tuple[str, str], int] = {}
    specs: list[CandidateSpec] = []
    for method, params, original_index in raw_candidates:
        canonical_params = canonical_json(params)
        occurrence_key = (method, canonical_params)
        occurrence = occurrences.get(occurrence_key, 0)
        occurrences[occurrence_key] = occurrence + 1
        fingerprint = sha256_text(canonical_params)[:16]
        suffix = f":{occurrence + 1:02d}" if occurrence else ""
        stable_key = f"{method}:{fingerprint}{suffix}"
        lexical = (method, canonical_params, f"{occurrence:03d}")
        display_id = f"{method}-{fingerprint[:8]}{suffix.replace(':', '-')}"
        specs.append(
            CandidateSpec(
                method=method,
                params=params,
                stable_key=stable_key,
                lexical_key=lexical,
                configured_index=original_index,
                display_id=display_id,
            )
        )
    return specs


def validate_candidate_params(
    spec: CandidateSpec,
    n_samples: int,
) -> tuple[str | None, str | None]:
    params = spec.params
    if spec.method == "KMeans":
        try:
            clusters = int(params.get("n_clusters"))
        except (TypeError, ValueError):
            return "CFG_PARAMETER_OUT_OF_RANGE", "n_clusters must be an integer"
        if clusters <= 0:
            return "CFG_CLUSTER_ZERO", "n_clusters must be greater than zero"
        if clusters > n_samples:
            return "CFG_PARAMETER_OUT_OF_RANGE", "n_clusters cannot exceed sample count"
        if int(params.get("max_iter", 1)) <= 0:
            return "CFG_PARAMETER_OUT_OF_RANGE", "max_iter must be positive"
    elif spec.method == "DBSCAN":
        if float(params.get("eps", 0.0)) <= 0:
            return "CFG_PARAMETER_OUT_OF_RANGE", "eps must be positive"
        if int(params.get("min_samples", 0)) <= 0:
            return "CFG_PARAMETER_OUT_OF_RANGE", "min_samples must be positive"
    elif spec.method == "GaussianMixture":
        try:
            components = int(params.get("n_components"))
        except (TypeError, ValueError):
            return "CFG_PARAMETER_OUT_OF_RANGE", "n_components must be an integer"
        if components <= 0:
            return "CFG_CLUSTER_ZERO", "n_components must be greater than zero"
        if components > n_samples:
            return "CFG_PARAMETER_OUT_OF_RANGE", "n_components cannot exceed sample count"
        if int(params.get("max_iter", 1)) <= 0:
            return "CFG_PARAMETER_OUT_OF_RANGE", "max_iter must be positive"
        if str(params.get("covariance_type")) not in {"full", "diag", "spherical", "tied"}:
            return "CFG_PARAMETER_OUT_OF_RANGE", "unsupported covariance_type"
        if float(params.get("reg_covar", -1.0)) < 0:
            return "CFG_PARAMETER_OUT_OF_RANGE", "reg_covar must be non-negative"
    return None, None


def _squared_distances(X: np.ndarray, centroids: np.ndarray) -> np.ndarray:
    distances = np.empty((X.shape[0], centroids.shape[0]), dtype=np.float64)
    for index in range(centroids.shape[0]):
        difference = X - centroids[index]
        distances[:, index] = np.sum(difference * difference, axis=1, dtype=np.float64)
    return distances


def _points_to_centroids(X: np.ndarray, centroids: np.ndarray) -> np.ndarray:
    return np.argmin(_squared_distances(X, centroids), axis=1).astype(np.int64)


def _farthest_first_initialization(X: np.ndarray, k: int) -> tuple[np.ndarray, list[int]]:
    if X.shape[0] == 0:
        raise ValueError("Cannot initialize centroids on an empty matrix")
    first_index = min(
        range(X.shape[0]),
        key=lambda index: tuple(float(value) for value in X[index]) + (index,),
    )
    selected = [int(first_index)]
    distances = np.sum((X - X[first_index]) ** 2, axis=1, dtype=np.float64)
    selected_mask = np.zeros(X.shape[0], dtype=bool)
    selected_mask[first_index] = True
    while len(selected) < k:
        masked_distances = distances.copy()
        masked_distances[selected_mask] = -1.0
        # np.argmax returns the first maximum, giving an index-order tie break.
        next_index = int(np.argmax(masked_distances))
        selected.append(next_index)
        selected_mask[next_index] = True
        candidate_distances = np.sum((X - X[next_index]) ** 2, axis=1, dtype=np.float64)
        distances = np.minimum(distances, candidate_distances)
    return X[np.asarray(selected, dtype=np.int64)].copy(), selected


def _run_kmeans(X: np.ndarray, params: dict[str, Any], grant: int) -> MethodExecution:
    k = int(params["n_clusters"])
    requested = int(params.get("max_iter", 30))
    effective = max(0, min(requested, int(grant)))
    tol = float(params.get("tol", 1e-6))
    centroids, initial_indices = _farthest_first_initialization(X, k)
    labels = np.full(X.shape[0], -1, dtype=np.int64)
    trajectory: list[float] = []
    converged = False
    shift = float("inf")
    for iteration in range(effective):
        distances = _squared_distances(X, centroids)
        new_labels = np.argmin(distances, axis=1).astype(np.int64)
        inertia = float(np.sum(distances[np.arange(X.shape[0]), new_labels], dtype=np.float64))
        trajectory.append(inertia)
        new_centroids = centroids.copy()
        for cluster in range(k):
            members = new_labels == cluster
            if np.any(members):
                new_centroids[cluster] = np.mean(X[members], axis=0, dtype=np.float64)
        shift = float(np.max(np.linalg.norm(new_centroids - centroids, axis=1)))
        changed = not np.array_equal(new_labels, labels)
        labels = new_labels
        centroids = new_centroids
        if not changed and shift <= tol:
            converged = True
            break
    if effective == 0:
        labels = np.full(X.shape[0], -1, dtype=np.int64)
    counts = np.bincount(labels, minlength=k).tolist() if np.all(labels >= 0) else [0] * k
    empty_clusters = [index for index, count in enumerate(counts) if count == 0]
    status = "converged" if converged else "truncated"
    if effective == 0:
        status = "aborted"
    flags = ["EMPTY_CLUSTER"] if empty_clusters else []
    diagnostics = {
        "algorithm": "KMeans",
        "initial_centroid_indices": initial_indices,
        "initialization": "lexicographic-first + farthest-first",
        "cluster_sizes": counts,
        "empty_clusters": empty_clusters,
        "converged": converged,
        "final_centroid_shift": shift,
        "iterations_requested": requested,
        "iterations_admitted": effective,
        "tie_break": "lowest_cluster_index",
    }
    return MethodExecution(
        labels=labels,
        status=status,
        iterations_executed=len(trajectory),
        iterations_requested=requested,
        diagnostics=diagnostics,
        trajectory=trajectory,
        diagnostic_flags=flags,
    )


def _run_dbscan(X: np.ndarray, params: dict[str, Any], grant: int) -> MethodExecution:
    requested = 1
    if grant < 1:
        return MethodExecution(
            labels=np.full(X.shape[0], -1, dtype=np.int64),
            status="aborted",
            iterations_executed=0,
            iterations_requested=requested,
            diagnostics={"algorithm": "DBSCAN", "reason": "no admitted budget"},
            trajectory=[],
            diagnostic_flags=["ALL_NOISE"],
        )
    eps = float(params["eps"])
    min_samples = int(params["min_samples"])
    n_rows = X.shape[0]
    neighbor_lists: list[np.ndarray] = []
    point_scans = 0
    for index in range(n_rows):
        difference = X - X[index]
        distances = np.sqrt(np.sum(difference * difference, axis=1, dtype=np.float64))
        neighbors = np.flatnonzero(distances <= eps + 1e-12).astype(np.int64)
        neighbor_lists.append(neighbors)
        point_scans += 1
    core = np.asarray([neighbors.size >= min_samples for neighbors in neighbor_lists], dtype=bool)
    labels = np.full(n_rows, -1, dtype=np.int64)
    visited = np.zeros(n_rows, dtype=bool)
    cluster_id = 0
    for point in range(n_rows):
        if visited[point]:
            continue
        visited[point] = True
        if not core[point]:
            continue
        labels[point] = cluster_id
        queue = list(int(value) for value in neighbor_lists[point])
        queue_index = 0
        while queue_index < len(queue):
            candidate = queue[queue_index]
            queue_index += 1
            if not visited[candidate]:
                visited[candidate] = True
                if core[candidate]:
                    for neighbor in neighbor_lists[candidate]:
                        neighbor_int = int(neighbor)
                        if not visited[neighbor_int] and neighbor_int not in queue[queue_index:]:
                            queue.append(neighbor_int)
            if labels[candidate] == -1:
                labels[candidate] = cluster_id
        cluster_id += 1
    counts = np.bincount(labels[labels >= 0], minlength=cluster_id).tolist() if cluster_id else []
    noise_count = int(np.count_nonzero(labels == -1))
    flags = []
    if noise_count == n_rows:
        flags.append("ALL_NOISE")
    if any(count == 0 for count in counts):
        flags.append("EMPTY_CLUSTER")
    diagnostics = {
        "algorithm": "DBSCAN",
        "eps": eps,
        "min_samples": min_samples,
        "point_scans": point_scans,
        "core_points": int(np.count_nonzero(core)),
        "noise_points": noise_count,
        "cluster_count": int(cluster_id),
        "cluster_sizes": counts,
        "neighbor_order": "ascending point index",
        "budget_cost_model": "one complete density pass",
        "iterations_requested": requested,
        "iterations_admitted": 1,
    }
    return MethodExecution(
        labels=labels,
        status="completed",
        iterations_executed=1,
        iterations_requested=requested,
        diagnostics=diagnostics,
        trajectory=[],
        diagnostic_flags=flags,
    )


def _log_gaussian_components(
    X: np.ndarray,
    means: np.ndarray,
    covariances: np.ndarray,
    covariance_type: str,
) -> tuple[np.ndarray, int]:
    n_rows = X.shape[0]
    n_components, n_features = means.shape
    log_prob = np.empty((n_rows, n_components), dtype=np.float64)
    regularization_events = 0
    common_covariance = covariances[0] if covariance_type == "tied" else None
    for component in range(n_components):
        difference = X - means[component]
        if covariance_type in {"full", "tied"}:
            covariance = common_covariance if covariance_type == "tied" else covariances[component]
            sign, logdet = np.linalg.slogdet(covariance)
            if sign <= 0 or not np.isfinite(logdet):
                covariance = covariance + np.eye(n_features, dtype=np.float64) * 1e-8
                sign, logdet = np.linalg.slogdet(covariance)
                regularization_events += 1
            try:
                inverse = np.linalg.inv(covariance)
            except np.linalg.LinAlgError:
                inverse = np.linalg.pinv(covariance)
                regularization_events += 1
            quadratic = np.einsum("ij,jk,ik->i", difference, inverse, difference, optimize=False)
        elif covariance_type == "diag":
            variance = np.asarray(covariances[component], dtype=np.float64)
            if np.any(variance <= 0) or not np.all(np.isfinite(variance)):
                variance = np.maximum(variance, 1e-8)
                regularization_events += 1
            quadratic = np.sum(difference * difference / variance, axis=1, dtype=np.float64)
            logdet = float(np.sum(np.log(variance), dtype=np.float64))
        else:
            variance = float(covariances[component])
            if variance <= 0 or not math.isfinite(variance):
                variance = 1e-8
                regularization_events += 1
            quadratic = np.sum(difference * difference, axis=1, dtype=np.float64) / variance
            logdet = float(n_features * math.log(variance))
        log_prob[:, component] = -0.5 * (
            n_features * math.log(2.0 * math.pi) + logdet + quadratic
        )
    return log_prob, regularization_events


def _logsumexp_weights(log_prob: np.ndarray, weights: np.ndarray) -> np.ndarray:
    weighted = log_prob + np.log(np.maximum(weights, 1e-300))[None, :]
    maximum = np.max(weighted, axis=1, keepdims=True)
    return (maximum + np.log(np.sum(np.exp(weighted - maximum), axis=1, keepdims=True)))[:, 0]


def _regularize_covariance(
    covariance: np.ndarray,
    covariance_type: str,
    reg_covar: float,
) -> np.ndarray:
    if covariance_type in {"full", "tied"}:
        return covariance + np.eye(covariance.shape[0], dtype=np.float64) * reg_covar
    if covariance_type == "diag":
        return covariance + reg_covar
    return float(covariance + reg_covar)


def _run_gaussian_mixture(X: np.ndarray, params: dict[str, Any], grant: int) -> MethodExecution:
    n_rows, n_features = X.shape
    k = int(params["n_components"])
    requested = int(params.get("max_iter", 40))
    effective = max(0, min(requested, int(grant)))
    tol = float(params.get("tol", 1e-5))
    reg_covar = float(params.get("reg_covar", 1e-6))
    covariance_type = str(params.get("covariance_type", "full"))
    means, _ = _farthest_first_initialization(X, k)
    initial_labels = _points_to_centroids(X, means)
    responsibilities = np.zeros((n_rows, k), dtype=np.float64)
    responsibilities[np.arange(n_rows), initial_labels] = 1.0
    weights = responsibilities.mean(axis=0)
    trajectory: list[float] = []
    converged = False
    singular_events = 0
    empty_components: list[int] = []
    previous_loglik = float("-inf")
    labels = initial_labels

    def normalize_weights(raw_weights: np.ndarray) -> np.ndarray:
        raw_weights = np.maximum(raw_weights, 1e-12)
        return raw_weights / np.sum(raw_weights, dtype=np.float64)

    for _iteration in range(effective):
        counts = np.sum(responsibilities, axis=0, dtype=np.float64)
        weights = normalize_weights(counts / max(1, n_rows))
        new_means = means.copy()
        for component in range(k):
            if counts[component] > 1e-12:
                new_means[component] = (
                    np.sum(responsibilities[:, component, None] * X, axis=0, dtype=np.float64)
                    / counts[component]
                )
        means = new_means
        covariances = np.zeros((k,), dtype=object)
        if covariance_type == "full":
            for component in range(k):
                if counts[component] > 1e-12:
                    difference = X - means[component]
                    covariance = (
                        np.einsum("i,ij,ik->jk", responsibilities[:, component], difference, difference, optimize=False)
                        / counts[component]
                    )
                else:
                    covariance = np.eye(n_features, dtype=np.float64)
                covariances[component] = _regularize_covariance(covariance, "full", reg_covar)
        elif covariance_type == "diag":
            for component in range(k):
                if counts[component] > 1e-12:
                    difference = X - means[component]
                    variance = (
                        np.sum(responsibilities[:, component, None] * difference * difference, axis=0, dtype=np.float64)
                        / counts[component]
                    )
                else:
                    variance = np.ones(n_features, dtype=np.float64)
                covariances[component] = _regularize_covariance(variance, "diag", reg_covar)
        elif covariance_type == "spherical":
            for component in range(k):
                if counts[component] > 1e-12:
                    difference = X - means[component]
                    variance = (
                        np.sum(responsibilities[:, component, None] * difference * difference, dtype=np.float64)
                        / (counts[component] * n_features)
                    )
                else:
                    variance = 1.0
                covariances[component] = _regularize_covariance(variance, "spherical", reg_covar)
        else:
            total_count = max(float(np.sum(counts)), 1e-12)
            covariance = np.zeros((n_features, n_features), dtype=np.float64)
            for component in range(k):
                difference = X - means[component]
                covariance += (
                    np.einsum("i,ij,ik->jk", responsibilities[:, component], difference, difference, optimize=False)
                    / total_count
                )
            covariances[0] = _regularize_covariance(covariance, "tied", reg_covar)

        log_prob, events = _log_gaussian_components(X, means, covariances, covariance_type)
        singular_events += events
        loglik_values = _logsumexp_weights(log_prob, weights)
        loglik = float(np.sum(loglik_values, dtype=np.float64))
        trajectory.append(loglik)
        weighted = log_prob + np.log(np.maximum(weights, 1e-300))[None, :]
        maximum = np.max(weighted, axis=1, keepdims=True)
        responsibilities = np.exp(weighted - maximum)
        responsibilities /= np.sum(responsibilities, axis=1, keepdims=True)
        labels = np.argmax(responsibilities, axis=1).astype(np.int64)
        if math.isfinite(previous_loglik) and abs(loglik - previous_loglik) <= tol:
            converged = True
            previous_loglik = loglik
            break
        previous_loglik = loglik

    counts = np.bincount(labels, minlength=k).tolist()
    empty_components = [index for index, count in enumerate(counts) if count == 0]
    flags = []
    if empty_components:
        flags.append("EMPTY_CLUSTER")
    if singular_events:
        flags.append("COVARIANCE_REGULARIZED")
    if effective == 0:
        labels = np.full(n_rows, -1, dtype=np.int64)
        status = "aborted"
        loglik_per_sample = None
    else:
        status = "converged" if converged else "truncated"
        log_prob, _ = _log_gaussian_components(X, means, covariances, covariance_type)
        loglik_per_sample = _logsumexp_weights(log_prob, weights)
    diagnostics = {
        "algorithm": "GaussianMixture",
        "covariance_type": covariance_type,
        "reg_covar": reg_covar,
        "component_weights": weights.tolist(),
        "component_sizes": counts,
        "empty_components": empty_components,
        "converged": converged,
        "singular_regularization_events": singular_events,
        "iterations_requested": requested,
        "iterations_admitted": effective,
        "initialization": "lexicographic-first + farthest-first hard responsibilities",
        "means": means.tolist(),
    }
    return MethodExecution(
        labels=labels,
        status=status,
        iterations_executed=len(trajectory),
        iterations_requested=requested,
        diagnostics=diagnostics,
        trajectory=trajectory,
        loglik_per_sample=loglik_per_sample,
        diagnostic_flags=flags,
    )


def execute_method(
    spec: CandidateSpec,
    X: np.ndarray,
    grant: int,
    algorithm_runner: Callable[[str, np.ndarray, dict[str, Any], int], MethodExecution] | None = None,
) -> MethodExecution:
    if algorithm_runner is not None:
        return algorithm_runner(spec.method, X, spec.params, grant)
    if spec.method == "KMeans":
        return _run_kmeans(X, spec.params, grant)
    if spec.method == "DBSCAN":
        return _run_dbscan(X, spec.params, grant)
    if spec.method == "GaussianMixture":
        return _run_gaussian_mixture(X, spec.params, grant)
    return MethodExecution(
        labels=np.full(X.shape[0], -1, dtype=np.int64),
        status="error",
        iterations_executed=0,
        iterations_requested=0,
        diagnostics={"algorithm": spec.method},
        trajectory=[],
        error=f"Unsupported method {spec.method}",
        error_code="CANDIDATE_RUNTIME_ERROR",
    )
