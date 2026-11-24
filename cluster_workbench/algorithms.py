from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .errors import E_COVARIANCE_SINGULAR, E_MAX_ITER_REACHED, Issue


@dataclass
class ClusterResult:
    labels: np.ndarray
    iterations: int
    converged: bool
    objective_trace: list[float]
    issues: list[Issue] = field(default_factory=list)
    extras: dict[str, Any] = field(default_factory=dict)
    requested_iterations: int = 1

    @property
    def budget_limited(self) -> bool:
        return self.iterations >= self.requested_iterations and not self.converged


def _squared_distances(matrix: np.ndarray, centroids: np.ndarray) -> np.ndarray:
    # Explicit broadcasting keeps tie order and arithmetic order deterministic.
    differences = matrix[:, None, :] - centroids[None, :, :]
    return np.sum(differences * differences, axis=2, dtype=np.float64)


def _farthest_initial_centroids(matrix: np.ndarray, k: int) -> np.ndarray:
    n_samples = matrix.shape[0]
    # First point: lexicographically smallest row, implemented without sorting
    # all rows through an unstable external library.
    first = 0
    for index in range(1, n_samples):
        if tuple(matrix[index]) < tuple(matrix[first]):
            first = index
    selected = [first]
    min_distance = np.sum((matrix - matrix[first]) ** 2, axis=1, dtype=np.float64)
    while len(selected) < k:
        best_index = -1
        best_distance = -1.0
        for index in range(n_samples):
            if index in selected:
                continue
            distance = float(min_distance[index])
            if distance > best_distance:
                best_distance = distance
                best_index = index
        if best_index < 0:
            # Duplicate rows can exhaust geometrically distinct choices.
            for index in range(n_samples):
                if index not in selected:
                    best_index = index
                    break
        if best_index < 0:
            break
        selected.append(best_index)
        candidate_distance = np.sum((matrix - matrix[best_index]) ** 2, axis=1, dtype=np.float64)
        min_distance = np.minimum(min_distance, candidate_distance)
    if len(selected) < k:
        selected.extend(range(k - len(selected)))
    return matrix[np.asarray(selected[:k], dtype=np.int64)].copy()


def kmeans(
    matrix: np.ndarray,
    *,
    n_clusters: int,
    max_iter: int,
    tolerance: float,
    requested_iterations: int | None = None,
) -> ClusterResult:
    centroids = _farthest_initial_centroids(matrix, n_clusters)
    previous_labels: np.ndarray | None = None
    trace: list[float] = []
    converged = False
    iterations = 0
    issues: list[Issue] = []
    requested = int(requested_iterations if requested_iterations is not None else max_iter)
    for iteration in range(1, max_iter + 1):
        distances = _squared_distances(matrix, centroids)
        labels = np.argmin(distances, axis=1).astype(np.int64)
        inertia = float(np.sum(distances[np.arange(matrix.shape[0]), labels], dtype=np.float64))
        trace.append(inertia)
        iterations = iteration
        new_centroids = np.empty_like(centroids)
        for cluster in range(n_clusters):
            members = matrix[labels == cluster]
            if members.shape[0] == 0:
                # Keep the centroid stable.  Empty clusters are reported by the
                # evaluation layer instead of being silently reseeded.
                new_centroids[cluster] = centroids[cluster]
            else:
                total = np.zeros(matrix.shape[1], dtype=np.float64)
                for row in members:
                    total += row
                new_centroids[cluster] = total / float(members.shape[0])
        centroid_shift = float(np.max(np.abs(new_centroids - centroids))) if centroids.size else 0.0
        labels_stable = previous_labels is not None and np.array_equal(labels, previous_labels)
        centroids = new_centroids
        previous_labels = labels
        if labels_stable or centroid_shift <= tolerance:
            converged = True
            break
    if not converged and iterations >= requested:
        issues.append(
            Issue(
                E_MAX_ITER_REACHED,
                "kmeans reached the configured iteration limit before convergence",
                severity="warning",
                context={"scope": "algorithm", "method": "kmeans"},
            )
        )
    distances = _squared_distances(matrix, centroids)
    labels = np.argmin(distances, axis=1).astype(np.int64)
    return ClusterResult(
        labels=labels,
        iterations=iterations,
        converged=converged,
        objective_trace=trace,
        issues=issues,
        extras={"centroids": centroids.tolist()},
        requested_iterations=requested,
    )


def _distance_matrix(matrix: np.ndarray) -> np.ndarray:
    differences = matrix[:, None, :] - matrix[None, :, :]
    return np.sqrt(np.sum(differences * differences, axis=2, dtype=np.float64))


def dbscan(matrix: np.ndarray, *, eps: float, min_samples: int) -> ClusterResult:
    n_samples = matrix.shape[0]
    distances = _distance_matrix(matrix)
    neighbors: list[list[int]] = []
    for index in range(n_samples):
        row = np.flatnonzero(distances[index] <= eps)
        neighbors.append([int(item) for item in row])
    core = np.asarray([len(items) >= min_samples for items in neighbors], dtype=bool)
    labels = np.full(n_samples, -1, dtype=np.int64)
    visited = np.zeros(n_samples, dtype=bool)
    cluster = 0
    expansion_count = 0
    for index in range(n_samples):
        if visited[index]:
            continue
        visited[index] = True
        if not core[index]:
            continue
        queue = list(neighbors[index])
        cursor = 0
        labels[index] = cluster
        while cursor < len(queue):
            point = queue[cursor]
            cursor += 1
            if not visited[point]:
                visited[point] = True
                if core[point]:
                    for neighbor in neighbors[point]:
                        if neighbor not in queue:
                            queue.append(neighbor)
            if labels[point] == -1:
                labels[point] = cluster
        expansion_count += 1
        cluster += 1
    return ClusterResult(
        labels=labels,
        iterations=1,
        converged=True,
        objective_trace=[float(expansion_count)],
        extras={
            "core_point_count": int(core.sum()),
            "noise_count": int((labels == -1).sum()),
            "cluster_count": int(cluster),
        },
        requested_iterations=1,
    )


def _logsumexp(values: np.ndarray, axis: int) -> np.ndarray:
    maximum = np.max(values, axis=axis, keepdims=True)
    stabilized = np.exp(values - maximum)
    return np.squeeze(maximum, axis=axis) + np.log(np.sum(stabilized, axis=axis))


def _regularized_covariance(
    points: np.ndarray,
    responsibilities: np.ndarray,
    *,
    covariance_type: str,
    regularization: float,
    global_covariance: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, bool]:
    dimension = points.shape[1]
    weight = float(np.sum(responsibilities, dtype=np.float64))
    singular = False
    if weight <= 1e-12:
        return np.eye(dimension, dtype=np.float64) * regularization, np.full(dimension, regularization), True
    mean = np.zeros(dimension, dtype=np.float64)
    for index, value in enumerate(responsibilities):
        mean += float(value) * points[index]
    mean /= weight
    centered = points - mean
    if covariance_type in {"full", "tied"}:
        covariance = np.zeros((dimension, dimension), dtype=np.float64)
        for index, value in enumerate(responsibilities):
            covariance += float(value) * np.outer(centered[index], centered[index])
        covariance /= weight
        covariance = (covariance + covariance.T) / 2.0
        pre_eigenvalues = np.linalg.eigvalsh(covariance)
        if float(np.min(pre_eigenvalues)) <= 1e-12:
            singular = True
        covariance = covariance + np.eye(dimension, dtype=np.float64) * regularization
        eigenvalues = np.linalg.eigvalsh(covariance)
        if float(np.min(eigenvalues)) <= 0:
            singular = True
            covariance += np.eye(dimension, dtype=np.float64) * max(regularization, 1e-12)
        try:
            precision = np.linalg.solve(
                covariance, np.eye(dimension, dtype=np.float64)
            )
        except np.linalg.LinAlgError:
            precision = np.linalg.pinv(covariance)
            singular = True
        return covariance, precision, singular
    variance = np.zeros(dimension, dtype=np.float64)
    for index, value in enumerate(responsibilities):
        variance += float(value) * centered[index] * centered[index]
    variance /= weight
    if covariance_type == "spherical":
        if float(np.mean(variance, dtype=np.float64)) <= 1e-12:
            singular = True
        scalar = float(np.mean(variance, dtype=np.float64)) + regularization
        if scalar <= 0:
            singular = True
            scalar = max(regularization, 1e-12)
        variance = np.full(dimension, scalar, dtype=np.float64)
    else:
        if float(np.min(variance)) <= 1e-12:
            singular = True
        variance = variance + regularization
        if float(np.min(variance)) <= 0:
            singular = True
            variance = np.maximum(variance, max(regularization, 1e-12))
    covariance = np.diag(variance)
    return covariance, variance, singular


def _density_log_probabilities(
    points: np.ndarray,
    means: np.ndarray,
    covariances: list[np.ndarray],
    precisions: list[np.ndarray],
    covariance_type: str,
) -> np.ndarray:
    n_samples, dimension = points.shape
    n_components = means.shape[0]
    log_probabilities = np.empty((n_samples, n_components), dtype=np.float64)
    normalizer = dimension * math.log(2.0 * math.pi)
    for component in range(n_components):
        centered = points - means[component]
        covariance = covariances[component]
        if covariance_type in {"full", "tied"}:
            sign, log_determinant = np.linalg.slogdet(covariance)
            if sign <= 0 or not math.isfinite(float(log_determinant)):
                log_determinant = dimension * math.log(1e-12)
            precision = precisions[component]
            if precision.ndim == 2 and precision.shape == covariance.shape:
                mahalanobis = np.sum((centered @ precision) * centered, axis=1)
            else:
                mahalanobis = np.sum(centered * centered, axis=1)
        else:
            variance = np.asarray(precisions[component], dtype=np.float64).reshape(-1)
            variance = np.maximum(variance, 1e-15)
            log_determinant = float(np.sum(np.log(variance), dtype=np.float64))
            mahalanobis = np.sum((centered * centered) / variance, axis=1)
        log_probabilities[:, component] = -0.5 * (normalizer + float(log_determinant) + mahalanobis)
    return log_probabilities


def gaussian_mixture(
    matrix: np.ndarray,
    *,
    n_components: int,
    max_iter: int,
    tolerance: float,
    covariance_type: str,
    regularization: float,
    requested_iterations: int | None = None,
) -> ClusterResult:
    n_samples, dimension = matrix.shape
    means = _farthest_initial_centroids(matrix, n_components)
    global_mean = np.mean(matrix, axis=0, dtype=np.float64)
    global_variance = np.var(matrix, axis=0, dtype=np.float64) + regularization
    if covariance_type in {"full", "tied"}:
        base_covariance = np.diag(global_variance)
    else:
        base_covariance = global_variance
    covariances = [np.array(base_covariance, copy=True) for _ in range(n_components)]
    if covariance_type in {"full", "tied"}:
        precisions = [np.linalg.pinv(item) for item in covariances]
    else:
        precisions = [np.array(item, copy=True) for item in covariances]
    weights = np.full(n_components, 1.0 / n_components, dtype=np.float64)
    previous_log_likelihood: float | None = None
    trace: list[float] = []
    issues: list[Issue] = []
    converged = False
    iterations = 0
    singular_seen = False
    requested = int(requested_iterations if requested_iterations is not None else max_iter)
    responsibilities = np.full((n_samples, n_components), 1.0 / n_components, dtype=np.float64)

    for iteration in range(1, max_iter + 1):
        iterations = iteration
        log_probabilities = _density_log_probabilities(
            matrix, means, covariances, precisions, covariance_type
        )
        weighted = log_probabilities + np.log(np.maximum(weights, 1e-300))[None, :]
        log_norm = _logsumexp(weighted, axis=1)
        log_likelihood = float(np.sum(log_norm, dtype=np.float64))
        trace.append(log_likelihood)
        responsibilities = np.exp(weighted - log_norm[:, None])
        if not np.all(np.isfinite(responsibilities)):
            responsibilities = np.full((n_samples, n_components), 1.0 / n_components)
            issues.append(
                Issue(
                    E_COVARIANCE_SINGULAR,
                    "non-finite responsibilities were replaced by a uniform fallback",
                    severity="warning",
                    context={"scope": "algorithm", "method": "gmm", "iteration": iteration},
                )
            )
        totals = np.sum(responsibilities, axis=0, dtype=np.float64)
        weights = totals / float(n_samples)
        weights = np.maximum(weights, 1e-15)
        weights /= np.sum(weights, dtype=np.float64)
        new_means = np.empty_like(means)
        new_covariances: list[np.ndarray] = []
        new_precisions: list[np.ndarray] = []
        for component in range(n_components):
            total = float(totals[component])
            if total <= 1e-15:
                new_means[component] = means[component]
            else:
                mean = np.zeros(dimension, dtype=np.float64)
                for index in range(n_samples):
                    mean += float(responsibilities[index, component]) * matrix[index]
                new_means[component] = mean / total
            covariance, precision, singular = _regularized_covariance(
                matrix,
                responsibilities[:, component],
                covariance_type=covariance_type,
                regularization=regularization,
                global_covariance=base_covariance,
            )
            if singular:
                singular_seen = True
            new_covariances.append(covariance)
            new_precisions.append(precision)
        if covariance_type == "tied":
            combined = np.mean(np.stack(new_covariances, axis=0), axis=0, dtype=np.float64)
            combined = combined + np.eye(dimension, dtype=np.float64) * regularization
            new_covariances = [combined for _ in range(n_components)]
            new_precisions = [np.linalg.pinv(combined) for _ in range(n_components)]
        means = new_means
        covariances = new_covariances
        precisions = new_precisions
        if previous_log_likelihood is not None:
            delta = abs(log_likelihood - previous_log_likelihood)
            threshold = tolerance * max(1.0, abs(previous_log_likelihood))
            if delta <= threshold:
                converged = True
                previous_log_likelihood = log_likelihood
                break
        previous_log_likelihood = log_likelihood

    final_log_probabilities = _density_log_probabilities(
        matrix, means, covariances, precisions, covariance_type
    )
    final_weighted = final_log_probabilities + np.log(np.maximum(weights, 1e-300))[None, :]
    final_log_norm = _logsumexp(final_weighted, axis=1)
    final_responsibilities = np.exp(final_weighted - final_log_norm[:, None])
    labels = np.argmax(final_responsibilities, axis=1).astype(np.int64)
    if singular_seen:
        issues.append(
            Issue(
                E_COVARIANCE_SINGULAR,
                "covariance singularity was isolated with the configured regularization",
                severity="warning",
                context={"scope": "algorithm", "method": "gmm"},
            )
        )
    if not converged and iterations >= requested:
        issues.append(
            Issue(
                E_MAX_ITER_REACHED,
                "gmm reached the configured iteration limit before convergence",
                severity="warning",
                context={"scope": "algorithm", "method": "gmm"},
            )
        )
    return ClusterResult(
        labels=labels,
        iterations=iterations,
        converged=converged,
        objective_trace=trace,
        issues=issues,
        extras={
            "means": means.tolist(),
            "weights": weights.tolist(),
            "mean_log_likelihood": float(np.mean(final_log_norm, dtype=np.float64)),
        },
        requested_iterations=requested,
    )
