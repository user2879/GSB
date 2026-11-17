"""Deterministic clustering implementations and hard/soft assignments."""

from __future__ import annotations

import hashlib
import math
from collections import deque
from dataclasses import dataclass
from typing import Any

import numpy as np
from sklearn.mixture import GaussianMixture

from .config import Candidate, validate_candidate_params


def seed_for_candidate(seed: int, stable_key: str) -> int:
    digest = hashlib.sha256(f"{seed}:{stable_key}".encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "big") % (2**31 - 1)


def stable_argmin(values: np.ndarray, row_index: int) -> int:
    minimum = float(np.min(values))
    tied = np.flatnonzero(np.isclose(values, minimum, rtol=0.0, atol=1e-15))
    if len(tied) == 0:
        tied = np.flatnonzero(values == minimum)
    return int(tied[row_index % len(tied)])


def stable_argmax(values: np.ndarray, row_index: int) -> int:
    maximum = float(np.max(values))
    tied = np.flatnonzero(np.isclose(values, maximum, rtol=0.0, atol=1e-15))
    if len(tied) == 0:
        tied = np.flatnonzero(values == maximum)
    return int(tied[row_index % len(tied)])


def _mean_columns(matrix: np.ndarray, indices: list[int]) -> list[float]:
    if not indices:
        return []
    result: list[float] = []
    for column in range(matrix.shape[1]):
        result.append(math.fsum(float(matrix[index, column]) for index in indices) / len(indices))
    return result


def _cluster_diagnostics(labels: list[int], expected_clusters: int | None, noise_label: int | None = None) -> dict[str, Any]:
    sizes: dict[str, int] = {}
    for label in labels:
        key = str(label)
        sizes[key] = sizes.get(key, 0) + 1
    noise_count = sizes.get(str(noise_label), 0) if noise_label is not None else 0
    found = sorted({label for label in labels if label != noise_label})
    if expected_clusters is None:
        expected = sorted({label for label in labels if label >= 0})
    else:
        expected = list(range(expected_clusters))
    empty = [label for label in expected if sizes.get(str(label), 0) == 0]
    return {
        "cluster_sizes": sizes,
        "n_clusters_found": len(found),
        "empty_clusters": empty,
        "noise_count": noise_count,
        "all_noise": bool(labels) and noise_count == len(labels),
    }


@dataclass
class FitResult:
    method: str
    labels: list[int]
    responsibilities: list[list[float]] | None
    n_iter: int
    converged: bool
    centers: list[list[float]]
    diagnostics: dict[str, Any]
    seed: int
    params: dict[str, Any]
    density_log_likelihood: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "method": self.method,
            "labels": self.labels,
            "responsibilities": self.responsibilities,
            "n_iter": self.n_iter,
            "converged": self.converged,
            "centers": self.centers,
            "diagnostics": self.diagnostics,
            "seed": self.seed,
            "params": self.params,
            "density_log_likelihood": self.density_log_likelihood,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "FitResult":
        return cls(**payload)


def _kmeans_initial_centers(matrix: np.ndarray, n_clusters: int) -> np.ndarray:
    selected = [0]
    min_distance = np.sum((matrix - matrix[0]) ** 2, axis=1)
    while len(selected) < n_clusters:
        best_index = -1
        best_distance = -1.0
        for index in range(matrix.shape[0]):
            if index in selected:
                continue
            distance = float(min_distance[index])
            if distance > best_distance + 1e-15:
                best_distance = distance
                best_index = index
        if best_index < 0 or best_distance <= 1e-15:
            for index in range(matrix.shape[0]):
                if index not in selected:
                    best_index = index
                    break
        selected.append(best_index)
        distances = np.sum((matrix - matrix[best_index]) ** 2, axis=1)
        min_distance = np.minimum(min_distance, distances)
    return matrix[selected].copy()


def fit_kmeans(matrix: np.ndarray, params: dict[str, Any], seed: int, stable_key: str) -> FitResult:
    del seed, stable_key
    n_clusters = int(params["n_clusters"])
    max_iter = int(params["max_iter"])
    tol = float(params["tol"])
    if n_clusters <= 0 or n_clusters > matrix.shape[0]:
        raise ValueError(f"invalid KMeans n_clusters: {n_clusters}")
    centers = _kmeans_initial_centers(matrix, n_clusters)
    labels = np.zeros(matrix.shape[0], dtype=int)
    converged = False
    iterations = 0
    for iterations in range(1, max_iter + 1):
        for row in range(matrix.shape[0]):
            squared = np.sum((centers - matrix[row]) ** 2, axis=1)
            labels[row] = stable_argmin(squared, row)
        new_centers = centers.copy()
        for cluster in range(n_clusters):
            members = np.flatnonzero(labels == cluster).tolist()
            if members:
                new_centers[cluster] = np.asarray(
                    _mean_columns(matrix, [int(index) for index in members]),
                    dtype=float,
                )
        shift = float(np.max(np.sqrt(np.sum((new_centers - centers) ** 2, axis=1))))
        centers = new_centers
        if shift <= tol:
            converged = True
            break
    for row in range(matrix.shape[0]):
        squared = np.sum((centers - matrix[row]) ** 2, axis=1)
        labels[row] = stable_argmin(squared, row)
    label_list = [int(label) for label in labels]
    diagnostics = _cluster_diagnostics(label_list, n_clusters)
    if diagnostics["empty_clusters"]:
        diagnostics["empty_cluster_policy"] = "retained_and_recorded"
    return FitResult(
        method="kmeans",
        labels=label_list,
        responsibilities=None,
        n_iter=iterations,
        converged=converged,
        centers=[[float(value) for value in row] for row in centers],
        diagnostics=diagnostics,
        seed=0,
        params=params,
        density_log_likelihood=None,
    )


def fit_dbscan(matrix: np.ndarray, params: dict[str, Any], seed: int, stable_key: str) -> FitResult:
    del seed, stable_key
    eps = float(params["eps"])
    min_samples = int(params["min_samples"])
    if eps < 0:
        raise ValueError(f"negative DBSCAN eps: {eps}")
    squared_distances = np.sum(
        (matrix[:, None, :] - matrix[None, :, :]) ** 2,
        axis=2,
    )
    radius_squared = eps * eps
    neighbor_cache = [
        [int(index) for index in np.flatnonzero(squared_distances[row] <= radius_squared)]
        for row in range(matrix.shape[0])
    ]
    visited = [False] * matrix.shape[0]
    labels = [-1] * matrix.shape[0]
    cluster_id = -1
    queue_order: list[int] = []

    for row in range(matrix.shape[0]):
        if visited[row]:
            continue
        visited[row] = True
        neighbors = neighbor_cache[row]
        queue_order.append(row)
        if len(neighbors) < min_samples:
            continue
        cluster_id += 1
        labels[row] = cluster_id
        pending = deque(neighbors)
        queued = set(neighbors)
        while pending:
            current = pending.popleft()
            queue_order.append(current)
            if not visited[current]:
                visited[current] = True
                current_neighbors = neighbor_cache[current]
                if len(current_neighbors) >= min_samples:
                    for candidate in current_neighbors:
                        if candidate not in queued:
                            queued.add(candidate)
                            pending.append(candidate)
            if labels[current] == -1:
                labels[current] = cluster_id

    centers: list[list[float]] = []
    for cluster in range(cluster_id + 1):
        members = [index for index, label in enumerate(labels) if label == cluster]
        centers.append(_mean_columns(matrix, members))
    diagnostics = _cluster_diagnostics(labels, None, noise_label=-1)
    diagnostics["noise_policy"] = "retained_and_recorded"
    diagnostics["traversal_order_hash"] = hashlib.sha256(
        ",".join(str(index) for index in queue_order).encode("utf-8")
    ).hexdigest()
    return FitResult(
        method="dbscan",
        labels=labels,
        responsibilities=None,
        n_iter=1,
        converged=True,
        centers=centers,
        diagnostics=diagnostics,
        seed=0,
        params=params,
        density_log_likelihood=None,
    )


def fit_gaussian_mixture(
    matrix: np.ndarray,
    params: dict[str, Any],
    seed: int,
    stable_key: str,
) -> FitResult:
    candidate_seed = seed_for_candidate(seed, stable_key)
    model = GaussianMixture(
        n_components=int(params["n_components"]),
        covariance_type=str(params["covariance_type"]),
        max_iter=int(params["max_iter"]),
        tol=float(params["tol"]),
        reg_covar=1e-6,
        init_params="kmeans",
        n_init=1,
        random_state=candidate_seed,
    )
    model.fit(matrix)
    responsibilities = model.predict_proba(matrix)
    labels = [stable_argmax(responsibilities[row], row) for row in range(matrix.shape[0])]
    diagnostics = _cluster_diagnostics(labels, int(params["n_components"]))
    if diagnostics["empty_clusters"]:
        diagnostics["empty_cluster_policy"] = "retained_and_recorded"
    density_log_likelihood = float(np.mean(model.score_samples(matrix)))
    return FitResult(
        method="gaussian_mixture",
        labels=labels,
        responsibilities=[[float(value) for value in row] for row in responsibilities],
        n_iter=int(model.n_iter_),
        converged=bool(model.converged_),
        centers=[[float(value) for value in row] for row in model.means_],
        diagnostics=diagnostics,
        seed=candidate_seed,
        params=params,
        density_log_likelihood=density_log_likelihood,
    )


def fit_candidate(
    matrix: np.ndarray,
    candidate: Candidate,
    seed: int,
) -> FitResult:
    params = validate_candidate_params(candidate.method, candidate.params, matrix.shape[0])
    if candidate.method == "kmeans":
        return fit_kmeans(matrix, params, seed, candidate.sort_key)
    if candidate.method == "dbscan":
        return fit_dbscan(matrix, params, seed, candidate.sort_key)
    if candidate.method == "gaussian_mixture":
        return fit_gaussian_mixture(matrix, params, seed, candidate.sort_key)
    raise ValueError(f"unsupported candidate method: {candidate.method}")
