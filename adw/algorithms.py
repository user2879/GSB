"""Three deterministic anomaly scoring algorithms implemented with NumPy."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np

from .canonical import StableRNG, fsum, quantize_score, sha256_text, stable_median


@dataclass
class MahalanobisState:
    centers: np.ndarray
    scales: np.ndarray
    cholesky: np.ndarray
    regularization: float


@dataclass
class HistogramState:
    bins_per_dimension: int
    neighborhood: int
    bin_indices: np.ndarray
    neighbor_counts: np.ndarray
    sample_count: int


@dataclass
class PartitionTree:
    root: dict[str, Any]
    effective_sample_size: int


@dataclass
class PartitionState:
    trees: list[PartitionTree]
    effective_sample_size: int


def _robust_column(values: np.ndarray) -> tuple[float, float]:
    center = stable_median(values)
    mad = stable_median(np.abs(values - center))
    scale = 1.4826 * mad
    if scale > 1.0e-12:
        return center, scale
    sorted_values = np.sort(values, kind="mergesort")
    if sorted_values.size >= 2:
        q1 = _quantile(sorted_values, 0.25)
        q3 = _quantile(sorted_values, 0.75)
        iqr = (q3 - q1) / 1.349
        if iqr > 1.0e-12:
            return center, float(iqr)
    return center, 1.0


def _quantile(sorted_values: np.ndarray, q: float) -> float:
    if sorted_values.size == 0:
        return 0.0
    position = (sorted_values.size - 1) * q
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return float(sorted_values[lower])
    fraction = position - lower
    return float(sorted_values[lower] * (1.0 - fraction) + sorted_values[upper] * fraction)


def _cholesky(matrix: np.ndarray) -> np.ndarray:
    d = int(matrix.shape[0])
    lower = np.zeros((d, d), dtype=np.float64, order="C")
    for j in range(d):
        diagonal = float(matrix[j, j]) - fsum(float(lower[j, k]) ** 2 for k in range(j))
        if diagonal <= 0.0:
            raise ValueError("robust covariance is not positive definite after regularization")
        lower[j, j] = math.sqrt(diagonal)
        for i in range(j + 1, d):
            numerator = float(matrix[i, j]) - fsum(float(lower[i, k]) * float(lower[j, k]) for k in range(j))
            lower[i, j] = numerator / float(lower[j, j])
    return lower


def fit_robust_mahalanobis(X: np.ndarray, params: dict[str, Any]) -> MahalanobisState:
    n, d = X.shape
    if n < 2 or d < 1:
        raise ValueError("robust Mahalanobis needs at least two samples and one feature")
    centers = np.empty(d, dtype=np.float64)
    scales = np.empty(d, dtype=np.float64)
    Z = np.empty_like(X, order="C")
    for j in range(d):
        center, scale = _robust_column(X[:, j])
        centers[j] = center
        scales[j] = scale
        for i in range(n):
            Z[i, j] = (X[i, j] - center) / scale

    covariance = np.zeros((d, d), dtype=np.float64, order="C")
    denominator = float(n - 1)
    for i in range(d):
        for j in range(i, d):
            value = fsum(float(Z[k, i]) * float(Z[k, j]) for k in range(n)) / denominator
            covariance[i, j] = value
            covariance[j, i] = value
    regularization = float(params["regularization"])
    for j in range(d):
        covariance[j, j] += regularization
    lower = _cholesky(covariance)
    return MahalanobisState(centers=centers, scales=scales, cholesky=lower, regularization=regularization)


def _solve_cholesky(lower: np.ndarray, value: np.ndarray) -> np.ndarray:
    d = int(lower.shape[0])
    intermediate = np.empty(d, dtype=np.float64)
    for i in range(d):
        residual = float(value[i]) - fsum(float(lower[i, k]) * float(intermediate[k]) for k in range(i))
        intermediate[i] = residual / float(lower[i, i])
    solution = np.empty(d, dtype=np.float64)
    for i in range(d - 1, -1, -1):
        residual = float(intermediate[i]) - fsum(float(lower[k, i]) * float(solution[k]) for k in range(i + 1, d))
        solution[i] = residual / float(lower[i, i])
    return solution


def score_robust_mahalanobis(state: MahalanobisState, x: np.ndarray) -> float:
    if not np.all(np.isfinite(x)):
        raise ValueError("non-finite feature in robust Mahalanobis scoring")
    d = int(x.shape[0])
    standardized = np.empty(d, dtype=np.float64)
    for j in range(d):
        standardized[j] = (float(x[j]) - float(state.centers[j])) / float(state.scales[j])
    solution = _solve_cholesky(state.cholesky, standardized)
    return quantize_score(fsum(float(solution[j]) ** 2 for j in range(d)))


def fit_histogram_density(X: np.ndarray, params: dict[str, Any]) -> HistogramState:
    n, d = X.shape
    bins_per_dimension = int(params["bins"])
    neighborhood = int(params["neighborhood"])
    if n < 1 or d < 1:
        raise ValueError("histogram density needs at least one sample and one feature")
    minimum = np.empty(d, dtype=np.float64)
    maximum = np.empty(d, dtype=np.float64)
    for j in range(d):
        minimum[j] = float(np.min(X[:, j]))
        maximum[j] = float(np.max(X[:, j]))
    bin_indices = np.empty((n, d), dtype=np.int64, order="C")
    for j in range(d):
        width = float(maximum[j] - minimum[j])
        for i in range(n):
            if width <= 0.0:
                bin_indices[i, j] = 0
            else:
                ratio = (float(X[i, j]) - float(minimum[j])) / width
                code = int(math.floor(ratio * bins_per_dimension))
                if code < 0:
                    code = 0
                elif code >= bins_per_dimension:
                    code = bins_per_dimension - 1
                bin_indices[i, j] = code

    counts = np.empty(n, dtype=np.int64)
    chunk_size = 256
    for start in range(0, n, chunk_size):
        stop = min(n, start + chunk_size)
        mask = np.ones((stop - start, n), dtype=bool, order="C")
        for j in range(d):
            mask &= np.abs(bin_indices[start:stop, j, None] - bin_indices[None, :, j]) <= neighborhood
        counts[start:stop] = np.sum(mask, axis=1, dtype=np.int64)
    return HistogramState(
        bins_per_dimension=bins_per_dimension,
        neighborhood=neighborhood,
        bin_indices=bin_indices,
        neighbor_counts=counts,
        sample_count=n,
    )


def score_histogram_density(state: HistogramState, sample_index: int, x: np.ndarray) -> float:
    if not np.all(np.isfinite(x)):
        raise ValueError("non-finite feature in histogram density scoring")
    density = int(state.neighbor_counts[sample_index])
    score = math.log((float(state.sample_count) + 0.5) / (float(density) + 0.5))
    return quantize_score(score)


def _c_factor(n: int) -> float:
    if n <= 1:
        return 0.0
    if n == 2:
        return 1.0
    return 2.0 * (math.log(float(n - 1)) + 0.5772156649015329) - 2.0 * float(n - 1) / float(n)


def _sample_indices(n: int, size: int, rng: StableRNG) -> list[int]:
    indices = list(range(n))
    take = min(n, size)
    for i in range(take):
        j = i + rng.randbelow(n - i)
        indices[i], indices[j] = indices[j], indices[i]
    chosen = indices[:take]
    chosen.sort()
    return chosen


def _build_tree(X: np.ndarray, indices: list[int], depth: int, max_depth: int, rng: StableRNG) -> dict[str, Any]:
    size = len(indices)
    if size < 2 or depth >= max_depth:
        return {"leaf": True, "size": size}
    feature_start = rng.randbelow(X.shape[1])
    chosen_feature = -1
    minimum = maximum = 0.0
    for offset in range(X.shape[1]):
        feature = (feature_start + offset) % X.shape[1]
        values = [float(X[index, feature]) for index in indices]
        local_min = min(values)
        local_max = max(values)
        if local_max > local_min:
            chosen_feature = feature
            minimum = local_min
            maximum = local_max
            break
    if chosen_feature < 0:
        return {"leaf": True, "size": size}
    threshold = minimum + (maximum - minimum) * (0.15 + 0.70 * rng.random())
    if not (minimum < threshold < maximum):
        threshold = (minimum + maximum) / 2.0
    left: list[int] = []
    right: list[int] = []
    for index in indices:
        if float(X[index, chosen_feature]) <= threshold:
            left.append(index)
        else:
            right.append(index)
    if not left or not right:
        return {"leaf": True, "size": size}
    return {
        "leaf": False,
        "feature": chosen_feature,
        "threshold": threshold,
        "left": _build_tree(X, left, depth + 1, max_depth, rng),
        "right": _build_tree(X, right, depth + 1, max_depth, rng),
        "size": size,
    }


def _path_length(node: dict[str, Any], x: np.ndarray) -> float:
    depth = 0
    while not node.get("leaf", False):
        if float(x[int(node["feature"])]) <= float(node["threshold"]):
            node = node["left"]
        else:
            node = node["right"]
        depth += 1
    return float(depth) + _c_factor(int(node["size"]))


def fit_random_partition_depth(
    X: np.ndarray,
    params: dict[str, Any],
    seed_material: str,
) -> PartitionState:
    n, d = X.shape
    if n < 2 or d < 1:
        raise ValueError("random partition depth needs at least two samples and one feature")
    tree_count = int(params["trees"])
    requested_size = int(params["sample_size"])
    max_depth = int(params["max_depth"])
    effective_size = min(n, requested_size)
    base_seed = int(sha256_text("partition-seed\n" + seed_material)[:16], 16)
    trees: list[PartitionTree] = []
    for tree_index in range(tree_count):
        rng = StableRNG((base_seed + tree_index * 0x9E3779B97F4A7C15) & ((1 << 64) - 1))
        indices = _sample_indices(n, effective_size, rng)
        root = _build_tree(X, indices, 0, max_depth, rng)
        trees.append(PartitionTree(root=root, effective_sample_size=effective_size))
    return PartitionState(trees=trees, effective_sample_size=effective_size)


def score_random_partition_depth(state: PartitionState, x: np.ndarray) -> float:
    if not np.all(np.isfinite(x)):
        raise ValueError("non-finite feature in random partition scoring")
    paths = [_path_length(tree.root, x) for tree in state.trees]
    average_path = fsum(paths) / float(len(paths))
    factor = _c_factor(state.effective_sample_size)
    score = math.pow(2.0, -average_path / factor) if factor > 0.0 else 0.0
    return quantize_score(score)


def algorithm_ranking(scores: dict[str, float], failed_ids: set[str] | None = None) -> list[dict[str, Any]]:
    failed_ids = failed_ids or set()
    items = [(sample_id, float(score)) for sample_id, score in scores.items() if sample_id not in failed_ids]
    items.sort(key=lambda item: (-item[1], item[0]))
    return [{"sample_id": sample_id, "score": score} for sample_id, score in items]
