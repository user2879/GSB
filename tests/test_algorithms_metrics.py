from __future__ import annotations

import numpy as np

from cluster_workbench.algorithms import dbscan, gaussian_mixture, kmeans
from cluster_workbench.errors import E_ALL_NOISE, E_EMPTY_CLUSTER
from cluster_workbench.metrics import evaluate_labels, rank_candidates


def clustered_matrix() -> np.ndarray:
    rows = []
    for center in (-4.0, 0.0, 4.0):
        for offset in range(10):
            rows.append([center + 0.03 * offset, center - 0.02 * offset, 0.1 * (offset % 3)])
    return np.asarray(rows, dtype=np.float64)


def test_kmeans_dbscan_and_gmm_repeat_exactly():
    matrix = clustered_matrix()
    first = kmeans(matrix, n_clusters=3, max_iter=30, tolerance=0.0, requested_iterations=30)
    second = kmeans(matrix, n_clusters=3, max_iter=30, tolerance=0.0, requested_iterations=30)
    assert np.array_equal(first.labels, second.labels)
    assert first.objective_trace == second.objective_trace
    assert first.iterations >= 1

    first_db = dbscan(matrix, eps=0.2, min_samples=3)
    second_db = dbscan(matrix, eps=0.2, min_samples=3)
    assert np.array_equal(first_db.labels, second_db.labels)
    assert first_db.iterations == 1

    for covariance_type in ("full", "diag", "spherical", "tied"):
        left = gaussian_mixture(
            matrix,
            n_components=3,
            max_iter=20,
            tolerance=1e-6,
            covariance_type=covariance_type,
            regularization=1e-5,
        )
        right = gaussian_mixture(
            matrix,
            n_components=3,
            max_iter=20,
            tolerance=1e-6,
            covariance_type=covariance_type,
            regularization=1e-5,
        )
        assert np.array_equal(left.labels, right.labels)
        assert np.all(np.isfinite(left.objective_trace))


def test_gmm_singular_covariance_is_regularized_without_exception():
    matrix = np.zeros((12, 3), dtype=np.float64)
    result = gaussian_mixture(
        matrix,
        n_components=3,
        max_iter=10,
        tolerance=1e-6,
        covariance_type="full",
        regularization=1e-4,
    )
    assert result.labels.shape == (12,)
    assert np.all(np.isfinite(result.objective_trace))
    assert any(issue.code == "E_COVARIANCE_SINGULAR" for issue in result.issues)


def test_empty_cluster_and_all_noise_are_noncomparable_warnings():
    matrix = clustered_matrix()
    metrics, issues, reasons = evaluate_labels(
        matrix,
        np.zeros(matrix.shape[0], dtype=np.int64),
        method="kmeans",
        expected_clusters=3,
        metrics=["silhouette", "calinski_harabasz", "density_log_likelihood"],
        density_regularization=1e-6,
        candidate_id="empty",
        params={"n_clusters": 3},
    )
    assert any(issue.code == E_EMPTY_CLUSTER for issue in issues)
    assert reasons
    assert all(not item["comparable"] for item in metrics.values())

    metrics, issues, reasons = evaluate_labels(
        matrix,
        np.full(matrix.shape[0], -1, dtype=np.int64),
        method="dbscan",
        expected_clusters=None,
        metrics=["silhouette", "calinski_harabasz", "density_log_likelihood"],
        density_regularization=1e-6,
        candidate_id="noise",
        params={"eps": 0.1, "min_samples": 5},
    )
    assert any(issue.code == E_ALL_NOISE for issue in issues)
    assert "all_noise" in reasons
    assert all(not item["comparable"] for item in metrics.values())


def test_ranking_ties_use_method_then_canonical_params():
    metric = {
        "silhouette": {"value": 0.5, "comparable": True, "reason_code": None, "covered_samples": 20},
        "calinski_harabasz": {"value": 2.0, "comparable": True, "reason_code": None, "covered_samples": 20},
        "density_log_likelihood": {"value": -1.0, "comparable": True, "reason_code": None, "covered_samples": 20},
    }
    candidates = [
        {"id": "z", "method": "kmeans", "params": {"n_clusters": 3}, "metrics": metric, "metrics_comparable": True},
        {"id": "a", "method": "gmm", "params": {"n_components": 3}, "metrics": metric, "metrics_comparable": True},
        {"id": "b", "method": "gmm", "params": {"n_components": 2}, "metrics": metric, "metrics_comparable": True},
    ]
    ranked = rank_candidates(candidates)
    assert [item["id"] for item in ranked] == ["b", "a", "z"]
