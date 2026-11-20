"""Deterministic KMeans.

- Fixed initial centroid rule: first centroid is the sample closest to the
  global mean (ties -> lowest index); subsequent centroids are chosen by
  deterministic farthest-point selection (ties -> lowest index).
- Fixed iteration cap; per-iteration within-cluster sum of squares (inertia)
  is recorded.
- Empty clusters are re-seeded to the point farthest from its assigned
  centroid and explicitly recorded as events, never silently dropped.
"""

from __future__ import annotations

from typing import Any

import numpy as np

CONVERGE_TOL = 1e-12


def _pairwise_sq(X: np.ndarray, C: np.ndarray) -> np.ndarray:
    x2 = np.sum(X * X, axis=1, keepdims=True)
    c2 = np.sum(C * C, axis=1)
    d2 = x2 + c2[None, :] - 2.0 * (X @ C.T)
    return np.maximum(d2, 0.0)


def _initial_centroids(X: np.ndarray, k: int) -> np.ndarray:
    mean = X.mean(axis=0)
    d = np.sum((X - mean) ** 2, axis=1)
    first = int(np.argmin(d))  # ties -> lowest index
    chosen = [first]
    min_dist = np.sum((X - X[first]) ** 2, axis=1)
    while len(chosen) < k:
        nxt = int(np.argmax(min_dist))  # ties -> lowest index
        chosen.append(nxt)
        min_dist = np.minimum(min_dist, np.sum((X - X[nxt]) ** 2, axis=1))
    return X[np.array(chosen)].copy()


def kmeans_train(X: np.ndarray, n_clusters: int, max_iter: int,
                 quota: int) -> dict[str, Any]:
    """Returns labels, per-iteration inertia trace, events, iterations used."""
    n = X.shape[0]
    k = int(n_clusters)
    cap = max(0, min(int(max_iter), int(quota)))
    events: list[dict] = []
    if k >= n:
        k = max(1, n)
        events.append({"event": "n_clusters_clamped", "value": k})
    centroids = _initial_centroids(X, k)
    labels = np.zeros(n, dtype=np.int64)
    trace: list[float] = []
    prev_labels: np.ndarray | None = None
    converged = False
    for it in range(cap):
        d2 = _pairwise_sq(X, centroids)
        labels = np.argmin(d2, axis=1).astype(np.int64)  # ties -> lowest cluster
        inertia = float(np.sum(d2[np.arange(n), labels]))
        trace.append(inertia)
        new_centroids = centroids.copy()
        for c in range(k):
            members = X[labels == c]
            if members.shape[0] == 0:
                own = d2[np.arange(n), labels]
                idx = int(np.argmax(own))
                new_centroids[c] = X[idx]
                events.append({"event": "empty_cluster", "iteration": it,
                               "cluster": c, "reseeded_row": idx})
            else:
                new_centroids[c] = members.mean(axis=0)
        shift = float(np.max(np.sqrt(np.sum((new_centroids - centroids) ** 2,
                                            axis=1))))
        centroids = new_centroids
        if prev_labels is not None and np.array_equal(labels, prev_labels):
            converged = True
            break
        if shift <= CONVERGE_TOL:
            converged = True
            break
        prev_labels = labels
    d2 = _pairwise_sq(X, centroids)
    labels = np.argmin(d2, axis=1).astype(np.int64)
    return {
        "labels": labels,
        "centroids": centroids,
        "inertia_trace": trace,
        "iterations": len(trace),
        "converged": converged,
        "events": events,
        "quota_exhausted": (not converged) and cap == int(quota),
    }
