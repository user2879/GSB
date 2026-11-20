"""Deterministic DBSCAN.

- Points are visited in row-index order; neighbor lists are kept in
  row-index order; cluster expansion is breadth-first with a FIFO queue.
- Iteration budget consumption: one unit per point visit.
- Noise points keep label -1; an all-noise outcome is recorded explicitly.
"""

from __future__ import annotations

from typing import Any

import numpy as np


def dbscan_train(X: np.ndarray, eps: float, min_samples: int,
                 quota: int) -> dict[str, Any]:
    n = X.shape[0]
    eps = float(eps)
    min_samples = int(min_samples)
    x2 = np.sum(X * X, axis=1, keepdims=True)
    d2 = np.maximum(x2 + x2.T - 2.0 * (X @ X.T), 0.0)
    neighbors = [np.where(d2[i] <= eps * eps)[0].tolist() for i in range(n)]
    core = [len(nb) >= min_samples for nb in neighbors]

    labels = np.full(n, -1, dtype=np.int64)
    visited = np.zeros(n, dtype=bool)
    events: list[dict] = []
    consumed = 0
    cluster_id = 0
    exhausted = False
    for i in range(n):
        if visited[i]:
            continue
        consumed += 1
        if consumed > quota:
            exhausted = True
            consumed -= 1
            break
        visited[i] = True
        if not core[i]:
            labels[i] = -1  # noise for now; may be claimed as border later
            continue
        labels[i] = cluster_id
        queue = list(neighbors[i])  # fixed index order
        in_queue = set(queue)
        head = 0
        while head < len(queue):
            j = queue[head]
            head += 1
            if not visited[j]:
                consumed += 1
                if consumed > quota:
                    exhausted = True
                    consumed -= 1
                    break
                visited[j] = True
                if core[j]:
                    for m in neighbors[j]:
                        if m not in in_queue and not visited[m]:
                            in_queue.add(m)
                            queue.append(m)
            if labels[j] < 0:
                labels[j] = cluster_id
        if exhausted:
            break
        cluster_id += 1
    n_clusters = cluster_id
    if n_clusters == 0:
        events.append({"event": "all_noise", "n_points": n})
    noise_count = int(np.sum(labels < 0))
    if noise_count > 0:
        events.append({"event": "noise_points", "count": noise_count})
    return {
        "labels": labels,
        "n_clusters": n_clusters,
        "noise_count": noise_count,
        "iterations": consumed,
        "converged": not exhausted,
        "events": events,
        "quota_exhausted": exhausted,
    }
