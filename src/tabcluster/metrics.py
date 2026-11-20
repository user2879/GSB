"""Shared evaluation metrics for clustering and density estimation.

All metrics are computed in fixed row order. Noise points (label -1) are
excluded from silhouette / Calinski-Harabasz and the exclusion is recorded.
Undefined metrics are reported as NaN (serialized as null) with an event.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np


def _valid_mask(labels: np.ndarray) -> np.ndarray:
    return labels >= 0


def silhouette_score(X: np.ndarray, labels: np.ndarray) -> tuple[float, dict]:
    mask = _valid_mask(labels)
    meta = {"excluded_noise": int(np.sum(~mask))}
    Xv, lv = X[mask], labels[mask]
    uniq = sorted(set(lv.tolist()))
    if len(uniq) < 2 or Xv.shape[0] < 2:
        meta["event"] = "silhouette_undefined"
        return float("nan"), meta
    x2 = np.sum(Xv * Xv, axis=1, keepdims=True)
    d = np.sqrt(np.maximum(x2 + x2.T - 2.0 * (Xv @ Xv.T), 0.0))
    n = Xv.shape[0]
    s = np.empty(n, dtype=np.float64)
    for i in range(n):
        same = lv == lv[i]
        a = float(np.sum(d[i, same]) / max(int(np.sum(same)) - 1, 1))
        b = math.inf
        for u in uniq:
            if u == lv[i]:
                continue
            other = lv == u
            if np.any(other):
                b = min(b, float(np.mean(d[i, other])))
        denom = max(a, b)
        s[i] = 0.0 if denom == 0 or not math.isfinite(denom) else (b - a) / denom
    return float(np.mean(s)), meta


def calinski_harabasz(X: np.ndarray, labels: np.ndarray) -> tuple[float, dict]:
    mask = _valid_mask(labels)
    meta = {"excluded_noise": int(np.sum(~mask))}
    Xv, lv = X[mask], labels[mask]
    uniq = sorted(set(lv.tolist()))
    n, k = Xv.shape[0], len(uniq)
    if k < 2 or n <= k:
        meta["event"] = "calinski_harabasz_undefined"
        return float("nan"), meta
    overall = Xv.mean(axis=0)
    between = 0.0
    within = 0.0
    for u in uniq:
        members = Xv[lv == u]
        centroid = members.mean(axis=0)
        between += members.shape[0] * float(np.sum((centroid - overall) ** 2))
        within += float(np.sum((members - centroid) ** 2))
    if within == 0.0:
        meta["event"] = "calinski_harabasz_zero_within"
        return float("nan"), meta
    return float((between / (k - 1)) / (within / (n - k))), meta


def gaussian_log_likelihood(X: np.ndarray, labels: np.ndarray,
                            reg: float) -> tuple[float, dict]:
    """Per-cluster full-covariance Gaussian log-likelihood (regularized)."""
    mask = _valid_mask(labels)
    meta = {"excluded_noise": int(np.sum(~mask))}
    Xv, lv = X[mask], labels[mask]
    uniq = sorted(set(lv.tolist()))
    if not uniq:
        meta["event"] = "log_likelihood_undefined"
        return float("nan"), meta
    d = X.shape[1]
    total = 0.0
    log_2pi = float(np.log(2.0 * np.pi))
    for u in uniq:
        members = Xv[lv == u]
        mean = members.mean(axis=0)
        diff = members - mean
        cov = (diff.T @ diff) / max(members.shape[0], 1)
        reg_use = reg
        logdet = 0.0
        for _ in range(7):
            sign, logdet = np.linalg.slogdet(cov + np.eye(d) * reg_use)
            if sign > 0:
                break
            reg_use *= 10.0
        covr = cov + np.eye(d) * reg_use
        sol = np.linalg.solve(covr, diff.T).T
        maha = np.sum(diff * sol, axis=1)
        total += float(np.sum(-0.5 * (d * log_2pi + logdet + maha)))
    return total, meta


def evaluate_candidate(X: np.ndarray, labels: np.ndarray, metrics: list[str],
                       reg_covar: float,
                       model_log_likelihood: float | None = None) -> dict[str, Any]:
    """Compute the configured metric set for one candidate, fixed order."""
    out: dict[str, Any] = {"metrics": {}, "meta": {}}
    for m in metrics:  # config order is part of the artifact
        if m == "silhouette":
            v, meta = silhouette_score(X, labels)
        elif m == "calinski_harabasz":
            v, meta = calinski_harabasz(X, labels)
        elif m == "log_likelihood":
            if model_log_likelihood is not None:
                v, meta = float(model_log_likelihood), {"source": "model"}
            else:
                v, meta = gaussian_log_likelihood(X, labels, reg_covar)
        else:  # pragma: no cover - validated at load
            continue
        out["metrics"][m] = v
        out["meta"][m] = meta
    return out
