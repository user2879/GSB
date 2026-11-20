"""Deterministic Gaussian Mixture Model (EM with fixed rules).

Fixed initialization: responsibilities seeded from the deterministic
KMeans in this package (no RNG anywhere).
Fixed convergence rule: stop when |ll - prev_ll| < tol.
Covariance singularities are absorbed by the configured regularization
coefficient (escalating x10 up to a fixed bound) and recorded as events;
no exception is raised.
Covariance types: full / diag / spherical / tied.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .kmeans import kmeans_train

_LOG_2PI = float(np.log(2.0 * np.pi))
MAX_REG_ESCALATIONS = 6


def _regularize(covs: np.ndarray, reg: float,
                events: list[dict], iteration: int) -> np.ndarray:
    """Add reg*I, escalating x10 until every component is PD (bounded)."""
    k, d = covs.shape[0], covs.shape[1]
    reg_use = reg
    for _esc in range(MAX_REG_ESCALATIONS + 1):
        out = covs + np.eye(d)[None, :, :] * reg_use
        ok = True
        for j in range(k):
            sign, _ = np.linalg.slogdet(out[j])
            if sign <= 0:
                ok = False
                events.append({"event": "covariance_regularized",
                               "iteration": iteration, "component": j,
                               "reg": reg_use})
                break
        if ok:
            return out
        reg_use *= 10.0
    out = covs + np.eye(d)[None, :, :] * (reg_use + 1e-12)
    for j in range(k):
        sign, _ = np.linalg.slogdet(out[j])
        if sign <= 0:
            out[j] = np.eye(d) * reg_use
            events.append({"event": "covariance_forced_identity",
                           "iteration": iteration, "component": j,
                           "reg": reg_use})
    return out


def _estimate_log_gauss(X: np.ndarray, means: np.ndarray,
                        covs: np.ndarray) -> np.ndarray:
    """log N(x; mean, cov) for each component -> (n, k)."""
    n, d = X.shape
    k = means.shape[0]
    out = np.empty((n, k), dtype=np.float64)
    for j in range(k):
        cov = covs[j]
        sign, logdet = np.linalg.slogdet(cov)
        if sign <= 0:  # should not happen after regularization
            logdet = float(np.log(max(np.linalg.det(cov), 1e-300)))
        diff = X - means[j]
        sol = np.linalg.solve(cov, diff.T).T
        maha = np.sum(diff * sol, axis=1)
        out[:, j] = -0.5 * (d * _LOG_2PI + logdet + maha)
    return out


def _logsumexp_rows(a: np.ndarray) -> np.ndarray:
    m = np.max(a, axis=1, keepdims=True)
    return (m[:, 0] + np.log(np.sum(np.exp(a - m), axis=1)))


def gmm_train(X: np.ndarray, n_components: int, covariance_type: str,
              reg_covar: float, tol: float, max_iter: int,
              quota: int) -> dict[str, Any]:
    n, d = X.shape
    k = int(n_components)
    cov_type = str(covariance_type)
    cap = max(0, min(int(max_iter), int(quota)))
    events: list[dict] = []
    if k >= n:
        k = max(1, n)
        events.append({"event": "n_components_clamped", "value": k})

    km = kmeans_train(X, k, max_iter=100, quota=10 ** 9)
    init_labels = km["labels"]
    resp = np.zeros((n, k), dtype=np.float64)
    resp[np.arange(n), init_labels] = 1.0
    for j in range(k):
        if resp[:, j].sum() == 0.0:
            idx = int(np.argmax(np.sum((X - X.mean(axis=0)) ** 2, axis=1)))
            resp[idx, :] = 0.0
            resp[idx, j] = 1.0
            events.append({"event": "empty_component_reseeded", "component": j,
                           "row": idx})

    def m_step(resp_: np.ndarray):
        nk = resp_.sum(axis=0) + 1e-12
        weights = nk / n
        means = (resp_.T @ X) / nk[:, None]
        covs = np.empty((k, d, d))
        if cov_type == "tied":
            shared = np.zeros((d, d))
            for j in range(k):
                diff = X - means[j]
                shared += (resp_[:, j:j + 1] * diff).T @ diff
            shared /= n
            for j in range(k):
                covs[j] = shared
        elif cov_type == "diag":
            for j in range(k):
                diff = X - means[j]
                var = (resp_[:, j:j + 1] * diff * diff).sum(axis=0) / nk[j]
                covs[j] = np.diag(var)
        elif cov_type == "spherical":
            for j in range(k):
                diff = X - means[j]
                var = float((resp_[:, j:j + 1] * diff * diff).sum() / (nk[j] * d))
                covs[j] = np.eye(d) * var
        else:  # full
            for j in range(k):
                diff = X - means[j]
                covs[j] = (resp_[:, j:j + 1] * diff).T @ diff / nk[j]
        return weights, means, covs

    weights, means, covs = m_step(resp)
    ll_trace: list[float] = []
    prev_ll: float | None = None
    converged = False
    for it in range(cap):
        covs_reg = _regularize(covs, float(reg_covar), events, it)
        log_prob = _estimate_log_gauss(X, means, covs_reg)
        log_weighted = log_prob + np.log(np.maximum(weights, 1e-300))[None, :]
        ll = float(np.sum(_logsumexp_rows(log_weighted)))
        ll_trace.append(ll)
        if prev_ll is not None and abs(ll - prev_ll) < float(tol):
            converged = True
            break
        prev_ll = ll
        log_resp = log_weighted - _logsumexp_rows(log_weighted)[:, None]
        resp = np.exp(log_resp)
        weights, means, covs = m_step(resp)
    covs_reg = _regularize(covs, float(reg_covar), events, len(ll_trace))
    log_prob = _estimate_log_gauss(X, means, covs_reg)
    log_weighted = log_prob + np.log(np.maximum(weights, 1e-300))[None, :]
    log_resp = log_weighted - _logsumexp_rows(log_weighted)[:, None]
    labels = np.argmax(log_resp, axis=1).astype(np.int64)
    final_ll = float(np.sum(_logsumexp_rows(log_weighted)))
    return {
        "labels": labels,
        "weights": weights,
        "means": means,
        "covariances": covs_reg,
        "covariance_type": cov_type,
        "log_likelihood": final_ll,
        "ll_trace": ll_trace,
        "iterations": len(ll_trace),
        "converged": converged,
        "events": events,
        "quota_exhausted": (not converged) and cap == int(quota),
    }
