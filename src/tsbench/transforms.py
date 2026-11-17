"""Preprocessing, seasonal decomposition, differencing and lag features.

All transformations align on integer time indices. Every function is pure and
deterministic, and each produced value exposes the exact raw-index interval it
depends on so that incremental invalidation can be computed precisely.
"""

from __future__ import annotations

import numpy as np


def trend_terms(x: np.ndarray, window: int, idx: int) -> np.ndarray:
    """Centered moving-average trend terms for the requested index."""
    half = window // 2
    length = x.shape[0]
    terms = []
    for offset in range(-half, window - half):
        source = idx + offset
        source = max(0, min(length - 1, source))
        terms.append(x[source])
    return np.asarray(terms, dtype=np.float64)


def trend_dependency(idx: int, length: int, window: int) -> tuple[int, int]:
    half = window // 2
    return max(0, idx - half), min(length - 1, idx + window - half - 1)


def seasonal_index(
    x: np.ndarray, window: int, period: int, terms: int, idx: int
) -> float:
    """Bounded-lookback mean of detrended values at the same season phase."""
    members = []
    for step in range(1, terms + 1):
        source = idx - step * period
        if source >= 0:
            t_lo, t_hi = trend_dependency(source, x.shape[0], window)
            trend = float(np.mean(trend_terms(x, window, source)))
            members.append((t_lo, t_hi, x[source] - trend))
    if not members:
        return 0.0
    return float(np.mean(np.asarray([m[2] for m in members], dtype=np.float64)))


def seasonal_dependency(
    idx: int, length: int, window: int, period: int, terms: int
) -> tuple[int, int]:
    sources = [idx - step * period for step in range(1, terms + 1)]
    sources = [s for s in sources if s >= 0]
    if not sources:
        return idx, idx
    # Each source contributes both its own raw value (detrended in-place) and
    # its centered trend window, so the source indices themselves must be
    # included in the dependency interval.
    bounds = [trend_dependency(s, length, window) for s in sources]
    lo = min(min(lo for lo, _ in bounds), min(sources))
    hi = max(max(hi for _, hi in bounds), max(sources))
    return lo, hi


def preprocess(x: np.ndarray, params: dict) -> dict:
    """Seasonal decomposition then differencing, with fixed operation order."""
    window = int(params["rolling_window"])
    period = int(params["seasonal_period"])
    terms = int(params["seasonal_terms"])
    diff_order = int(params["diff_order"])
    length = x.shape[0]

    trend = np.zeros(length, dtype=np.float64)
    seasonal = np.zeros(length, dtype=np.float64)
    adjusted = np.zeros(length, dtype=np.float64)
    pre_deps: list[list[int]] = []

    for idx in range(length):
        trend[idx] = float(np.mean(trend_terms(x, window, idx)))
        seasonal[idx] = seasonal_index(x, window, period, terms, idx)
        # Fixed order: raw - trend first, then subtract the seasonal term.
        detrended = x[idx] - trend[idx]
        adjusted[idx] = detrended - seasonal[idx]
        s_lo, s_hi = seasonal_dependency(idx, length, window, period, terms)
        t_lo, t_hi = trend_dependency(idx, length, window)
        pre_deps.append([min(idx, s_lo, t_lo), max(idx, s_hi, t_hi)])

    differenced = adjusted.copy()
    diff_deps = [list(pair) for pair in pre_deps]
    for _ in range(diff_order):
        next_values = np.zeros(differenced.shape[0], dtype=np.float64)
        next_deps: list[list[int]] = []
        for idx in range(differenced.shape[0]):
            if idx == 0:
                next_values[0] = 0.0
                next_deps.append(diff_deps[0])
            else:
                next_values[idx] = differenced[idx] - differenced[idx - 1]
                next_deps.append(
                    [
                        min(diff_deps[idx][0], diff_deps[idx - 1][0]),
                        max(diff_deps[idx][1], diff_deps[idx - 1][1]),
                    ]
                )
        differenced = next_values
        diff_deps = next_deps

    return {
        "trend": trend,
        "seasonal": seasonal,
        "adjusted": adjusted,
        "differenced": differenced,
        "preprocess_deps": pre_deps,
        "diff_deps": diff_deps,
    }


def inverse_difference(history: list[float], value: float, diff_order: int) -> float:
    """Undo differencing using a fixed right-to-left accumulation order."""
    restored = float(value)
    for level in range(diff_order - 1, -1, -1):
        restored = restored + history[level]
    return restored


def build_feature_matrix(pre: dict, params: dict) -> dict:
    """Construct deterministic lag (+ optional rolling-mean) feature rows."""
    differenced = pre["differenced"]
    diff_deps = pre["diff_deps"]
    lags = [int(v) for v in params["lag_orders"]]
    max_lag = max(lags)
    roll_window = int(params["rolling_window"])
    rules = list(params["feature_rules"])

    names = [f"lag{lag}" for lag in lags]
    use_rolling = "rolling_mean" in rules
    if use_rolling:
        names.append(f"roll_mean_{roll_window}")

    rows: list[np.ndarray] = []
    row_indices: list[int] = []
    row_deps: list[list[int]] = []
    for idx in range(max_lag, differenced.shape[0]):
        features = [float(differenced[idx - lag]) for lag in lags]
        dep_values = list(diff_deps[idx - max_lag : idx])
        if use_rolling:
            start = max(0, idx - roll_window)
            window_slice = differenced[start:idx]
            features.append(float(np.mean(window_slice)))
            dep_values.extend(diff_deps[start:idx])
        rows.append(np.asarray(features, dtype=np.float64))
        row_indices.append(idx)
        row_deps.append(
            [
                min(pair[0] for pair in dep_values),
                max(pair[1] for pair in dep_values),
            ]
        )

    matrix = np.vstack(rows) if rows else np.zeros((0, len(names)), dtype=np.float64)
    targets = np.asarray(
        [float(differenced[idx]) for idx in row_indices], dtype=np.float64
    )
    target_deps = [list(diff_deps[idx]) for idx in row_indices]
    return {
        "feature_names": names,
        "matrix": matrix,
        "targets": targets,
        "row_indices": row_indices,
        "row_deps": row_deps,
        "target_deps": target_deps,
        "max_lag": max_lag,
    }


def standardize_fit(matrix: np.ndarray) -> dict:
    """Per-column mean/std used both for OLS and gradient descent."""
    if matrix.shape[0] == 0:
        return {"mean": np.zeros(matrix.shape[1]), "std": np.ones(matrix.shape[1])}
    mean = np.mean(matrix, axis=0)
    std = np.std(matrix, axis=0)
    std = np.where(std < 1e-12, 1.0, std)
    return {"mean": mean, "std": std}


def standardize_apply(matrix: np.ndarray, scaler: dict) -> np.ndarray:
    return (matrix - scaler["mean"]) / scaler["std"]
