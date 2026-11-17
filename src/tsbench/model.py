"""Deterministic least-squares fitting, gradient descent and rolling backtest."""

from __future__ import annotations

import numpy as np

from .transforms import seasonal_index, standardize_apply, standardize_fit


def fit_ols(x_train: np.ndarray, y_train: np.ndarray) -> dict:
    """Closed-form least squares on standardized features."""
    scaler = standardize_fit(x_train)
    xs = standardize_apply(x_train, scaler)
    design = np.column_stack([np.ones(xs.shape[0]), xs])
    beta, *_ = np.linalg.lstsq(design, y_train, rcond=None)
    return {
        "beta": np.asarray(beta, dtype=np.float64),
        "scaler": scaler,
    }


def predict_linear(beta: np.ndarray, scaler: dict, x_rows: np.ndarray) -> np.ndarray:
    xs = standardize_apply(np.atleast_2d(x_rows), scaler)
    design = np.column_stack([np.ones(xs.shape[0]), xs])
    return design @ beta


def run_gradient_descent(
    x_train: np.ndarray, y_train: np.ndarray, tier: dict
) -> list[dict]:
    """Deterministic mini-batch GD; batch order is always ascending index."""
    scaler = standardize_fit(x_train)
    xs = standardize_apply(x_train, scaler)
    design = np.column_stack([np.ones(xs.shape[0]), xs])
    params = np.zeros(design.shape[1], dtype=np.float64)
    batch_size = int(tier["batch_size"])
    lr = float(tier["learning_rate"])
    trace: list[dict] = []

    for epoch in range(int(tier["epochs"])):
        for start in range(0, design.shape[0], batch_size):
            batch_x = design[start : start + batch_size]
            batch_y = y_train[start : start + batch_size]
            residual = batch_x @ params - batch_y
            grad = (batch_x.T @ residual) / batch_x.shape[0]
            params = params - lr * grad
        residual_all = design @ params - y_train
        loss = float(np.mean(residual_all * residual_all))
        trace.append({"epoch": epoch + 1, "loss": loss, "params": params.copy()})
    return trace


def _forecast_raw(
    pre: dict,
    feat: dict,
    fit: dict,
    target_idx: int,
    horizon: int,
    x_raw: np.ndarray,
    params: dict,
) -> list[float]:
    """Recursive multi-step forecast on differenced scale, restored to raw."""
    lags = [int(v) for v in params["lag_orders"]]
    roll_window = int(params["rolling_window"])
    use_rolling = "rolling_mean" in params["feature_rules"]
    diff_order = int(params["diff_order"])
    period = int(params["seasonal_period"])
    terms = int(params["seasonal_terms"])

    differenced = list(pre["differenced"][:target_idx])
    adjusted_history = list(pre["adjusted"][:target_idx])

    # levels[0] is the differenced series; higher entries are deeper diffs.
    levels: list[list[float]] = [differenced]
    current = list(pre["adjusted"][:target_idx])
    for _ in range(diff_order):
        current = [current[i] - current[i - 1] for i in range(1, len(current))]
        levels.append(current)

    raw_forecasts: list[float] = []
    for step in range(1, horizon + 1):
        idx = target_idx + step - 1
        features = [float(differenced[idx - lag]) for lag in lags]
        if use_rolling:
            start = max(0, idx - roll_window)
            features.append(float(np.mean(np.asarray(differenced[start:idx]))))
        x_row = np.asarray(features, dtype=np.float64)
        delta = float(predict_linear(fit["beta"], fit["scaler"], x_row)[0])
        differenced.append(delta)

        if diff_order == 0:
            adjusted_pred = delta
        else:
            # Invert in fixed ascending level order.
            levels[0].append(delta)
            value = delta
            for level in range(1, diff_order + 1):
                value = value + levels[level][-1]
                levels[level].append(value)
            adjusted_pred = value
        adjusted_pred = float(adjusted_history[-1]) + adjusted_pred
        adjusted_history.append(adjusted_pred)

        trend_pred = float(pre["trend"][target_idx - 1])
        seasonal_pred = seasonal_index(
            x_raw, roll_window, period, terms, idx
        )
        raw_forecasts.append(adjusted_pred + trend_pred + seasonal_pred)
    return raw_forecasts


def error_metrics(actual: np.ndarray, predicted: np.ndarray) -> dict:
    actual = np.asarray(actual, dtype=np.float64)
    predicted = np.asarray(predicted, dtype=np.float64)
    error = predicted - actual
    mae = float(np.mean(np.abs(error)))
    rmse = float(np.sqrt(np.mean(error * error)))
    denom = np.maximum(np.abs(actual), 1e-9)
    mape = float(np.mean(np.abs(error) / denom))
    return {"mae": mae, "rmse": rmse, "mape": mape}


def run_backtest(
    x_raw: np.ndarray,
    pre: dict,
    feat: dict,
    params: dict,
    tier: dict,
    min_train: int,
) -> dict:
    """Fixed-step rolling backtest covering all configured forecast steps."""
    horizon = int(params["forecast_steps"])
    matrix = feat["matrix"]
    targets = feat["targets"]
    row_indices = feat["row_indices"]
    row_deps = feat["row_deps"]
    target_deps = feat["target_deps"]

    folds: list[dict] = []
    for position in range(min_train, len(row_indices)):
        target_idx = row_indices[position]
        if target_idx + horizon > x_raw.shape[0]:
            break
        x_train = matrix[:position]
        y_train = targets[:position]
        fit = fit_ols(x_train, y_train)
        gd_trace = run_gradient_descent(x_train, y_train, tier)
        raw_forecasts = _forecast_raw(
            pre, feat, fit, target_idx, horizon, x_raw, params
        )
        actual = [float(x_raw[target_idx + step]) for step in range(horizon)]
        metrics = error_metrics(np.asarray(actual), np.asarray(raw_forecasts))

        covered_deps = row_deps[:position] + target_deps[position : position + 1]
        folds.append(
            {
                "fold_id": f"fold-{position - min_train:05d}",
                "target_index": int(target_idx),
                "train_rows": int(position),
                "predictions": raw_forecasts,
                "actuals": actual,
                "metrics": metrics,
                "beta": fit["beta"],
                "gd_final_loss": gd_trace[-1]["loss"],
                "gd_loss_first": gd_trace[0]["loss"],
                "gd_trace": gd_trace,
                "dependency": [
                    min(p[0] for p in covered_deps),
                    max(p[1] for p in covered_deps),
                ],
            }
        )

    all_actual = np.asarray([v for fold in folds for v in fold["actuals"]])
    all_pred = np.asarray([v for fold in folds for v in fold["predictions"]])
    aggregate = (
        error_metrics(all_actual, all_pred)
        if folds
        else {"mae": 0.0, "rmse": 0.0, "mape": 0.0}
    )
    return {"folds": folds, "aggregate_metrics": aggregate}
