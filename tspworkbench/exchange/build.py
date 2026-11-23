"""把引擎运行结果组装为交换 IR。"""
from __future__ import annotations

from ..config import Config
from .ir import ExchangeIR, WindowWeight, WindowPrediction, WindowMetric


def build_ir(cfg: Config, run_results: list) -> ExchangeIR:
    weights: list[WindowWeight] = []
    predictions: list[WindowPrediction] = []
    metrics: list[WindowMetric] = []

    for win in run_results:
        for mo in win.models:
            weights.append(WindowWeight(
                series_id=win.series_id,
                window_index=win.window_index,
                model=mo.model,
                weight=mo.weight,
                lag_order=mo.lag_order,
                precision_label="exact",
            ))
            for step_index, pred_q in enumerate(mo.predictions_q):
                predictions.append(WindowPrediction(
                    series_id=win.series_id,
                    window_index=win.window_index,
                    model=mo.model,
                    step_index=step_index,
                    pred_q=pred_q,
                    actual_q=win.actual_q[step_index],
                ))
        metrics.append(WindowMetric(
            series_id=win.series_id,
            window_index=win.window_index,
            mae_q=win.metric.mae_q,
            rmse_q=win.metric.rmse_q,
            mape_bp=win.metric.mape_bp,
            near_zero=win.metric.near_zero,
        ))

    ir = ExchangeIR(
        exchange_version=cfg.exchange_version,
        config_fingerprint=cfg.config_hash(),
        version_fingerprint="",
        missing_strategy=cfg.missing_strategy,
        weights=weights,
        predictions=predictions,
        metrics=metrics,
    )
    ir.validate()
    return ir
