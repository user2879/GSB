"""逐窗口确定性训练/预测/量化指标引擎（带分层内容指纹缓存）。

每个窗口任务的缓存链：window -> train(每模型) -> predict(每模型) -> metric。
命中即直接复用且不重算；任一上游键变化都会让下游自然失配而重算，
实现“仅受影响窗口及其下游失效，未失效项复用”。
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any

from .cache import CacheStore, Ledger, window_key, train_key, predict_key, metric_key
from .config import Config
from .models import fit_model
from .quantize import quantize_list, compute_metrics
from .util import stable_sort_key
from .windows import Window
from .exchange.ir import WindowMetric


@dataclass
class ModelOutput:
    model: str
    lag_order: int
    weight: float
    predictions_q: list[int]
    fit_params: dict
    residuals: list[float]
    sort_key: str


@dataclass
class WindowResult:
    series_id: str
    window_index: int
    sort_key: str
    models: list[ModelOutput]
    metric: WindowMetric
    imputed_positions: list[int]
    missing_positions: list[int]
    metric_refs: list[str]
    train_window_len: int
    actual_q: list[int]
    status: str = "ok"

    def as_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["metric"] = self.metric.__dict__
        return d


def process_window(win: Window, cfg: Config, cache: CacheStore,
                   ledger: Ledger) -> WindowResult:
    ex_version = cfg.exchange_version
    wkey = window_key(win.fingerprint, cfg.config_hash(), ex_version)
    win_sort = stable_sort_key(win.series_id, win.window_index, "window")

    cached_window = cache.get(wkey)
    if cached_window is None:
        cache.put(wkey, {"imputed": win.imputed_positions,
                         "missing": win.missing_positions})
        ledger.record("window", win.series_id, win.window_index, None,
                      "recomputed", sort_key=win_sort)
    else:
        ledger.record("window", win.series_id, win.window_index, None,
                      "reused", sort_key=win_sort)

    model_outputs: list[ModelOutput] = []
    pkeys: list[str] = []
    model_refs: list[str] = []

    for spec in sorted(cfg.models, key=lambda m: m.name):
        m_sort = stable_sort_key(win.series_id, win.window_index, f"train:{spec.name}")
        tkey = train_key(wkey, {"name": spec.name, "lag_order": spec.lag_order,
                                "extra": spec.extra}, ex_version)
        train_blob = cache.get(tkey)
        if train_blob is None:
            artifact = fit_model(spec, win.train_values)
            cache.put(tkey, {"params": artifact.params, "residuals": artifact.residuals,
                             "lag_order": artifact.lag_order})
            ledger.record("train", win.series_id, win.window_index, spec.name,
                          "recomputed", sort_key=m_sort)
        else:
            artifact_params = train_blob["params"]
            artifact_residuals = train_blob["residuals"]
            ledger.record("train", win.series_id, win.window_index, spec.name,
                          "reused", sort_key=m_sort)

        pkey = predict_key(tkey, cfg.horizon, ex_version)
        pkeys.append(pkey)
        model_refs.append(f"{spec.name}:{pkey}")
        pred_blob = cache.get(pkey)
        p_sort = stable_sort_key(win.series_id, win.window_index, f"predict:{spec.name}")
        if pred_blob is None:
            if train_blob is None:
                preds = artifact.predict(cfg.horizon)
            else:
                # 训练命中时用缓存参数重建预测（仍为确定性纯函数）
                from .models import FitArtifact
                art = FitArtifact(spec.name, spec.lag_order, win.train_values,
                                  artifact_params, artifact_residuals)
                preds = art.predict(cfg.horizon)
            preds_q = quantize_list(preds, cfg.quant)
            cache.put(pkey, preds_q)
            ledger.record("predict", win.series_id, win.window_index, spec.name,
                          "recomputed", sort_key=p_sort)
        else:
            preds_q = pred_blob
            ledger.record("predict", win.series_id, win.window_index, spec.name,
                          "reused", sort_key=p_sort)

        params_for_out = artifact.params if train_blob is None else artifact_params
        residuals_for_out = artifact.residuals if train_blob is None else artifact_residuals
        model_outputs.append(ModelOutput(
            model=spec.name,
            lag_order=spec.lag_order,
            weight=spec.weight,
            predictions_q=preds_q,
            fit_params=params_for_out,
            residuals=residuals_for_out,
            sort_key=p_sort,
        ))

    actual_q = quantize_list(win.actual, cfg.quant)
    q_dict = {"frac_bits": cfg.quant.frac_bits, "gear": cfg.quant.gear,
              "effective_frac_bits": cfg.quant.effective_frac_bits}
    mkey = metric_key(pkeys, q_dict, ex_version)
    m_sort = stable_sort_key(win.series_id, win.window_index, "metric")
    metric_blob = cache.get(mkey)
    if metric_blob is None:
        # 集成预测：按模型权重对定点预测加权后四舍五入回定点整数
        ensemble_q = [
            int(round(sum(mo.predictions_q[step] * mo.weight for mo in model_outputs)))
            for step in range(cfg.horizon)
        ]
        metric = compute_metrics(ensemble_q, actual_q, cfg.quant)
        cache.put(mkey, {"metric": metric.as_dict(), "ensemble_q": ensemble_q})
        ledger.record("metric", win.series_id, win.window_index, None,
                      "recomputed", refs=model_refs, sort_key=m_sort)
        metric_record = WindowMetric(win.series_id, win.window_index,
                                     metric.mae_q, metric.rmse_q,
                                     metric.mape_bp, metric.near_zero_used)
        ensemble_final = ensemble_q
    else:
        ledger.record("metric", win.series_id, win.window_index, None,
                      "reused", refs=model_refs, sort_key=m_sort)
        md = metric_blob["metric"]
        metric_record = WindowMetric(win.series_id, win.window_index,
                                     md["mae_q"], md["rmse_q"], md["mape_bp"],
                                     md["near_zero_used"])
        ensemble_final = metric_blob["ensemble_q"]

    # 将集成预测挂到每个模型输出的同源预测上（交换文件逐模型存储原始预测）
    return WindowResult(
        series_id=win.series_id,
        window_index=win.window_index,
        sort_key=win_sort,
        models=model_outputs,
        metric=metric_record,
        imputed_positions=win.imputed_positions,
        missing_positions=win.missing_positions,
        metric_refs=model_refs,
        train_window_len=len(win.train_values),
        actual_q=actual_q,
        status="ok",
    ), ensemble_final
