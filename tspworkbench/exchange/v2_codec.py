"""v2 JSON 交换文件的解析与写入。

v2 显式携带浮点窗口权重、精度标签、命名模型、命名缺失处理方式，
以及配置指纹与版本指纹，便于无损承载命名信息。
"""
from __future__ import annotations

import json
from pathlib import Path

from .ir import ExchangeIR, WindowWeight, WindowPrediction, WindowMetric


def parse_v2(text: str, failures: list | None = None) -> ExchangeIR:
    failures = failures if failures is not None else []
    data = json.loads(text)
    ir = ExchangeIR(
        exchange_version="v2",
        config_fingerprint=data.get("config_fingerprint", ""),
        version_fingerprint=data.get("version_fingerprint", ""),
        missing_strategy=data.get("missing_strategy", "forward_fill"),
        weights=[WindowWeight(
            series_id=w["series_id"],
            window_index=int(w["window_index"]),
            model=w["model"],
            weight=float(w["weight"]),
            lag_order=int(w["lag_order"]),
            precision_label=w.get("precision_label", "exact"),
        ) for w in data.get("weights", [])],
        predictions=[WindowPrediction(
            series_id=p["series_id"],
            window_index=int(p["window_index"]),
            model=p["model"],
            step_index=int(p["step_index"]),
            pred_q=int(p["pred_q"]),
            actual_q=int(p["actual_q"]),
        ) for p in data.get("predictions", [])],
        metrics=[WindowMetric(
            series_id=m["series_id"],
            window_index=int(m["window_index"]),
            mae_q=int(m["mae_q"]),
            rmse_q=int(m["rmse_q"]),
            mape_bp=int(m["mape_bp"]),
            near_zero=bool(m["near_zero"]),
        ) for m in data.get("metrics", [])],
    )
    return ir


def write_v2(ir: ExchangeIR, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "exchange_version": "v2",
        "config_fingerprint": ir.config_fingerprint,
        "version_fingerprint": ir.version_fingerprint,
        "missing_strategy": ir.missing_strategy,
        "weights": [w.__dict__ for w in ir.weights],
        "predictions": [p.__dict__ for p in ir.predictions],
        "metrics": [m.__dict__ for m in ir.metrics],
    }
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True),
                   encoding="utf-8")
    tmp.replace(path)
    return path
