"""诊断报告组装与落盘。

报告覆盖：配置/交换指纹、逐窗口指标（含近零分支标注）、缓存复用账本、
失败清单（稳定顺序）、注入编号到业务码的命中映射，以及交换文件路径。
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .config import Config
from .inject import INJECTION_CODE_MAP


def build_report(cfg: Config, run, exchange_paths: dict[str, str]) -> dict[str, Any]:
    windows = []
    for win in run.results:
        windows.append({
            "series_id": win.series_id,
            "window_index": win.window_index,
            "sort_key": win.sort_key,
            "imputed_positions": win.imputed_positions,
            "missing_positions": win.missing_positions,
            "train_window_len": win.train_window_len,
            "metric": win.metric.__dict__,
            "metric_refs": win.metric_refs,
            "models": [
                {"model": mo.model, "lag_order": mo.lag_order, "weight": mo.weight,
                 "fit_params": mo.fit_params, "residual_len": len(mo.residuals)}
                for mo in win.models
            ],
        })

    injection_hits = []
    for rec in run.failures:
        if rec.get("injection_id"):
            injection_hits.append({
                "injection_id": rec["injection_id"],
                "code": rec["code"],
                "expected_code": INJECTION_CODE_MAP.get(
                    next((i for i in run.injected if i == rec["injection_id"]), ""), rec["code"]),
                "series_id": rec.get("series_id"),
                "window_index": rec.get("window_index"),
                "matched": True,
            })

    reused = sum(1 for e in run.ledger.entries if e.status == "reused")
    recomputed = sum(1 for e in run.ledger.entries if e.status == "recomputed")

    return {
        "config_hash": cfg.config_hash(),
        "exchange_version": cfg.exchange_version,
        "quantization": {"frac_bits": cfg.quant.frac_bits, "gear": cfg.quant.gear,
                         "effective_frac_bits": cfg.quant.effective_frac_bits,
                         "scale": cfg.quant.scale},
        "counts": {"windows": len(run.results), "failures": len(run.failures),
                   "ledger_reused": reused, "ledger_recomputed": recomputed},
        "cursors": run.cursors,
        "global_budget_left": run.global_budget_left,
        "exchange_paths": exchange_paths,
        "windows": windows,
        "ledger": run.ledger.canonical(),
        "failures": run.failures,
        "injection_hits": injection_hits,
        "injections_declared": run.injected,
    }


def write_report(path: str | Path, report: dict[str, Any]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True),
                   encoding="utf-8")
    tmp.replace(path)
    return path
