"""逐段指标与跨格式/跨版本一致性口径。"""
from __future__ import annotations

from typing import Any

import numpy as np

# 声明的容差口径（验收只依据这里，不依赖具体机器）
TOLERANCE = {
    "rms_db": 0.5,          # 每段 RMS(dB) 允许差
    "peak": 0.02,           # 每段峰值绝对差
    "max_abs_sample": 2.0,  # 16bit PCM 量化单位下的最大样本差
    "rms_abs": 1.0e-3,      # 每段 RMS 绝对差（低能量段友好）
}

EPS = 1e-12


def segment_metrics(signal: np.ndarray, segments: list[dict]) -> list[dict[str, Any]]:
    rows = []
    for seg in segments:
        start, end = seg["start_sample"], seg["end_sample"]
        piece = signal[start:end]
        rms = float(np.sqrt(np.mean(piece.astype(np.float64) ** 2) + EPS))
        rms_db = float(20.0 * np.log10(rms))
        peak = float(np.max(np.abs(piece))) if piece.size else 0.0
        rows.append(
            {
                "kind": seg["kind"],
                "start_sample": start,
                "end_sample": end,
                "rms": rms,
                "rms_db": rms_db,
                "peak": peak,
            }
        )
    return rows


def compare_conclusions(
    rows_a: list[dict[str, Any]], rows_b: list[dict[str, Any]]
) -> dict[str, Any]:
    """按声明容差比较两份逐段指标，返回逐项差异与总体结论。"""
    if len(rows_a) != len(rows_b):
        return {"consistent": False, "reason": "segment count differs", "diffs": []}
    diffs = []
    consistent = True
    for i, (a, b) in enumerate(zip(rows_a, rows_b)):
        d_rms_db = abs(a["rms_db"] - b["rms_db"])
        d_peak = abs(a["peak"] - b["peak"])
        d_rms = abs(a["rms"] - b["rms"])
        ok = (
            d_rms_db <= TOLERANCE["rms_db"]
            and d_peak <= TOLERANCE["peak"]
            and d_rms <= TOLERANCE["rms_abs"]
        )
        consistent = consistent and ok
        diffs.append(
            {
                "segment": i,
                "kind": a["kind"],
                "abs_diff_rms_db": d_rms_db,
                "abs_diff_peak": d_peak,
                "abs_diff_rms": d_rms,
                "within_tolerance": ok,
            }
        )
    return {"consistent": consistent, "tolerance": TOLERANCE, "diffs": diffs}


def compare_pcm_int16(a: np.ndarray, b: np.ndarray) -> dict[str, Any]:
    """比较两份 int16 PCM，长度对齐后给最大样本差。"""
    n = min(a.shape[0], b.shape[0])
    if a.shape != b.shape:
        return {
            "consistent": False,
            "reason": "length/shape differs",
            "max_abs_sample": None,
            "compared_samples": n,
        }
    diff = np.abs(a.astype(np.int32) - b.astype(np.int32))
    max_diff = int(np.max(diff)) if diff.size else 0
    return {
        "consistent": max_diff <= TOLERANCE["max_abs_sample"],
        "max_abs_sample": max_diff,
        "compared_samples": n,
        "tolerance": TOLERANCE["max_abs_sample"],
    }
