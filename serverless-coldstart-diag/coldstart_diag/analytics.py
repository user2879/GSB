"""基于 NumPy 的可视化派生分析。

注意：本模块只服务于热力分档、等待曲线分位数等展示性指标，
所有账本/定点核算仍由 engine.py 以纯整数运算完成，保证可复现性。
"""

import numpy as np


def wait_percentiles(payloads):
    """平均等待毫秒的 P50/P90/P95/P99（确定性，无随机量）。"""
    values = np.array([p["avg_wait_ms"] for p in payloads], dtype=np.int64)
    if values.size == 0:
        return {"p50": 0, "p90": 0, "p95": 0, "p99": 0}
    return {
        "p50": int(np.percentile(values, 50, method="linear")),
        "p90": int(np.percentile(values, 90, method="linear")),
        "p95": int(np.percentile(values, 95, method="linear")),
        "p99": int(np.percentile(values, 99, method="linear")),
    }


def cold_ratio_heat_hist(payloads, scale):
    """冷启动定点比例的热力直方图（4 档）。"""
    arr = np.array([p["cold_ratio_q"] for p in payloads], dtype=np.int64) / scale
    bins = np.array([0.05, 0.20, 0.50])
    counts = [0, 0, 0, 0]
    for v in arr:
        if v <= bins[0]:
            counts[0] += 1
        elif v <= bins[1]:
            counts[1] += 1
        elif v <= bins[2]:
            counts[2] += 1
        else:
            counts[3] += 1
    return {"low": counts[0], "medium": counts[1], "high": counts[2], "severe": counts[3]}


def queue_pressure_series(payloads):
    """排队压力序列（NumPy 向量化），供等待曲线绘制。"""
    minutes = np.array([p["minute"] for p in payloads], dtype=np.int64)
    queued = np.array([p["queued_calls"] for p in payloads], dtype=np.int64)
    waits = np.array([p["avg_wait_ms"] for p in payloads], dtype=np.int64)
    return [
        {"minute": int(m), "queued_calls": int(q), "avg_wait_ms": int(w)}
        for m, q, w in zip(minutes, queued, waits, strict=True)
    ]
