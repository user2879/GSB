"""NumPy 数值分析：模型动作权重统计与诊断指标聚合。"""
from __future__ import annotations

from collections import Counter
from typing import Any

import numpy as np


def model_weight_stats(model_params: dict[str, Any]) -> dict[str, Any]:
    """四类转移动作权重矩阵的确定性统计。"""
    actions = model_params["actions"]
    matrix = np.array([model_params["weights"][a] for a in actions], dtype=np.float64)
    return {
        "actions": list(actions),
        "shape": list(matrix.shape),
        "mean_per_action": [float(x) for x in matrix.mean(axis=1)],
        "norm_per_action": [float(x) for x in np.linalg.norm(matrix, axis=1)],
        "global_std": float(matrix.std()),
        "argmax_action_per_feature": [int(x) for x in matrix.argmax(axis=0)],
        "hash_buckets": int(model_params["hash_buckets"]),
        "feature_dim": int(model_params["feature_dim"]),
    }


def aggregate_head_histogram(histograms: list[dict[int, int]]) -> dict[str, int]:
    """合并多批头词偏差直方图。"""
    counter: Counter[int] = Counter()
    for hist in histograms:
        for key, value in hist.items():
            counter[int(key)] += int(value)
    return {str(k): counter[k] for k in sorted(counter)}


def max_score_deviation(values: list[float]) -> float:
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return 0.0
    return float(np.max(np.abs(arr)))
