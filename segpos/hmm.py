"""对数域维特比：未登录词字符级识别与词性标注。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Sequence, Tuple

import numpy as np

NEG_INF = -1.0e300


@dataclass
class ViterbiResult:
    tags: List[str]
    deltas: List[float]
    normalized_logprob: float
    margin_confidence: List[float]
    smoothing_hits: int


def _safe_log_sum_exp(values: np.ndarray) -> float:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return NEG_INF
    max_value = finite.max()
    return float(max_value + np.log(np.exp(finite - max_value).sum()))


def viterbi_decode(
    observations: Sequence[str],
    tags: Sequence[str],
    start_scores: np.ndarray,
    transition_score: Callable[[int, int], Tuple[float, bool]],
    emission_score: Callable[[str, int], Tuple[float, bool]],
) -> ViterbiResult:
    """通用一阶 HMM 维特比解码（自然对数域，NumPy 矩阵运算）。

    返回最优标签序列、逐位置局部路径得分、按长度归一化的整句对数概率、
    每位置后验边际置信度以及平滑命中次数。
    """

    tag_count = len(tags)
    length = len(observations)
    if length == 0:
        return ViterbiResult([], [], 0.0, [], 0)

    delta = np.full((length, tag_count), NEG_INF, dtype=np.float64)
    back = np.full((length, tag_count), -1, dtype=np.int64)
    forward = np.full((length, tag_count), NEG_INF, dtype=np.float64)
    smoothing_hits = 0

    for tag_idx in range(tag_count):
        emit_value, smoothed = emission_score(observations[0], tag_idx)
        smoothing_hits += int(smoothed)
        delta[0, tag_idx] = start_scores[tag_idx] + emit_value
        forward[0, tag_idx] = delta[0, tag_idx]

    for step in range(1, length):
        for tag_idx in range(tag_count):
            emit_value, emit_smoothed = emission_score(observations[step], tag_idx)
            smoothing_hits += int(emit_smoothed)
            candidate = np.empty(tag_count, dtype=np.float64)
            for prev_idx in range(tag_count):
                trans_value, trans_smoothed = transition_score(prev_idx, tag_idx)
                smoothing_hits += int(trans_smoothed)
                candidate[prev_idx] = delta[step - 1, prev_idx] + trans_value
            best_prev = int(np.argmax(candidate))
            back[step, tag_idx] = best_prev
            delta[step, tag_idx] = candidate[best_prev] + emit_value
            forward[step, tag_idx] = _safe_log_sum_exp(candidate + emit_value)

    best_last = int(np.argmax(delta[length - 1]))
    best_path = np.empty(length, dtype=np.int64)
    best_path[length - 1] = best_last
    for step in range(length - 1, 0, -1):
        best_path[step - 1] = back[step, best_path[step]]

    tags_out = [tags[index] for index in best_path]
    deltas_out = [float(delta[step, best_path[step]]) for step in range(length)]
    total_logprob = _safe_log_sum_exp(forward[length - 1])
    normalized = total_logprob / length
    margins: List[float] = []
    for step in range(length):
        row = forward[step]
        finite = row[np.isfinite(row)]
        if finite.size == 0:
            margins.append(0.0)
            continue
        centered = finite - finite.max()
        weights = np.exp(centered)
        margins.append(float(weights.max() / weights.sum()))
    return ViterbiResult(
        tags=tags_out,
        deltas=deltas_out,
        normalized_logprob=float(normalized),
        margin_confidence=margins,
        smoothing_hits=smoothing_hits,
    )
