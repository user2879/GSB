"""词性维特比解码与低置信度定点量化。"""
from __future__ import annotations

import math

import numpy as np

from .fingerprint import combine
from .models import Config, ProbTable, Token

NEG_INF = -1e18


def _fixed_quantize(value: float, bits: int, levels: int) -> int:
    """把 [0, 1] 置信度按位宽与量化档位转为非负定点整数。"""
    clipped = min(1.0, max(0.0, value))
    scale = (1 << bits) - 1
    step_count = levels - 1
    scaled = clipped * scale
    bucket = int(math.floor(scaled * step_count / scale + 1e-12)) if scale else 0
    if bucket >= step_count:
        return scale
    lower = bucket * scale / step_count
    upper = (bucket + 1) * scale / step_count
    if abs(scaled - lower) <= abs(upper - scaled):
        return int(round(lower))
    return int(round(upper))


def viterbi(tokens: list[Token], table: ProbTable, config: Config, oov_meta: dict) -> tuple[list[Token], dict]:
    tags = table.tags
    tag_index = {tag: idx for idx, tag in enumerate(tags)}
    n_tags = len(tags)
    n_words = len(tokens)

    trans = np.full((n_tags, n_tags), table.log_smoothing, dtype=np.float64)
    for outer, row in table.transition_logp.items():
        i = tag_index[outer]
        for inner, value in row.items():
            trans[i, tag_index[inner]] = value
    start = np.full(n_tags, table.log_smoothing, dtype=np.float64)
    for tag, value in table.start_logp.items():
        start[tag_index[tag]] = value

    emissions = np.empty((n_words, n_tags), dtype=np.float64)
    smoothed_positions: list[dict] = []
    for word_idx in range(n_words):
        word = tokens[word_idx].text
        for tag_idx in range(n_tags):
            value, is_smoothed = table.emission(tags[tag_idx], word)
            emissions[word_idx, tag_idx] = value
            if is_smoothed:
                smoothed_positions.append({"word_index": word_idx, "word": word, "tag": tags[tag_idx]})

    backpointers = np.full((n_words, n_tags), -1, dtype=np.int64)
    dp = np.empty((n_words, n_tags), dtype=np.float64)
    # 初始列：词性编码升序显式累加
    for tag_idx in range(n_tags):
        dp[0, tag_idx] = start[tag_idx] + emissions[0, tag_idx]

    for word_idx in range(1, n_words):
        for tag_idx in range(n_tags):
            best_score = NEG_INF
            best_prev = 0
            # 前驱按词性编码升序，严格大于才替换 -> 并列保留最小编码（字典序）
            for prev_idx in range(n_tags):
                score = dp[word_idx - 1, prev_idx] + trans[prev_idx, tag_idx]
                if score > best_score:
                    best_score = score
                    best_prev = prev_idx
            dp[word_idx, tag_idx] = best_score + emissions[word_idx, tag_idx]
            backpointers[word_idx, tag_idx] = best_prev

    last_best = 0
    last_score = NEG_INF
    for tag_idx in range(n_tags):
        if dp[n_words - 1, tag_idx] > last_score:
            last_score = dp[n_words - 1, tag_idx]
            last_best = tag_idx

    path = [0] * n_words
    path[n_words - 1] = last_best
    for word_idx in range(n_words - 1, 0, -1):
        path[word_idx - 1] = int(backpointers[word_idx, path[word_idx]])

    marginals = np.full(n_words, NEG_INF, dtype=np.float64)
    for word_idx in range(n_words):
        # 显式按词序号、词性编码有序累加，禁用默认数组归约
        acc = 0.0
        for tag_idx in range(n_tags):
            acc += math.exp(dp[word_idx, tag_idx])
        marginals[word_idx] = math.log(acc) if acc > 0.0 else NEG_INF

    low_confidence: list[dict] = []
    threshold_fixed = _fixed_quantize(
        config.low_confidence_threshold, config.fixed_point_bits, config.quantization_levels
    )
    normalized_components: list[float] = []
    for word_idx in range(n_words):
        tag = tags[path[word_idx]]
        token = tokens[word_idx]
        token.tag = tag
        confidence = float(math.exp(dp[word_idx, path[word_idx]] - marginals[word_idx]))
        token.confidence = confidence
        token.confidence_fixed = _fixed_quantize(
            confidence, config.fixed_point_bits, config.quantization_levels
        )
        prev_score = 0.0 if word_idx == 0 else dp[word_idx - 1, path[word_idx - 1]]
        component = dp[word_idx, path[word_idx]] - prev_score
        token.path_score = float(component)
        normalized_components.append(component)
        token.smoothed = any(
            item["word_index"] == word_idx and item["tag"] == tag for item in smoothed_positions
        )
        if token.confidence_fixed < threshold_fixed:
            low_confidence.append(
                {
                    "word_index": word_idx,
                    "text": token.text,
                    "tag": tag,
                    "confidence": confidence,
                    "confidence_fixed": token.confidence_fixed,
                    "threshold_fixed": threshold_fixed,
                }
            )

    # 整句归一化：逐词有序累加，不使用 numpy 默认归约顺序
    normalized_logp = 0.0
    for component in normalized_components:
        normalized_logp += component

    content_fp = combine(
        "tag-content",
        oov_meta["content_fingerprint"],
        table.table_fingerprint(),
        config.param_fingerprint(),
    )
    meta = {
        "content_fingerprint": content_fp,
        "tags": [tags[idx] for idx in path],
        "normalized_log_prob": normalized_logp,
        "smoothed_positions": smoothed_positions,
        "low_confidence": low_confidence,
        "threshold_fixed": threshold_fixed,
        "marginal_log_sum": [float(value) for value in marginals],
    }
    return tokens, meta
