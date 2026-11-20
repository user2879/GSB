"""窗口内多样性重排：相似度判定与得分折减全部在定点整数域执行。"""

from __future__ import annotations

from fractions import Fraction

from .config import RerankConfig
from .fixedpoint import round_half_away
from .models import Candidate, DiscountRecord, FusionResult, RerankResult


def _scale(frac_bits: int) -> int:
    return 10**frac_bits


def similarity_fixed(
    features_a: list[int], features_b: list[int], scale: int, quant_levels: int
) -> int:
    """定点余弦相似度，返回 scale 语义下的整数（范围约 [-scale, scale]）。"""
    dot = sum(a * b for a, b in zip(features_a, features_b))
    norm_a_sq = sum(a * a for a in features_a)
    norm_b_sq = sum(b * b for b in features_b)
    if norm_a_sq == 0 or norm_b_sq == 0:
        return 0
    denom = isqrt_int(norm_a_sq) * isqrt_int(norm_b_sq)
    return round_half_away(Fraction(dot * scale, denom))


def isqrt_int(value: int) -> int:
    """标准库整数平方根：纯整数运算、结果确定、无浮点。"""
    import math

    if value < 0:
        raise ValueError("负数不能开方")
    return math.isqrt(value)


def rerank_request(
    request_id: str,
    fusion_order: list[FusionResult],
    candidates: list[Candidate],
    quantized: list[list[int]],
    config: RerankConfig,
) -> RerankResult:
    by_item = {candidate.item_id: candidate for candidate in candidates}
    features_by_item = {
        candidate.item_id: row for candidate, row in zip(candidates, quantized)
    }
    window_size = min(config.rerank_window, len(fusion_order))
    window = list(fusion_order[:window_size])
    outside = [result.item_id for result in fusion_order[window_size:]]
    initial_window_order = [result.item_id for result in window]

    scale = _scale(config.frac_bits)
    threshold = round_half_away(config.similarity_threshold * scale)
    penalty = round_half_away(config.diversity_penalty * scale)

    scores = {result.item_id: result.score_fixed for result in window}
    remaining = [result.item_id for result in window]
    selected: list[str] = []
    discounts: list[DiscountRecord] = []
    step = 0

    def window_sort_key(item_id: str):
        return (-scores[item_id], item_id)

    # 确定性贪心 MMR：每轮取当前折减后得分最高的候选，对其与已选集合
    # 做一次相似度判定与定点折减，然后固定入选；每个候选恰好处理一次，
    # 窗口大小为 W 时循环 W 轮，必然收敛且可复现。
    while remaining:
        remaining.sort(key=window_sort_key)
        current = remaining.pop(0)

        best: tuple[int, str, int] | None = None  # (相似度, 参照物品, 参照序位)
        for position, reference in enumerate(selected):
            sim = similarity_fixed(
                features_by_item[current],
                features_by_item[reference],
                scale,
                config.quant_levels,
            )
            if sim >= threshold:
                candidate_key = (sim, reference, position)
                if best is None or candidate_key > best:
                    best = candidate_key

        if best is not None:
            sim, reference, _ = best
            before = scores[current]
            # 定点折减：new = score * (1 - penalty)，penalty 为 0 时严格不变
            after = round_half_away(Fraction(before * (scale - penalty), scale))
            scores[current] = after
            step += 1
            discounts.append(
                DiscountRecord(
                    target_item=current,
                    reference_item=reference,
                    similarity_fixed=sim,
                    threshold_fixed=threshold,
                    before_score_fixed=before,
                    after_score_fixed=after,
                    penalty_fixed=penalty,
                    step=step,
                )
            )

        selected.append(current)

    return RerankResult(
        request_id=request_id,
        window_size=window_size,
        final_order=selected,
        outside_order=outside,
        discounts=discounts,
        initial_window_order=initial_window_order,
    )
