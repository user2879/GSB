"""定点整数加权融合打分。

所有运算均为 Python 整数（配合 Fraction 做确定性四舍五入），运算顺序固定：
对每个候选按特征下标 0..N-1 依次计算贡献并累加，再应用融合温度、裁剪。
"""

from __future__ import annotations

from fractions import Fraction

from .config import RerankConfig
from .fixedpoint import round_half_away, stable_task_key
from .models import Candidate, Contribution, FusionResult


def _scale(frac_bits: int) -> int:
    return 10**frac_bits


def quantize_weights(config: RerankConfig, frac_bits: int) -> list[int]:
    scale = _scale(frac_bits)
    return [round_half_away(weight * scale) for weight in config.weights]


def clip_bounds_fixed(config: RerankConfig, frac_bits: int) -> tuple[int, int]:
    scale = _scale(frac_bits)
    low = round_half_away(config.score_clip[0] * scale)
    high = round_half_away(config.score_clip[1] * scale)
    return low, high


def fuse_candidate(
    candidate: Candidate,
    quantized_features: list[int],
    config: RerankConfig,
    frac_bits: int,
    quant_levels: int,
    weights_fixed: list[int],
    temperature_fixed: int,
    clip_low: int,
    clip_high: int,
    group_mapping: dict[int, str],
) -> FusionResult:
    scale = _scale(frac_bits)
    contributions: list[Contribution] = []
    raw_sum = 0
    for index, feature_fixed in enumerate(quantized_features):
        product = Fraction(feature_fixed * weights_fixed[index], scale)
        contribution = round_half_away(product)
        raw_sum += contribution
        contributions.append(
            Contribution(
                feature_index=index,
                feature_name=config.feature_names[index],
                group=group_mapping[index],
                normalized_fixed=feature_fixed,
                weight_fixed=weights_fixed[index],
                contribution_fixed=contribution,
                frac_bits=frac_bits,
            )
        )

    if temperature_fixed != scale:
        score = round_half_away(Fraction(raw_sum * scale, temperature_fixed))
    else:
        score = raw_sum

    clipped = not (clip_low <= score <= clip_high)
    score = max(clip_low, min(clip_high, score))

    recall_exact = {
        channel: Fraction(candidate.recall_scores[channel])
        for channel in config.recall_channels
    }
    # 同分决断：最高召回通道分数，分数并列时取通道名字典序最小者
    top_channel = min(
        config.recall_channels,
        key=lambda channel: (-recall_exact[channel], channel),
    )

    return FusionResult(
        request_id=candidate.request_id,
        item_id=candidate.item_id,
        score_fixed=score,
        score_clip_low=clip_low,
        score_clip_high=clip_high,
        clipped=clipped,
        contributions=contributions,
        frac_bits=frac_bits,
        quant_levels=quant_levels,
        task_key=stable_task_key(candidate.request_id, candidate.item_id),
        recall_tie_channel=top_channel,
        recall_tie_score_exact=(
            f"{recall_exact[top_channel].numerator}/"
            f"{recall_exact[top_channel].denominator}"
        ),
    )


def fuse_request(
    candidates: list[Candidate],
    quantized: list[list[int]],
    config: RerankConfig,
    frac_bits: int | None = None,
    quant_levels: int | None = None,
) -> list[FusionResult]:
    bits = config.frac_bits if frac_bits is None else frac_bits
    levels = config.quant_levels if quant_levels is None else quant_levels
    weights_fixed = quantize_weights(config, bits)
    temperature_fixed = round_half_away(config.fusion_temperature * _scale(bits))
    clip_low, clip_high = clip_bounds_fixed(config, bits)
    from .config import group_of_feature

    group_mapping = group_of_feature(config)
    return [
        fuse_candidate(
            candidate,
            features,
            config,
            bits,
            levels,
            weights_fixed,
            temperature_fixed,
            clip_low,
            clip_high,
            group_mapping,
        )
        for candidate, features in zip(candidates, quantized)
    ]


def fusion_sort_key(result: FusionResult, config: RerankConfig, candidate: Candidate):
    """排序键：融合分降序；同分时按最高召回通道精确分数降序、通道名升序、物品 ID 升序。"""
    best_channel = min(
        config.recall_channels,
        key=lambda channel: (-Fraction(candidate.recall_scores[channel]), channel),
    )
    best_score = Fraction(candidate.recall_scores[best_channel])
    return (-result.score_fixed, -best_score, best_channel, result.item_id)


def sort_fusion(
    results: list[FusionResult], candidates: list[Candidate], config: RerankConfig
) -> list[FusionResult]:
    by_item = {candidate.item_id: candidate for candidate in candidates}
    return sorted(
        results,
        key=lambda result: fusion_sort_key(result, config, by_item[result.item_id]),
    )
