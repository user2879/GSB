"""Stage-by-stage fixed-point error propagation diagnostics."""

from __future__ import annotations

from typing import Any

from .config import QuantizationProfile
from .features import VisualStage
from .fusion import FusionStage
from .text import TextStage
from .util import q_to_string, round_div


def _align(value: int, source_bits: int, target_bits: int) -> int:
    if source_bits == target_bits:
        return int(value)
    if source_bits > target_bits:
        return round_div(int(value), 1 << (source_bits - target_bits))
    return int(value) << (target_bits - source_bits)


def _item(key: str, low: int, high: int, low_bits: int, high_bits: int) -> dict[str, Any]:
    aligned_high = _align(high, high_bits, low_bits)
    delta = aligned_high - int(low)
    return {
        "item": key,
        "configured_q": int(low),
        "higher_aligned_q": aligned_high,
        "higher_original_q": int(high),
        "delta_q": delta,
        "abs_delta_q": abs(delta),
        "configured_value": q_to_string(int(low), low_bits),
        "higher_value": q_to_string(int(high), high_bits),
    }


def _exact_item(key: str, low: int, high: int) -> dict[str, Any]:
    delta = int(high) - int(low)
    return {
        "item": key,
        "configured_q": int(low),
        "higher_aligned_q": int(high),
        "higher_original_q": int(high),
        "delta_q": delta,
        "abs_delta_q": abs(delta),
        "configured_value": str(int(low)),
        "higher_value": str(int(high)),
    }


def _stage(
    name: str,
    items: list[dict[str, Any]],
    tolerance_q: int,
    *,
    exact: bool = False,
) -> dict[str, Any]:
    max_delta = max((item["abs_delta_q"] for item in items), default=0)
    first = next(
        (
            item["item"]
            for item in items
            if (item["delta_q"] != 0 if exact else item["abs_delta_q"] > tolerance_q)
        ),
        None,
    )
    exceeds = first is not None
    return {
        "stage": name,
        "item_count": len(items),
        "max_abs_delta_q": max_delta,
        "exceeds_tolerance": exceeds,
        "first_exceeding_item": first,
        "items": items,
    }


def compare_precisions(
    *,
    configured_visual: VisualStage,
    configured_text: TextStage,
    configured_fusion: FusionStage,
    higher_visual: VisualStage,
    higher_text: TextStage,
    higher_fusion: FusionStage,
    configured_profile: QuantizationProfile,
    higher_profile: QuantizationProfile,
    tolerance_q: int,
) -> dict[str, Any]:
    low_bits = configured_profile.fractional_bits
    high_bits = higher_profile.fractional_bits

    quantization_items: list[dict[str, Any]] = []
    for block_index in range(min(len(configured_visual.blocks), len(higher_visual.blocks))):
        low_block = configured_visual.blocks[block_index]
        high_block = higher_visual.blocks[block_index]
        for dimension in range(min(len(low_block["values"]), len(high_block["values"]))):
            quantization_items.append(
                _item(
                    f"block.{block_index}.dim.{dimension}.quantized",
                    low_block["values"][dimension],
                    high_block["values"][dimension],
                    low_bits,
                    high_bits,
                )
            )
            quantization_items.append(
                _item(
                    f"block.{block_index}.dim.{dimension}.residual",
                    low_block["residuals"][dimension],
                    high_block["residuals"][dimension],
                    low_bits,
                    high_bits,
                )
            )
    for dimension in range(min(len(configured_visual.vector), len(higher_visual.vector))):
        quantization_items.append(
            _item(
                f"visual.aggregate.dim.{dimension}",
                configured_visual.vector[dimension],
                higher_visual.vector[dimension],
                low_bits,
                high_bits,
            )
        )

    text_items: list[dict[str, Any]] = []
    for dimension in range(min(len(configured_text.vector), len(higher_text.vector))):
        text_items.append(
            _item(
                f"text.dim.{dimension}",
                configured_text.vector[dimension],
                higher_text.vector[dimension],
                low_bits,
                high_bits,
            )
        )
        text_items.append(
            _item(
                f"text.dim.{dimension}.residual",
                configured_text.residuals[dimension],
                higher_text.residuals[dimension],
                low_bits,
                high_bits,
            )
        )

    fusion_items: list[dict[str, Any]] = []
    low_modalities = {item["modality"]: item for item in configured_fusion.modality_contributions}
    high_modalities = {item["modality"]: item for item in higher_fusion.modality_contributions}
    for modality in sorted(low_modalities):
        for field in ("mean_q", "exp_q", "softmax_probability_q", "activation_q", "score_contribution_q"):
            fusion_items.append(
                _item(
                    f"fusion.{modality}.{field}",
                    int(low_modalities[modality][field]),
                    int(high_modalities[modality][field]),
                    low_bits,
                    high_bits,
                )
            )
    fusion_items.append(
        _item(
            "fusion.score",
            configured_fusion.score_q,
            higher_fusion.score_q,
            low_bits,
            high_bits,
        )
    )

    decision_items = [
        _exact_item(
            "decision.positive",
            1 if configured_fusion.decision == "positive" else 0,
            1 if higher_fusion.decision == "positive" else 0,
        )
    ]

    stages = [
        _stage("quantization", quantization_items, tolerance_q),
        _stage("text_encoding", text_items, tolerance_q),
        _stage("weighted_fusion", fusion_items, tolerance_q),
        _stage("threshold_decision", decision_items, tolerance_q, exact=True),
    ]
    first_exceeded_stage = next(
        (stage["stage"] for stage in stages if stage["exceeds_tolerance"]),
        None,
    )
    decision_flip = configured_fusion.decision != higher_fusion.decision
    if not decision_flip:
        attribution = None
    elif first_exceeded_stage in {"quantization", "text_encoding"}:
        attribution = "quantization_truncation"
    else:
        attribution = "fusion_rounding_accumulation"

    return {
        "configured_profile": configured_profile.as_dict(),
        "higher_profile": higher_profile.as_dict(),
        "tolerance_q": tolerance_q,
        "tolerance": q_to_string(tolerance_q, low_bits),
        "first_exceeded_stage": first_exceeded_stage,
        "decision_flip": decision_flip,
        "flip_attribution": attribution,
        "stages": stages,
    }
