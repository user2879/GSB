"""领域对象与快照/报告 JSON 之间的确定性转换。"""

from __future__ import annotations

from .models import (
    Candidate,
    Contribution,
    DiscountRecord,
    FusionResult,
    RerankResult,
)


def contribution_to_dict(obj: Contribution) -> dict:
    return {
        "feature_index": obj.feature_index,
        "feature_name": obj.feature_name,
        "group": obj.group,
        "normalized_fixed": obj.normalized_fixed,
        "weight_fixed": obj.weight_fixed,
        "contribution_fixed": obj.contribution_fixed,
        "frac_bits": obj.frac_bits,
    }


def contribution_from_dict(data: dict) -> Contribution:
    return Contribution(**data)


def fusion_to_dict(result: FusionResult) -> dict:
    return {
        "request_id": result.request_id,
        "item_id": result.item_id,
        "score_fixed": result.score_fixed,
        "score_clip_low": result.score_clip_low,
        "score_clip_high": result.score_clip_high,
        "clipped": result.clipped,
        "contributions": [contribution_to_dict(c) for c in result.contributions],
        "frac_bits": result.frac_bits,
        "quant_levels": result.quant_levels,
        "task_key": result.task_key,
        "recall_tie_channel": result.recall_tie_channel,
        "recall_tie_score_exact": result.recall_tie_score_exact,
    }


def fusion_from_dict(data: dict) -> FusionResult:
    data = {k: v for k, v in data.items() if k != "key"}
    data["contributions"] = [contribution_from_dict(c) for c in data["contributions"]]
    return FusionResult(**data)


def discount_to_dict(record: DiscountRecord) -> dict:
    return {
        "target_item": record.target_item,
        "reference_item": record.reference_item,
        "similarity_fixed": record.similarity_fixed,
        "threshold_fixed": record.threshold_fixed,
        "before_score_fixed": record.before_score_fixed,
        "after_score_fixed": record.after_score_fixed,
        "penalty_fixed": record.penalty_fixed,
        "step": record.step,
    }


def discount_from_dict(data: dict) -> DiscountRecord:
    return DiscountRecord(**data)


def rerank_to_dict(result: RerankResult) -> dict:
    return {
        "request_id": result.request_id,
        "window_size": result.window_size,
        "final_order": list(result.final_order),
        "outside_order": list(result.outside_order),
        "discounts": [discount_to_dict(d) for d in result.discounts],
        "initial_window_order": list(result.initial_window_order),
    }


def rerank_from_dict(data: dict) -> RerankResult:
    data = {k: v for k, v in data.items() if k not in ("key", "discount_keys")}
    data["discounts"] = [discount_from_dict(d) for d in data["discounts"]]
    return RerankResult(**data)


def candidate_to_dict(candidate: Candidate) -> dict:
    return candidate.to_dict()
