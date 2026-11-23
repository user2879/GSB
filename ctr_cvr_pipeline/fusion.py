from __future__ import annotations

import math
from typing import Any

from .canonical import fingerprint
from .config import FeatureConfig
from .fixed import clamp_fixed, fraction_scale, quantize_real, trace_rounding


FUSION_SCHEMA = "fixed-fusion-v1"


def compute_fusion(
    training_payload: dict[str, Any],
    config: FeatureConfig,
    quantization_tier: str,
    fractional_bits: int,
) -> dict[str, Any]:
    scale = fraction_scale(fractional_bits)
    ctr_fixed = training_payload["ctr_fixed"]
    cvr_fixed = training_payload["cvr_fixed"]
    ctr_ref = training_payload["ctr_reference"]
    cvr_ref = training_payload["cvr_reference"]
    fusion_config = config.fusion
    ctr_weight_q = quantize_real(fusion_config.ctr_weight, fractional_bits, quantization_tier).value
    cvr_weight_q = quantize_real(fusion_config.cvr_weight, fractional_bits, quantization_tier).value
    records: list[dict[str, Any]] = []
    for sample_id in sorted(ctr_fixed):
        ctr_q = int(ctr_fixed[sample_id]["score_q"])
        cvr_q = int(cvr_fixed[sample_id]["score_q"])
        if fusion_config.mode == "product":
            numerator = ctr_q * cvr_q
            denominator = scale
            reference = float(ctr_ref[sample_id]) * float(cvr_ref[sample_id])
        else:
            numerator = ctr_weight_q * ctr_q + cvr_weight_q * cvr_q
            denominator = scale
            reference = (
                fusion_config.ctr_weight * float(ctr_ref[sample_id])
                + fusion_config.cvr_weight * float(cvr_ref[sample_id])
            )
        fused, record = trace_rounding(
            "fusion",
            "fusion_rescale",
            numerator,
            denominator,
            sample=sample_id,
            extra={
                "mode": fusion_config.mode,
                "ctr_weight_q": ctr_weight_q,
                "cvr_weight_q": cvr_weight_q,
            },
        )
        fixed_value, clipped, clip_direction = clamp_fixed(fused, quantization_tier)
        if clipped:
            record["clipped"] = True
            record["clip_direction"] = clip_direction
        request_id, item_id = sample_id.split(":", 1)
        records.append(
            {
                "sample_id": sample_id,
                "request_id": request_id,
                "item_id": item_id,
                "ctr_score_q": ctr_q,
                "cvr_score_q": cvr_q,
                "ctr_score": ctr_q / scale,
                "cvr_score": cvr_q / scale,
                "ctr_reference_score": float(ctr_ref[sample_id]),
                "cvr_reference_score": float(cvr_ref[sample_id]),
                "fusion_score_q": fixed_value,
                "fusion_score": fixed_value / scale,
                "fusion_reference_score": reference,
                "fusion_rounding": record,
            }
        )
    ranked = assign_ranks(records, "fusion_score")
    assign_reference_ranks(ranked, "fusion_reference_score")
    return {
        "schema": FUSION_SCHEMA,
        "mode": fusion_config.mode,
        "ctr_weight": fusion_config.ctr_weight,
        "cvr_weight": fusion_config.cvr_weight,
        "ctr_weight_q": ctr_weight_q,
        "cvr_weight_q": cvr_weight_q,
        "tier": quantization_tier,
        "fractional_bits": fractional_bits,
        "records": ranked,
    }


def assign_ranks(records: list[dict[str, Any]], score_field: str) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        grouped.setdefault(record["request_id"], []).append(record)
    for request_id, group in grouped.items():
        ordered = sorted(group, key=lambda item: (-float(item[score_field]), item["item_id"]))
        for rank, record in enumerate(ordered, start=1):
            record["fixed_rank"] = rank
    for rank, record in enumerate(
        sorted(records, key=lambda item: (-float(item[score_field]), item["request_id"], item["item_id"])),
        start=1,
    ):
        record["global_fixed_rank"] = rank
    return records


def assign_reference_ranks(records: list[dict[str, Any]], score_field: str) -> None:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        grouped.setdefault(record["request_id"], []).append(record)
    for group in grouped.values():
        ordered = sorted(group, key=lambda item: (-float(item[score_field]), item["item_id"]))
        for rank, record in enumerate(ordered, start=1):
            record["reference_rank"] = rank
    for rank, record in enumerate(
        sorted(records, key=lambda item: (-float(item[score_field]), item["request_id"], item["item_id"])),
        start=1,
    ):
        record["global_reference_rank"] = rank


def fusion_rank_map(records: list[dict[str, Any]], score_field: str) -> dict[tuple[str, str], int]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        grouped.setdefault(record["request_id"], []).append(record)
    result: dict[tuple[str, str], int] = {}
    for group in grouped.values():
        ordered = sorted(group, key=lambda item: (-float(item[score_field]), item["item_id"]))
        for rank, record in enumerate(ordered, start=1):
            result[(record["request_id"], record["item_id"])] = rank
    return result


def _safe_log(value: float) -> float:
    return math.log(max(value, 1e-15))


def _pair_contribution(
    a: dict[str, Any],
    b: dict[str, Any],
    mode: str,
    ctr_weight: float,
    cvr_weight: float,
) -> tuple[float, float, float]:
    if mode == "product":
        ctr = _safe_log(float(a["ctr_score"])) - _safe_log(float(b["ctr_score"]))
        cvr = _safe_log(float(a["cvr_score"])) - _safe_log(float(b["cvr_score"]))
        total = ctr + cvr
    else:
        ctr = ctr_weight * (float(a["ctr_score"]) - float(b["ctr_score"]))
        cvr = cvr_weight * (float(a["cvr_score"]) - float(b["cvr_score"]))
        total = ctr + cvr
    return ctr, cvr, total


def compare_rankings(
    before_records: list[dict[str, Any]],
    after_records: list[dict[str, Any]],
    *,
    mode: str,
    ctr_weight: float,
    cvr_weight: float,
    comparison: dict[str, Any],
) -> list[dict[str, Any]]:
    before = {(row["request_id"], row["item_id"]): row for row in before_records}
    after = {(row["request_id"], row["item_id"]): row for row in after_records}
    common = sorted(set(before) & set(after))
    rank_before = fusion_rank_map(before_records, "fusion_score")
    rank_after = fusion_rank_map(after_records, "fusion_score")
    flips: list[dict[str, Any]] = []
    for left_index in range(len(common)):
        for right_index in range(left_index + 1, len(common)):
            key_a = common[left_index]
            key_b = common[right_index]
            if key_a[0] != key_b[0]:
                continue
            a_before = before[key_a]
            b_before = before[key_b]
            a_after = after[key_a]
            b_after = after[key_b]
            delta_before = float(a_before["fusion_score"]) - float(b_before["fusion_score"])
            delta_after = float(a_after["fusion_score"]) - float(b_after["fusion_score"])
            a_first_before = rank_before[key_a] < rank_before[key_b]
            a_first_after = rank_after[key_a] < rank_after[key_b]
            if a_first_before == a_first_after:
                continue
            cb_before, cc_before, total_before = _pair_contribution(
                a_before, b_before, mode, ctr_weight, cvr_weight
            )
            cb_after, cc_after, total_after = _pair_contribution(
                a_after, b_after, mode, ctr_weight, cvr_weight
            )
            ctr_delta = cb_after - cb_before
            cvr_delta = cc_after - cc_before
            expected_total_delta = total_after - total_before
            if mode == "product":
                actual_total_delta = (
                    _safe_log(float(a_after["fusion_score"]))
                    - _safe_log(float(b_after["fusion_score"]))
                    - _safe_log(float(a_before["fusion_score"]))
                    + _safe_log(float(b_before["fusion_score"]))
                )
            else:
                actual_total_delta = delta_after - delta_before
            fusion_quant = actual_total_delta - expected_total_delta
            abs_parts = {
                "CTR_TOWER": abs(ctr_delta),
                "CVR_TOWER": abs(cvr_delta),
                "FUSION_QUANTIZATION": abs(fusion_quant),
            }
            cause = max(("CTR_TOWER", "CVR_TOWER", "FUSION_QUANTIZATION"), key=lambda name: abs_parts[name])
            flips.append(
                {
                    "request_id": key_a[0],
                    "item_id": key_a[1],
                    "other_item_id": key_b[1],
                    "rank_before": rank_before[key_a],
                    "rank_after": rank_after[key_a],
                    "other_rank_before": rank_before[key_b],
                    "other_rank_after": rank_after[key_b],
                    "score_gap_before": delta_before,
                    "score_gap_after": delta_after,
                    "ctr_contribution_delta": ctr_delta,
                    "cvr_contribution_delta": cvr_delta,
                    "expected_fusion_gap_delta": expected_total_delta,
                    "fusion_quantization_component": fusion_quant,
                    "cause": cause,
                    "cause_detail": {
                        "dominant_absolute_component": abs_parts[cause],
                        "parts": abs_parts,
                    },
                    **comparison,
                }
            )
    return flips


def _auc(labels: list[int], scores: list[float]) -> float | None:
    positives = sum(1 for label in labels if label == 1)
    negatives = len(labels) - positives
    if positives == 0 or negatives == 0:
        return None
    wins = 0.0
    for i, (label_i, score_i) in enumerate(zip(labels, scores)):
        if label_i != 1:
            continue
        for label_j, score_j in zip(labels[i + 1 :], scores[i + 1 :]):
            if label_j == 0:
                if score_i > score_j:
                    wins += 1.0
                elif score_i == score_j:
                    wins += 0.5
        for label_j, score_j in zip(labels[:i], scores[:i]):
            if label_j == 0:
                if score_i > score_j:
                    wins += 1.0
                elif score_i == score_j:
                    wins += 0.5
    return wins / (positives * negatives)


def _ndcg(relevances: list[float], order: list[int], k: int) -> float:
    def dcg(indices: list[int]) -> float:
        return sum(
            (2 ** relevances[index] - 1) / math.log2(position + 2)
            for position, index in enumerate(indices[:k])
        )

    actual = dcg(order)
    ideal_indices = sorted(range(len(relevances)), key=lambda idx: (-relevances[idx], idx))
    ideal = dcg(ideal_indices)
    return 0.0 if ideal == 0 else actual / ideal


def evaluation_metrics(
    fusion_payload: dict[str, Any],
    labels: dict[tuple[str, str], dict[str, int]],
) -> dict[str, Any]:
    records = fusion_payload["records"]
    ctr_labels = [labels[(row["request_id"], row["item_id"])]["click"] for row in records]
    cvr_labels = [labels[(row["request_id"], row["item_id"])]["conversion"] for row in records]
    ctr_fixed = [float(row["ctr_score"]) for row in records]
    ctr_ref = [float(row["ctr_reference_score"]) for row in records]
    cvr_fixed = [float(row["cvr_score"]) for row in records]
    cvr_ref = [float(row["cvr_reference_score"]) for row in records]
    fixed_metrics = {
        "ctr_auc": _auc(ctr_labels, ctr_fixed),
        "cvr_auc": _auc(cvr_labels, cvr_fixed),
    }
    reference_metrics = {
        "ctr_auc": _auc(ctr_labels, ctr_ref),
        "cvr_auc": _auc(cvr_labels, cvr_ref),
    }
    ndcg_fixed: list[float] = []
    ndcg_ref: list[float] = []
    top1_matches = 0
    request_count = 0
    for request_id in sorted({row["request_id"] for row in records}):
        group = [row for row in records if row["request_id"] == request_id]
        order_fixed = sorted(group, key=lambda row: (row["fixed_rank"], row["item_id"]))
        order_ref = sorted(group, key=lambda row: (row["reference_rank"], row["item_id"]))
        relevances = [float(labels[(request_id, row["item_id"])]["click"]) for row in group]
        fixed_indices = [group.index(row) for row in order_fixed]
        ref_indices = [group.index(row) for row in order_ref]
        ndcg_fixed.append(_ndcg(relevances, fixed_indices, 5))
        ndcg_ref.append(_ndcg(relevances, ref_indices, 5))
        request_count += 1
        if order_fixed and order_ref and order_fixed[0]["item_id"] == order_ref[0]["item_id"]:
            top1_matches += 1
    fixed_metrics["fusion_ndcg_at_5"] = sum(ndcg_fixed) / request_count if request_count else None
    reference_metrics["fusion_ndcg_at_5"] = sum(ndcg_ref) / request_count if request_count else None

    def metric_error(name: str) -> float | None:
        left = fixed_metrics.get(name)
        right = reference_metrics.get(name)
        if left is None or right is None:
            return None
        return abs(float(left) - float(right))

    return {
        "fixed": fixed_metrics,
        "reference": reference_metrics,
        "absolute_error": {
            "ctr_auc": metric_error("ctr_auc"),
            "cvr_auc": metric_error("cvr_auc"),
            "fusion_ndcg_at_5": metric_error("fusion_ndcg_at_5"),
        },
        "top1_agreement": top1_matches / request_count if request_count else None,
    }


def fusion_reference_flips(fusion_payload: dict[str, Any], mode: str, ctr_weight: float, cvr_weight: float) -> list[dict[str, Any]]:
    fixed_rows = []
    reference_rows = []
    for row in fusion_payload["records"]:
        fixed_row = dict(row)
        fixed_row["fusion_score"] = row["fusion_score"]
        reference_row = dict(row)
        reference_row["fusion_score"] = row["fusion_reference_score"]
        fixed_rows.append(fixed_row)
        reference_rows.append(reference_row)
    return compare_rankings(
        reference_rows,
        fixed_rows,
        mode=mode,
        ctr_weight=ctr_weight,
        cvr_weight=cvr_weight,
        comparison={"comparison_kind": "float64_reference_vs_fixed"},
    )
