from __future__ import annotations

import math
from typing import Any, Iterable, Mapping

from .histogram import Histogram, overlap_pairs, overlap_row_ratios
from .util import FIXED_SCALE, sha256_json


def histogram_join_estimate(
    left: Histogram,
    right: Histogram,
    *,
    left_filtered_rows: int,
    right_filtered_rows: int,
) -> dict[str, Any]:
    pairs = overlap_pairs(left, right)
    max_possible = float(left_filtered_rows * right_filtered_rows)
    max_fixed = int(round(max_possible * FIXED_SCALE))
    raw_fixed = sum(int(pair["histogram_fixed"]) for pair in pairs)
    if raw_fixed > max_fixed and raw_fixed > 0:
        scaled: list[int] = []
        running = 0
        for pair in pairs:
            value = int(pair["histogram_fixed"])
            scaled_value = (value * max_fixed) // raw_fixed
            scaled.append(scaled_value)
            running += scaled_value
        remainder = max_fixed - running
        for index in range(len(scaled)):
            if remainder <= 0:
                break
            scaled[index] += 1
            remainder -= 1
        for pair, value in zip(pairs, scaled):
            pair["histogram_fixed"] = value
            pair["histogram_estimate"] = value / FIXED_SCALE
    fixed_total = sum(int(pair["histogram_fixed"]) for pair in pairs)
    fixed_total = max(0, min(fixed_total, max_fixed))
    estimate = fixed_total / FIXED_SCALE
    ratios = overlap_row_ratios(left, right, pairs)
    payload = {
        "estimate": estimate,
        "fixed_scale": FIXED_SCALE,
        "estimate_fixed": fixed_total,
        "quantization_error_upper": len(pairs) * (0.5 / FIXED_SCALE),
        "overlap_pair_count": len(pairs),
        "overlap_row_ratio": ratios,
        "overlap_row_ratio_combined": (ratios["left"] + ratios["right"]) / 2.0,
        "max_possible": max_possible,
        "pairs": pairs,
        "path_digest": sha256_json(
            {
                "estimate_fixed": fixed_total,
                "overlap_pair_count": len(pairs),
                "pairs": [
                    {
                        "pair_id": pair["pair_id"],
                        "histogram_fixed": pair["histogram_fixed"],
                    }
                    for pair in pairs
                ],
            }
        ),
    }
    return payload


def sampling_path_summary(estimate: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in estimate.items()
        if key != "pair_contributions"
    }


def derive_tolerance_interval(
    *,
    truth_rows: int,
    left_filtered_rows: int,
    right_filtered_rows: int,
    buckets: int,
    sample_size: int,
) -> dict[str, Any]:
    effective_sample = max(1, min(sample_size, left_filtered_rows, right_filtered_rows))
    sampling_term = 1.96 / math.sqrt(effective_sample)
    bucket_term = 1.0 / (2.0 * buckets)
    base_term = 0.02
    relative_tolerance = min(0.80, base_term + sampling_term + bucket_term)
    if truth_rows > 0:
        lower = truth_rows * (1.0 - relative_tolerance)
        upper = truth_rows * (1.0 + relative_tolerance)
        lower_int = max(0, int(math.floor(lower)))
        upper_int = max(lower_int, int(math.ceil(upper)))
        absolute_floor = 0.0
    else:
        absolute_floor = max(
            1.0,
            0.05 * math.sqrt(max(1, left_filtered_rows) * max(1, right_filtered_rows)),
        )
        lower_int = 0
        upper_int = max(1, int(math.ceil(absolute_floor)))
    return {
        "lower": float(lower_int),
        "upper": float(upper_int),
        "lower_inclusive": True,
        "upper_inclusive": True,
        "relative_tolerance": relative_tolerance,
        "sampling_term": sampling_term,
        "bucket_term": bucket_term,
        "base_term": base_term,
        "effective_sample": effective_sample,
        "truth_center": truth_rows,
        "formula": (
            "relative_tolerance = min(0.80, 0.02 + 1.96/sqrt(min(sample_size, n_left, n_right)) "
            "+ 1/(2*buckets)); interval = [floor(truth*(1-tol)), ceil(truth*(1+tol))]"
        ),
        "zero_truth_absolute_floor": absolute_floor,
    }


def _sample_quality(
    *,
    buckets: int,
    sample_size: int,
    left_rows: int,
    right_rows: int,
    sample_hit_pairs: int,
    observed_common_keys: int,
    hit_pair_top_share: float,
) -> dict[str, Any]:
    effective_sample = max(1, min(sample_size, left_rows, right_rows))
    histogram_model_term = bucket_term = 1.0 / (2.0 * buckets)
    sampling_stat_term = 1.96 / math.sqrt(effective_sample) + 1.0 / math.sqrt(sample_hit_pairs + 1.0)
    histogram_score = bucket_term
    sampling_score = sampling_stat_term
    if sample_hit_pairs < max(8, buckets) or observed_common_keys < 2:
        selected = "histogram"
        reason = "样本命中键对过少，抽样路径方差主导"
    elif sample_size < 2 * buckets:
        selected = "histogram"
        reason = "抽样规模不足分桶数的两倍，直方图路径更稳定"
    elif hit_pair_top_share >= 0.50:
        selected = "histogram"
        reason = "样本命中对由富集键主导，抽样重标定方差偏大"
    elif sampling_score <= histogram_score:
        selected = "sampling"
        reason = "抽样命中键对充足且统计误差项不超过桶模型项"
    else:
        selected = "histogram"
        reason = "桶模型误差项低于抽样统计误差项"
    return {
        "selected_path": selected,
        "reason": reason,
        "histogram_score": histogram_score,
        "sampling_score": sampling_score,
        "sampling_stat_term": sampling_stat_term,
        "histogram_bucket_term": histogram_model_term,
        "effective_sample": effective_sample,
        "sample_hit_pairs": sample_hit_pairs,
        "observed_common_keys": observed_common_keys,
        "hit_pair_top_share": hit_pair_top_share,
        "rule": (
            "先检查命中稀疏、样本规模与富集键主导；否则比较 "
            "1/(2*buckets) 与 1.96/sqrt(n_eff)+1/sqrt(hit_pairs+1)。"
        ),
    }


def decide_estimates(
    *,
    histogram_estimate: Mapping[str, Any],
    sampling_estimate: Mapping[str, Any],
    truth: Mapping[str, Any],
    buckets: int,
    sample_size: int,
) -> dict[str, Any]:
    interval = derive_tolerance_interval(
        truth_rows=int(truth["row_count"]),
        left_filtered_rows=int(truth["left_filtered_rows"]),
        right_filtered_rows=int(truth["right_filtered_rows"]),
        buckets=buckets,
        sample_size=sample_size,
    )
    histogram_value = float(histogram_estimate["estimate_from_fixed"] if "estimate_from_fixed" in histogram_estimate else histogram_estimate["estimate"])
    sampling_value = float(sampling_estimate["estimate_from_fixed"])
    truth_rows = int(truth["row_count"])
    denominator = max(1.0, float(truth_rows))

    def evaluate(value: float) -> dict[str, Any]:
        within = interval["lower"] <= value <= interval["upper"]
        if value < interval["lower"]:
            overshoot = interval["lower"] - value
        elif value > interval["upper"]:
            overshoot = value - interval["upper"]
        else:
            overshoot = 0.0
        return {
            "estimate": value,
            "within_interval": within,
            "absolute_error": abs(value - truth_rows),
            "relative_error": abs(value - truth_rows) / denominator,
            "out_of_bounds_ratio": overshoot / denominator,
        }

    histogram_decision = evaluate(histogram_value)
    sampling_decision = evaluate(sampling_value)
    pair_contributions = sampling_estimate.get("pair_contributions", [])
    top_share = 0.0
    if pair_contributions:
        hit_total = sum(int(item["sample_hit_pairs"]) for item in pair_contributions)
        top_hit = max(int(item["sample_hit_pairs"]) for item in pair_contributions)
        top_share = (top_hit / hit_total) if hit_total else 0.0
    quality = _sample_quality(
        buckets=buckets,
        sample_size=sample_size,
        left_rows=int(truth["left_filtered_rows"]),
        right_rows=int(truth["right_filtered_rows"]),
        sample_hit_pairs=int(sampling_estimate["sample_hit_pairs"]),
        observed_common_keys=int(sampling_estimate["observed_common_keys"]),
        hit_pair_top_share=top_share,
    )
    if histogram_decision["within_interval"] and sampling_decision["within_interval"]:
        selected = quality["selected_path"]
    elif histogram_decision["within_interval"]:
        selected = "histogram"
    elif sampling_decision["within_interval"]:
        selected = "sampling"
    else:
        selected = (
            "histogram"
            if histogram_decision["relative_error"] <= sampling_decision["relative_error"]
            else "sampling"
        )
    outcome = "OK" if histogram_decision["within_interval"] and sampling_decision["within_interval"] else "E_EST"
    decision = {
        "outcome": outcome,
        "within_interval": outcome == "OK",
        "histogram": histogram_decision,
        "sampling": sampling_decision,
        "interval": interval,
        "selected_path": selected,
        "path_selection": quality,
        "estimates": {"histogram": histogram_value, "sampling": sampling_value, "truth": truth_rows},
    }
    decision["digest"] = sha256_json(
        {
            "outcome": outcome,
            "histogram": histogram_decision,
            "sampling": sampling_decision,
            "interval": interval,
            "selected_path": selected,
            "path_selection": quality,
        }
    )
    return decision


def dominant_deviation_pairs(
    estimator_pairs: Mapping[str, int],
    truth_pair_rows: Mapping[str, int],
    pair_meta: Iterable[Mapping[str, Any]],
    *,
    top_k: int = 5,
) -> list[dict[str, Any]]:
    meta_by_id = {str(item["pair_id"]): item for item in pair_meta}
    all_ids = set(estimator_pairs) | set(truth_pair_rows)
    records: list[dict[str, Any]] = []
    for pair_id in all_ids:
        meta = meta_by_id.get(pair_id, {})
        estimate_fixed = int(estimator_pairs.get(pair_id, 0))
        truth_fixed = int(truth_pair_rows.get(pair_id, 0)) * FIXED_SCALE
        difference = estimate_fixed - truth_fixed
        records.append(
            {
                "pair_id": pair_id,
                "left_bucket": int(meta.get("left_bucket", int(pair_id.split(":")[0][1:]))),
                "right_bucket": int(meta.get("right_bucket", int(pair_id.split(":")[1][1:]))),
                "left_lower": meta.get("left_lower"),
                "right_lower": meta.get("right_lower"),
                "estimated_rows": estimate_fixed / FIXED_SCALE,
                "truth_rows": truth_fixed / FIXED_SCALE,
                "difference_rows": difference / FIXED_SCALE,
                "absolute_difference_rows": abs(difference) / FIXED_SCALE,
            }
        )
    records.sort(
        key=lambda item: (
            -item["absolute_difference_rows"],
            item["left_lower"] if item["left_lower"] is not None else float("-inf"),
            item["left_bucket"],
            item["right_lower"] if item["right_lower"] is not None else float("-inf"),
            item["right_bucket"],
        )
    )
    return records[:top_k]
