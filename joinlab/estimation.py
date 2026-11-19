from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from typing import Any

from .canonical import fixed_text
from .histogram import Bucket, Histogram
from .sampling import SampleResult


FIXED_SCALE = 1_000_000
TRUNCATION = "floor_toward_zero"


def _fraction(value: int | float) -> Fraction:
    if isinstance(value, int):
        return Fraction(value, 1)
    return Fraction(str(value))


def _bucket_geometry(bucket: Bucket, key_type: str) -> tuple[Fraction, Fraction, Fraction]:
    lower = _fraction(bucket.lower_bound)
    upper = _fraction(bucket.upper_bound)
    width = upper - lower
    if width <= 0:
        width = Fraction(1, 1) if bucket.row_count > 0 else Fraction(0, 1)
    return lower, upper, width


def _covers(bucket: Bucket, point: Fraction) -> bool:
    lower = _fraction(bucket.lower_bound)
    upper = _fraction(bucket.upper_bound)
    lower_ok = point > lower or (point == lower and bucket.lower_inclusive)
    upper_ok = point < upper or (point == upper and bucket.upper_inclusive)
    return lower_ok and upper_ok


def _overlap_geometry(
    left: Bucket,
    right: Bucket,
    left_type: str,
    right_type: str,
) -> tuple[Fraction, Fraction, Fraction, Fraction] | None:
    left_lower, left_upper, left_width = _bucket_geometry(left, left_type)
    right_lower, right_upper, right_width = _bucket_geometry(right, right_type)
    lower = max(left_lower, right_lower)
    upper = min(left_upper, right_upper)
    if upper < lower:
        return None
    measure = upper - lower
    if measure == 0:
        measure = Fraction(1, 1) if _covers(left, lower) and _covers(right, lower) else Fraction(0, 1)
    if measure == 0 or left_width == 0 or right_width == 0:
        return None
    left_ratio = min(Fraction(1, 1), measure / left_width)
    right_ratio = min(Fraction(1, 1), measure / right_width)
    return lower, upper, left_ratio, right_ratio


def _floor_fraction(value: Fraction, scale: int = FIXED_SCALE) -> int:
    return (value.numerator * scale) // value.denominator


def _fraction_text(value: Fraction) -> str:
    return fixed_text(_floor_fraction(value), FIXED_SCALE)


def estimate_join(
    left_hist: Histogram,
    right_hist: Histogram,
    left_sample: SampleResult,
    right_sample: SampleResult,
    left_rows: int,
    right_rows: int,
    attribution_k: int,
) -> dict[str, Any]:
    hist_pair_values: dict[tuple[int, int], dict[str, Any]] = {}
    total_hist = Fraction(0, 1)
    overlap_left_rows = Fraction(0, 1)
    overlap_right_rows = Fraction(0, 1)
    left_bucket_map = {bucket.index: bucket for bucket in left_hist.buckets}
    right_bucket_map = {bucket.index: bucket for bucket in right_hist.buckets}
    for left_bucket in sorted(left_hist.buckets, key=lambda bucket: bucket.index):
        for right_bucket in sorted(right_hist.buckets, key=lambda bucket: bucket.index):
            geometry = _overlap_geometry(
                left_bucket,
                right_bucket,
                left_hist.key_type,
                right_hist.key_type,
            )
            if geometry is None:
                continue
            lower, upper, left_ratio, right_ratio = geometry
            left_candidate_rows = Fraction(left_bucket.row_count, 1) * left_ratio
            right_candidate_rows = Fraction(right_bucket.row_count, 1) * right_ratio
            if left_hist.key_type == "integer" and right_hist.key_type == "integer":
                domain_size = max(1, int(upper - lower))
            else:
                domain_size = max(1, left_bucket.unique_count, right_bucket.unique_count)
            contribution = left_candidate_rows * right_candidate_rows / Fraction(domain_size, 1)
            total_hist += contribution
            overlap_left_rows += left_candidate_rows
            overlap_right_rows += right_candidate_rows
            hist_pair_values[(left_bucket.index, right_bucket.index)] = {
                "left_bucket": left_bucket,
                "right_bucket": right_bucket,
                "lower": lower,
                "upper": upper,
                "overlap_fraction_product": left_ratio * right_ratio,
                "hist_contribution": contribution,
            }

    sample_pair_hits: dict[tuple[int, int], int] = {}
    sample_hits = 0
    matched_sample_keys = 0
    matched_sample_rows_left = 0
    matched_sample_rows_right = 0
    for key, left_frequency in left_sample.frequencies.items():
        right_frequency = right_sample.frequencies.get(key)
        if not right_frequency:
            continue
        matched_sample_keys += 1
        matched_sample_rows_left += left_frequency
        matched_sample_rows_right += right_frequency
        hits = left_frequency * right_frequency
        sample_hits += hits
        pair = (left_hist.bucket_index(key), right_hist.bucket_index(key))
        sample_pair_hits[pair] = sample_pair_hits.get(pair, 0) + hits

    sample_total_numerator = (
        sample_hits * left_rows * right_rows * FIXED_SCALE
    )
    sample_total_denominator = left_sample.sampled_row_count * right_sample.sampled_row_count
    sample_fixed = sample_total_numerator // sample_total_denominator if sample_total_denominator else 0

    all_pairs = sorted(set(hist_pair_values) | set(sample_pair_hits))
    pair_details: list[dict[str, Any]] = []
    for pair in all_pairs:
        left_index, right_index = pair
        left_bucket = left_bucket_map[left_index]
        right_bucket = right_bucket_map[right_index]
        if pair in hist_pair_values:
            hist_value = hist_pair_values[pair]
            hist_fixed = _floor_fraction(hist_value["hist_contribution"])
            lower: Fraction = hist_value["lower"]
            upper: Fraction = hist_value["upper"]
            overlap_product: Fraction = hist_value["overlap_fraction_product"]
        else:
            hist_fixed = 0
            lower = max(_fraction(left_bucket.lower_bound), _fraction(right_bucket.lower_bound))
            upper = min(_fraction(left_bucket.upper_bound), _fraction(right_bucket.upper_bound))
            overlap_product = Fraction(0, 1)
        hits = sample_pair_hits.get(pair, 0)
        sample_pair_fixed = (
            hits * left_rows * right_rows * FIXED_SCALE
            // (left_sample.sampled_row_count * right_sample.sampled_row_count)
            if left_sample.sampled_row_count and right_sample.sampled_row_count
            else 0
        )
        difference_fixed = hist_fixed - sample_pair_fixed
        left_sample_rows = sum(
            count for key, count in left_sample.frequencies.items()
            if left_hist.bucket_index(key) == left_index
        )
        right_sample_rows = sum(
            count for key, count in right_sample.frequencies.items()
            if right_hist.bucket_index(key) == right_index
        )
        source_code, source = _classify_difference(
            left_bucket,
            right_bucket,
            left_sample_rows,
            right_sample_rows,
            hits,
        )
        pair_details.append(
            {
                "left_bucket_index": left_index,
                "right_bucket_index": right_index,
                "key_interval": {
                    "lower": _fraction_to_number(lower),
                    "upper": _fraction_to_number(upper),
                },
                "left_bucket": left_bucket.as_dict(),
                "right_bucket": right_bucket.as_dict(),
                "histogram_contribution_fixed": hist_fixed,
                "sample_contribution_fixed": sample_pair_fixed,
                "difference_fixed": difference_fixed,
                "sample_hit_pairs": hits,
                "sample_key_rows_left": left_sample_rows,
                "sample_key_rows_right": right_sample_rows,
                "overlap_fraction_product": _fraction_text(overlap_product),
                "difference_source_code": source_code,
                "difference_source": source,
            }
        )

    pair_details.sort(
        key=lambda item: (
            -abs(item["difference_fixed"]),
            Fraction(str(item["key_interval"]["lower"])),
            item["left_bucket_index"],
            item["right_bucket_index"],
        )
    )
    differing_pairs = [item for item in pair_details if item["difference_fixed"] != 0]
    attribution = differing_pairs[:attribution_k]
    first_difference_interval = None
    if differing_pairs:
        first_difference_interval = min(
            differing_pairs,
            key=lambda item: (
                Fraction(str(item["key_interval"]["lower"])),
                item["left_bucket_index"],
                item["right_bucket_index"],
            ),
        )["key_interval"]

    hist_fixed = _floor_fraction(total_hist)
    combined_factor = Fraction(
        left_rows * right_rows * FIXED_SCALE,
        left_sample.sampled_row_count * right_sample.sampled_row_count,
    ) if left_sample.sampled_row_count and right_sample.sampled_row_count else Fraction(0, 1)
    left_factor = Fraction(left_rows * FIXED_SCALE, left_sample.sampled_row_count) if left_sample.sampled_row_count else Fraction(0, 1)
    right_factor = Fraction(right_rows * FIXED_SCALE, right_sample.sampled_row_count) if right_sample.sampled_row_count else Fraction(0, 1)
    ratio_denominator = max(1, left_hist.non_null_rows + right_hist.non_null_rows)
    overlap_ratio = (overlap_left_rows + overlap_right_rows) / Fraction(ratio_denominator, 1)

    return {
        "fixed_point": {
            "scale": FIXED_SCALE,
            "truncation": TRUNCATION,
            "note": "所有 fixed 值为行数乘以 scale 后向下截断；逐桶差异同样截断。",
        },
        "histogram_path": {
            "method": "分位桶区间重叠 + 两侧候选行数比例；整数域按重叠区间点数归一，浮点域按桶内唯一键数归一",
            "estimate_fixed": hist_fixed,
            "estimate_rows": fixed_text(hist_fixed, FIXED_SCALE),
            "overlap_bucket_pairs": len(hist_pair_values),
            "overlap_left_row_ratio": _fraction_text(min(Fraction(1,1), overlap_left_rows / max(1, left_hist.non_null_rows))),
            "overlap_right_row_ratio": _fraction_text(min(Fraction(1,1), overlap_right_rows / max(1, right_hist.non_null_rows))),
            "overlap_row_ratio": _fraction_text(min(Fraction(1,1), overlap_ratio)),
            "first_difference_key_interval": first_difference_interval,
        },
        "sample_path": {
            "method": "两侧样本键频次乘积按全表/样本重标定因子放大",
            "estimate_fixed": sample_fixed,
            "estimate_rows": fixed_text(sample_fixed, FIXED_SCALE),
            "sample_hit_pairs": sample_hits,
            "sample_matched_keys": matched_sample_keys,
            "sample_matched_rows_left": matched_sample_rows_left,
            "sample_matched_rows_right": matched_sample_rows_right,
            "sample_hit_rate_left": _fraction_text(Fraction(matched_sample_rows_left, max(1, left_sample.sampled_row_count))),
            "sample_hit_rate_right": _fraction_text(Fraction(matched_sample_rows_right, max(1, right_sample.sampled_row_count))),
            "left_rescale_factor_fixed": _floor_fraction(left_factor),
            "right_rescale_factor_fixed": _floor_fraction(right_factor),
            "combined_rescale_factor_fixed": _floor_fraction(combined_factor),
            "left_rescale_factor": _fraction_text(left_factor),
            "right_rescale_factor": _fraction_text(right_factor),
            "combined_rescale_factor": _fraction_text(combined_factor),
        },
        "difference_fixed": hist_fixed - sample_fixed,
        "difference_rows": fixed_text(hist_fixed - sample_fixed, FIXED_SCALE),
        "attribution": attribution,
        "attribution_candidates": len([item for item in pair_details if item["difference_fixed"] != 0]),
    }


def _fraction_to_number(value: Fraction) -> int | float:
    if value.denominator == 1:
        return int(value.numerator)
    return float(value)


def _classify_difference(
    left_bucket: Bucket,
    right_bucket: Bucket,
    left_sample_rows: int,
    right_sample_rows: int,
    sample_hit_pairs: int,
) -> tuple[str, str]:
    left_concentration = (
        left_bucket.top_count / left_bucket.row_count if left_bucket.row_count else 0.0
    )
    right_concentration = (
        right_bucket.top_count / right_bucket.row_count if right_bucket.row_count else 0.0
    )
    if max(left_concentration, right_concentration) >= 0.45:
        return "skew_enrichment", "倾斜键富集"
    if min(left_sample_rows, right_sample_rows) < 5 or sample_hit_pairs < 2:
        return "sample_sparsity", "样本稀疏"
    return "boundary_ownership", "边界归属"
