from __future__ import annotations

import math
from typing import Any, Iterable, Mapping

import numpy as np

from .histogram import Histogram, bucket_for_values
from .util import FIXED_SCALE, json_number, sha256_json


def _pair_id(left_index: int, right_index: int) -> str:
    return f"L{left_index}:R{right_index}"


def _boundary_risk(
    key_values: Iterable[int | float],
    *,
    left_bucket: Mapping[str, Any],
    right_bucket: Mapping[str, Any],
) -> tuple[bool, int]:
    boundary_hits = 0
    left_lower = left_bucket["lower"]
    left_upper = left_bucket["upper"]
    right_lower = right_bucket["lower"]
    right_upper = right_bucket["upper"]
    point_bucket = left_lower == left_upper or right_lower == right_upper
    for key in key_values:
        if key in {left_lower, left_upper, right_lower, right_upper}:
            boundary_hits += 1
    return point_bucket or boundary_hits > 0, boundary_hits


def attribute_differences(
    *,
    left_histogram: Histogram,
    right_histogram: Histogram,
    histogram_pairs: list[dict[str, Any]],
    sampling_pair_contributions: list[dict[str, Any]],
    truth_pair_rows: Mapping[str, int],
    sample_size: int,
    buckets: int,
    top_k: int,
    left_skewed: bool,
    right_skewed: bool,
) -> dict[str, Any]:
    pair_meta = {str(pair["pair_id"]): pair for pair in histogram_pairs}
    sample_fixed: dict[str, int] = {}
    sample_hits: dict[str, int] = {}
    sample_left_frequency: dict[str, int] = {}
    sample_right_frequency: dict[str, int] = {}
    sample_key_contrib: dict[str, list[tuple[int | float, int, int]]] = {}
    for contribution in sampling_pair_contributions:
        key = contribution["key"]
        key_array = np.asarray([key])
        left_index = int(bucket_for_values(left_histogram, key_array)[0])
        right_index = int(bucket_for_values(right_histogram, key_array)[0])
        if left_index < 0 or right_index < 0:
            continue
        pair_id = _pair_id(left_index, right_index)
        sample_fixed[pair_id] = sample_fixed.get(pair_id, 0) + int(contribution["contribution_fixed"])
        sample_hits[pair_id] = sample_hits.get(pair_id, 0) + int(contribution["sample_hit_pairs"])
        sample_left_frequency[pair_id] = sample_left_frequency.get(pair_id, 0) + int(
            contribution["left_sample_frequency"]
        )
        sample_right_frequency[pair_id] = sample_right_frequency.get(pair_id, 0) + int(
            contribution["right_sample_frequency"]
        )
        sample_key_contrib.setdefault(pair_id, []).append(
            (
                key,
                int(contribution["left_sample_frequency"]),
                int(contribution["right_sample_frequency"]),
            )
        )
    sparse_threshold = max(5, int(math.ceil(buckets / 2.0)))
    records: list[dict[str, Any]] = []
    all_pair_ids = set(pair_meta) | set(sample_fixed)
    for pair_id in all_pair_ids:
        meta = pair_meta.get(pair_id)
        if meta is None:
            left_index = int(pair_id.split(":")[0][1:])
            right_index = int(pair_id.split(":")[1][1:])
            left_bucket = left_histogram.buckets[left_index]
            right_bucket = right_histogram.buckets[right_index]
            meta = {
                "pair_id": pair_id,
                "left_bucket": left_index,
                "right_bucket": right_index,
                "left_lower": left_bucket.lower,
                "left_upper": left_bucket.upper,
                "right_lower": right_bucket.lower,
                "right_upper": right_bucket.upper,
                "left_interval": left_bucket.interval_label(),
                "right_interval": right_bucket.interval_label(),
                "left_rows": left_bucket.row_count,
                "right_rows": right_bucket.row_count,
                "overlap_width": 0.0,
                "histogram_fixed": 0,
            }
        left_index = int(meta["left_bucket"])
        right_index = int(meta["right_bucket"])
        left_bucket_payload = {
            "lower": left_histogram.buckets[left_index].lower,
            "upper": left_histogram.buckets[left_index].upper,
        }
        right_bucket_payload = {
            "lower": right_histogram.buckets[right_index].lower,
            "upper": right_histogram.buckets[right_index].upper,
        }
        histogram_fixed = int(meta.get("histogram_fixed", 0))
        sampled_fixed = int(sample_fixed.get(pair_id, 0))
        difference_fixed = histogram_fixed - sampled_fixed
        hits = int(sample_hits.get(pair_id, 0))
        pair_key_stats = sample_key_contrib.get(pair_id, [])
        key_values = [item[0] for item in pair_key_stats]
        boundary, boundary_hits = _boundary_risk(
            key_values,
            left_bucket=left_bucket_payload,
            right_bucket=right_bucket_payload,
        )
        top_key_share = 0.0
        if pair_key_stats:
            weights = [left * right for _, left, right in pair_key_stats]
            denominator = sum(weights)
            if denominator > 0:
                top_key_share = max(weights) / denominator
        left_sample_rows = int(sample_left_frequency.get(pair_id, 0))
        right_sample_rows = int(sample_right_frequency.get(pair_id, 0))
        if hits < sparse_threshold or left_sample_rows == 0 or right_sample_rows == 0:
            source = "sample_sparsity"
            basis = (
                f"样本命中对 {hits} 小于稀疏阈值 {sparse_threshold}，"
                f"左/右样本桶行数为 {left_sample_rows}/{right_sample_rows}"
            )
        elif boundary:
            source = "boundary_membership"
            basis = (
                "该桶对包含等分位重复边界或点桶"
                if boundary_hits == 0
                else f"样本中有 {boundary_hits} 个键恰好落在显式桶边界"
            )
        elif top_key_share >= 0.35 or ((left_skewed or right_skewed) and top_key_share >= 0.20):
            source = "skew_key_enrichment"
            basis = (
                f"样本命中对的首键占比 {top_key_share:.6f}，"
                f"左表倾斜={left_skewed}，右表倾斜={right_skewed}"
            )
        else:
            source = "sample_sparsity"
            basis = f"残差由抽样波动主导，首键占比 {top_key_share:.6f}"
        records.append(
            {
                "pair_id": pair_id,
                "left_bucket": left_index,
                "right_bucket": right_index,
                "left_lower": json_number(meta.get("left_lower")),
                "left_upper": json_number(meta.get("left_upper")),
                "right_lower": json_number(meta.get("right_lower")),
                "right_upper": json_number(meta.get("right_upper")),
                "left_interval": meta.get("left_interval"),
                "right_interval": meta.get("right_interval"),
                "left_rows": int(meta.get("left_rows", 0)),
                "right_rows": int(meta.get("right_rows", 0)),
                "overlap_width": float(meta.get("overlap_width", 0.0)),
                "histogram_estimate": histogram_fixed / FIXED_SCALE,
                "sampling_estimate": sampled_fixed / FIXED_SCALE,
                "estimate_difference": difference_fixed / FIXED_SCALE,
                "absolute_difference": abs(difference_fixed) / FIXED_SCALE,
                "histogram_fixed": histogram_fixed,
                "sampling_fixed": sampled_fixed,
                "difference_fixed": difference_fixed,
                "sample_hit_pairs": hits,
                "sample_left_bucket_rows": left_sample_rows,
                "sample_right_bucket_rows": right_sample_rows,
                "top_key_hit_share": top_key_share,
                "boundary_hits": boundary_hits,
                "truth_rows": int(truth_pair_rows.get(pair_id, 0)),
                "source": source,
                "basis": basis,
                "sparse_threshold": sparse_threshold,
            }
        )
    records.sort(
        key=lambda item: (
            -item["absolute_difference"],
            item["left_lower"],
            item["left_bucket"],
            item["right_lower"],
            item["right_bucket"],
        )
    )
    digest = sha256_json(
        [
            {
                "pair_id": record["pair_id"],
                "histogram_fixed": record["histogram_fixed"],
                "sampling_fixed": record["sampling_fixed"],
                "difference_fixed": record["difference_fixed"],
                "source": record["source"],
                "sample_hit_pairs": record["sample_hit_pairs"],
                "boundary_hits": record["boundary_hits"],
                "top_key_hit_share": record["top_key_hit_share"],
            }
            for record in records
        ]
    )
    return {
        "pair_count": len(records),
        "top_k": records[:top_k],
        "pairs": records,
        "digest": digest,
        "sparse_threshold": sparse_threshold,
        "classification_rule": (
            "稳定优先级：样本命中对低于稀疏阈值→sample_sparsity；"
            "点桶或样本键命中显式边界→boundary_membership；"
            "首样本键命中占比达阈值→skew_key_enrichment；否则归入残差抽样波动。"
        ),
        "sort_rule": "absolute_difference 降序，left_lower 升序，left_bucket/right_lower/right_bucket 稳定升序",
    }
