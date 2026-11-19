from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from .filtering import null_mask
from .histogram import Histogram, bucket_for_values
from .util import FIXED_SCALE, json_number, sha256_json, sort_key_value


def _clean_keys(values: np.ndarray) -> np.ndarray:
    return values[~null_mask(values)]


def execute_equi_join(
    left_arrays: Mapping[str, np.ndarray],
    right_arrays: Mapping[str, np.ndarray],
    *,
    left_key: str,
    right_key: str,
    left_histogram: Histogram,
    right_histogram: Histogram,
    left_predicate_digest: str,
    right_predicate_digest: str,
) -> dict[str, Any]:
    left_keys = _clean_keys(left_arrays[left_key])
    right_keys = _clean_keys(right_arrays[right_key])
    left_unique, left_counts = np.unique(left_keys, return_counts=True)
    right_unique, right_counts = np.unique(right_keys, return_counts=True)
    common, left_indices, right_indices = np.intersect1d(
        left_unique,
        right_unique,
        assume_unique=True,
        return_indices=True,
    )
    common_left_counts = left_counts[left_indices]
    common_right_counts = right_counts[right_indices]
    left_buckets = bucket_for_values(left_histogram, common)
    right_buckets = bucket_for_values(right_histogram, common)
    pair_rows: dict[str, int] = {}
    left_bucket_rows: dict[int, int] = {bucket.index: 0 for bucket in left_histogram.buckets}
    right_bucket_rows: dict[int, int] = {bucket.index: 0 for bucket in right_histogram.buckets}
    key_distribution: list[dict[str, Any]] = []
    total_rows = 0
    for index, key in enumerate(common.tolist()):
        contribution = int(common_left_counts[index]) * int(common_right_counts[index])
        left_bucket = int(left_buckets[index])
        right_bucket = int(right_buckets[index])
        pair_id = f"L{left_bucket}:R{right_bucket}"
        pair_rows[pair_id] = pair_rows.get(pair_id, 0) + contribution
        if left_bucket >= 0:
            left_bucket_rows[left_bucket] = left_bucket_rows.get(left_bucket, 0) + contribution
        if right_bucket >= 0:
            right_bucket_rows[right_bucket] = right_bucket_rows.get(right_bucket, 0) + contribution
        total_rows += contribution
        key_distribution.append(
            {
                "key": json_number(key),
                "left_frequency": int(common_left_counts[index]),
                "right_frequency": int(common_right_counts[index]),
                "join_rows": contribution,
            }
        )
    key_distribution.sort(key=lambda item: sort_key_value(item["key"]))
    pair_rows_sorted = {
        pair_id: pair_rows[pair_id]
        for pair_id in sorted(
            pair_rows,
            key=lambda value: (
                int(value.split(":")[0][1:]),
                int(value.split(":")[1][1:]),
            ),
        )
    }
    top_by_rows = sorted(
        key_distribution,
        key=lambda item: (-int(item["join_rows"]), sort_key_value(item["key"])),
    )[:50]
    distribution_digest = sha256_json(key_distribution)
    payload = {
        "row_count": int(total_rows),
        "common_key_count": int(len(common)),
        "left_filtered_rows": int(len(left_keys)),
        "right_filtered_rows": int(len(right_keys)),
        "max_possible": float(len(left_keys) * len(right_keys)),
        "left_key_column": left_key,
        "right_key_column": right_key,
        "left_predicate_digest": left_predicate_digest,
        "right_predicate_digest": right_predicate_digest,
        "left_bucket_rows": [left_bucket_rows.get(bucket.index, 0) for bucket in left_histogram.buckets],
        "right_bucket_rows": [right_bucket_rows.get(bucket.index, 0) for bucket in right_histogram.buckets],
        "pair_rows": pair_rows_sorted,
        "pair_rows_fixed": {
            pair_id: int(rows) * FIXED_SCALE for pair_id, rows in pair_rows_sorted.items()
        },
        "key_distribution": key_distribution,
        "key_distribution_digest": distribution_digest,
        "key_distribution_top": top_by_rows,
    }
    payload["digest"] = sha256_json(
        {
            "row_count": payload["row_count"],
            "common_key_count": payload["common_key_count"],
            "pair_rows": payload["pair_rows"],
            "left_bucket_rows": payload["left_bucket_rows"],
            "right_bucket_rows": payload["right_bucket_rows"],
            "key_distribution_digest": distribution_digest,
        }
    )
    return payload


def truth_summary(truth: Mapping[str, Any], *, top_keys: int = 20) -> dict[str, Any]:
    full_distribution = truth.get("key_distribution", [])
    include_full = len(full_distribution) <= 5000
    return {
        "row_count": truth["row_count"],
        "common_key_count": truth["common_key_count"],
        "left_filtered_rows": truth["left_filtered_rows"],
        "right_filtered_rows": truth["right_filtered_rows"],
        "max_possible": truth["max_possible"],
        "digest": truth["digest"],
        "key_distribution_digest": truth["key_distribution_digest"],
        "key_distribution_top": truth.get("key_distribution_top", [])[:top_keys],
        "key_distribution": full_distribution if include_full else [],
        "key_distribution_complete": include_full,
        "key_distribution_truncated_at": None if include_full else 5000,
        "left_bucket_rows": truth["left_bucket_rows"],
        "right_bucket_rows": truth["right_bucket_rows"],
        "pair_rows": truth["pair_rows"],
    }
