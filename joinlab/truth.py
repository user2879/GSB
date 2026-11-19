from __future__ import annotations

import hashlib
import math
from typing import Any, Iterable

try:
    import numpy as np
except ModuleNotFoundError as exc:  # pragma: no cover - exercised by broken installs
    raise RuntimeError("NumPy 未安装；请执行 python -m pip install numpy") from exc

from .canonical import canonical_json
from .errors import JoinLabError


SEQUENCE_RULE = "key ASC, left.__rid ASC, right.__rid ASC; null keys excluded"


def _key_array(rows: list[dict[str, Any]], key: str, use_float: bool) -> np.ndarray:
    values = [row[key] for row in rows if row.get(key) is not None]
    dtype = np.float64 if use_float else np.int64
    return np.asarray(values, dtype=dtype)


def _sequence_digest(rows: Iterable[tuple[int | float, str, str]]) -> str:
    digest = hashlib.sha256()
    count = 0
    for key, left_rid, right_rid in rows:
        digest.update(canonical_json([key, left_rid, right_rid]).encode("utf-8"))
        digest.update(b"\n")
        count += 1
    return digest.hexdigest()


def execute_truth_join(
    left_rows: list[dict[str, Any]],
    right_rows: list[dict[str, Any]],
    left_key: str,
    right_key: str,
    left_key_type: str,
    right_key_type: str,
    preview_limit: int = 20,
) -> dict[str, Any]:
    use_float = left_key_type == "float" or right_key_type == "float"
    left_rows_nonnull = [row for row in left_rows if row.get(left_key) is not None]
    right_rows_nonnull = [row for row in right_rows if row.get(right_key) is not None]
    left_keys = _key_array(left_rows_nonnull, left_key, use_float)
    right_keys = _key_array(right_rows_nonnull, right_key, use_float)
    left_rids = [row["__rid"] for row in left_rows_nonnull]
    right_rids = [row["__rid"] for row in right_rows_nonnull]

    if left_keys.size == 0 or right_keys.size == 0:
        common = np.asarray([], dtype=left_keys.dtype)
    else:
        common = np.intersect1d(left_keys, right_keys, assume_unique=False)

    key_chunks: list[np.ndarray] = []
    left_chunks: list[np.ndarray] = []
    right_chunks: list[np.ndarray] = []
    for key in common:
        left_positions = np.flatnonzero(left_keys == key)
        right_positions = np.flatnonzero(right_keys == key)
        if left_positions.size == 0 or right_positions.size == 0:
            continue
        pair_count = left_positions.size * right_positions.size
        key_chunks.append(np.full(pair_count, key, dtype=left_keys.dtype))
        left_chunks.append(
            np.repeat(np.asarray(left_rids, dtype=object)[left_positions], right_positions.size)
        )
        right_chunks.append(
            np.tile(np.asarray(right_rids, dtype=object)[right_positions], left_positions.size)
        )

    if key_chunks:
        pair_keys = np.concatenate(key_chunks)
        pair_left = np.concatenate(left_chunks)
        pair_right = np.concatenate(right_chunks)
        order = np.lexsort((pair_right, pair_left, pair_keys))
        pair_keys = pair_keys[order]
        pair_left = pair_left[order]
        pair_right = pair_right[order]
        grouped_keys, grouped_counts = np.unique(pair_keys, return_counts=True)
    else:
        pair_keys = np.asarray([], dtype=left_keys.dtype if left_keys.size else np.float64)
        pair_left = np.asarray([], dtype=object)
        pair_right = np.asarray([], dtype=object)
        grouped_keys = np.asarray([], dtype=pair_keys.dtype)
        grouped_counts = np.asarray([], dtype=np.int64)

    true_count = int(pair_keys.size)
    pair_rows = [
        (
            float(pair_keys[index]) if use_float else int(pair_keys[index]),
            str(pair_left[index]),
            str(pair_right[index]),
        )
        for index in range(true_count)
    ]
    grouped_distribution = [
        {
            "key": float(grouped_keys[index]) if use_float else int(grouped_keys[index]),
            "count": int(grouped_counts[index]),
        }
        for index in range(len(grouped_keys))
    ]
    group_digest = hashlib.sha256(
        canonical_json(grouped_distribution).encode("utf-8")
    ).hexdigest()
    return {
        "true_row_count": true_count,
        "left_non_null_rows": len(left_rows_nonnull),
        "right_non_null_rows": len(right_rows_nonnull),
        "grouped_row_count": len(grouped_distribution),
        "grouped_distribution": grouped_distribution,
        "grouped_distribution_digest": group_digest,
        "output_sequence_rule": SEQUENCE_RULE,
        "output_sequence_digest": _sequence_digest(pair_rows),
        "output_preview": [
            {"key": key, "left_rid": left_rid, "right_rid": right_rid}
            for key, left_rid, right_rid in pair_rows[:preview_limit]
        ],
        "engine": f"NumPy {np.__version__}",
    }


def _tolerance_bounds(true_rows: int, basis_points: int, absolute_slack: int) -> tuple[int, int]:
    lower = max(0, (true_rows * (10000 - basis_points)) // 10000 - absolute_slack)
    upper = math.ceil(true_rows * (10000 + basis_points) / 10000) + absolute_slack
    return lower, upper


def validate_estimates(
    truth: dict[str, Any],
    histogram_estimate: dict[str, Any],
    sample_estimate: dict[str, Any],
    bucket_count: int,
    sample_size: int,
    attribution: list[dict[str, Any]],
    fixed_scale: int,
) -> dict[str, Any]:
    true_rows = int(truth["true_row_count"])
    hist_tolerance_bp = min(5000, max(1000, 20000 // max(1, bucket_count)))
    sample_tolerance_bp = min(
        5000,
        max(1000, 20000 // max(1, math.isqrt(sample_size))),
    )
    absolute_slack = max(1, bucket_count // 2)
    hist_lower, hist_upper = _tolerance_bounds(true_rows, hist_tolerance_bp, absolute_slack)
    sample_lower, sample_upper = _tolerance_bounds(true_rows, sample_tolerance_bp, absolute_slack)
    hist_fixed = int(histogram_estimate["estimate_fixed"])
    sample_fixed = int(sample_estimate["estimate_fixed"])
    hist_in_range = hist_lower * fixed_scale <= hist_fixed <= hist_upper * fixed_scale
    sample_in_range = sample_lower * fixed_scale <= sample_fixed <= sample_upper * fixed_scale

    failing_interval = None
    if not (hist_in_range and sample_in_range):
        failing_interval = histogram_estimate.get("first_difference_key_interval")
        candidates = sorted(
            attribution,
            key=lambda item: (
                item["key_interval"]["lower"],
                item["left_bucket_index"],
                item["right_bucket_index"],
            ),
        )
        if failing_interval is None and candidates:
            failing_interval = candidates[0]["key_interval"]

    result = {
        "status": "PASS" if hist_in_range and sample_in_range else "FAIL",
        "true_row_count": true_rows,
        "histogram": {
            "estimate_fixed": hist_fixed,
            "estimate_rows": histogram_estimate["estimate_rows"],
            "tolerance_basis_points": hist_tolerance_bp,
            "absolute_slack_rows": absolute_slack,
            "lower_rows": hist_lower,
            "upper_rows": hist_upper,
            "in_range": hist_in_range,
        },
        "sample": {
            "estimate_fixed": sample_fixed,
            "estimate_rows": sample_estimate["estimate_rows"],
            "tolerance_basis_points": sample_tolerance_bp,
            "absolute_slack_rows": absolute_slack,
            "lower_rows": sample_lower,
            "upper_rows": sample_upper,
            "in_range": sample_in_range,
        },
        "first_failing_key_interval": failing_interval,
        "truth_digest": truth["output_sequence_digest"],
    }
    if result["status"] == "FAIL":
        raise JoinLabError(
            code="E_ESTIMATE",
            message="至少一条估计路径未落入真实结果容差区间",
            details={
                "first_failing_key_interval": failing_interval,
                "histogram_estimate_rows": histogram_estimate["estimate_rows"],
                "histogram_estimate_fixed": hist_fixed,
                "sample_estimate_rows": sample_estimate["estimate_rows"],
                "sample_estimate_fixed": sample_fixed,
                "true_row_count": true_rows,
                "histogram_interval_rows": [hist_lower, hist_upper],
                "sample_interval_rows": [sample_lower, sample_upper],
                "validation": result,
            },
        )
    return result
