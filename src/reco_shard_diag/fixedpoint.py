from __future__ import annotations

from typing import Any

import numpy as np

from .canonical import fingerprint


ROUNDING_POLICY = {
    "division": "floor_exact_integer",
    "aggregate_overflow": "saturate_to_signed_width",
    "soft_boundary": "equal_is_ok",
    "hard_boundary": "greater_than_hard_is_rejected",
    "tie_break": "floor_then_hard_boundary_wins",
    "watermark_domain": "unbounded_python_integer_with_width_in_artifact_key",
}


def signed_bounds(width: int) -> tuple[int, int]:
    if width < 2:
        raise ValueError("fixed-point width must be >= 2")
    return -(1 << (width - 1)), (1 << (width - 1)) - 1


def quantize_signed(value: int, width: int) -> int:
    low, high = signed_bounds(width)
    return max(low, min(high, int(value)))


def aggregate_groups(
    samples: list[dict[str, Any]],
    dimensions: int,
    width: int,
    fixed_scale: int = 1000,
) -> dict[str, Any]:
    if not samples:
        body = {
            "width": width,
            "fixed_scale": fixed_scale,
            "group_order": [],
            "groups": [],
            "sample_count": 0,
            "overflow_count": 0,
        }
        return {**body, "summary_fingerprint": fingerprint(body)}

    grouped: dict[str, list[list[int]]] = {}
    for sample in samples:
        values = sample.get("values")
        if not isinstance(values, list) or len(values) != dimensions:
            raise ValueError("sample dimension mismatch")
        key = str(sample["group"])
        grouped.setdefault(key, []).append([int(value) for value in values])

    ordered_parts: list[dict[str, Any]] = []
    total_count = 0
    total_overflow = 0
    low, high = signed_bounds(width)
    for key in sorted(grouped):
        matrix = np.asarray(grouped[key], dtype=np.int64)
        sums = matrix.sum(axis=0, dtype=np.int64)
        raw_sums = [int(value) for value in sums.tolist()]
        clipped = [max(low, min(high, value)) for value in raw_sums]
        overflow = [value != clipped_value for value, clipped_value in zip(raw_sums, clipped)]
        total_count += int(matrix.shape[0])
        total_overflow += sum(overflow)
        ordered_parts.append(
            {
                "group": key,
                "count": int(matrix.shape[0]),
                "raw_sum": raw_sums,
                "quantized_sum": clipped,
                "overflow": overflow,
            }
        )

    body = {
        "width": width,
        "fixed_scale": fixed_scale,
        "group_order": [part["group"] for part in ordered_parts],
        "groups": ordered_parts,
        "sample_count": total_count,
        "overflow_count": total_overflow,
    }
    return {**body, "summary_fingerprint": fingerprint(body)}


def water_limit(capacity_slots: int, watermark_bp: int, width: int) -> tuple[int, int]:
    product = int(capacity_slots) * int(watermark_bp)
    return product // 10000, product


def evaluate_watermark(
    capacity_slots: int,
    resource_units: int,
    soft_watermark_bp: int,
    hard_watermark_bp: int,
    width: int,
) -> dict[str, Any]:
    soft_limit, soft_product = water_limit(capacity_slots, soft_watermark_bp, width)
    hard_limit, hard_product = water_limit(capacity_slots, hard_watermark_bp, width)
    if resource_units > hard_limit:
        state = "HARD_REJECT"
        code = "E_QUOTA_HARD_LIMIT"
    elif resource_units > soft_limit:
        state = "SOFT_WARN"
        code = "E_QUOTA_SOFT_LIMIT"
    else:
        state = "OK"
        code = None
    return {
        "capacity_slots": capacity_slots,
        "resource_units": resource_units,
        "soft_watermark_bp": soft_watermark_bp,
        "hard_watermark_bp": hard_watermark_bp,
        "soft_limit": soft_limit,
        "hard_limit": hard_limit,
        "soft_product": soft_product,
        "hard_product": hard_product,
        "width": width,
        "rounding_policy": ROUNDING_POLICY.copy(),
        "state": state,
        "code": code,
    }
