"""Group/bit-width sensitivity layers and deterministic conclusions."""

from __future__ import annotations

from typing import Any

from .config import CalibrationConfig
from .keys import sensitivity_keys
from .quantize import SOURCE_PRIORITY


def _dominant_share(shares: dict[str, float]) -> str:
    maximum = max((shares.get(source, 0.0) for source in SOURCE_PRIORITY), default=0.0)
    if maximum <= 0.0:
        return "none"
    for source in SOURCE_PRIORITY:
        if abs(shares.get(source, 0.0) - maximum) <= 1e-15:
            return source
    return "none"


def summarise_group(
    config: CalibrationConfig,
    attribution: dict[str, Any],
) -> dict[str, Any]:
    group_id = attribution["group_id"]
    bit_width = int(attribution["bit_width"])
    fractional_bits = int(attribution["fractional_bits"])
    outlier_ratio_limit = config.tolerance.outlier_ratio_limit
    outlier_ratio = float(attribution["outlier_ratio"])
    threshold_exceeded = outlier_ratio > outlier_ratio_limit
    outlier_shares = dict(attribution["outlier_source_shares"])
    all_shares = dict(attribution["dominant_source_shares"])
    dominant_outlier_source = _dominant_share(outlier_shares)
    dominant_all_source = _dominant_share(all_shares)
    keys = sensitivity_keys(
        config,
        content_fingerprint=attribution["content_fingerprint"],
        attribution_cache_key=attribution["cache_key"],
        bit_width=bit_width,
    )

    max_relative = attribution["max_relative_error"]
    if max_relative is None:
        max_relative_text = "n/a"
    else:
        max_relative_text = f"{float(max_relative):.12g}"
    conclusion = (
        f"group={group_id} bit_width={bit_width} "
        f"max_relative_error={max_relative_text} "
        f"truncation_ratio={float(attribution['truncation_ratio']):.12g} "
        f"outlier_ratio={outlier_ratio:.12g} "
        f"threshold_exceeded={'yes' if threshold_exceeded else 'no'} "
        f"dominant_outlier_source={dominant_outlier_source}"
    )
    if attribution["zero_value_branch_count"]:
        conclusion += (
            f" zero_branch_count={attribution['zero_value_branch_count']}"
            f" zero_branch_max_abs={attribution['max_zero_branch_absolute_error']:.12g}"
        )

    return {
        "stage": "sensitivity",
        "task_sort_key": f"{group_id}|sensitivity|{bit_width:03d}",
        "layer": f"group:{group_id}/bit_width:{bit_width}",
        "group_id": group_id,
        "columns": attribution["columns"],
        "row_ids": attribution["row_ids"],
        "bit_width": bit_width,
        "fractional_bits": fractional_bits,
        "active_bit_width": attribution["active_bit_width"],
        "element_count": attribution["element_count"],
        "nonmissing_count": attribution["nonmissing_count"],
        "missing_count": attribution["missing_count"],
        "max_relative_error": max_relative,
        "max_absolute_error": attribution["max_absolute_error"],
        "max_zero_branch_absolute_error": attribution["max_zero_branch_absolute_error"],
        "zero_value_branch_count": attribution["zero_value_branch_count"],
        "truncation_count": attribution["truncation_count"],
        "truncation_ratio": attribution["truncation_ratio"],
        "outlier_count": attribution["outlier_count"],
        "outlier_ratio": outlier_ratio,
        "outlier_ratio_limit": outlier_ratio_limit,
        "threshold_exceeded": threshold_exceeded,
        "group_width_threshold_key": f"{group_id}@{bit_width}",
        "dominant_source_counts": attribution["dominant_source_counts"],
        "dominant_source_shares": all_shares,
        "outlier_source_counts": attribution["outlier_source_counts"],
        "outlier_source_shares": outlier_shares,
        "dominant_error_source": dominant_outlier_source
        if attribution["outlier_count"]
        else dominant_all_source,
        "dominant_outlier_error_source": dominant_outlier_source,
        "conclusion": conclusion,
        "layer_status": "THRESHOLD_EXCEEDED" if threshold_exceeded else "WITHIN_THRESHOLD",
        "upstream_cache_key": attribution["cache_key"],
        "content_fingerprint": attribution["content_fingerprint"],
        "parameter_fingerprint": keys["parameter_fingerprint"],
        "cache_key": keys["cache_key"],
    }
