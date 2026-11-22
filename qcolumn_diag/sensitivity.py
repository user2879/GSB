"""Layered column-group sensitivity analysis across bit-width candidates."""

from __future__ import annotations

from typing import Any, Iterable

from .kernel import ATTRIBUTION_PRIORITY, ATTRIBUTION_SOURCES
from .models import TensorConfig


def _empty_source_sums() -> dict[str, float]:
    return {source: 0.0 for source in ATTRIBUTION_SOURCES}


def _normalize(sums: dict[str, float]) -> dict[str, float]:
    total = sum(float(value) for value in sums.values())
    if total <= 0.0:
        return _empty_source_sums()
    return {source: float(sums[source]) / total for source in ATTRIBUTION_SOURCES}


def _record_width_summary(
    record: dict[str, Any] | None, bit_width: int
) -> dict[str, Any] | None:
    if not record:
        return None
    stage = record.get("stages", {}).get("requantize_errors", {}).get(str(bit_width))
    if not isinstance(stage, dict):
        return None
    artifact = stage.get("artifact")
    if not isinstance(artifact, dict):
        return None
    summary = artifact.get("summary")
    return summary if isinstance(summary, dict) else None


def build_group_sensitivity(
    tensor_configs: Iterable[TensorConfig],
    records: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    entries: list[dict[str, Any]] = []
    violations: list[dict[str, Any]] = []
    for tensor in sorted(tensor_configs, key=lambda item: item.tensor_id):
        groups = sorted(tensor.groups, key=lambda group: group.group_id)
        for group in groups:
            for bit_width in sorted(tensor.bit_width_candidates):
                analyzed_columns: list[int] = []
                missing_columns: list[int] = []
                analyzed_count = 0
                missing_count = 0
                saturated_count = 0
                over_tolerance_count = 0
                zero_reference_count = 0
                max_abs_error = 0.0
                max_relative_error = 0.0
                max_effective_relative_error = 0.0
                source_error_sums = _empty_source_sums()
                over_source_error_sums = _empty_source_sums()
                attribution_total = 0.0
                over_attribution_total = 0.0
                for column in group.columns:
                    column_id = f"{tensor.tensor_id}:c{column:06d}"
                    summary = _record_width_summary(records.get(column_id), bit_width)
                    if summary is None:
                        missing_columns.append(column)
                        continue
                    analyzed_columns.append(column)
                    analyzed_count += int(summary.get("analyzed_count", 0))
                    missing_count += int(summary.get("missing_count", 0))
                    saturated_count += int(summary.get("saturated_count", 0))
                    over_tolerance_count += int(
                        summary.get("over_tolerance_count", 0)
                    )
                    zero_reference_count += int(
                        summary.get("zero_reference_count", 0)
                    )
                    max_abs_error = max(
                        max_abs_error, float(summary.get("max_abs_error", 0.0))
                    )
                    max_relative_error = max(
                        max_relative_error,
                        float(summary.get("max_relative_error_nonzero", 0.0)),
                    )
                    max_effective_relative_error = max(
                        max_effective_relative_error,
                        float(
                            summary.get("max_effective_relative_error", 0.0)
                        ),
                    )
                    column_error_total = float(
                        summary.get("attribution_error_total", 0.0)
                    )
                    column_over_total = float(
                        summary.get("over_tolerance_attribution_error_total", 0.0)
                    )
                    attribution_total += column_error_total
                    over_attribution_total += column_over_total
                    all_shares = summary.get("attribution_share_by_source", {})
                    over_shares = summary.get(
                        "over_tolerance_attribution_share_by_source", {}
                    )
                    for source in ATTRIBUTION_SOURCES:
                        source_error_sums[source] += (
                            float(all_shares.get(source, 0.0)) * column_error_total
                        )
                        over_source_error_sums[source] += (
                            float(over_shares.get(source, 0.0))
                            * column_over_total
                        )
                source_shares = _normalize(source_error_sums)
                over_source_shares = _normalize(over_source_error_sums)
                dominant_source = max(
                    ATTRIBUTION_PRIORITY,
                    key=lambda source: (
                        source_shares[source],
                        -ATTRIBUTION_PRIORITY.index(source),
                    ),
                )
                if attribution_total <= 0.0:
                    dominant_source = "none"
                saturation_ratio = (
                    saturated_count / analyzed_count if analyzed_count else 0.0
                )
                over_tolerance_ratio = (
                    over_tolerance_count / analyzed_count if analyzed_count else 0.0
                )
                limit_exceeded = bool(
                    analyzed_count > 0
                    and over_tolerance_ratio > tensor.group_over_tolerance_limit
                )
                if analyzed_count == 0:
                    status = "no_data"
                elif limit_exceeded:
                    status = "LIMIT_EXCEEDED"
                else:
                    status = "WITHIN_LIMIT"
                conclusion = (
                    f"tensor={tensor.tensor_id} group={group.group_id} width={bit_width} "
                    f"q=[{tensor.quantile_lower_permille},{tensor.quantile_upper_permille}] "
                    f"scheme={tensor.scheme} frac={tensor.fractional_bits}: "
                    f"over_tolerance_ratio={over_tolerance_ratio:.6f} "
                    f"limit={tensor.group_over_tolerance_limit:.6f} "
                    f"saturated_ratio={saturation_ratio:.6f} "
                    f"max_relative_error={max_relative_error:.6f} "
                    f"max_effective_relative_error={max_effective_relative_error:.6f} "
                    f"dominant_source={dominant_source} "
                    f"coverage={len(analyzed_columns)}/{len(group.columns)}; {status}"
                )
                entry = {
                    "tensor_id": tensor.tensor_id,
                    "group_id": group.group_id,
                    "columns": list(group.columns),
                    "bit_width": int(bit_width),
                    "quantile_lower_permille": tensor.quantile_lower_permille,
                    "quantile_upper_permille": tensor.quantile_upper_permille,
                    "scheme": tensor.scheme,
                    "fractional_bits": tensor.fractional_bits,
                    "analyzed_columns": analyzed_columns,
                    "missing_columns": missing_columns,
                    "column_coverage_ratio": (
                        len(analyzed_columns) / len(group.columns)
                        if group.columns
                        else 0.0
                    ),
                    "analyzed_count": analyzed_count,
                    "missing_count": missing_count,
                    "saturated_count": saturated_count,
                    "saturation_ratio": saturation_ratio,
                    "over_tolerance_count": over_tolerance_count,
                    "over_tolerance_ratio": over_tolerance_ratio,
                    "max_abs_error": max_abs_error,
                    "max_relative_error": max_relative_error,
                    "max_relative_error_excludes_zero_reference": True,
                    "max_effective_relative_error": max_effective_relative_error,
                    "zero_reference_count": zero_reference_count,
                    "attribution_share_by_source": source_shares,
                    "over_tolerance_attribution_share_by_source": over_source_shares,
                    "dominant_source": dominant_source,
                    "group_over_tolerance_limit": tensor.group_over_tolerance_limit,
                    "limit_exceeded": limit_exceeded,
                    "status": status,
                    "conclusion": conclusion,
                }
                entries.append(entry)
                if limit_exceeded:
                    violations.append(
                        {
                            "tensor_id": tensor.tensor_id,
                            "group_id": group.group_id,
                            "bit_width": int(bit_width),
                            "columns": list(group.columns),
                            "over_tolerance_ratio": over_tolerance_ratio,
                            "group_over_tolerance_limit": tensor.group_over_tolerance_limit,
                        }
                    )
    entries.sort(
        key=lambda item: (
            item["tensor_id"],
            item["group_id"],
            item["bit_width"],
        )
    )
    violations.sort(
        key=lambda item: (
            item["tensor_id"],
            item["group_id"],
            item["bit_width"],
        )
    )
    return {"groups": entries, "violations": violations}
