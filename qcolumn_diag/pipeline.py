"""Concurrent column pipeline, incremental cache coordination and reporting."""

from __future__ import annotations

import json
import platform
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from .cache import SnapshotStore, empty_snapshot, write_text_atomic
from .calibration import derive_quant_params, quantile_stage_artifact
from .config import load_config
from .errors import make_failure, sort_failures
from .fingerprint import (
    column_content_fingerprint,
    sha256_json,
    stage_cache_key,
)
from .kernel import ATTRIBUTION_SOURCES, analyze_column_width
from .models import TensorConfig, WorkbenchConfig
from .reference import verify_active_width
from .sample import SampleLoadResult, TensorSample, load_sample
from .sensitivity import build_group_sensitivity

ENGINE_VERSION = "column-calibration-diagnostics-v1"
REPORT_SCHEMA_VERSION = "1.0"
ATTRIBUTION_PRIORITY = (
    "saturation_clipping",
    "dequant_rounding",
    "scale_quantization",
    "quantile_discretization",
)


@dataclass
class ColumnPlan:
    tensor: TensorConfig
    sample: TensorSample
    column: int
    column_id: str
    sort_key: str
    values: np.ndarray
    missing_mask: np.ndarray
    content_fingerprint: str
    sample_element_refs: list[dict[str, Any]]
    calibration_key: str
    quant_keys: dict[int, str]
    requant_keys: dict[int, str]
    cost: int
    previous_record: dict[str, Any] | None
    cached_complete: bool
    blocking_failure: dict[str, Any] | None = None

    @property
    def active_width(self) -> int:
        return int(self.tensor.active_bit_width)


@dataclass
class ColumnExecution:
    record: dict[str, Any] | None
    status: dict[str, Any]
    failures: list[dict[str, Any]]
    timings_ms: dict[str, float] = field(default_factory=dict)


def _decorate_failure(
    failure: dict[str, Any],
    plan: ColumnPlan,
    *,
    bit_width: int | None = None,
    stage: str | None = None,
) -> dict[str, Any]:
    item = dict(failure)
    item["tensor_id"] = plan.tensor.tensor_id
    item["column"] = plan.column
    item["bit_width"] = bit_width if bit_width is not None else item.get("bit_width")
    item["stage"] = stage if stage is not None else item.get("stage")
    return item


def _calibration_params(config: WorkbenchConfig, plan: ColumnPlan) -> dict[str, Any]:
    return {
        "column_id": plan.column_id,
        "quantile_lower_permille": plan.tensor.quantile_lower_permille,
        "quantile_upper_permille": plan.tensor.quantile_upper_permille,
        "interpolation": config.interpolation,
        "tie_break": config.tie_break,
        "sort_order": "ascending_stable",
        "stage_version": 1,
    }


def _quant_params(config: WorkbenchConfig, plan: ColumnPlan, bit_width: int) -> dict[str, Any]:
    return {
        "column_id": plan.column_id,
        "scheme": plan.tensor.scheme,
        "bit_width": int(bit_width),
        "fractional_bits": plan.tensor.fractional_bits,
        "scale_quantization": config.scale_quantization,
        "zero_scale_policy": plan.tensor.zero_scale_policy,
        "stage_version": 1,
    }


def _make_plan(
    *,
    config: WorkbenchConfig,
    tensor: TensorConfig,
    sample: TensorSample,
    column: int,
    previous_snapshot_columns: dict[str, Any],
) -> ColumnPlan:
    values = np.asarray(sample.matrix[:, column], dtype=np.float64)
    missing = np.asarray(sample.missing_mask[:, column], dtype=bool)
    refs: list[dict[str, Any]] = []
    for row in range(sample.row_count):
        ref = sample.element_ref(row, column)
        ref["missing"] = bool(missing[row])
        refs.append(ref)
    value_fingerprint = column_content_fingerprint(
        values, missing, sample.original_dtype
    )
    reference_fingerprint = sha256_json(refs)
    content_fingerprint = sha256_json(
        {
            "value_fingerprint": value_fingerprint,
            "sample_reference_fingerprint": reference_fingerprint,
        }
    )
    column_id = f"{tensor.tensor_id}:c{column:06d}"
    calibration_params = {
        "column_id": column_id,
        "quantile_lower_permille": tensor.quantile_lower_permille,
        "quantile_upper_permille": tensor.quantile_upper_permille,
        "interpolation": config.interpolation,
        "tie_break": config.tie_break,
        "sort_order": "ascending_stable",
        "stage_version": 1,
    }
    calibration_key = stage_cache_key(
        "quantile_calibration", content_fingerprint, params=calibration_params
    )
    quant_keys: dict[int, str] = {}
    requant_keys: dict[int, str] = {}
    for bit_width in sorted(tensor.bit_width_candidates):
        quant_key = stage_cache_key(
            "scale_zero_point",
            content_fingerprint,
            calibration_key,
            params={
                "column_id": column_id,
                "scheme": tensor.scheme,
                "bit_width": int(bit_width),
                "fractional_bits": tensor.fractional_bits,
                "scale_quantization": config.scale_quantization,
                "zero_scale_policy": tensor.zero_scale_policy,
                "stage_version": 1,
            },
        )
        quant_keys[bit_width] = quant_key
        requant_keys[bit_width] = stage_cache_key(
            "requantize_dequantize_errors",
            content_fingerprint,
            quant_key,
            params={
                "column_id": column_id,
                "active_bit_width": tensor.active_bit_width,
                "saturation_strategy": tensor.saturation_strategy,
                "tolerance": tensor.tolerance,
                "tolerance_branch": "relative_except_absolute_at_zero",
                "code_rounding": "round_half_away_from_zero",
                "attribution_priority": list(ATTRIBUTION_PRIORITY),
                "stage_version": 1,
            },
        )
    previous_record = previous_snapshot_columns.get(column_id)
    if not isinstance(previous_record, dict):
        previous_record = None
    cached_complete = _record_is_complete(
        previous_record,
        content_fingerprint=content_fingerprint,
        calibration_key=calibration_key,
        quant_keys=quant_keys,
        requant_keys=requant_keys,
    )
    blocking_failure: dict[str, Any] | None = None
    if not np.any(~missing):
        blocking_failure = {
            "error_code": "E_COLUMN_EMPTY",
            "field": "column.elements",
            "message": "column has no non-missing elements",
            "tensor_id": tensor.tensor_id,
            "column": column,
            "bit_width": None,
            "stage": "quantile_calibration",
        }
    return ColumnPlan(
        tensor=tensor,
        sample=sample,
        column=column,
        column_id=column_id,
        sort_key=f"{column_id}|calibration+requantization",
        values=values,
        missing_mask=missing,
        content_fingerprint=content_fingerprint,
        sample_element_refs=refs,
        calibration_key=calibration_key,
        quant_keys=quant_keys,
        requant_keys=requant_keys,
        cost=int(np.count_nonzero(~missing)),
        previous_record=previous_record,
        cached_complete=cached_complete,
        blocking_failure=blocking_failure,
    )


def _record_is_complete(
    record: dict[str, Any] | None,
    *,
    content_fingerprint: str,
    calibration_key: str,
    quant_keys: dict[int, str],
    requant_keys: dict[int, str],
) -> bool:
    if not isinstance(record, dict):
        return False
    if record.get("content_fingerprint") != content_fingerprint:
        return False
    stages = record.get("stages")
    if not isinstance(stages, dict):
        return False
    calibration = stages.get("calibration")
    if not isinstance(calibration, dict) or calibration.get("key") != calibration_key:
        return False
    quant_stage = stages.get("quant_params")
    requant_stage = stages.get("requantize_errors")
    if not isinstance(quant_stage, dict) or not isinstance(requant_stage, dict):
        return False
    for bit_width, key in quant_keys.items():
        stage = quant_stage.get(str(bit_width))
        if not isinstance(stage, dict) or stage.get("key") != key:
            return False
    for bit_width, key in requant_keys.items():
        stage = requant_stage.get(str(bit_width))
        if not isinstance(stage, dict) or stage.get("key") != key:
            return False
    return True


def _compute_column(plan: ColumnPlan, config: WorkbenchConfig) -> ColumnExecution:
    started = time.perf_counter()
    stage_timings = {
        "calibration_ms": 0.0,
        "quant_params_ms": 0.0,
        "requantize_errors_ms": 0.0,
    }
    previous_stages = (
        plan.previous_record.get("stages", {})
        if isinstance(plan.previous_record, dict)
        else {}
    )
    status: dict[str, Any] = {
        "calibration": "computed",
        "quant_params": {},
        "requantize_errors": {},
    }
    any_computed = False
    calibration_stage = previous_stages.get("calibration")
    if (
        isinstance(calibration_stage, dict)
        and calibration_stage.get("key") == plan.calibration_key
        and isinstance(calibration_stage.get("artifact"), dict)
    ):
        calibration = calibration_stage["artifact"]
        status["calibration"] = "reused"
    else:
        calibration_started = time.perf_counter()
        calibration, failure = quantile_stage_artifact(
            plan.values,
            plan.missing_mask,
            plan.tensor,
            interpolation=config.interpolation,
            tie_break=config.tie_break,
        )
        stage_timings["calibration_ms"] += _elapsed_ms(calibration_started)
        if calibration is None:
            status["any_computed"] = True
            decorated = _decorate_failure(
                failure or {
                    "error_code": "E_COLUMN_EMPTY",
                    "field": "column.elements",
                    "message": "quantile calibration failed",
                },
                plan,
                stage="quantile_calibration",
            )
            return ColumnExecution(
                record=None,
                status=status,
                failures=[decorated],
                timings_ms={
                    "column_total": _elapsed_ms(started),
                    **stage_timings,
                },
            )
        any_computed = True
        status["calibration"] = "computed"

    quant_artifacts: dict[int, dict[str, Any]] = {}
    analyses: dict[int, dict[str, Any]] = {}
    failures: list[dict[str, Any]] = []
    previous_quant = previous_stages.get("quant_params", {})
    previous_requant = previous_stages.get("requantize_errors", {})
    for bit_width in sorted(plan.tensor.bit_width_candidates):
        width_key = str(bit_width)
        quant_stage = (
            previous_quant.get(width_key) if isinstance(previous_quant, dict) else None
        )
        if (
            isinstance(quant_stage, dict)
            and quant_stage.get("key") == plan.quant_keys[bit_width]
            and isinstance(quant_stage.get("artifact"), dict)
        ):
            params = quant_stage["artifact"]
            status["quant_params"][width_key] = "reused"
        else:
            quant_started = time.perf_counter()
            params = derive_quant_params(
                lower_quantile=float(calibration["lower_quantile"]),
                upper_quantile=float(calibration["upper_quantile"]),
                tensor=plan.tensor,
                bit_width=bit_width,
            )
            stage_timings["quant_params_ms"] += _elapsed_ms(quant_started)
            any_computed = True
            status["quant_params"][width_key] = "computed"
        quant_artifacts[bit_width] = params

        analysis_stage = (
            previous_requant.get(width_key)
            if isinstance(previous_requant, dict)
            else None
        )
        if (
            isinstance(analysis_stage, dict)
            and analysis_stage.get("key") == plan.requant_keys[bit_width]
            and isinstance(analysis_stage.get("artifact"), dict)
        ):
            analysis = analysis_stage["artifact"]
            status["requantize_errors"][width_key] = "reused"
        else:
            requant_started = time.perf_counter()
            analysis, width_failures = analyze_column_width(
                values=plan.values,
                missing_mask=plan.missing_mask,
                tensor=plan.tensor,
                calibration=calibration,
                params=params,
                sample_refs=plan.sample_element_refs,
            )
            stage_timings["requantize_errors_ms"] += _elapsed_ms(requant_started)
            analysis["column_index"] = plan.column
            for failure in width_failures:
                failures.append(
                    _decorate_failure(
                        failure,
                        plan,
                        bit_width=bit_width,
                        stage="requantize_dequantize_errors",
                    )
                )
            any_computed = True
            status["requantize_errors"][width_key] = "computed"
        analyses[bit_width] = analysis

    record = {
        "column_id": plan.column_id,
        "tensor_id": plan.tensor.tensor_id,
        "column_index": plan.column,
        "content_fingerprint": plan.content_fingerprint,
        "sample_element_refs": plan.sample_element_refs,
        "sample_spec_hash": plan.sample.spec_hash,
        "stages": {
            "calibration": {
                "key": plan.calibration_key,
                "artifact": calibration,
            },
            "quant_params": {
                str(width): {
                    "key": plan.quant_keys[width],
                    "artifact": quant_artifacts[width],
                }
                for width in sorted(quant_artifacts)
            },
            "requantize_errors": {
                str(width): {
                    "key": plan.requant_keys[width],
                    "artifact": analyses[width],
                }
                for width in sorted(analyses)
            },
        },
    }
    status["any_computed"] = bool(any_computed)
    return ColumnExecution(
        record=record,
        status=status,
        failures=failures,
        timings_ms={
            "column_total": _elapsed_ms(started),
            **stage_timings,
        },
    )


def _elapsed_ms(started: float) -> float:
    return round((time.perf_counter() - started) * 1000.0, 6)


def _build_batches(
    plans: list[ColumnPlan], budget: int
) -> tuple[list[list[ColumnPlan]], list[ColumnPlan]]:
    batches: list[list[ColumnPlan]] = []
    pending: list[ColumnPlan] = []
    current: list[ColumnPlan] = []
    current_cost = 0
    for plan in sorted(plans, key=lambda item: item.column_id):
        if plan.cost > budget:
            pending.append(plan)
            continue
        if current and current_cost + plan.cost > budget:
            batches.append(current)
            current = []
            current_cost = 0
        current.append(plan)
        current_cost += plan.cost
    if current:
        batches.append(current)
    return batches, pending


def _complete_column_ids(
    plans: list[ColumnPlan], records: dict[str, dict[str, Any]]
) -> list[str]:
    completed: list[str] = []
    for plan in plans:
        record = records.get(plan.column_id)
        if _record_is_complete(
            record,
            content_fingerprint=plan.content_fingerprint,
            calibration_key=plan.calibration_key,
            quant_keys=plan.quant_keys,
            requant_keys=plan.requant_keys,
        ):
            completed.append(plan.column_id)
    return sorted(completed)


def _artifact_versions(
    plans: list[ColumnPlan], records: dict[str, dict[str, Any]]
) -> dict[str, Any]:
    versions: dict[str, Any] = {}
    for plan in plans:
        record = records.get(plan.column_id)
        if not _record_is_complete(
            record,
            content_fingerprint=plan.content_fingerprint,
            calibration_key=plan.calibration_key,
            quant_keys=plan.quant_keys,
            requant_keys=plan.requant_keys,
        ):
            continue
        versions[plan.column_id] = {
            "calibration": plan.calibration_key,
            "quant_params": {
                str(width): plan.quant_keys[width] for width in sorted(plan.quant_keys)
            },
            "requantize_errors": {
                str(width): plan.requant_keys[width]
                for width in sorted(plan.requant_keys)
            },
        }
    return {key: versions[key] for key in sorted(versions)}


def _collect_failures(
    *,
    config: WorkbenchConfig,
    config_failures: list[dict[str, Any]],
    sample_failures: list[dict[str, Any]],
    execution_failures: list[dict[str, Any]],
    plans: list[ColumnPlan],
    records: dict[str, dict[str, Any]],
    pending_plans: list[ColumnPlan],
) -> list[dict[str, Any]]:
    failures: list[dict[str, Any]] = []
    failures.extend(config_failures)
    failures.extend(sample_failures)
    failures.extend(execution_failures)
    for plan in plans:
        if plan.blocking_failure is not None:
            failures.append(plan.blocking_failure)
            continue
        record = records.get(plan.column_id)
        if not _record_is_complete(
            record,
            content_fingerprint=plan.content_fingerprint,
            calibration_key=plan.calibration_key,
            quant_keys=plan.quant_keys,
            requant_keys=plan.requant_keys,
        ):
            continue
        for bit_width in sorted(plan.tensor.bit_width_candidates):
            artifact = record["stages"]["requantize_errors"][str(bit_width)]["artifact"]
            for failure in artifact.get("failures", []):
                failures.append(
                    _decorate_failure(
                        failure,
                        plan,
                        bit_width=bit_width,
                        stage="requantize_dequantize_errors",
                    )
                )
    for plan in pending_plans:
        failures.append(
            make_failure(
                "E_CALIB_BUDGET_PENDING",
                "column.calibration_budget",
                (
                    f"column requires {plan.cost} calibration elements, exceeding "
                    f"batch budget {config.calibration_element_budget}; kept pending"
                ),
                tensor_id=plan.tensor.tensor_id,
                column=plan.column,
                stage="quantile_calibration",
            )
        )
    return sort_failures(failures)


def _column_report(
    plan: ColumnPlan,
    record: dict[str, Any],
    config: WorkbenchConfig,
) -> dict[str, Any]:
    active_width = plan.tensor.active_bit_width
    calibration = record["stages"]["calibration"]["artifact"]
    active_quant = record["stages"]["quant_params"][str(active_width)]["artifact"]
    active_analysis = record["stages"]["requantize_errors"][str(active_width)][
        "artifact"
    ]
    candidate_summaries = []
    quant_params_by_width: dict[str, Any] = {}
    for bit_width in sorted(plan.tensor.bit_width_candidates):
        quant_artifact = record["stages"]["quant_params"][str(bit_width)]["artifact"]
        analysis_artifact = record["stages"]["requantize_errors"][str(bit_width)][
            "artifact"
        ]
        quant_params_by_width[str(bit_width)] = quant_artifact
        candidate_summaries.append(analysis_artifact["summary"])
    verification = verify_active_width(
        values=plan.values,
        missing_mask=plan.missing_mask,
        tensor=plan.tensor,
        calibration=calibration,
        params=active_quant,
        analysis=active_analysis,
    )
    return {
        "column_id": plan.column_id,
        "tensor_id": plan.tensor.tensor_id,
        "column_index": int(plan.column),
        "active_bit_width": int(active_width),
        "column_content_fingerprint": plan.content_fingerprint,
        "sample_element_refs": plan.sample_element_refs,
        "calibration": calibration,
        "quant_params_by_width": quant_params_by_width,
        "active_quant_params": active_quant,
        "candidate_summaries": candidate_summaries,
        "active_summary": active_analysis["summary"],
        "elements": active_analysis["elements"],
        "verification": verification,
        "stage_cache_keys": {
            "calibration": plan.calibration_key,
            "quant_params": {
                str(width): plan.quant_keys[width]
                for width in sorted(plan.quant_keys)
            },
            "requantize_errors": {
                str(width): plan.requant_keys[width]
                for width in sorted(plan.requant_keys)
            },
        },
    }


def _aggregate_attribution(column_reports: list[dict[str, Any]]) -> tuple[dict[str, float], dict[str, float]]:
    all_sums = {source: 0.0 for source in ATTRIBUTION_SOURCES}
    over_sums = {source: 0.0 for source in ATTRIBUTION_SOURCES}
    for column in column_reports:
        summary = column["active_summary"]
        total = float(summary.get("attribution_error_total", 0.0))
        over_total = float(
            summary.get("over_tolerance_attribution_error_total", 0.0)
        )
        all_share = summary.get("attribution_share_by_source", {})
        over_share = summary.get("over_tolerance_attribution_share_by_source", {})
        for source in ATTRIBUTION_SOURCES:
            all_sums[source] += float(all_share.get(source, 0.0)) * total
            over_sums[source] += float(over_share.get(source, 0.0)) * over_total
    def normalize(values: dict[str, float]) -> dict[str, float]:
        total = sum(values.values())
        if total <= 0.0:
            return {source: 0.0 for source in ATTRIBUTION_SOURCES}
        return {source: values[source] / total for source in ATTRIBUTION_SOURCES}
    return normalize(all_sums), normalize(over_sums)


def _render_report(
    *,
    config: WorkbenchConfig,
    sample_result: SampleLoadResult,
    config_failures: list[dict[str, Any]],
    sample_failures: list[dict[str, Any]],
    plans: list[ColumnPlan],
    records: dict[str, dict[str, Any]],
    execution_failures: list[dict[str, Any]],
    pending_plans: list[ColumnPlan],
    run_version: str,
    resume_status: str,
    previous_run_version: str | None,
    mode: str,
    snapshot_generation: int,
    max_workers_observed: int,
    batch_count: int,
    timings_ms: dict[str, float],
    cache_status: dict[str, Any],
) -> dict[str, Any]:
    column_reports: list[dict[str, Any]] = []
    for plan in plans:
        if plan.blocking_failure is not None:
            continue
        record = records.get(plan.column_id)
        if record is None:
            continue
        if not _record_is_complete(
            record,
            content_fingerprint=plan.content_fingerprint,
            calibration_key=plan.calibration_key,
            quant_keys=plan.quant_keys,
            requant_keys=plan.requant_keys,
        ):
            continue
        column_reports.append(_column_report(plan, record, config))
    column_reports.sort(key=lambda item: item["column_id"])
    failures = _collect_failures(
        config=config,
        config_failures=config_failures,
        sample_failures=sample_failures,
        execution_failures=execution_failures,
        plans=plans,
        records=records,
        pending_plans=pending_plans,
    )

    sensitivity = build_group_sensitivity(config.tensors, records)
    all_shares, over_shares = _aggregate_attribution(column_reports)
    element_count = 0
    missing_count = 0
    saturated_count = 0
    over_tolerance_count = 0
    zero_reference_count = 0
    max_abs_error = 0.0
    max_effective_relative_error = 0.0
    attribution_share_error_max = 0.0
    attribution_share_violation_count = 0
    dominant_counts = {source: 0 for source in ATTRIBUTION_SOURCES}
    for column in column_reports:
        summary = column["active_summary"]
        element_count += int(summary["analyzed_count"]) + int(summary["missing_count"])
        missing_count += int(summary["missing_count"])
        saturated_count += int(summary["saturated_count"])
        over_tolerance_count += int(summary["over_tolerance_count"])
        zero_reference_count += int(summary["zero_reference_count"])
        max_abs_error = max(max_abs_error, float(summary["max_abs_error"]))
        max_effective_relative_error = max(
            max_effective_relative_error,
            float(summary["max_effective_relative_error"]),
        )
        attribution_share_error_max = max(
            attribution_share_error_max,
            float(summary["attribution_share_error_max"]),
        )
        for element in column["elements"]:
            attribution = element.get("attribution")
            if attribution is not None and not bool(
                attribution.get("normalized_to_total_error")
            ):
                attribution_share_violation_count += 1
        for source, count in summary["dominant_source_counts"].items():
            dominant_counts[source] += int(count)
    analyzed_count = element_count - missing_count
    dominant_source = "none"
    if any(dominant_counts.values()):
        dominant_source = max(
            ATTRIBUTION_PRIORITY,
            key=lambda source: (
                dominant_counts[source],
                -ATTRIBUTION_PRIORITY.index(source),
            ),
        )
    injection_hits: list[str] = []
    injection_attempted = 0
    injection_by_tensor: dict[str, list[str]] = {}
    for tensor_id in sorted(sample_result.tensors):
        sample = sample_result.tensors[tensor_id]
        hits = []
        for injection in sample.applied_injections:
            injection_attempted += 1
            if injection.get("hit"):
                injection_id = str(injection["id"])
                injection_hits.append(injection_id)
                hits.append(injection_id)
        injection_by_tensor[tensor_id] = sorted(set(hits))
    injection_hits = sorted(set(injection_hits))

    verifications = [column["verification"] for column in column_reports]
    reference_summary = {
        "checked_columns": len(verifications),
        "passed_columns": sum(1 for item in verifications if item["passed"]),
        "failed_columns": sum(1 for item in verifications if not item["passed"]),
        "quantile_max_abs_delta": max(
            (float(item["quantile_max_abs_delta"]) for item in verifications),
            default=0.0,
        ),
        "scale_max_abs_delta": max(
            (float(item["scale_abs_delta"]) for item in verifications),
            default=0.0,
        ),
        "code_mismatch_count": sum(
            int(item["code_mismatch_count"]) for item in verifications
        ),
        "dequant_max_abs_delta": max(
            (float(item["dequant_max_abs_delta"]) for item in verifications),
            default=0.0,
        ),
        "abs_error_max_abs_delta": max(
            (float(item["abs_error_max_abs_delta"]) for item in verifications),
            default=0.0,
        ),
        "tolerance_branch_mismatch_count": sum(
            int(item["tolerance_branch_mismatch_count"])
            for item in verifications
        ),
        "passed": all(item["passed"] for item in verifications),
    }
    failures_by_code: dict[str, int] = {}
    for failure in failures:
        code = str(failure["error_code"])
        failures_by_code[code] = failures_by_code.get(code, 0) + 1

    summary = {
        "tensor_count_loaded": len(sample_result.tensors),
        "tensor_count_configured": len(config.tensors),
        "column_count_configured": len(plans),
        "column_count_completed": len(column_reports),
        "element_count_completed": element_count,
        "analyzed_element_count": analyzed_count,
        "missing_element_count": missing_count,
        "saturated_element_count": saturated_count,
        "saturation_ratio": saturated_count / analyzed_count if analyzed_count else 0.0,
        "over_tolerance_element_count": over_tolerance_count,
        "over_tolerance_ratio": (
            over_tolerance_count / analyzed_count if analyzed_count else 0.0
        ),
        "zero_reference_count": zero_reference_count,
        "max_abs_error": max_abs_error,
        "max_effective_relative_error": max_effective_relative_error,
        "attribution_share_by_source": all_shares,
        "over_tolerance_attribution_share_by_source": over_shares,
        "dominant_source": dominant_source,
        "dominant_source_counts": dominant_counts,
        "attribution_share_error_max": attribution_share_error_max,
        "attribution_share_violation_count": attribution_share_violation_count,
        "attribution_share_sum_within_tolerance": (
            attribution_share_violation_count == 0
        ),
        "sensitivity_violation_count": len(sensitivity["violations"]),
        "failure_count": len(failures),
        "failures_by_code": dict(sorted(failures_by_code.items())),
        "budget_pending_count": len(pending_plans),
        "float64_reference": reference_summary,
    }
    metadata = {
        "quantile_interpolation": config.interpolation,
        "tie_break": config.tie_break,
        "sort_order": "ascending_stable",
        "scale_quantization": config.scale_quantization,
        "code_rounding": "round_half_away_from_zero",
        "tolerance_interpretation": "relative error, absolute branch at true value zero",
        "attribution_priority": list(ATTRIBUTION_PRIORITY),
        "attribution_sources": list(ATTRIBUTION_SOURCES),
        "attribution_share_normalization": (
            "four causal magnitudes are normalized to each element's absolute error; "
            "the largest raw component determines the dominant source"
        ),
        "float_reference": "float64",
        "engine_version": ENGINE_VERSION,
    }
    normalized_config = config.to_dict()
    normalized_config["config_hash"] = config.config_hash
    sample_metadata = {
        "sample_id": sample_result.sample_id,
        "sample_hash": sample_result.sample_hash,
        "normalized": sample_result.normalized,
    }
    deterministic_core = {
        "schema_version": REPORT_SCHEMA_VERSION,
        "report_type": "per_column_quantization_diagnostics",
        "metadata": metadata,
        "configuration": normalized_config,
        "sample_metadata": sample_metadata,
        "injection_hits": {
            "attempted_count": injection_attempted,
            "hit_count": len(injection_hits),
            "ids": injection_hits,
            "by_tensor": {
                key: injection_by_tensor[key] for key in sorted(injection_by_tensor)
            },
        },
        "columns": column_reports,
        "sensitivity": sensitivity,
        "failures": failures,
        "summary": summary,
    }
    deterministic_hash = sha256_json(deterministic_core)
    report = {
        "schema_version": REPORT_SCHEMA_VERSION,
        "report_type": "per_column_quantization_diagnostics",
        "run": {
            "mode": mode,
            "run_version": run_version,
            "config_hash": config.config_hash,
            "sample_hash": sample_result.sample_hash,
            "sample_id": sample_result.sample_id,
            "resume_status": resume_status,
            "previous_run_version": previous_run_version,
            "max_workers_configured": config.max_workers,
            "max_workers_observed": max_workers_observed,
            "batch_count": batch_count,
            "snapshot_commit_interval_batches": config.snapshot_commit_interval_batches,
            "snapshot_generation": snapshot_generation,
            "complete": len(pending_plans) == 0,
            "environment": {
                "python": platform.python_version(),
                "numpy": np.__version__,
                "platform": platform.system(),
            },
        },
        "metadata": metadata,
        "configuration": normalized_config,
        "sample_metadata": sample_metadata,
        "injection_hits": deterministic_core["injection_hits"],
        "columns": column_reports,
        "sensitivity": sensitivity,
        "failures": failures,
        "summary": summary,
        "cache": cache_status,
        "timings_ms": timings_ms,
        "equivalence": {
            "deterministic_result_hash": deterministic_hash,
            "excluded_from_deterministic_core": [
                "run.mode",
                "run.max_workers_observed",
                "run.snapshot_generation",
                "cache.execution_provenance",
                "timings_ms",
            ],
        },
    }
    return report


def run_workbench(
    *,
    config_path: str | Path,
    sample_path: str | Path,
    state_dir: str | Path,
    report_path: str | Path,
    full_rebuild: bool = False,
) -> dict[str, Any]:
    total_started = time.perf_counter()
    validation_started = time.perf_counter()
    config_result = load_config(config_path)
    config = config_result.config
    config_failures = list(config_result.failures)
    if config is None:
        config = WorkbenchConfig(
            config_id="invalid-config",
            interpolation="linear",
            tie_break="stable_ascending_first_occurrence",
            scale_quantization="round_half_away_from_zero",
            allowed_bit_widths=(4, 8, 12),
            max_workers=1,
            calibration_element_budget=1,
            snapshot_commit_interval_batches=1,
            tensors=(),
            config_hash=sha256_json({"invalid_config": True}),
        )
    sample_result = load_sample(sample_path, config.tensors)
    validation_ms = _elapsed_ms(validation_started)

    store = SnapshotStore(state_dir)
    loaded_snapshot, previous_generation = store.load()
    previous_run_version = loaded_snapshot.get("run_version")
    resume_status: str
    if full_rebuild:
        previous_snapshot_columns: dict[str, Any] = {}
        resume_status = "full_rebuild"
    else:
        previous_snapshot_columns = loaded_snapshot.get("columns", {})
        if not isinstance(previous_snapshot_columns, dict):
            previous_snapshot_columns = {}
        if previous_run_version:
            resume_status = f"resume_from_generation_{previous_generation}"
        else:
            resume_status = "new_workspace"

    run_version = sha256_json(
        {
            "engine_version": ENGINE_VERSION,
            "config_hash": config.config_hash,
            "sample_hash": sample_result.sample_hash,
        }
    )
    if not full_rebuild and previous_run_version == run_version:
        resume_status = "version_match_reuse"

    plans: list[ColumnPlan] = []
    for tensor in config.tensors:
        sample = sample_result.tensors.get(tensor.tensor_id)
        if sample is None:
            continue
        for column in range(tensor.column_count):
            plans.append(
                _make_plan(
                    config=config,
                    tensor=tensor,
                    sample=sample,
                    column=column,
                    previous_snapshot_columns=previous_snapshot_columns,
                )
            )
    plans.sort(key=lambda item: item.column_id)

    records: dict[str, dict[str, Any]] = {}
    if not full_rebuild:
        valid_column_ids = {plan.column_id for plan in plans}
        records = {
            column_id: record
            for column_id, record in previous_snapshot_columns.items()
            if column_id in valid_column_ids and isinstance(record, dict)
        }

    reused_columns: list[str] = []
    compute_plans: list[ColumnPlan] = []
    invalidated_columns: list[str] = []
    execution_failures: list[dict[str, Any]] = []
    for plan in plans:
        if plan.blocking_failure is not None:
            records.pop(plan.column_id, None)
            invalidated_columns.append(plan.column_id)
            continue
        if plan.cached_complete:
            reused_columns.append(plan.column_id)
            continue
        compute_plans.append(plan)
        if plan.previous_record is not None:
            invalidated_columns.append(plan.column_id)
        else:
            invalidated_columns.append(plan.column_id)

    batches, pending_plans = _build_batches(
        compute_plans, config.calibration_element_budget
    )
    execution_status: dict[str, Any] = {}
    for plan in plans:
        if plan.cached_complete:
            execution_status[plan.column_id] = {
                "calibration": "reused",
                "quant_params": {
                    str(width): "reused"
                    for width in sorted(plan.tensor.bit_width_candidates)
                },
                "requantize_errors": {
                    str(width): "reused"
                    for width in sorted(plan.tensor.bit_width_candidates)
                },
                "any_computed": False,
            }
    recomputed_columns: list[str] = []
    max_workers_observed = 0
    worker_stage_timings = {
        "calibration_ms": 0.0,
        "quant_params_ms": 0.0,
        "requantize_errors_ms": 0.0,
    }
    lock = threading.Lock()
    active_workers = 0
    worker_timings_total = 0.0
    scheduler_started = time.perf_counter()
    batch_count = 0
    snapshot_time_ms = 0.0
    commit_interval = max(1, config.snapshot_commit_interval_batches)

    def snapshot_payload(complete: bool) -> dict[str, Any]:
        payload = empty_snapshot()
        payload.update(
            {
                "columns": {key: records[key] for key in sorted(records)},
                "completed_columns": _complete_column_ids(plans, records),
                "pending_columns": sorted(plan.column_id for plan in pending_plans),
                "artifact_versions": _artifact_versions(plans, records),
                "failures": _collect_failures(
                    config=config,
                    config_failures=config_failures,
                    sample_failures=sample_result.failures,
                    execution_failures=execution_failures,
                    plans=plans,
                    records=records,
                    pending_plans=pending_plans,
                ),
                "complete": complete,
            }
        )
        return payload

    with ThreadPoolExecutor(max_workers=config.max_workers) as executor:
        def tracked_compute(plan: ColumnPlan) -> ColumnExecution:
            nonlocal active_workers, max_workers_observed
            with lock:
                active_workers += 1
                max_workers_observed = max(max_workers_observed, active_workers)
            try:
                return _compute_column(plan, config)
            finally:
                with lock:
                    active_workers -= 1

        for batch_index, batch in enumerate(batches, start=1):
            batch_count += 1
            batch_started = time.perf_counter()
            futures = []
            for plan in sorted(batch, key=lambda item: item.sort_key):
                futures.append((plan, executor.submit(tracked_compute, plan)))
            for plan, future in futures:
                try:
                    execution = future.result()
                except Exception as exc:  # pragma: no cover - defensive boundary
                    execution_failures.append(
                        make_failure(
                            "E_COLUMN_PROCESSING",
                            "column",
                            f"unexpected column processing error: {type(exc).__name__}: {exc}",
                            tensor_id=plan.tensor.tensor_id,
                            column=plan.column,
                            stage="calibration+requantization",
                        )
                    )
                    records.pop(plan.column_id, None)
                    recomputed_columns.append(plan.column_id)
                    continue
                worker_timings_total += execution.timings_ms.get("column_total", 0.0)
                for timing_key in worker_stage_timings:
                    worker_stage_timings[timing_key] += execution.timings_ms.get(
                        timing_key, 0.0
                    )
                execution_status[plan.column_id] = execution.status
                if execution.record is not None:
                    records[plan.column_id] = execution.record
                else:
                    records.pop(plan.column_id, None)
                execution_failures.extend(execution.failures)
                recomputed_columns.append(plan.column_id)
            _ = _elapsed_ms(batch_started)
            if batch_index % commit_interval == 0 and batch_index < len(batches):
                commit_started = time.perf_counter()
                store.commit(
                    snapshot_payload(False),
                    run_version=run_version,
                    config_hash=config.config_hash,
                    sample_hash=sample_result.sample_hash,
                    complete=False,
                )
                snapshot_time_ms += _elapsed_ms(commit_started)
    scheduler_ms = _elapsed_ms(scheduler_started)

    # Remove stale records from columns that have no current successful result.
    valid_completed = set(_complete_column_ids(plans, records))
    records = {
        column_id: record
        for column_id, record in records.items()
        if column_id in valid_completed
    }
    invalidated_columns = sorted(set(invalidated_columns))
    recomputed_columns = sorted(set(recomputed_columns))
    reused_columns = sorted(set(reused_columns))
    pending_ids = sorted(plan.column_id for plan in pending_plans)

    stage_computed_counts = {
        "calibration": 0,
        "quant_params": 0,
        "requantize_errors": 0,
    }
    stage_reused_counts = {
        "calibration": 0,
        "quant_params": 0,
        "requantize_errors": 0,
    }
    for status in execution_status.values():
        if status["calibration"] == "computed":
            stage_computed_counts["calibration"] += 1
        else:
            stage_reused_counts["calibration"] += 1
        for value in status["quant_params"].values():
            if value == "computed":
                stage_computed_counts["quant_params"] += 1
            else:
                stage_reused_counts["quant_params"] += 1
        for value in status["requantize_errors"].values():
            if value == "computed":
                stage_computed_counts["requantize_errors"] += 1
            else:
                stage_reused_counts["requantize_errors"] += 1

    commit_started = time.perf_counter()
    final_generation = store.commit(
        snapshot_payload(True),
        run_version=run_version,
        config_hash=config.config_hash,
        sample_hash=sample_result.sample_hash,
        complete=not pending_plans,
    )
    snapshot_time_ms += _elapsed_ms(commit_started)
    timings = {
        "validation_ms": validation_ms,
        "scheduler_and_execution_ms": scheduler_ms,
        "column_worker_sum_ms": round(worker_timings_total, 6),
        "calibration_worker_ms": round(worker_stage_timings["calibration_ms"], 6),
        "quant_params_worker_ms": round(worker_stage_timings["quant_params_ms"], 6),
        "requantize_errors_worker_ms": round(
            worker_stage_timings["requantize_errors_ms"], 6
        ),
        "sensitivity_and_report_ms": 0.0,
        "snapshot_ms": round(snapshot_time_ms, 6),
        "total_ms": _elapsed_ms(total_started),
    }
    cache_status = {
        "mode": "full_rebuild" if full_rebuild else "incremental",
        "previous_snapshot_generation": previous_generation,
        "final_snapshot_generation": final_generation,
        "run_version": run_version,
        "reused_columns": reused_columns,
        "recomputed_columns": recomputed_columns,
        "invalidated_columns": invalidated_columns,
        "pending_columns": pending_ids,
        "stage_reused_counts": stage_reused_counts,
        "stage_computed_counts": stage_computed_counts,
        "execution_provenance": {
            key: execution_status[key] for key in sorted(execution_status)
        },
        "artifact_versions": _artifact_versions(plans, records),
    }
    report_started = time.perf_counter()
    report = _render_report(
        config=config,
        sample_result=sample_result,
        config_failures=config_failures,
        sample_failures=sample_result.failures,
        plans=plans,
        records=records,
        execution_failures=execution_failures,
        pending_plans=pending_plans,
        run_version=run_version,
        resume_status=resume_status,
        previous_run_version=previous_run_version,
        mode="full_rebuild" if full_rebuild else "incremental",
        snapshot_generation=final_generation,
        max_workers_observed=max_workers_observed,
        batch_count=batch_count,
        timings_ms=timings,
        cache_status=cache_status,
    )
    report["timings_ms"]["sensitivity_and_report_ms"] = _elapsed_ms(report_started)
    report["timings_ms"]["total_ms"] = _elapsed_ms(total_started)
    output_path = Path(report_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_text_atomic(
        output_path,
        json.dumps(
            report,
            ensure_ascii=False,
            allow_nan=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
    )
    return report
