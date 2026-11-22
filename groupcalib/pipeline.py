"""End-to-end column-group calibration pipeline and CLI."""

from __future__ import annotations

import argparse
import shutil
import threading
import traceback
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Iterable

from .cache import STAGES, StageCache
from .config import CalibrationConfig, load_config
from .errors import (
    CONFIG_CODES,
    DATA_CODES,
    E_CALIB_BUDGET_PENDING,
    E_CONFIG_INVALID,
    E_DATA_INVALID,
    BusinessEvent,
    Failure,
    GroupCalibError,
    failure_sort_key,
)
from .hashing import canonical_json
from .io_utils import atomic_write_json, atomic_write_jsonl, ensure_dir
from .keys import expected_stage_keys
from .plan import BatchApplyResult, EditPlan, apply_batch, load_plan, select_batches
from .quantile import calibrate_group_columns
from .quantize import attribute_group_errors, fixed_rescale_group
from .scheduler import BudgetTask, schedule_budget
from .sensitivity import summarise_group
from .table import FeatureTable, TableLoadResult, parse_feature_table, validate_nonmissing_columns, validate_table_schema
from .version import SCHEMA_VERSION


@dataclass
class PipelineArgs:
    config: str
    table: str
    plan: str
    out: str
    through_batch: str | None = None
    full_rebuild: bool = False


@dataclass
class PairOutcome:
    group_id: str
    bit_width: int
    artifacts: dict[str, dict[str, Any]]
    statuses: list[dict[str, Any]]


@dataclass
class PipelineResult:
    exit_code: int
    report: dict[str, Any]


def _business_event(
    code: str,
    message: str,
    details: dict[str, Any],
) -> BusinessEvent:
    return BusinessEvent(code=code, message=message, sort_key=(code, canonical_json(details)), details=details)


def _pair_sort_key(pair: tuple[str, int]) -> tuple[str, int]:
    return pair


def _eligible_pairs(
    config: CalibrationConfig,
    table: FeatureTable,
    column_groups: dict[str, tuple[str, ...]],
) -> list[tuple[str, int]]:
    pairs: list[tuple[str, int]] = []
    for group_id in sorted(column_groups):
        rows = table.rows_for_group(group_id)
        if not rows:
            continue
        if any(
            not any(row.values.get(column) is not None for row in rows)
            for column in column_groups[group_id]
        ):
            continue
        for bit_width in sorted(set(config.bit_widths)):
            pairs.append((group_id, bit_width))
    return pairs


def _nonmissing_counts(
    table: FeatureTable,
    column_groups: dict[str, tuple[str, ...]],
) -> dict[tuple[str, str], int]:
    counts: dict[tuple[str, str], int] = {}
    for group_id, columns in sorted(column_groups.items()):
        rows = table.rows_for_group(group_id)
        for column in columns:
            counts[(group_id, column)] = sum(
                1 for row in rows if row.values.get(column) is not None
            )
    return counts


def _group_schedule_tasks(
    pairs: Iterable[tuple[str, int]],
    column_groups: dict[str, tuple[str, ...]],
    counts: dict[tuple[str, str], int],
) -> list[BudgetTask]:
    tasks = [
        BudgetTask(
            group_id=group_id,
            column=column,
            bit_width=bit_width,
            cost=counts.get((group_id, column), 0),
        )
        for group_id, bit_width in sorted(set(pairs))
        for column in sorted(column_groups[group_id])
    ]
    return sorted(tasks, key=lambda task: task.sort_key)


def _record_stage_statuses(
    *,
    edit_batch_id: str,
    edit_batch_index: int,
    group_id: str,
    bit_width: int,
    expected_keys: dict[str, str],
    force_recompute: bool,
    cache_hits: dict[str, bool],
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for stage_index, stage in enumerate(STAGES):
        if force_recompute:
            status = "recomputed"
            reason = "affected_group"
        elif cache_hits.get(stage, False):
            status = "reused"
            reason = "cache_hit"
        else:
            status = "recomputed"
            reason = "cache_miss"
        records.append(
            {
                "edit_batch_id": edit_batch_id,
                "edit_batch_index": edit_batch_index,
                "stage_index": stage_index,
                "group_id": group_id,
                "bit_width": bit_width,
                "stage": stage,
                "cache_key": expected_keys[stage],
                "status": status,
                "reason": reason,
            }
        )
    return records


def _process_pair(
    *,
    config: CalibrationConfig,
    table: FeatureTable,
    column_groups: dict[str, tuple[str, ...]],
    cache: StageCache,
    group_id: str,
    bit_width: int,
    force_recompute: bool,
    edit_batch_id: str,
    edit_batch_index: int,
    expected_keys: dict[str, str],
) -> PairOutcome:
    artifacts: dict[str, dict[str, Any]] = {}
    cache_hits: dict[str, bool] = {}

    calibration = None if force_recompute else cache.get("quantile_calibration", expected_keys["quantile_calibration"])
    cache_hits["quantile_calibration"] = calibration is not None
    if calibration is None:
        calibration = calibrate_group_columns(config, table, column_groups, group_id, bit_width)
    artifacts["quantile_calibration"] = calibration

    rescaled = None if force_recompute else cache.get("fixed_rescale", expected_keys["fixed_rescale"])
    cache_hits["fixed_rescale"] = rescaled is not None
    if rescaled is None:
        rescaled = fixed_rescale_group(config, table, column_groups, calibration)
    artifacts["fixed_rescale"] = rescaled

    attributed = None if force_recompute else cache.get("error_attribution", expected_keys["error_attribution"])
    cache_hits["error_attribution"] = attributed is not None
    if attributed is None:
        attributed = attribute_group_errors(config, calibration, rescaled)
    artifacts["error_attribution"] = attributed

    sensitivity = None if force_recompute else cache.get("sensitivity", expected_keys["sensitivity"])
    cache_hits["sensitivity"] = sensitivity is not None
    if sensitivity is None:
        sensitivity = summarise_group(config, attributed)
    artifacts["sensitivity"] = sensitivity

    for stage in STAGES:
        actual = artifacts[stage].get("cache_key")
        if actual != expected_keys[stage]:
            raise GroupCalibError(
                E_DATA_INVALID,
                f"cache key drift in {stage} for group {group_id!r} width {bit_width}",
                {"group_id": group_id, "bit_width": bit_width, "stage": stage},
            )

    statuses = _record_stage_statuses(
        edit_batch_id=edit_batch_id,
        edit_batch_index=edit_batch_index,
        group_id=group_id,
        bit_width=bit_width,
        expected_keys=expected_keys,
        force_recompute=force_recompute,
        cache_hits=cache_hits,
    )
    return PairOutcome(group_id=group_id, bit_width=bit_width, artifacts=artifacts, statuses=statuses)


def _flatten_stage_results(
    results: dict[str, dict[tuple[str, int], dict[str, Any]]],
    stage: str,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for pair in sorted(results[stage], key=_pair_sort_key):
        record = results[stage][pair]
        records.append(record)
    return records


def _write_canonical_outputs(
    out_dir: Path,
    config: CalibrationConfig,
    table: FeatureTable,
    column_groups: dict[str, tuple[str, ...]],
    results: dict[str, dict[tuple[str, int], dict[str, Any]]],
) -> None:
    calibration_records: list[dict[str, Any]] = []
    quantized_records: list[dict[str, Any]] = []
    attribution_records: list[dict[str, Any]] = []
    sensitivity_records: list[dict[str, Any]] = []
    group_records: list[dict[str, Any]] = []
    threshold_records: list[dict[str, Any]] = []

    for group_id, bit_width in sorted(results["sensitivity"]):
        calibration_stage = results["quantile_calibration"][(group_id, bit_width)]
        rescale_stage = results["fixed_rescale"][(group_id, bit_width)]
        attribution_stage = results["error_attribution"][(group_id, bit_width)]
        sensitivity_stage = results["sensitivity"][(group_id, bit_width)]
        for column in sorted(calibration_stage["columns_calibration"]):
            details = calibration_stage["columns_calibration"][column]
            calibration_records.append(
                {
                    "group_id": group_id,
                    "column": column,
                    "bit_width": bit_width,
                    "fractional_bits": calibration_stage["fractional_bits"],
                    "active_bit_width": bit_width == config.active_bit_width,
                    "scheme": config.scheme,
                    "qmin": details["qmin"],
                    "qmax": details["qmax"],
                    "quantile_low_permille": config.quantile_lower_permille,
                    "quantile_high_permille": config.quantile_upper_permille,
                    "interpolation": calibration_stage["interpolation"],
                    "tie_break": calibration_stage["tie_break"],
                    "tie_policy": calibration_stage["tie_policy"],
                    "nonmissing_count": details["nonmissing_count"],
                    "missing_count": details["missing_count"],
                    "observed_min": details["observed_min"],
                    "observed_max": details["observed_max"],
                    "duplicate_value_count": details["duplicate_value_count"],
                    "quantile_low": details["quantile_low"],
                    "quantile_high": details["quantile_high"],
                    "scale_raw": details["scale_raw"],
                    "scale_factor_fixed_units": details["scale_factor_fixed_units"],
                    "scale_factor": details["scale_factor"],
                    "zero_point_unclipped": details["zero_point_unclipped"],
                    "zero_point": details["zero_point"],
                    "zero_point_clipped": details["zero_point_clipped"],
                    "degenerate_range": details["degenerate_range"],
                    "column_content_fingerprint": details["column_content_fingerprint"],
                    "cache_key": calibration_stage["cache_key"],
                }
            )
        for element in rescale_stage["elements"]:
            record = dict(element)
            record["active_bit_width"] = bit_width == config.active_bit_width
            record["clipping_strategy"] = rescale_stage["clipping_strategy"]
            quantized_records.append(record)
        for element in attribution_stage["elements"]:
            record = dict(element)
            record["active_bit_width"] = bit_width == config.active_bit_width
            attribution_records.append(record)
        sensitivity_records.append(dict(sensitivity_stage))
        if sensitivity_stage["threshold_exceeded"]:
            threshold_records.append(
                {
                    "group_id": group_id,
                    "bit_width": bit_width,
                    "outlier_ratio": sensitivity_stage["outlier_ratio"],
                    "outlier_ratio_limit": sensitivity_stage["outlier_ratio_limit"],
                    "layer": sensitivity_stage["layer"],
                    "conclusion": sensitivity_stage["conclusion"],
                }
            )
        group_records.append(
            {
                "group_id": group_id,
                "bit_width": bit_width,
                "active_bit_width": bit_width == config.active_bit_width,
                "columns": list(calibration_stage["columns"]),
                "row_ids": list(calibration_stage["row_ids"]),
                "row_count": len(calibration_stage["row_ids"]),
                "stage_cache_keys": {
                    stage: results[stage][(group_id, bit_width)]["cache_key"] for stage in STAGES
                },
                "stage_parameter_fingerprints": {
                    stage: results[stage][(group_id, bit_width)]["parameter_fingerprint"] for stage in STAGES
                },
                "content_fingerprint": calibration_stage["content_fingerprint"],
                "qmin": calibration_stage["qmin"],
                "qmax": calibration_stage["qmax"],
                "truncation_count": rescale_stage["truncation_count"],
                "truncation_ratio": rescale_stage["truncation_ratio"],
                "outlier_count": sensitivity_stage["outlier_count"],
                "outlier_ratio": sensitivity_stage["outlier_ratio"],
            }
        )

    calibration_records.sort(key=lambda item: (item["group_id"], item["bit_width"], item["column"]))
    quantized_records.sort(
        key=lambda item: (item["group_id"], item["bit_width"], item["row_id"], item["column"])
    )
    attribution_records.sort(
        key=lambda item: (item["group_id"], item["bit_width"], item["row_id"], item["column"])
    )
    sensitivity_records.sort(key=lambda item: (item["group_id"], item["bit_width"]))
    group_records.sort(key=lambda item: (item["group_id"], item["bit_width"]))
    threshold_records.sort(key=lambda item: (item["group_id"], item["bit_width"]))

    atomic_write_jsonl(out_dir / "calibration.jsonl", calibration_records)
    atomic_write_jsonl(out_dir / "quantized.jsonl", quantized_records)
    atomic_write_jsonl(out_dir / "attribution.jsonl", attribution_records)
    atomic_write_jsonl(out_dir / "sensitivity.jsonl", sensitivity_records)
    atomic_write_jsonl(out_dir / "groups.jsonl", group_records)
    atomic_write_jsonl(out_dir / "threshold_exceedances.jsonl", threshold_records)

    effective_rows = []
    for row in sorted(table.rows, key=lambda item: item.row_id):
        values = {column: row.values.get(column) for column in table.columns}
        effective_rows.append({"row_id": row.row_id, "group_id": row.group_id, **values})
    atomic_write_jsonl(out_dir / "effective_table.jsonl", effective_rows)
    atomic_write_json(out_dir / "effective_groups.json", {gid: list(cols) for gid, cols in sorted(column_groups.items())})


def _report_payload(
    *,
    status: str,
    exit_code: int,
    config: CalibrationConfig | None,
    table: FeatureTable | None,
    plan: EditPlan | None,
    failures: list[Failure],
    events: list[BusinessEvent],
    stage_versions: dict[str, str],
    completed_groups: list[str],
    completed_group_widths: list[dict[str, Any]],
    consumed_budget: int,
    ledger: list[dict[str, Any]],
    reuse_records: list[dict[str, Any]],
    batches: list[dict[str, Any]],
    peak_concurrency: int,
    initial_group_hash: str | None,
    final_group_hash: str | None,
    thresholds: list[dict[str, Any]],
) -> dict[str, Any]:
    failure_codes = Counter(failure.code for failure in failures)
    event_codes = Counter(event.code for event in events)
    reused = sum(1 for record in reuse_records if record["status"] == "reused")
    recomputed = sum(1 for record in reuse_records if record["status"] == "recomputed")
    reused_groups = sorted(
        {
            record["group_id"]
            for record in reuse_records
            if record["status"] == "reused" and record["stage"] == "sensitivity"
        }
    )
    recomputed_groups = sorted(
        {
            record["group_id"]
            for record in reuse_records
            if record["status"] == "recomputed"
        }
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "status": status,
        "exit_code": exit_code,
        "config_hash": config.config_hash if config else None,
        "table_hash": table.table_hash() if table else None,
        "plan_hash": plan.plan_hash if plan else None,
        "initial_group_partition_hash": initial_group_hash,
        "final_group_partition_hash": final_group_hash,
        "active_bit_width": config.active_bit_width if config else None,
        "warning_codes": sorted(code for code in CONFIG_CODES if code in failure_codes),
        "business_code_counts": dict(sorted(failure_codes.items())),
        "business_event_counts": dict(sorted(event_codes.items())),
        "failure_count": len(failures),
        "event_count": len(events),
        "completed_groups": completed_groups,
        "completed_group_widths": completed_group_widths,
        "stage_versions": dict(sorted(stage_versions.items())),
        "consumed_budget": consumed_budget,
        "budget_cycle_count": len(ledger),
        "budget_cycle_events": [
            {
                "edit_batch_id": cycle["edit_batch_id"],
                "cycle_index": cycle["cycle_index"],
                "consumed": cycle["consumed"],
                "pending_count": cycle["pending_count"],
                "event_code": cycle.get("event_code"),
            }
            for cycle in ledger
        ],
        "reused_stage_count": reused,
        "recomputed_stage_count": recomputed,
        "reused_groups": reused_groups,
        "recomputed_groups": recomputed_groups,
        "batch_summaries": batches,
        "threshold_exceedances": thresholds,
        "thread_pool_size": config.thread_pool_size if config else None,
        "peak_concurrency": peak_concurrency,
    }


def _write_failure_only_outputs(
    out_dir: Path,
    failures: list[Failure],
    events: list[BusinessEvent],
    *,
    exit_code: int,
    config: CalibrationConfig | None,
    table: FeatureTable | None,
    plan: EditPlan | None,
) -> dict[str, Any]:
    ensure_dir(out_dir)
    stale_products = [
        "manifest.json",
        "calibration.jsonl",
        "quantized.jsonl",
        "attribution.jsonl",
        "sensitivity.jsonl",
        "groups.jsonl",
        "threshold_exceedances.jsonl",
        "effective_table.jsonl",
        "effective_groups.json",
        "reuse_recompute.jsonl",
        "canonical_reuse_plan.jsonl",
        "cache_operations.jsonl",
        "batch_ledger.jsonl",
        "batch_history.jsonl",
    ]
    for name in stale_products:
        path = out_dir / name
        if path.exists():
            path.unlink()
    stages_dir = out_dir / "stages"
    if stages_dir.exists():
        shutil.rmtree(stages_dir)
    failure_records = [failure.as_record() for failure in sorted(failures, key=lambda item: item.sort_key)]
    event_records = [event.as_record() for event in events]
    atomic_write_jsonl(out_dir / "failures.jsonl", failure_records)
    atomic_write_jsonl(out_dir / "business_events.jsonl", event_records)
    report = _report_payload(
        status="failed",
        exit_code=exit_code,
        config=config,
        table=table,
        plan=plan,
        failures=failures,
        events=events,
        stage_versions={},
        completed_groups=[],
        completed_group_widths=[],
        consumed_budget=0,
        ledger=[],
        reuse_records=[],
        batches=[],
        peak_concurrency=0,
        initial_group_hash=None,
        final_group_hash=None,
        thresholds=[],
    )
    report["diagnostic_detail"] = "Validation failed before a complete stage snapshot could be committed."
    atomic_write_json(out_dir / "diagnostics.json", report)
    atomic_write_json(out_dir / "run_report.json", report)
    return report


def run_pipeline(args: PipelineArgs) -> PipelineResult:
    out_dir = ensure_dir(args.out)
    failures: list[Failure] = []
    events: list[BusinessEvent] = []
    ledger: list[dict[str, Any]] = []
    reuse_records: list[dict[str, Any]] = []
    canonical_reuse_records: list[dict[str, Any]] = []
    batch_summaries: list[dict[str, Any]] = []
    consumed_budget = 0

    config_result = load_config(args.config)
    if config_result.config is None or config_result.failures:
        failures.extend(config_result.failures)
        exit_code = 2 if any(failure.code in CONFIG_CODES for failure in failures) else 3
        report = _write_failure_only_outputs(
            out_dir,
            failures,
            events,
            exit_code=exit_code,
            config=config_result.config,
            table=None,
            plan=None,
        )
        return PipelineResult(exit_code, report)
    config = config_result.config

    stage_cache = StageCache(out_dir)
    stage_cache.load(config_hash=config.config_hash, full_rebuild=args.full_rebuild)
    events.extend(stage_cache.events)

    table_result: TableLoadResult = parse_feature_table(args.table, config)
    table = table_result.table
    failures.extend(table_result.failures)
    if table.rows:
        schema_failures = validate_table_schema(config, table)
    else:
        message = "feature table contains no valid rows"
        schema_failures = [
            Failure(
                code=E_DATA_INVALID,
                message=message,
                sort_key=failure_sort_key(source="table", code=E_DATA_INVALID, field="rows", message=message),
                details={"field": "rows", "tag": "table"},
            )
        ]
    failures.extend(schema_failures)

    plan_result = load_plan(args.plan)
    if plan_result.plan is None or plan_result.failures:
        failures.extend(plan_result.failures)
        exit_code = 2 if any(failure.code in CONFIG_CODES for failure in failures) else 3
        report = _write_failure_only_outputs(
            out_dir,
            failures,
            events,
            exit_code=exit_code,
            config=config,
            table=table,
            plan=plan_result.plan,
        )
        return PipelineResult(exit_code, report)
    plan = plan_result.plan

    try:
        selected_batches = select_batches(plan, args.through_batch)
    except KeyError:
        message = f"through_batch {args.through_batch!r} does not exist in the edit plan"
        failures.append(
            Failure(
                code=E_CONFIG_INVALID,
                message=message,
                sort_key=failure_sort_key(source="plan", code=E_CONFIG_INVALID, field="through_batch", message=message),
                details={"field": "through_batch", "tag": "plan"},
            )
        )
        report = _write_failure_only_outputs(
            out_dir,
            failures,
            events,
            exit_code=2,
            config=config,
            table=table,
            plan=plan,
        )
        return PipelineResult(2, report)

    fatal_schema_failure = any(failure.code == E_CONFIG_INVALID for failure in failures)
    if fatal_schema_failure or not table.rows:
        exit_code = 2 if fatal_schema_failure else 3
        report = _write_failure_only_outputs(
            out_dir,
            failures,
            events,
            exit_code=exit_code,
            config=config,
            table=table,
            plan=plan,
        )
        return PipelineResult(exit_code, report)

    initial_groups = {group_id: tuple(columns) for group_id, columns in config.column_groups.items()}
    initial_group_hash = table.group_partition_hash(initial_groups)
    current_table = table
    current_groups = initial_groups
    results: dict[str, dict[tuple[str, int], dict[str, Any]]] = {stage: {} for stage in STAGES}
    completed_groups: set[str] = set()
    completed_pairs: set[tuple[str, int]] = set()
    thresholds: list[dict[str, Any]] = []
    peak_concurrency = 0
    execute_batches: list[tuple[int, str, Iterable[dict[str, Any]], set[str]]] = []

    if selected_batches:
        for batch_index, batch in enumerate(selected_batches):
            execute_batches.append((batch_index, batch.batch_id, batch.actions, set()))
    else:
        execute_batches.append((0, "initial", (), set()))

    for batch_index, batch_id, actions, _placeholder in execute_batches:
        if selected_batches:
            batch = selected_batches[batch_index]
            applied: BatchApplyResult = apply_batch(
                current_table,
                current_groups,
                config,
                batch,
                batch_index,
            )
            current_table = applied.table
            current_groups = applied.column_groups
            failures.extend(applied.failures)
            affected = set(applied.affected_groups)
            effective_config = replace(config, column_groups=current_groups)
            nonmissing_failures = validate_nonmissing_columns(effective_config, current_table)
            failures.extend(nonmissing_failures)
        else:
            affected = set()
            effective_config = replace(config, column_groups=current_groups)
            nonmissing_failures = validate_nonmissing_columns(effective_config, current_table)
            failures.extend(nonmissing_failures)

        current_pairs = _eligible_pairs(config, current_table, current_groups)
        expected_keys_by_pair = {
            pair: expected_stage_keys(config, current_table, pair[0], current_groups[pair[0]], pair[1])
            for pair in current_pairs
        }
        compute_pairs: set[tuple[str, int]] = set()
        for pair in current_pairs:
            force = pair[0] in affected
            keys = expected_keys_by_pair[pair]
            all_cached = all(
                stage_cache.get(stage, keys[stage]) is not None for stage in STAGES
            )
            if force or not all_cached:
                compute_pairs.add(pair)

        for pair in sorted(current_pairs):
            expected_keys = expected_keys_by_pair[pair]
            for stage_index, stage in enumerate(STAGES):
                if batch_index == 0:
                    status = "recomputed"
                    reason = "initial_materialization"
                elif pair[0] in affected:
                    status = "recomputed"
                    reason = "affected_group"
                else:
                    status = "reused"
                    reason = "unaffected_group"
                canonical_reuse_records.append(
                    {
                        "edit_batch_id": batch_id,
                        "edit_batch_index": batch_index,
                        "stage_index": stage_index,
                        "group_id": pair[0],
                        "bit_width": pair[1],
                        "stage": stage,
                        "cache_key": expected_keys[stage],
                        "status": status,
                        "reason": reason,
                    }
                )

        counts = _nonmissing_counts(current_table, current_groups)
        tasks = _group_schedule_tasks(compute_pairs, current_groups, counts)
        schedule = schedule_budget(tasks, budget_per_cycle=config.element_budget_per_batch)
        for cycle in schedule.cycles:
            cycle_record = {
                "edit_batch_id": batch_id,
                "edit_batch_index": batch_index,
                **cycle,
            }
            ledger.append(cycle_record)
            consumed_budget += cycle["consumed"]
            if cycle.get("event_code") == E_CALIB_BUDGET_PENDING:
                events.append(
                    _business_event(
                        E_CALIB_BUDGET_PENDING,
                        "Calibration element budget exhausted; pending queue retained",
                        {
                            "edit_batch_id": batch_id,
                            "edit_batch_index": batch_index,
                            "cycle_index": cycle["cycle_index"],
                            "pending_count": cycle["pending_count"],
                            "pending": cycle["pending"],
                        },
                    )
                )

        if schedule.pending:
            pending_pairs = {(task.group_id, task.bit_width) for task in schedule.pending}
            for pair in sorted(pending_pairs):
                message = f"calibration budget remains pending for group {pair[0]!r} width {pair[1]}"
                failures.append(
                    Failure(
                        code=E_DATA_INVALID,
                        message=message,
                        sort_key=failure_sort_key(
                            source="budget",
                            batch_index=batch_index,
                            code=E_DATA_INVALID,
                            field="pending",
                            message=message,
                        ),
                        details={
                            "field": "pending",
                            "tag": "budget",
                            "group_id": pair[0],
                            "bit_width": pair[1],
                        },
                    )
                )
            current_pairs = [pair for pair in current_pairs if pair not in pending_pairs]

        concurrency_lock = threading.Lock()
        active_tasks = 0

        def wrapped_process(pair: tuple[str, int]) -> PairOutcome:
            nonlocal active_tasks, peak_concurrency
            with concurrency_lock:
                active_tasks += 1
                peak_concurrency = max(peak_concurrency, active_tasks)
            try:
                return _process_pair(
                    config=config,
                    table=current_table,
                    column_groups=current_groups,
                    cache=stage_cache,
                    group_id=pair[0],
                    bit_width=pair[1],
                    force_recompute=pair[0] in affected,
                    edit_batch_id=batch_id,
                    edit_batch_index=batch_index,
                    expected_keys=expected_keys_by_pair[pair],
                )
            finally:
                with concurrency_lock:
                    active_tasks -= 1

        pair_outcomes: list[PairOutcome] = []
        if current_pairs:
            with ThreadPoolExecutor(max_workers=config.thread_pool_size) as executor:
                futures = [executor.submit(wrapped_process, pair) for pair in sorted(current_pairs)]
                for future in futures:
                    try:
                        pair_outcomes.append(future.result())
                    except Exception as exc:  # noqa: BLE001 - retain a diagnostic and continue other groups
                        message = f"stage computation failed: {type(exc).__name__}: {exc}"
                        failures.append(
                            Failure(
                                code=E_DATA_INVALID,
                                message=message,
                                sort_key=failure_sort_key(
                                    source="stage",
                                    batch_index=batch_index,
                                    code=E_DATA_INVALID,
                                    field="compute",
                                    message=message,
                                ),
                                details={
                                    "field": "compute",
                                    "tag": "stage",
                                    "edit_batch_id": batch_id,
                                    "traceback": traceback.format_exc(limit=4),
                                },
                            )
                        )

        pair_outcomes.sort(key=lambda outcome: (outcome.group_id, outcome.bit_width))
        new_results: dict[str, dict[tuple[str, int], dict[str, Any]]] = {stage: {} for stage in STAGES}
        for outcome in pair_outcomes:
            pair = (outcome.group_id, outcome.bit_width)
            for stage in STAGES:
                artifact = outcome.artifacts[stage]
                new_results[stage][pair] = artifact
                stage_cache.put(stage, artifact)
            reuse_records.extend(outcome.statuses)
        results = new_results
        completed_pairs = set(results["sensitivity"])
        completed_groups = sorted({group_id for group_id, _ in completed_pairs})

        batches_summary = {
            "edit_batch_id": batch_id,
            "edit_batch_index": batch_index,
            "affected_groups": sorted(affected),
            "eligible_pairs": len(current_pairs),
            "computed_pairs": len(compute_pairs),
            "budget_cycles": len(schedule.cycles),
            "consumed_budget": schedule.consumed,
            "pending_count": len(schedule.pending),
        }
        batch_summaries.append(batches_summary)

        for stage in STAGES:
            stage_records = [
                results[stage][pair] for pair in sorted(results[stage], key=_pair_sort_key)
            ]
            stage_cache.commit_stage(stage, stage_records)

    final_group_hash = current_table.group_partition_hash(current_groups)
    stage_versions = dict(stage_cache.loaded_stage_versions)
    _write_canonical_outputs(out_dir, config, current_table, current_groups, results)

    sensitivity_records = _flatten_stage_results(results, "sensitivity")
    thresholds = [
        {
            "group_id": record["group_id"],
            "bit_width": record["bit_width"],
            "outlier_ratio": record["outlier_ratio"],
            "outlier_ratio_limit": record["outlier_ratio_limit"],
            "layer": record["layer"],
            "conclusion": record["conclusion"],
        }
        for record in sensitivity_records
        if record["threshold_exceeded"]
    ]
    thresholds.sort(key=lambda record: (record["group_id"], record["bit_width"]))
    atomic_write_jsonl(out_dir / "threshold_exceedances.jsonl", thresholds)

    reuse_records.sort(
        key=lambda record: (
            record["edit_batch_index"],
            record["group_id"],
            record["bit_width"],
            record["stage_index"],
        )
    )
    canonical_reuse_records.sort(
        key=lambda record: (
            record["edit_batch_index"],
            record["group_id"],
            record["bit_width"],
            record["stage_index"],
        )
    )
    atomic_write_jsonl(out_dir / "canonical_reuse_plan.jsonl", canonical_reuse_records)
    atomic_write_jsonl(out_dir / "reuse_recompute.jsonl", reuse_records)
    ledger.sort(key=lambda cycle: (cycle["edit_batch_index"], cycle["cycle_index"]))
    atomic_write_jsonl(out_dir / "batch_ledger.jsonl", ledger)
    atomic_write_jsonl(out_dir / "batch_history.jsonl", batch_summaries)

    failure_records = [failure.as_record() for failure in sorted(failures, key=lambda item: item.sort_key)]
    event_records = [event.as_record() for event in events]
    atomic_write_jsonl(out_dir / "failures.jsonl", failure_records)
    atomic_write_jsonl(out_dir / "business_events.jsonl", event_records)

    if any(failure.code in CONFIG_CODES for failure in failures):
        exit_code = 2
    elif any(failure.code in DATA_CODES for failure in failures):
        exit_code = 3
    else:
        exit_code = 0

    completed_group_widths = [
        {"group_id": pair[0], "bit_width": pair[1]} for pair in sorted(completed_pairs)
    ]
    manifest = {
        "global_version": SCHEMA_VERSION,
        "config_hash": config.config_hash,
        "plan_hash": plan.plan_hash,
        "through_batch": args.through_batch,
        "table_hash": current_table.table_hash(),
        "initial_group_partition_hash": initial_group_hash,
        "group_partition_hash": final_group_hash,
        "active_bit_width": config.active_bit_width,
        "completed_groups": sorted(completed_groups),
        "completed_group_widths": completed_group_widths,
        "completed_stages": list(STAGES),
        "consumed_budget": consumed_budget,
        "stage_versions": dict(sorted(stage_versions.items())),
        "stage_artifact_counts": {stage: len(results[stage]) for stage in STAGES},
        "element_budget_per_batch": config.element_budget_per_batch,
        "thread_pool_size": config.thread_pool_size,
        "failure_count": len(failures),
        "event_count": len(events),
        "status": "ok" if exit_code == 0 else "failed",
    }
    atomic_write_json(out_dir / "manifest.json", manifest)

    report = _report_payload(
        status="ok" if exit_code == 0 else "failed",
        exit_code=exit_code,
        config=config,
        table=current_table,
        plan=plan,
        failures=failures,
        events=events,
        stage_versions=stage_versions,
        completed_groups=sorted(completed_groups),
        completed_group_widths=completed_group_widths,
        consumed_budget=consumed_budget,
        ledger=ledger,
        reuse_records=reuse_records,
        batches=batch_summaries,
        peak_concurrency=peak_concurrency,
        initial_group_hash=initial_group_hash,
        final_group_hash=final_group_hash,
        thresholds=thresholds,
    )
    atomic_write_json(out_dir / "diagnostics.json", report)
    atomic_write_json(out_dir / "run_report.json", report)
    return PipelineResult(exit_code, report)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Column-group quantile calibration pipeline")
    parser.add_argument("--config", required=True, help="column-group calibration JSON")
    parser.add_argument("--table", required=True, help="feature table JSONL")
    parser.add_argument("--plan", required=True, help="edit plan JSON")
    parser.add_argument("--out", required=True, help="output directory")
    parser.add_argument(
        "--through-batch",
        default=None,
        help="apply batches through this batch_id (inclusive); useful for incremental snapshots",
    )
    parser.add_argument(
        "--full-rebuild",
        action="store_true",
        help="ignore all prior stage snapshots for this run",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    namespace = parser.parse_args(argv)
    result = run_pipeline(
        PipelineArgs(
            config=namespace.config,
            table=namespace.table,
            plan=namespace.plan,
            out=namespace.out,
            through_batch=namespace.through_batch,
            full_rebuild=namespace.full_rebuild,
        )
    )
    summary = {
        "status": result.report.get("status"),
        "exit_code": result.exit_code,
        "completed_groups": result.report.get("completed_groups", []),
        "failure_count": result.report.get("failure_count", 0),
        "business_code_counts": result.report.get("business_code_counts", {}),
        "business_event_counts": result.report.get("business_event_counts", {}),
        "reused_stage_count": result.report.get("reused_stage_count", 0),
        "recomputed_stage_count": result.report.get("recomputed_stage_count", 0),
    }
    print(canonical_json(summary))
    return result.exit_code


if __name__ == "__main__":
    raise SystemExit(main())
