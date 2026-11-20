"""Incremental execution engine, reporting, and recovery semantics."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import os
from pathlib import Path
import time
from typing import Any, Callable

from .config import (
    config_hash,
    executable_config_hash,
    load_config,
    normalise_config,
    validate_config_against_schema,
)
from .errors import (
    E_CONFIG_INVALID,
    E_DATA_ROW_INVALID,
    E_GROUP_MAP_INVALID,
    E_PATCH_BUDGET_PENDING,
    E_PLAN_INVALID,
    WorkbenchError,
    diagnostic,
)
from .impute import model_dependency_fingerprint, run_imputation_slice
from .injection import (
    apply_raw_injections,
    finalize_row_injection_hits,
    load_injections,
    shadow_config_validate,
)
from .loader import (
    _decode_cell,
    data_fingerprint,
    group_coverage_failures,
    group_map_fingerprint,
    load_group_map,
    read_jsonl_drafts,
    validate_drafts,
    validate_group_map_against_config,
)
from .model import assert_model_fresh, train_and_evaluate
from .plan import IMPUTER_OPS, GROUP_OPS, load_plan, plan_hash
from .snapshot import SnapshotStore, clear_state
from .stats import aggregate_stats
from .util import canonical_json, sha256_obj, write_json_atomic


@dataclass
class WorkspacePaths:
    workspace: Path
    state: Path
    config: Path
    data: Path
    group_map: Path
    plan: Path
    injections: Path | None = None

    @classmethod
    def from_workspace(
        cls,
        workspace: Path,
        state: Path | None = None,
        injections: Path | None = None,
    ) -> "WorkspacePaths":
        workspace = Path(workspace)
        return cls(
            workspace=workspace,
            state=Path(state) if state is not None else workspace / "work",
            config=workspace / "imputation.yaml",
            data=workspace / "raw_data.jsonl",
            group_map=workspace / "group_map.json",
            plan=workspace / "edit_plan.json",
            injections=Path(injections) if injections is not None else workspace / "injections.json",
        )


@dataclass
class InputBundle:
    config: dict[str, Any]
    rows: list[dict[str, Any]]
    data_failures: list[dict[str, Any]]
    group_map: dict[str, Any]
    plan: dict[str, Any]
    injection_hits: list[dict[str, Any]]
    config_injection_diagnostics: list[dict[str, Any]]
    group_map_failures: list[dict[str, Any]]
    base_config_hash: str
    base_executable_config_hash: str
    base_data_fingerprint: str
    base_group_map_hash: str
    plan_hash: str


class RunContext:
    def __init__(self) -> None:
        self.started = time.perf_counter()
        self.stats_seconds = 0.0
        self.imputation_seconds = 0.0
        self.model_seconds = 0.0
        self.snapshot_seconds = 0.0
        self.stats_recomputed_columns: set[str] = set()
        self.stats_reused_columns: set[str] = set()
        self.stats_recomputed_groups: set[tuple[str, str]] = set()
        self.stats_reused_groups: set[tuple[str, str]] = set()
        self.computed_cells_by_column: dict[str, int] = {}
        self.budget_slices = 0
        self.pending_events: list[dict[str, Any]] = []
        self.model_recomputed = False
        self.model_reused = False
        self.stale_diagnostics: list[dict[str, Any]] = []
        self.batches_completed_this_run: list[str] = []
        self.snapshot_versions: list[int] = []
        self.commits = 0

    def merge_stats_metrics(self, metrics: dict[str, Any]) -> None:
        self.stats_recomputed_columns.update(metrics.get("recomputed_columns", []))
        self.stats_reused_columns.update(metrics.get("reused_columns", []))
        self.stats_recomputed_groups.update(
            (item["column"], item["group"]) for item in metrics.get("recomputed_groups", [])
        )
        self.stats_reused_groups.update(
            (item["column"], item["group"]) for item in metrics.get("reused_groups", [])
        )

    def merge_imputation_metrics(self, metrics: dict[str, Any]) -> None:
        self.budget_slices += 1
        for column, count in metrics.get("computed_cells_by_column", {}).items():
            self.computed_cells_by_column[column] = (
                self.computed_cells_by_column.get(column, 0) + count
            )
        if metrics.get("pending_count", 0):
            self.pending_events.append(
                {
                    "business_code": E_PATCH_BUDGET_PENDING,
                    "slice": self.budget_slices,
                    "pending_count": metrics["pending_count"],
                }
            )


def _new_state(bundle: InputBundle) -> dict[str, Any]:
    return {
        "state_format": 1,
        "config": deepcopy(bundle.config),
        "config_hash": config_hash(bundle.config),
        "executable_config_hash": executable_config_hash(bundle.config),
        "base_config_hash": bundle.base_config_hash,
        "base_executable_config_hash": bundle.base_executable_config_hash,
        "rows": deepcopy(bundle.rows),
        "data_failures": deepcopy(bundle.data_failures),
        "group_map": deepcopy(bundle.group_map),
        "group_map_hash": bundle.base_group_map_hash,
        "base_group_map_hash": bundle.base_group_map_hash,
        "group_map_failures": deepcopy(bundle.group_map_failures),
        "group_failures": [],
        "operation_failures": [],
        "imputation_failures": [],
        "plan_hash": bundle.plan_hash,
        "base_data_fingerprint": bundle.base_data_fingerprint,
        "data_fingerprint": data_fingerprint(bundle.rows),
        "stats": {},
        "imputations": {},
        "model": None,
        "previous_model": None,
        "pending_jobs": [],
        "completed_batches": [],
        "next_batch_index": 0,
        "in_progress_batch": None,
        "injection_hits": deepcopy(bundle.injection_hits),
        "config_injection_diagnostics": deepcopy(bundle.config_injection_diagnostics),
        "result_fingerprint": None,
    }


def load_input_bundle(paths: WorkspacePaths) -> InputBundle:
    config, config_errors = load_config(paths.config)
    injections_path = paths.injections if paths.injections and Path(paths.injections).exists() else None
    injection_items, injection_load_failures = load_injections(injections_path)
    config_hits, shadow_errors = shadow_config_validate(config, injection_items)

    drafts, parse_failures = read_jsonl_drafts(paths.data)
    drafts, row_hits = apply_raw_injections(drafts, injection_items)
    rows, data_failures, schema_columns = validate_drafts(drafts, config, parse_failures)
    config_errors.extend(validate_config_against_schema(config, schema_columns))
    config_errors.extend(injection_load_failures)
    if config_errors:
        raise WorkbenchError(
            E_CONFIG_INVALID,
            "configuration validation failed",
            {"diagnostics": config_errors},
        )

    group_map, group_map_failures = load_group_map(paths.group_map)
    group_map_failures.extend(validate_group_map_against_config(config, group_map))
    group_failures = group_coverage_failures(rows, config, group_map)
    all_hits = finalize_row_injection_hits(
        sorted(row_hits + config_hits, key=lambda item: item["injection_id"]),
        data_failures + group_failures,
    )
    plan, plan_failures = load_plan(paths.plan)
    if plan_failures:
        raise WorkbenchError(
            E_PLAN_INVALID,
            "edit plan validation failed",
            {"diagnostics": plan_failures},
        )
    base_data_fp = sha256_obj(
        {
            "rows": [
                {"row_id": row["row_id"], "values": row["values"]}
                for row in sorted(rows, key=lambda item: item["row_id"])
            ],
            "data_failures": data_failures,
        }
    )
    return InputBundle(
        config=config,
        rows=rows,
        data_failures=data_failures,
        group_map=group_map,
        plan=plan,
        injection_hits=all_hits,
        config_injection_diagnostics=shadow_errors,
        group_map_failures=group_map_failures,
        base_config_hash=config_hash(config),
        base_executable_config_hash=executable_config_hash(config),
        base_data_fingerprint=base_data_fp,
        base_group_map_hash=group_map_fingerprint(group_map),
        plan_hash=plan_hash(plan),
    )


def _failures_for_state(state: dict[str, Any]) -> list[dict[str, Any]]:
    failures = []
    failures.extend(state.get("data_failures", []))
    failures.extend(state.get("group_map_failures", []))
    failures.extend(state.get("group_failures", []))
    failures.extend(state.get("operation_failures", []))
    failures.extend(state.get("imputation_failures", []))
    for failure in failures:
        failure.setdefault("failed_row", bool(failure.get("row_id")))
    failures.sort(
        key=lambda item: (
            int(item.get("line_no") or 0),
            item.get("row_id", ""),
            item.get("column", ""),
            item.get("stage", ""),
            item.get("business_code", ""),
        )
    )
    return failures


def _failed_row_ids(state: dict[str, Any]) -> set[str]:
    out: set[str] = set()
    rows_by_id = {row["row_id"]: row for row in state.get("rows", [])}
    for failure in _failures_for_state(state):
        if not failure.get("row_id") or failure.get("stage") == "plan":
            continue
        row_id = str(failure["row_id"])
        row = rows_by_id.get(row_id)
        if row is None:
            out.add(row_id)
            continue
        failure_line = failure.get("line_no")
        row_line = row.get("line_no")
        if failure_line is None or row_line is None or failure_line == row_line:
            out.add(row_id)
    return out


def _refresh_group_failures(state: dict[str, Any]) -> None:
    state["group_failures"] = group_coverage_failures(
        state["rows"], state["config"], state["group_map"]
    )


def _valid_ids_for_stats(state: dict[str, Any]) -> set[str]:
    excluded = {
        str(failure["row_id"])
        for failure in state.get("group_failures", []) + state.get("imputation_failures", [])
        if failure.get("row_id")
    }
    return {row["row_id"] for row in state["rows"] if row["row_id"] not in excluded}


def _decode_plan_row(
    raw_row: dict[str, Any],
    config: dict[str, Any],
    existing_values: dict[str, Any] | None = None,
) -> tuple[str, dict[str, Any]]:
    id_column = config["id_column"]
    raw_id = raw_row.get(id_column)
    if isinstance(raw_id, bool) or not isinstance(raw_id, (str, int)) or str(raw_id) == "":
        raise WorkbenchError(E_DATA_ROW_INVALID, "row operation needs a non-empty row identifier")
    row_id = str(raw_id)
    values: dict[str, Any] = dict(existing_values or {})
    for column, spec in config["columns"].items():
        if column == id_column:
            continue
        if column in raw_row:
            value, error = _decode_cell(raw_row[column], spec, column)
            if error is not None:
                error.update({"row_id": row_id, "stage": "plan"})
                raise WorkbenchError(E_DATA_ROW_INVALID, error["message"], {"diagnostics": [error]})
            values[column] = value
        elif existing_values is None:
            values[column] = None
    target = config["target_column"]
    if values.get(target) is not None and values[target] not in (0, 1, 0.0, 1.0):
        raise WorkbenchError(E_DATA_ROW_INVALID, "classification target must be 0 or 1", {"row_id": row_id})
    return row_id, values


def _apply_batch(state: dict[str, Any], batch: dict[str, Any]) -> None:
    config = deepcopy(state["config"])
    rows = deepcopy(state["rows"])
    rows_by_id = {row["row_id"]: row for row in rows}
    group_map = deepcopy(state["group_map"])
    operation_failures: list[dict[str, Any]] = []

    for operation in batch["operations"]:
        op = operation.get("op", operation.get("kind"))
        try:
            if op in {"append_row", "row_append"}:
                raw_row = operation["row"]
                row_id, values = _decode_plan_row(raw_row, config)
                if row_id in rows_by_id:
                    raise WorkbenchError(E_DATA_ROW_INVALID, "duplicate row identifier")
                row = {"row_id": row_id, "line_no": None, "values": values}
                rows.append(row)
                rows_by_id[row_id] = row
            elif op in {"correct_row", "update_row"}:
                row_id = str(operation["row_id"])
                row = rows_by_id.get(row_id)
                if row is None:
                    raise WorkbenchError(E_DATA_ROW_INVALID, "corrected row does not exist")
                _decode_plan_row(
                    {config["id_column"]: row_id, **operation["changes"]},
                    config,
                    existing_values=row["values"],
                )
                for column, raw_cell in operation["changes"].items():
                    if column == config["id_column"]:
                        continue
                    spec = config["columns"].get(column)
                    if spec is None:
                        raise WorkbenchError(E_CONFIG_INVALID, f"correction refers to unknown column {column!r}")
                    value, error = _decode_cell(raw_cell, spec, column)
                    if error is not None:
                        raise WorkbenchError(E_DATA_ROW_INVALID, error["message"], {"diagnostics": [error]})
                    row["values"][column] = value
            elif op in IMPUTER_OPS:
                column = operation["column"]
                spec = config["columns"].get(column)
                if spec is None:
                    raise WorkbenchError(E_CONFIG_INVALID, f"imputer change refers to unknown column {column!r}")
                for key, value in operation["changes"].items():
                    normalized_key = "imputer" if key in {"method", "imputer"} else key
                    spec[normalized_key] = deepcopy(value)
            elif op in GROUP_OPS:
                key_column = operation["key_column"]
                table = group_map.setdefault("tables", {}).setdefault(
                    key_column, {"mapping": {}, "groups": []}
                )
                mapping = table.setdefault("mapping", {})
                for raw_key, raw_group in operation["changes"].items():
                    key = str(raw_key)
                    if raw_group is None:
                        mapping.pop(key, None)
                    elif isinstance(raw_group, str) and raw_group:
                        mapping[key] = raw_group
                    else:
                        raise WorkbenchError(
                            E_GROUP_MAP_INVALID,
                            "group mapping values must be non-empty strings or null",
                        )
                table["mapping"] = dict(sorted(mapping.items()))
                table["groups"] = sorted(set(table["mapping"].values()) | set(table.get("groups", [])))
                group_map["tables"] = dict(sorted(group_map["tables"].items()))
            else:
                raise WorkbenchError(E_PLAN_INVALID, f"unsupported operation {op!r}")
        except WorkbenchError as exc:
            operation_failures.append(
                diagnostic(
                    exc.code,
                    exc.message,
                    row_id=str(operation.get("row_id", "")) or None,
                    column=operation.get("column"),
                    stage="plan",
                    batch_id=batch["batch_id"],
                )
            )

    normalized_config, config_errors = normalise_config(config)
    if config_errors:
        raise WorkbenchError(
            E_CONFIG_INVALID,
            "configuration became invalid while applying a batch",
            {"diagnostics": config_errors, "batch_id": batch["batch_id"]},
        )
    rows.sort(key=lambda row: row["row_id"])
    state["config"] = normalized_config
    state["config_hash"] = config_hash(normalized_config)
    state["executable_config_hash"] = executable_config_hash(normalized_config)
    state["rows"] = rows
    state["group_map"] = group_map
    state["group_map_hash"] = group_map_fingerprint(group_map)
    state["data_fingerprint"] = data_fingerprint(rows)
    state["operation_failures"].extend(operation_failures)
    state["group_map_failures"] = validate_group_map_against_config(state["config"], group_map)
    _refresh_group_failures(state)


def semantic_result(state: dict[str, Any]) -> dict[str, Any]:
    config = deepcopy(state["config"])
    for key in ("thread_pool_workers", "batch_cell_budget", "snapshot_interval_batches"):
        config.pop(key, None)
    model = deepcopy(state.get("model"))
    if isinstance(model, dict):
        model.pop("reused", None)
    return {
        "config": config,
        "rows": deepcopy(state["rows"]),
        "group_map": deepcopy(state["group_map"]),
        "completed_batches": list(state.get("completed_batches", [])),
        "stats": deepcopy(state.get("stats", {})),
        "imputations": deepcopy(state.get("imputations", {})),
        "model": model,
        "failures": _failures_for_state(state),
        "injection_hits": deepcopy(state.get("injection_hits", [])),
    }


def result_fingerprint(state: dict[str, Any]) -> str:
    return sha256_obj(semantic_result(state))


def _report_stats(state: dict[str, Any]) -> dict[str, Any]:
    return {
        column: {
            key: value
            for key, value in record.items()
            if key not in {"referenced_row_ids"}
        }
        for column, record in sorted(state.get("stats", {}).items())
    }


def build_report(
    state: dict[str, Any],
    context: RunContext,
    status: str,
    plan: dict[str, Any],
    last_manifest: dict[str, Any] | None,
) -> dict[str, Any]:
    failures = _failures_for_state(state)
    injection_hits = finalize_row_injection_hits(
        deepcopy(state.get("injection_hits", [])), failures
    )
    business_codes = {failure.get("business_code") for failure in failures if failure.get("business_code")}
    for event in context.pending_events:
        business_codes.add(E_PATCH_BUDGET_PENDING)
    for record in state.get("stats", {}).values():
        for code in record.get("status_codes", []):
            business_codes.add(code)
    business_codes.update(
        item.get("business_code")
        for item in state.get("config_injection_diagnostics", [])
        if item.get("business_code")
    )
    business_codes.update(item.get("business_code") for item in context.stale_diagnostics)
    model = state.get("model")
    model_summary = None
    if isinstance(model, dict):
        model_summary = {
            "status": model.get("status"),
            "imputation_fingerprint": model.get("imputation_fingerprint"),
            "input_fingerprint": model.get("input_fingerprint"),
            "model_fingerprint": model.get("model_fingerprint"),
            "feature_names": model.get("feature_names", []),
            "training_row_ids": model.get("training_row_ids", []),
            "validation_row_ids": model.get("validation_row_ids", []),
            "training_row_count": model.get("training_row_count"),
            "validation_row_count": model.get("validation_row_count"),
            "excluded_row_ids": model.get("excluded_row_ids", []),
            "loss_trajectory": model.get("loss_trajectory", []),
            "accuracy": model.get("accuracy"),
            "log_loss": model.get("log_loss"),
            "reused": model.get("reused", False),
        }
    total_cells = sum(len(column_map) for column_map in state.get("imputations", {}).values())
    report = {
        "format": 1,
        "status": status,
        "business_codes": sorted(code for code in business_codes if code),
        "config_hash": state.get("config_hash"),
        "executable_config_hash": state.get("executable_config_hash"),
        "data_fingerprint": state.get("data_fingerprint"),
        "group_map_hash": state.get("group_map_hash"),
        "plan_hash": state.get("plan_hash"),
        "completed_batches": list(state.get("completed_batches", [])),
        "in_progress_batch": state.get("in_progress_batch"),
        "next_batch_index": state.get("next_batch_index"),
        "planned_batch_count": len(plan.get("batches", [])),
        "completed_batches_this_run": list(context.batches_completed_this_run),
        "stats": {
            "columns": _report_stats(state),
            "sample_counts": {
                column: record.get("sample_count", 0)
                for column, record in sorted(state.get("stats", {}).items())
            },
            "recomputed_columns": sorted(context.stats_recomputed_columns),
            "reused_columns": sorted(context.stats_reused_columns),
            "recomputed_groups": [
                {"column": column, "group": group}
                for column, group in sorted(context.stats_recomputed_groups)
            ],
            "reused_groups": [
                {"column": column, "group": group}
                for column, group in sorted(context.stats_reused_groups)
            ],
        },
        "imputation": {
            "imputed_cell_count": total_cells,
            "computed_cell_count": sum(context.computed_cells_by_column.values()),
            "computed_cells_by_column": dict(sorted(context.computed_cells_by_column.items())),
            "reused_cell_count": max(0, total_cells - sum(context.computed_cells_by_column.values())),
            "pending_count": len(state.get("pending_jobs", [])),
            "pending_jobs": deepcopy(state.get("pending_jobs", [])),
            "budget_slices": context.budget_slices,
            "pending_events": deepcopy(context.pending_events),
            "failures": deepcopy(state.get("imputation_failures", [])),
        },
        "reuse_recompute": {
            "stats": {
                "recomputed_columns": sorted(context.stats_recomputed_columns),
                "reused_columns": sorted(context.stats_reused_columns),
                "recomputed_groups": [
                    {"column": column, "group": group}
                    for column, group in sorted(context.stats_recomputed_groups)
                ],
                "reused_groups": [
                    {"column": column, "group": group}
                    for column, group in sorted(context.stats_reused_groups)
                ],
            },
            "imputation": {
                "computed_cells": sum(context.computed_cells_by_column.values()),
                "reused_cells": max(0, total_cells - sum(context.computed_cells_by_column.values())),
            },
            "model": {
                "recomputed": context.model_recomputed,
                "reused": context.model_reused,
            },
        },
        "model": model_summary,
        "injection_hits": injection_hits,
        "config_injection_diagnostics": deepcopy(state.get("config_injection_diagnostics", [])),
        "failures": failures,
        "failure_count": len(failures),
        "failed_row_ids": sorted(
            {str(failure["row_id"]) for failure in failures if failure.get("row_id")}
        ),
        "data_failure_count": len(state.get("data_failures", [])),
        "group_failure_count": len(state.get("group_failures", [])),
        "imputation_failure_count": len(state.get("imputation_failures", [])),
        "stale_diagnostics": deepcopy(context.stale_diagnostics),
        "timings_seconds": {
            "stats": context.stats_seconds,
            "imputation": context.imputation_seconds,
            "model": context.model_seconds,
            "snapshot": context.snapshot_seconds,
            "total": time.perf_counter() - context.started,
        },
        "snapshot_versions_committed_this_run": list(context.snapshot_versions),
        "snapshot": deepcopy(last_manifest),
    }
    report["result_fingerprint"] = result_fingerprint(state)
    return report


class WorkbenchEngine:
    def __init__(self, paths: WorkspacePaths):
        self.paths = paths
        self.store = SnapshotStore(paths.state)
        self._committed_version = 0

    def _commit(
        self,
        state: dict[str, Any],
        context: RunContext,
        status: str,
        plan: dict[str, Any],
        hard_kill_after_commits: int | None = None,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        state["result_fingerprint"] = result_fingerprint(state)
        report = build_report(state, context, status, plan, None)
        started = time.perf_counter()
        manifest, committed_report = self.store.commit(
            state,
            report,
            expected_parent_version=self._committed_version,
            stage_fingerprints={
                "stats": sha256_obj(state.get("stats", {})),
                "imputation": sha256_obj(state.get("imputations", {})),
                "model": sha256_obj(state.get("model")),
            },
        )
        context.snapshot_seconds += time.perf_counter() - started
        self._committed_version = manifest["global_version"]
        context.snapshot_versions.append(manifest["global_version"])
        context.commits += 1
        committed_report["snapshot"] = manifest
        committed_report["stages"] = manifest.get("stage_versions", {})
        write_json_atomic(self.paths.state / "report.json", committed_report)
        if (
            hard_kill_after_commits is not None
            and context.commits >= hard_kill_after_commits
        ):
            # Simulates an uncatchable process termination after a complete commit.
            os._exit(137)
        return manifest, committed_report

    def _restore_or_initialize(
        self, bundle: InputBundle, fresh: bool
    ) -> tuple[dict[str, Any], dict[str, Any] | None]:
        if fresh:
            clear_state(self.paths.state)
            self.store = SnapshotStore(self.paths.state)
            self._committed_version = 0
        loaded = None if fresh else self.store.load_state()
        if loaded is None:
            self._committed_version = 0
            state = _new_state(bundle)
            return state, None
        state, manifest = loaded
        self._committed_version = int(manifest["global_version"])
        checks = {
            "base_executable_config_hash": bundle.base_executable_config_hash,
            "base_data_fingerprint": bundle.base_data_fingerprint,
            "base_group_map_hash": bundle.base_group_map_hash,
            "plan_hash": bundle.plan_hash,
        }
        for key, expected in checks.items():
            if state.get(key) != expected:
                raise WorkbenchError(
                    E_CONFIG_INVALID,
                    f"base input changed since the snapshot ({key}); run a fresh rebuild",
                    {"field": key, "snapshot_value": state.get(key), "input_value": expected},
                )
        # Execution-only knobs may evolve without invalidating durable semantic
        # products.  Adopt the new values and continue from the persisted queue.
        for key in (
            "thread_pool_workers",
            "batch_cell_budget",
            "snapshot_interval_batches",
        ):
            state["config"][key] = deepcopy(bundle.config[key])
        state["base_config_hash"] = bundle.base_config_hash
        state["base_executable_config_hash"] = bundle.base_executable_config_hash
        state["config_hash"] = config_hash(state["config"])
        state["executable_config_hash"] = executable_config_hash(state["config"])
        return state, manifest

    def run(
        self,
        *,
        fresh: bool = False,
        max_slices: int | None = None,
        max_batches: int | None = None,
        workers_override: int | None = None,
        budget_override: int | None = None,
        stop_requested: Callable[[], bool] | None = None,
        hard_kill_after_commits: int | None = None,
    ) -> dict[str, Any]:
        stop_requested = stop_requested or (lambda: False)
        bundle = load_input_bundle(self.paths)
        state, manifest = self._restore_or_initialize(bundle, fresh)
        plan = bundle.plan
        context = RunContext()
        last_manifest = manifest
        workers = (
            int(workers_override)
            if workers_override is not None
            else int(state["config"]["thread_pool_workers"])
        )
        budget = (
            int(budget_override)
            if budget_override is not None
            else int(state["config"]["batch_cell_budget"])
        )
        snapshot_interval = int(state["config"]["snapshot_interval_batches"])
        if workers <= 0 or budget <= 0:
            raise WorkbenchError(E_CONFIG_INVALID, "worker count and cell budget must be positive")
        slices_started = 0

        while True:
            if stop_requested():
                last_manifest, report = self._commit(
                    state,
                    context,
                    "interrupted",
                    plan,
                    hard_kill_after_commits,
                )
                return {"status": "interrupted", "report": report, "manifest": last_manifest}

            if state.get("in_progress_batch") is None:
                if state["next_batch_index"] >= len(plan["batches"]):
                    # Completion/reuse pass: recompute dependencies from content keys,
                    # reuse all unchanged artifacts, and refresh the model dependency.
                    stats_started = time.perf_counter()
                    stats, stats_metrics = aggregate_stats(
                        state["config"],
                        state.get("stats", {}),
                        state["rows"],
                        _valid_ids_for_stats(state),
                        state["group_map"],
                        workers,
                    )
                    context.stats_seconds += time.perf_counter() - stats_started
                    state["stats"] = stats
                    context.merge_stats_metrics(stats_metrics)

                    impute_started = time.perf_counter()
                    cells, pending, failures, metrics = run_imputation_slice(
                        state.get("imputations", {}),
                        state["config"],
                        state["rows"],
                        _valid_ids_for_stats(state),
                        state["stats"],
                        state["group_map"],
                        state.get("imputation_failures", []),
                        budget,
                        workers,
                    )
                    context.imputation_seconds += time.perf_counter() - impute_started
                    state["imputations"] = cells
                    state["pending_jobs"] = pending
                    state["imputation_failures"] = failures
                    context.merge_imputation_metrics(metrics)
                    slices_started += 1
                    if pending:
                        last_manifest, report = self._commit(
                            state, context, "pending", plan, hard_kill_after_commits
                        )
                        return {"status": "pending", "report": report, "manifest": last_manifest}
                    excluded = _failed_row_ids(state)
                    imputed_fp = model_dependency_fingerprint(
                        state["stats"], state["rows"], state["imputations"], excluded
                    )
                    model_started = time.perf_counter()
                    try:
                        model, metrics, diagnostics = train_and_evaluate(
                            state["config"],
                            state["rows"],
                            state["imputations"],
                            excluded,
                            imputed_fp,
                            state.get("model") or state.get("previous_model"),
                        )
                    except WorkbenchError as exc:
                        state["operation_failures"].append(
                            diagnostic(exc.code, exc.message, stage="model")
                        )
                        last_manifest, report = self._commit(
                            state, context, "error", plan, hard_kill_after_commits
                        )
                        return {"status": "error", "report": report, "manifest": last_manifest}
                    context.model_seconds += time.perf_counter() - model_started
                    state["model"] = model
                    state["previous_model"] = None
                    context.model_reused = bool(metrics.get("reused"))
                    context.model_recomputed = bool(metrics.get("recomputed"))
                    context.stale_diagnostics.extend(diagnostics)
                    assert_model_fresh(state["model"], imputed_fp)
                    last_manifest, report = self._commit(
                        state, context, "complete", plan, hard_kill_after_commits
                    )
                    return {"status": "complete", "report": report, "manifest": last_manifest}

                batch_index = int(state["next_batch_index"])
                batch = plan["batches"][batch_index]
                old_model = state.get("model")
                _apply_batch(state, batch)
                state["previous_model"] = old_model
                state["model"] = None
                state["pending_jobs"] = []
                state["in_progress_batch"] = {
                    "index": batch_index,
                    "batch_id": batch["batch_id"],
                    "applied": True,
                }
                stats_started = time.perf_counter()
                stats, stats_metrics = aggregate_stats(
                    state["config"],
                    state.get("stats", {}),
                    state["rows"],
                    _valid_ids_for_stats(state),
                    state["group_map"],
                    workers,
                )
                context.stats_seconds += time.perf_counter() - stats_started
                state["stats"] = stats
                context.merge_stats_metrics(stats_metrics)
                self._refresh_injection_hits(state)
                # At the configured interval of one, persist the applied edit and
                # fresh statistics before consuming the budget.  With a coarser
                # interval, an uncommitted batch is safely reapplied after a crash.
                if snapshot_interval == 1:
                    last_manifest, _checkpoint = self._commit(
                        state, context, "stats_ready", plan, hard_kill_after_commits
                    )
            else:
                # A resumed partial batch has already applied its edit exactly once.
                stats_started = time.perf_counter()
                stats, stats_metrics = aggregate_stats(
                    state["config"],
                    state.get("stats", {}),
                    state["rows"],
                    _valid_ids_for_stats(state),
                    state["group_map"],
                    workers,
                )
                context.stats_seconds += time.perf_counter() - stats_started
                state["stats"] = stats
                context.merge_stats_metrics(stats_metrics)

            impute_started = time.perf_counter()
            cells, pending, failures, metrics = run_imputation_slice(
                state.get("imputations", {}),
                state["config"],
                state["rows"],
                _valid_ids_for_stats(state),
                state["stats"],
                state["group_map"],
                state.get("imputation_failures", []),
                budget,
                workers,
            )
            context.imputation_seconds += time.perf_counter() - impute_started
            state["imputations"] = cells
            state["pending_jobs"] = pending
            state["imputation_failures"] = failures
            context.merge_imputation_metrics(metrics)
            slices_started += 1

            if pending:
                last_manifest, report = self._commit(
                    state, context, "pending", plan, hard_kill_after_commits
                )
                if max_slices is not None and slices_started >= max_slices:
                    return {"status": "pending", "report": report, "manifest": last_manifest}
                if stop_requested():
                    last_manifest, report = self._commit(
                        state, context, "interrupted", plan, hard_kill_after_commits
                    )
                    return {"status": "interrupted", "report": report, "manifest": last_manifest}
                continue

            excluded = _failed_row_ids(state)
            imputed_fp = model_dependency_fingerprint(
                state["stats"], state["rows"], state["imputations"], excluded
            )
            model_started = time.perf_counter()
            try:
                model, model_metrics, diagnostics = train_and_evaluate(
                    state["config"],
                    state["rows"],
                    state["imputations"],
                    excluded,
                    imputed_fp,
                    state.get("previous_model"),
                )
            except WorkbenchError as exc:
                state["operation_failures"].append(diagnostic(exc.code, exc.message, stage="model"))
                last_manifest, report = self._commit(
                    state, context, "error", plan, hard_kill_after_commits
                )
                return {"status": "error", "report": report, "manifest": last_manifest}
            context.model_seconds += time.perf_counter() - model_started
            state["model"] = model
            state["previous_model"] = None
            context.model_reused = context.model_reused or bool(model_metrics.get("reused"))
            context.model_recomputed = context.model_recomputed or bool(model_metrics.get("recomputed"))
            context.stale_diagnostics.extend(diagnostics)
            assert_model_fresh(state["model"], imputed_fp)

            completed = state["in_progress_batch"]
            state["completed_batches"].append(completed["batch_id"])
            context.batches_completed_this_run.append(completed["batch_id"])
            state["next_batch_index"] = int(completed["index"]) + 1
            state["in_progress_batch"] = None
            state["pending_jobs"] = []
            remaining_batches = state["next_batch_index"] < len(plan["batches"])
            limit_reached = (
                (max_slices is not None and slices_started >= max_slices)
                or (
                    max_batches is not None
                    and len(context.batches_completed_this_run) >= max_batches
                )
            )
            if remaining_batches and stop_requested():
                batch_status = "interrupted"
            elif remaining_batches and limit_reached:
                batch_status = "partial"
            else:
                batch_status = "complete"
            commit_required = (
                not remaining_batches
                or batch_status != "complete"
                or len(state["completed_batches"]) % snapshot_interval == 0
            )
            if commit_required:
                last_manifest, report = self._commit(
                    state, context, batch_status, plan, hard_kill_after_commits
                )
                if not remaining_batches:
                    return {"status": "complete", "report": report, "manifest": last_manifest}
                if batch_status != "complete":
                    return {"status": batch_status, "report": report, "manifest": last_manifest}

    def _refresh_injection_hits(self, state: dict[str, Any]) -> None:
        state["injection_hits"] = finalize_row_injection_hits(
            deepcopy(state.get("injection_hits", [])), _failures_for_state(state)
        )


def compare_semantic_states(left: dict[str, Any], right: dict[str, Any]) -> dict[str, Any]:
    left_semantic = semantic_result(left)
    right_semantic = semantic_result(right)
    left_fp = sha256_obj(left_semantic)
    right_fp = sha256_obj(right_semantic)
    differences: list[dict[str, Any]] = []
    _collect_differences(left_semantic, right_semantic, "$", differences, limit=50)
    return {
        "equal": left_fp == right_fp,
        "left_fingerprint": left_fp,
        "right_fingerprint": right_fp,
        "difference_count": len(differences),
        "differences": differences,
        "left_canonical_bytes": len(canonical_json(left_semantic).encode("utf-8")),
        "right_canonical_bytes": len(canonical_json(right_semantic).encode("utf-8")),
    }


def _collect_differences(
    left: Any,
    right: Any,
    path: str,
    output: list[dict[str, Any]],
    *,
    limit: int,
) -> None:
    if len(output) >= limit:
        return
    if type(left) is not type(right) and not (
        isinstance(left, (int, float))
        and not isinstance(left, bool)
        and isinstance(right, (int, float))
        and not isinstance(right, bool)
    ):
        output.append({"path": path, "left": left, "right": right, "kind": "type"})
        return
    if isinstance(left, dict):
        for key in sorted(set(left) | set(right)):
            if key not in left:
                output.append(
                    {"path": f"{path}.{key}", "left": "<missing>", "right": right[key], "kind": "missing_left"}
                )
            elif key not in right:
                output.append(
                    {"path": f"{path}.{key}", "left": left[key], "right": "<missing>", "kind": "missing_right"}
                )
            else:
                _collect_differences(left[key], right[key], f"{path}.{key}", output, limit=limit)
            if len(output) >= limit:
                return
        return
    if isinstance(left, list):
        if len(left) != len(right):
            output.append(
                {
                    "path": path,
                    "left": f"list length {len(left)}",
                    "right": f"list length {len(right)}",
                    "kind": "length",
                }
            )
        for index, (left_item, right_item) in enumerate(zip(left, right)):
            _collect_differences(
                left_item,
                right_item,
                f"{path}[{index}]",
                output,
                limit=limit,
            )
            if len(output) >= limit:
                return
        return
    if left != right:
        output.append({"path": path, "left": left, "right": right, "kind": "value"})
