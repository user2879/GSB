from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import copy
from decimal import Decimal
import hashlib
import json
from typing import Any, Iterable

from .cost import PlanEstimate, estimate_plan, fixed, SATURATION_RULE
from .errors import LabError, replay_error, schema_error
from .executor import (
    ExecutionRun,
    bias_report,
    compare_execution_results,
    execute_plan,
)
from .graph import CandidateEnumeration, compile_join_graph, enumerate_candidates
from .ledger import Ledger
from .loader import load_workload_dict, rebuild_workload, validate_histogram_override
from .model import (
    Histogram,
    HistogramBin,
    Plan,
    Table,
    Tuning,
    Workload,
    canonical_key,
)
from .stats import StatisticsBundle, collect_statistics, fixed_signed


RUNNER_FORMAT_VERSION = 1


@dataclass(frozen=True)
class ExecutionPlanRecord:
    role: str
    plan: Plan
    estimate: PlanEstimate


class ExperimentSession:
    def __init__(self, workload: Workload, *, generation: int = 1) -> None:
        self.workload = workload
        self.generation = generation
        self.stale_conclusions: list[dict[str, Any]] = []
        self.execution_summaries: dict[str, dict[str, Any]] = {}
        self.execution_runs: dict[str, ExecutionRun] = {}
        self.progress: dict[str, Any] = {"cursor": 0, "total": 0}
        self.run_id = _run_id(workload)
        self.fingerprint = _fingerprint(workload)
        self._analyze()

    def _analyze(self) -> None:
        self.stats = collect_statistics(self.workload)
        self.graph = compile_join_graph(self.workload.query, self.workload.table_map())
        self.enumeration = enumerate_candidates(self.workload, self.graph)
        self.estimates = tuple(
            estimate_plan(plan, self.workload, self.stats)
            for plan in self.enumeration.plans
        )
        self.baseline_estimate = estimate_plan(
            self.enumeration.baseline, self.workload, self.stats
        )
        estimates_by_id = {estimate.plan.candidate_id: estimate for estimate in self.estimates}
        estimates_by_id[self.baseline_estimate.plan.candidate_id] = self.baseline_estimate
        self.estimates_by_id = estimates_by_id
        self.ranked = sorted(
            self.estimates,
            key=lambda estimate: (estimate.total_cost_fx, estimate.plan.candidate_id),
        )
        self.best_estimate = self.ranked[0]
        self.selection = self._selection_conclusion()
        execution_records: list[ExecutionPlanRecord] = [
            ExecutionPlanRecord("BEST", self.best_estimate.plan, self.best_estimate)
        ]
        if self.baseline_estimate.plan.candidate_id != self.best_estimate.plan.candidate_id:
            execution_records.append(
                ExecutionPlanRecord(
                    "BASELINE",
                    self.baseline_estimate.plan,
                    self.baseline_estimate,
                )
            )
        self.execution_records = tuple(execution_records)
        self.progress["total"] = len(self.execution_records)
        self.current_view = self._build_current_view()

    def _selection_conclusion(self) -> dict[str, Any]:
        return {
            "best_candidate_id": self.best_estimate.plan.candidate_id,
            "criterion": "total_cost_fx ascending, then candidate_id lexicographically ascending",
            "ranking": [
                {
                    "rank": index + 1,
                    "candidate_id": estimate.plan.candidate_id,
                    "total_cost_fx": estimate.total_cost_fx,
                    "plan_signature": estimate.plan.signature,
                }
                for index, estimate in enumerate(self.ranked)
            ],
            "generation": self.generation,
            "stale_generations_excluded": [
                stale["generation"] for stale in self.stale_conclusions
            ],
            "stale_candidate_ids_excluded": sorted(
                {
                    candidate_id
                    for stale in self.stale_conclusions
                    for candidate_id in stale.get("affected_candidate_ids", [])
                }
            ),
        }

    def _build_current_view(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "generation": self.generation,
            "statistics_digest": self.stats.digest,
            "graph_digest": self.graph.digest,
            "best_candidate_id": self.best_estimate.plan.candidate_id,
            "candidate_ids": [estimate.plan.candidate_id for estimate in self.estimates],
            "selection": self.selection,
            "estimates": {
                estimate.plan.candidate_id: estimate.as_dict(
                    self.workload.tuning.fixed_point_scale
                )
                for estimate in self.estimates
            },
            "baseline_candidate_id": self.baseline_estimate.plan.candidate_id,
            "execution_ordered_candidate_ids": [
                record.plan.candidate_id for record in self.execution_records
            ],
            "whole_replacement": True,
        }

    def analysis_report(self, *, include_current_estimates: bool = True) -> dict[str, Any]:
        scale = self.workload.tuning.fixed_point_scale
        report: dict[str, Any] = {
            "run_id": self.run_id,
            "fingerprint": self.fingerprint,
            "runner_format_version": RUNNER_FORMAT_VERSION,
            "generation": self.generation,
            "workload_digest": self.workload.digest,
            "query_digest": self.graph.query_digest,
            "graph": {
                "digest": self.graph.digest,
                "summary": json.loads(self.graph.summary_json),
            },
            "tuning": self.workload.tuning.canonical(),
            "fixed_point": {
                "scale": scale,
                "truncation": "toward_zero",
                "saturation": SATURATION_RULE,
                "max_fixed_value": (1 << 63) - 1,
            },
            "statistics": self.stats.as_dict(scale),
            "caliber_comparison": [
                {
                    "table": item.table,
                    "column": item.column,
                    "predicate": item.predicate_key,
                    "exact_selectivity_fx": fixed(item.exact.selectivity, scale),
                    "sampled_selectivity_fx": fixed(item.sampled.selectivity, scale),
                    "difference_fx": fixed_signed(item.difference, scale),
                    "selected_caliber": item.selected.caliber,
                    "selected_version": item.selected.version,
                }
                for item in self.stats.filter_stats
            ],
            "candidate_enumeration": {
                "candidate_count": len(self.enumeration.plans),
                "total_before_limit": self.enumeration.total_before_limit,
                "applied_limit": self.enumeration.applied_limit,
                "candidate_ids": [estimate.plan.candidate_id for estimate in self.estimates],
                "pruned_orders": list(self.enumeration.pruned_orders),
                "omitted_candidates": list(self.enumeration.omitted_candidates),
            },
            "selection": self.selection,
            "baseline": self.baseline_estimate.as_dict(scale),
            "execution_order": [
                {
                    "role": record.role,
                    "candidate_id": record.plan.candidate_id,
                }
                for record in self.execution_records
            ],
            "current_conclusion": self.current_view,
            "stale_conclusions": self.stale_conclusions,
            "progress": self.progress,
        }
        if include_current_estimates:
            report["candidates"] = [
                estimate.as_dict(scale) for estimate in self.estimates
            ]
        return report

    def run(
        self,
        ledger: Ledger,
        *,
        interrupt_after_execution: int | None = None,
        interrupt_before_commit: int | None = None,
        execution_faults: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        execution_faults = execution_faults or {}
        ledger.verify_integrity(self.run_id)
        ledger.ensure_run(
            self.run_id,
            query_digest=self.graph.query_digest,
            workload_digest=self.workload.digest,
            fingerprint=self.fingerprint,
        )
        recovered = ledger.recover_incomplete_attempts(self.run_id)
        committed = ledger.committed_batches(self.run_id)
        committed_by_index = {batch["batch_index"]: batch for batch in committed}
        if committed and len(committed) != max(committed_by_index) + 1:
            raise LabError(
                "E_RECOVERY",
                "已提交批次不连续，拒绝从不完整账本恢复",
                field="batches.batch_index",
                context={"committed_indices": sorted(committed_by_index)},
            )
        if len(committed) > len(self.execution_records):
            raise LabError(
                "E_RECOVERY",
                "账本批次数量超过当前执行计划数量",
                field="batches.batch_index",
                expected=len(self.execution_records),
                actual=len(committed),
            )

        for index, batch in enumerate(committed):
            record = self.execution_records[index]
            if batch["candidate_id"] != record.plan.candidate_id:
                raise LabError(
                    "E_REPLAY",
                    "账本批次与当前执行顺序的计划标识不一致",
                    candidate_id=record.plan.candidate_id,
                    field="batches.candidate_id",
                    expected=record.plan.candidate_id,
                    actual=batch["candidate_id"],
                )
            execution = batch["payload"]["execution"]
            self.execution_summaries[record.plan.candidate_id] = execution

        pending_attempts: dict[str, str] = {}
        missing_indices = [
            index for index in range(len(self.execution_records)) if index not in committed_by_index
        ]
        for index in missing_indices:
            record = self.execution_records[index]
            attempt_id = ledger.begin_attempt(self.run_id, index, record.plan.candidate_id)
            pending_attempts[record.plan.candidate_id] = attempt_id
        try:
            for index in missing_indices:
                record = self.execution_records[index]
                fault = execution_faults.get(record.role.lower())
                run = execute_plan(
                    self.workload,
                    record.plan,
                    record.estimate,
                    fault_mode=fault,
                )
                self.execution_runs[record.plan.candidate_id] = run
                if interrupt_after_execution == index:
                    raise LabError(
                        "E_INTERRUPTED",
                        "模拟进程在批次提交前被强制终止",
                        field="ledger.batch",
                        context={"batch_index": index, "attempt_left": "RUNNING"},
                    )
        except LabError as exc:
            if exc.code != "E_INTERRUPTED":
                for attempt_id in pending_attempts.values():
                    ledger.fail_attempt(attempt_id, f"{exc.code}: {exc.message}")
            raise
        except BaseException as exc:
            for attempt_id in pending_attempts.values():
                ledger.fail_attempt(attempt_id, f"{type(exc).__name__}: {exc}")
            raise

        try:
            consistency = self._check_consistency()
            bias = self._best_bias()
        except LabError as exc:
            for attempt_id in pending_attempts.values():
                ledger.fail_attempt(attempt_id, f"{exc.code}: {exc.message}")
            raise

        for index, record in enumerate(self.execution_records):
            execution = self._execution_summary(record, consistency, bias)
            self.execution_summaries[record.plan.candidate_id] = execution

        commits: list[dict[str, Any]] = []
        for index, record in enumerate(self.execution_records):
            if index in committed_by_index:
                batch = committed_by_index[index]
                commits.append(
                    {
                        "batch_id": batch["batch_id"],
                        "batch_index": index,
                        "candidate_id": batch["candidate_id"],
                        "content_digest": batch["content_digest"],
                        "snapshot_version": batch["snapshot_version"],
                        "reused": True,
                    }
                )
                continue
            if interrupt_before_commit == index:
                raise LabError(
                    "E_INTERRUPTED",
                    "模拟进程在批次事务提交前被强制终止",
                    field="ledger.batch",
                    context={"batch_index": index, "attempt_left": "RUNNING"},
                )
            progress = {
                "cursor": index + 1,
                "total": len(self.execution_records),
                "completed_candidate_ids": [
                    item.plan.candidate_id for item in self.execution_records[: index + 1]
                ],
                "next_candidate_id": (
                    self.execution_records[index + 1].plan.candidate_id
                    if index + 1 < len(self.execution_records)
                    else None
                ),
            }
            self.progress = progress
            snapshot_payload = self._snapshot_payload(
                progress=progress,
                summaries={
                    item.plan.candidate_id: self.execution_summaries[item.plan.candidate_id]
                    for item in self.execution_records[: index + 1]
                },
            )
            payload = self._batch_payload(
                batch_index=index,
                record=record,
                execution=self.execution_summaries[record.plan.candidate_id],
                progress=progress,
            )
            attempt_id = pending_attempts[record.plan.candidate_id]
            try:
                committed_info = ledger.commit_batch(
                    run_id=self.run_id,
                    attempt_id=attempt_id,
                    payload=payload,
                    snapshot_payload=snapshot_payload,
                )
            except BaseException as exc:
                ledger.fail_attempt(attempt_id, f"COMMIT: {type(exc).__name__}: {exc}")
                raise
            commits.append({**committed_info, "reused": False})

        self.progress = {
            "cursor": len(self.execution_records),
            "total": len(self.execution_records),
            "completed_candidate_ids": [
                record.plan.candidate_id for record in self.execution_records
            ],
            "next_candidate_id": None,
        }
        return {
            "run_id": self.run_id,
            "generation": self.generation,
            "recovered_unknown_attempts": recovered,
            "commits": commits,
            "consistency": consistency,
            "bias": bias,
            "execution_summaries": self.execution_summaries,
            "current_view": self.current_view,
            "progress": self.progress,
        }

    def _check_consistency(self) -> dict[str, Any]:
        if len(self.execution_records) == 1:
            return {
                "consistent": True,
                "raw_equal": True,
                "grouped_equal": True,
                "note": "best plan is also the unoptimized baseline candidate",
                "best_candidate_id": self.best_estimate.plan.candidate_id,
                "baseline_candidate_id": self.best_estimate.plan.candidate_id,
            }
        best_id = self.best_estimate.plan.candidate_id
        baseline_id = self.baseline_estimate.plan.candidate_id
        best_run = self.execution_runs.get(best_id)
        baseline_run = self.execution_runs.get(baseline_id)
        if best_run is not None and baseline_run is not None:
            return compare_execution_results(best_run, baseline_run)
        if best_run is not None and baseline_run is None:
            baseline_summary = self.execution_summaries.get(baseline_id, {})
            baseline_raw = baseline_summary.get("raw_digest") or baseline_summary.get("summary", {}).get("raw_digest")
            baseline_grouped = baseline_summary.get("grouped_digest") or baseline_summary.get("summary", {}).get("grouped_digest")
            if best_run.result.raw_digest == baseline_raw and best_run.result.grouped_digest == baseline_grouped:
                return {
                    "consistent": True,
                    "raw_equal": True,
                    "grouped_equal": True,
                    "best_candidate_id": best_id,
                    "baseline_candidate_id": baseline_id,
                    "comparison_source": "ledger_summary_plus_reexecution",
                }
            raise LabError(
                "E_DIVERGE",
                "恢复后最优计划与账本基线执行摘要不一致",
                candidate_id=best_id,
                field="grouped_digest" if best_run.result.grouped_digest != baseline_grouped else "raw_digest",
                expected=best_run.result.grouped_digest if best_run.result.grouped_digest != baseline_grouped else best_run.result.raw_digest,
                actual=baseline_grouped if best_run.result.grouped_digest != baseline_grouped else baseline_raw,
                context={"baseline_candidate_id": baseline_id},
            )
        if best_run is None and baseline_run is not None:
            best_summary = self.execution_summaries.get(best_id, {})
            best_raw = best_summary.get("raw_digest") or best_summary.get("summary", {}).get("raw_digest")
            best_grouped = best_summary.get("grouped_digest") or best_summary.get("summary", {}).get("grouped_digest")
            if best_raw == baseline_run.result.raw_digest and best_grouped == baseline_run.result.grouped_digest:
                return {
                    "consistent": True,
                    "raw_equal": True,
                    "grouped_equal": True,
                    "best_candidate_id": best_id,
                    "baseline_candidate_id": baseline_id,
                    "comparison_source": "ledger_summary_plus_reexecution",
                }
            raise LabError(
                "E_DIVERGE",
                "恢复后账本最优摘要与基线真实执行不一致",
                candidate_id=best_id,
                field="grouped_digest" if best_grouped != baseline_run.result.grouped_digest else "raw_digest",
                expected=best_grouped if best_grouped != baseline_run.result.grouped_digest else best_raw,
                actual=baseline_run.result.grouped_digest if best_grouped != baseline_run.result.grouped_digest else baseline_run.result.raw_digest,
                context={"baseline_candidate_id": baseline_id},
            )
        best_summary = self.execution_summaries.get(best_id, {})
        baseline_summary = self.execution_summaries.get(baseline_id, {})
        best_raw = best_summary.get("raw_digest") or best_summary.get("summary", {}).get("raw_digest")
        best_grouped = best_summary.get("grouped_digest") or best_summary.get("summary", {}).get("grouped_digest")
        baseline_raw = baseline_summary.get("raw_digest") or baseline_summary.get("summary", {}).get("raw_digest")
        baseline_grouped = baseline_summary.get("grouped_digest") or baseline_summary.get("summary", {}).get("grouped_digest")
        if best_raw == baseline_raw and best_grouped == baseline_grouped:
            return {
                "consistent": True,
                "raw_equal": True,
                "grouped_equal": True,
                "best_candidate_id": best_id,
                "baseline_candidate_id": baseline_id,
            }
        raise LabError(
            "E_DIVERGE",
            "恢复后最优计划与基线执行摘要不一致",
            candidate_id=best_id,
            field="grouped_digest" if best_grouped != baseline_grouped else "raw_digest",
            expected=best_grouped if best_grouped != baseline_grouped else best_raw,
            actual=baseline_grouped if best_grouped != baseline_grouped else baseline_raw,
            context={"baseline_candidate_id": baseline_id},
        )

    def _best_bias(self) -> dict[str, Any]:
        best_id = self.best_estimate.plan.candidate_id
        run = self.execution_runs.get(best_id)
        if run is None:
            summary = self.execution_summaries.get(best_id, {})
            return summary.get("bias", {})
        return bias_report(
            self.best_estimate,
            run,
            threshold_ratio=self.workload.tuning.bias_threshold_ratio,
            scale=self.workload.tuning.fixed_point_scale,
        )

    def _execution_summary(
        self,
        record: ExecutionPlanRecord,
        consistency: dict[str, Any],
        bias: dict[str, Any],
    ) -> dict[str, Any]:
        scale = self.workload.tuning.fixed_point_scale
        run = self.execution_runs.get(record.plan.candidate_id)
        if run is not None:
            summary = run.summary(scale)
            raw_digest = run.result.raw_digest
            grouped_digest = run.result.grouped_digest
            elapsed_ms = run.elapsed_ms
        else:
            existing = self.execution_summaries.get(record.plan.candidate_id, {})
            summary = existing.get("summary", {})
            raw_digest = existing.get("raw_digest") or summary.get("raw_digest")
            grouped_digest = existing.get("grouped_digest") or summary.get("grouped_digest")
            elapsed_ms = existing.get("elapsed_ms", summary.get("elapsed_ms", 0.0))
        return {
            "candidate_id": record.plan.candidate_id,
            "role": record.role,
            "raw_digest": raw_digest,
            "grouped_digest": grouped_digest,
            "elapsed_ms": elapsed_ms,
            "estimated_cost_fx": record.estimate.total_cost_fx,
            "summary": summary,
            "bias": bias if record.role == "BEST" else summary.get("bias", {}),
            "consistency": consistency,
        }

    def _batch_payload(
        self,
        *,
        batch_index: int,
        record: ExecutionPlanRecord,
        execution: dict[str, Any],
        progress: dict[str, Any],
    ) -> dict[str, Any]:
        batch_id = _batch_id(self.run_id, batch_index, record.plan.candidate_id)
        scale = self.workload.tuning.fixed_point_scale
        all_estimates = sorted(
            (
                estimate.as_dict(scale)
                for estimate in self.estimates_by_id.values()
            ),
            key=lambda item: item["candidate_id"],
        )
        return {
            "run_id": self.run_id,
            "batch_index": batch_index,
            "batch_id": batch_id,
            "candidate_id": record.plan.candidate_id,
            "estimates": all_estimates,
            "selection": self.selection,
            "execution": execution,
            "scheduler_tasks": sorted(
                execution.get("summary", {}).get("scheduler", {}).get("tasks", []),
                key=lambda item: item["task_id"],
            ),
            "progress": progress,
        }

    def _snapshot_payload(
        self,
        *,
        progress: dict[str, Any],
        summaries: dict[str, dict[str, Any]],
    ) -> dict[str, Any]:
        return {
            "snapshot_format_version": 1,
            "runner_format_version": RUNNER_FORMAT_VERSION,
            "run_id": self.run_id,
            "fingerprint": self.fingerprint,
            "generation": self.generation,
            "workload": workload_to_json(self.workload),
            "statistics": self.stats.as_dict(self.workload.tuning.fixed_point_scale),
            "current_view": self.current_view,
            "stale_conclusions": self.stale_conclusions,
            "execution_summaries": summaries,
            "progress": progress,
        }

    @classmethod
    def from_snapshot(cls, payload: dict[str, Any]) -> "ExperimentSession":
        version = payload.get("snapshot_format_version")
        if version != 1:
            raise LabError(
                "E_RECOVERY",
                "不支持的快照格式版本",
                field="snapshot.snapshot_format_version",
                expected=1,
                actual=version,
            )
        workload = load_workload_dict(payload["workload"], source="<snapshot>")
        session = cls(workload, generation=int(payload.get("generation", 1)))
        if session.run_id != payload.get("run_id"):
            raise LabError(
                "E_RECOVERY",
                "快照 run_id 与重算指纹不一致",
                field="snapshot.run_id",
                expected=payload.get("run_id"),
                actual=session.run_id,
            )
        recorded_statistics = payload.get("statistics")
        if isinstance(recorded_statistics, dict):
            if recorded_statistics.get("digest") != session.stats.digest:
                raise LabError(
                    "E_RECOVERY",
                    "快照统计口径与当前确定性重算结果不一致",
                    field="snapshot.statistics.digest",
                    expected=recorded_statistics.get("digest"),
                    actual=session.stats.digest,
                )
        snapshot_view = payload.get("current_view")
        if isinstance(snapshot_view, dict):
            expected_ids = [estimate.plan.candidate_id for estimate in session.estimates]
            if snapshot_view.get("candidate_ids") != expected_ids:
                raise LabError(
                    "E_RECOVERY",
                    "快照候选集合与当前确定性重算结果不一致",
                    field="snapshot.current_view.candidate_ids",
                    expected=snapshot_view.get("candidate_ids"),
                    actual=expected_ids,
                )
            session.current_view = snapshot_view
        session.stale_conclusions = list(payload.get("stale_conclusions", []))
        session.execution_summaries = dict(payload.get("execution_summaries", {}))
        session.progress = dict(payload.get("progress", session.progress))
        return session

    def restore_from_ledger(
        self,
        ledger: Ledger,
        *,
        snapshot_version: int | None = None,
    ) -> tuple["ExperimentSession", dict[str, Any]]:
        ledger.verify_integrity(self.run_id)
        if snapshot_version is None:
            snapshot = ledger.latest_snapshot(self.run_id)
        else:
            snapshot = ledger.get_snapshot(self.run_id, snapshot_version)
        if snapshot is None:
            raise LabError(
                "E_RECOVERY",
                "账本中没有可恢复的会话快照",
                field="snapshots",
                context={"run_id": self.run_id},
            )
        restored = self.from_snapshot(snapshot["payload"])
        return restored, snapshot

    def apply_change_set(
        self,
        changes: dict[str, Any],
        *,
        ledger: Ledger | None = None,
    ) -> dict[str, Any]:
        if not isinstance(changes, dict):
            raise schema_error("change_set 必须是对象", field="change_set")
        old_view = copy.deepcopy(self.current_view)
        old_run_id = self.run_id
        old_generation = self.generation
        old_candidate_ids = list(self.current_view["candidate_ids"])
        old_plan_tables = {
            estimate.plan.candidate_id: set(estimate.plan.table_order)
            for estimate in self.ranked
        }
        tables = {table.name: _clone_table(table) for table in self.workload.tables}
        tuning = self.workload.tuning
        changed_tables: set[str] = set()
        global_reasons: list[str] = []

        row_counts = changes.get("table_row_counts", {})
        if row_counts is not None and not isinstance(row_counts, dict):
            raise schema_error("table_row_counts 必须是对象", field="change_set.table_row_counts")
        for table_name, target in row_counts.items():
            if table_name not in tables:
                raise schema_error(
                    "变更集引用未知表", table=table_name, field="change_set.table_row_counts"
                )
            target_count = _require_nonnegative_int(
                value=target, field=f"change_set.table_row_counts.{table_name}"
            )
            tables[table_name] = _resize_table(tables[table_name], target_count)
            changed_tables.add(table_name)

        histogram_changes = changes.get("histograms", {})
        if histogram_changes is not None and not isinstance(histogram_changes, dict):
            raise schema_error("histograms 必须是对象", field="change_set.histograms")
        for table_name, column_changes in histogram_changes.items():
            if table_name not in tables:
                raise schema_error(
                    "直方图变更引用未知表", table=table_name, field="change_set.histograms"
                )
            if not isinstance(column_changes, dict):
                raise schema_error(
                    "表直方图变更必须是对象",
                    table=table_name,
                    field="change_set.histograms",
                )
            table = tables[table_name]
            updated = dict(table.histograms)
            for column_name, raw_histogram in column_changes.items():
                if column_name not in updated:
                    raise schema_error(
                        "直方图变更引用未知列",
                        table=table_name,
                        column=column_name,
                        field="change_set.histograms",
                    )
                updated[column_name] = validate_histogram_override(
                    raw_histogram, table, column_name
                )
            table.histograms = updated
            tables[table_name] = table
            changed_tables.add(table_name)

        tuning_updates: dict[str, Any] = {}
        for name in ("fixed_point_scale", "max_candidates", "exact_scan_threshold_rows"):
            if name in changes:
                value = _require_nonnegative_int(value=changes[name], field=f"change_set.{name}")
                if value < 1:
                    raise schema_error(
                        f"{name} 必须至少为 1", field=f"change_set.{name}"
                    )
                tuning_updates[name] = value
                global_reasons.append(name)
        if "sample_size" in changes:
            value = _require_nonnegative_int(
                value=changes["sample_size"], field="change_set.sample_size"
            )
            if value < 1:
                raise schema_error("sample_size 必须至少为 1", field="change_set.sample_size")
            tuning_updates["sample_size"] = value
            global_reasons.append("sample_size")
        if tuning_updates:
            tuning = replace(tuning, **tuning_updates)

        new_workload = rebuild_workload(tables.values(), self.workload.query, tuning)
        self.workload = new_workload
        self.generation = old_generation + 1
        self.execution_summaries.clear()
        self.execution_runs.clear()
        self.run_id = _run_id(new_workload)
        self.fingerprint = _fingerprint(new_workload)
        self._analyze()
        new_candidate_ids = [estimate.plan.candidate_id for estimate in self.estimates]

        affected_old_ids = [
            candidate_id
            for candidate_id, plan_tables in old_plan_tables.items()
            if bool(global_reasons) or bool(changed_tables.intersection(plan_tables))
        ]
        if not affected_old_ids and (changed_tables or global_reasons):
            affected_old_ids = list(old_candidate_ids)
        added_ids = [candidate_id for candidate_id in new_candidate_ids if candidate_id not in old_candidate_ids]
        removed_ids = [candidate_id for candidate_id in old_candidate_ids if candidate_id not in new_candidate_ids]
        retained_ids = [candidate_id for candidate_id in new_candidate_ids if candidate_id in old_candidate_ids]
        diff = {
            "from_generation": old_generation,
            "to_generation": self.generation,
            "old_run_id": old_run_id,
            "new_run_id": self.run_id,
            "changed_tables": sorted(changed_tables),
            "global_reasons": sorted(global_reasons),
            "added": [
                {"candidate_id": candidate_id, "reason": "NEW_ACTIVE_UNDER_CURRENT_LIMIT"}
                for candidate_id in added_ids
            ],
            "removed": [
                {"candidate_id": candidate_id, "reason": "CANDIDATE_LIMIT_OR_GRAPH_CHANGE"}
                for candidate_id in removed_ids
            ],
            "retained": [
                {
                    "candidate_id": candidate_id,
                    "reason": "STABLE_PLAN_ID_RECOMPUTED_IN_NEW_GENERATION",
                }
                for candidate_id in retained_ids
            ],
            "stale_estimate_ids": affected_old_ids,
            "stale_selection_generation": old_generation,
            "replacement_mode": "whole_view",
        }
        stale_entry = {
            "generation": old_generation,
            "run_id": old_run_id,
            "statistics_digest": old_view.get("statistics_digest"),
            "candidate_ids": old_candidate_ids,
            "affected_candidate_ids": affected_old_ids,
            "selection": old_view.get("selection"),
            "reason": "STATS_OR_DATA_OR_SCALE_OR_LIMIT_CHANGED",
        }
        self.stale_conclusions.append(stale_entry)
        self.selection = self._selection_conclusion()
        self.current_view = self._build_current_view()
        if ledger is not None:
            ledger.ensure_run(
                self.run_id,
                query_digest=self.graph.query_digest,
                workload_digest=self.workload.digest,
                fingerprint=self.fingerprint,
            )
            event_id = "I-" + hashlib.sha256(
                canonical_key(
                    {
                        "old_run_id": old_run_id,
                        "new_run_id": self.run_id,
                        "changes": changes,
                    }
                ).encode("utf-8")
            ).hexdigest()[:24]
            ledger.record_invalidation(
                event_id=event_id,
                run_id=self.run_id,
                generation=self.generation,
                affected={
                    "changed_tables": sorted(changed_tables),
                    "global_reasons": sorted(global_reasons),
                    "stale_estimate_ids": affected_old_ids,
                },
                diff=diff,
            )
        return {
            "generation": self.generation,
            "run_id": self.run_id,
            "affected": {
                "changed_tables": sorted(changed_tables),
                "global_reasons": sorted(global_reasons),
                "stale_estimate_ids": affected_old_ids,
            },
            "view_diff": diff,
            "current_view": self.current_view,
        }

    def replay_check(
        self,
        ledger: Ledger,
        *,
        snapshot_version: int | None = None,
        reexecute: bool = True,
    ) -> dict[str, Any]:
        integrity = ledger.verify_integrity(self.run_id)
        restored, snapshot = self.restore_from_ledger(
            ledger, snapshot_version=snapshot_version
        )
        batches = ledger.committed_batches(self.run_id)
        checked: list[dict[str, Any]] = []
        for batch in batches:
            payload = batch["payload"]
            candidate_id = payload["candidate_id"]
            for ledger_estimate in payload["estimates"]:
                estimate_candidate_id = ledger_estimate["candidate_id"]
                recomputed = restored.estimates_by_id.get(estimate_candidate_id)
                if recomputed is None:
                    raise replay_error(
                        "账本估算中的候选计划在当前候选集合中不存在",
                        candidate_id=estimate_candidate_id,
                        operator_id=None,
                        field="candidate_id",
                        expected=estimate_candidate_id,
                        actual=sorted(restored.estimates_by_id),
                        batch_index=payload["batch_index"],
                    )
                _compare_estimate_for_replay(
                    ledger_estimate,
                    recomputed.as_dict(restored.workload.tuning.fixed_point_scale),
                    candidate_id=estimate_candidate_id,
                    batch_index=payload["batch_index"],
                )
            expected_estimate = next(
                estimate
                for estimate in payload["estimates"]
                if estimate["candidate_id"] == candidate_id
            )
            actual_estimate = restored.estimates_by_id.get(candidate_id)
            if actual_estimate is None:
                raise replay_error(
                    "账本中的计划在当前候选集合中不存在",
                    candidate_id=candidate_id,
                    operator_id=None,
                    field="candidate_id",
                    expected=candidate_id,
                    actual=sorted(restored.estimates_by_id),
                    batch_index=payload["batch_index"],
                )
            selection_mismatch = _first_mismatch(
                payload["selection"], restored.selection, "selection"
            )
            if selection_mismatch is not None:
                raise replay_error(
                    "重算后的选择结论不一致",
                    candidate_id=candidate_id,
                    operator_id=None,
                    field=selection_mismatch["path"],
                    expected=selection_mismatch["expected"],
                    actual=selection_mismatch["actual"],
                    batch_index=payload["batch_index"],
                )
            execution = payload["execution"]
            execution_check: dict[str, Any] = {
                "candidate_id": candidate_id,
                "batch_index": payload["batch_index"],
                "digests_consistent": True,
                "scheduler_consistent": True,
            }
            if reexecute:
                new_run = execute_plan(restored.workload, actual_estimate.plan, actual_estimate)
                if new_run.result.raw_digest != execution["raw_digest"]:
                    raise replay_error(
                        "重放真实执行原始结果摘要不一致",
                        candidate_id=candidate_id,
                        operator_id="output",
                        field="execution.raw_digest",
                        expected=execution["raw_digest"],
                        actual=new_run.result.raw_digest,
                        batch_index=payload["batch_index"],
                    )
                if new_run.result.grouped_digest != execution["grouped_digest"]:
                    raise replay_error(
                        "重放分组聚合摘要不一致",
                        candidate_id=candidate_id,
                        operator_id="aggregate",
                        field="execution.grouped_digest",
                        expected=execution["grouped_digest"],
                        actual=new_run.result.grouped_digest,
                        batch_index=payload["batch_index"],
                    )
                _compare_scheduler_for_replay(
                    payload["scheduler_tasks"],
                    new_run.schedule.trace_dicts(),
                    candidate_id=candidate_id,
                    batch_index=payload["batch_index"],
                )
                expected_constraints = execution.get("summary", {}).get("scheduler", {}).get("constraints", {})
                actual_constraints = new_run.schedule.constraints
                for constraint_field in (
                    "global_concurrency_limit",
                    "per_table_quota_limit",
                    "memory_budget_bytes",
                    "global_limit_respected",
                    "table_quota_respected",
                    "memory_budget_respected",
                ):
                    if expected_constraints.get(constraint_field) != actual_constraints.get(constraint_field):
                        raise replay_error(
                            "重放调度参数或约束结论不一致",
                            candidate_id=candidate_id,
                            operator_id="scheduler",
                            field=f"scheduler.constraints.{constraint_field}",
                            expected=expected_constraints.get(constraint_field),
                            actual=actual_constraints.get(constraint_field),
                            batch_index=payload["batch_index"],
                        )
                new_bias = bias_report(
                    actual_estimate,
                    new_run,
                    threshold_ratio=restored.workload.tuning.bias_threshold_ratio,
                    scale=restored.workload.tuning.fixed_point_scale,
                    raise_on_exceed=False,
                )
                old_bias = execution.get("bias", {})
                if bool(old_bias.get("exceeds_threshold")) != bool(new_bias.get("exceeds_threshold")):
                    raise replay_error(
                        "重放偏差结论不一致",
                        candidate_id=candidate_id,
                        operator_id=new_bias.get("dominant_operator_id"),
                        field="bias.exceeds_threshold",
                        expected=old_bias.get("exceeds_threshold"),
                        actual=new_bias.get("exceeds_threshold"),
                        batch_index=payload["batch_index"],
                    )
                if old_bias.get("dominant_operator_id") not in (None, new_bias.get("dominant_operator_id")):
                    raise replay_error(
                        "重放主导偏差算子不一致",
                        candidate_id=candidate_id,
                        operator_id=new_bias.get("dominant_operator_id"),
                        field="bias.dominant_operator_id",
                        expected=old_bias.get("dominant_operator_id"),
                        actual=new_bias.get("dominant_operator_id"),
                        batch_index=payload["batch_index"],
                    )
                execution_check["recomputed_elapsed_ms"] = new_run.elapsed_ms
            if payload["progress"]["cursor"] != payload["batch_index"] + 1:
                raise replay_error(
                    "账本进度游标与批次序号不一致",
                    candidate_id=candidate_id,
                    operator_id=None,
                    field="progress.cursor",
                    expected=payload["batch_index"] + 1,
                    actual=payload["progress"]["cursor"],
                    batch_index=payload["batch_index"],
                )
            checked.append(execution_check)
        remaining = [
            record.plan.candidate_id
            for index, record in enumerate(restored.execution_records)
            if index >= restored.progress.get("cursor", 0)
        ]
        return {
            "ok": True,
            "run_id": self.run_id,
            "restored_snapshot_version": snapshot["version"],
            "snapshot_digest": snapshot["content_digest"],
            "ledger_integrity": integrity,
            "checked_batches": checked,
            "remaining_candidate_ids": remaining,
        }


def _compare_estimate_for_replay(
    expected: dict[str, Any],
    actual: dict[str, Any],
    *,
    candidate_id: str,
    batch_index: int,
) -> None:
    operator_ids = [
        operator.get("operator_id") for operator in actual.get("operators", [])
    ]
    mismatch = _first_mismatch(expected, actual, "estimate")
    if mismatch is None:
        return
    operator_id: str | None = None
    path = mismatch["path"]
    marker = "operators["
    if marker in path:
        tail = path.split(marker, 1)[1]
        index_text = tail.split("]", 1)[0]
        try:
            operator_id = operator_ids[int(index_text)]
        except (ValueError, IndexError):
            operator_id = None
    raise replay_error(
        "重放估算值与账本不一致",
        candidate_id=candidate_id,
        operator_id=operator_id,
        field=path,
        expected=mismatch["expected"],
        actual=mismatch["actual"],
        batch_index=batch_index,
    )


def _compare_scheduler_for_replay(
    expected_tasks: list[dict[str, Any]],
    actual_tasks: list[dict[str, Any]],
    *,
    candidate_id: str,
    batch_index: int,
) -> None:
    expected_map = {task["task_id"]: task for task in expected_tasks}
    actual_map = {task["task_id"]: task for task in actual_tasks}
    stable_fields = (
        "kind",
        "operator_id",
        "tables",
        "dependencies",
        "memory_reservation_bytes",
        "completed_rows",
        "peak_memory_bytes",
    )
    for task_id in sorted(set(expected_map) | set(actual_map)):
        if task_id not in expected_map or task_id not in actual_map:
            raise replay_error(
                "重放调度任务集合不一致",
                candidate_id=candidate_id,
                operator_id=task_id,
                field="scheduler.tasks.task_id",
                expected=task_id in expected_map,
                actual=task_id in actual_map,
                batch_index=batch_index,
            )
        for field in stable_fields:
            if expected_map[task_id].get(field) != actual_map[task_id].get(field):
                raise replay_error(
                    "重放调度轨迹稳定字段不一致",
                    candidate_id=candidate_id,
                    operator_id=expected_map[task_id].get("operator_id"),
                    field=f"scheduler.tasks.{task_id}.{field}",
                    expected=expected_map[task_id].get(field),
                    actual=actual_map[task_id].get(field),
                    batch_index=batch_index,
                )


def _first_mismatch(expected: Any, actual: Any, path: str) -> dict[str, Any] | None:
    if type(expected) is not type(actual):
        if isinstance(expected, (int, float)) and isinstance(actual, (int, float)):
            if expected == actual:
                return None
        return {"path": path, "expected": expected, "actual": actual}
    if isinstance(expected, dict):
        for key in sorted(set(expected) | set(actual)):
            if key not in expected:
                return {"path": f"{path}.{key}", "expected": None, "actual": actual[key]}
            if key not in actual:
                return {"path": f"{path}.{key}", "expected": expected[key], "actual": None}
            mismatch = _first_mismatch(expected[key], actual[key], f"{path}.{key}")
            if mismatch is not None:
                return mismatch
        return None
    if isinstance(expected, list):
        for index in range(min(len(expected), len(actual))):
            mismatch = _first_mismatch(expected[index], actual[index], f"{path}[{index}]")
            if mismatch is not None:
                return mismatch
        if len(expected) != len(actual):
            return {"path": f"{path}.length", "expected": len(expected), "actual": len(actual)}
        return None
    if expected != actual:
        return {"path": path, "expected": expected, "actual": actual}
    return None


def workload_to_json(workload: Workload) -> dict[str, Any]:
    tables = []
    for table in workload.tables:
        tables.append(
            {
                "name": table.name,
                "columns": [{"name": col.name, "type": col.type} for col in table.columns],
                "rows": [list(row) for row in table.rows],
                "null_bitmap": table.null_bitmap,
                "histograms": {
                    column.name: {
                        "version": table.histograms[column.name].version,
                        "bins": [
                            {
                                "lo": _decimal_json(bin.lo),
                                "hi": _decimal_json(bin.hi),
                                "count": bin.count,
                            }
                            for bin in table.histograms[column.name].bins
                        ],
                    }
                    for column in table.columns
                },
            }
        )
    return {
        "format_version": workload.format_version,
        "tables": sorted(tables, key=lambda item: item["name"]),
        "query": {"sql": workload.query.sql},
        "tuning": workload.tuning.canonical(),
    }


def _clone_table(table: Table) -> Table:
    return Table(
        name=table.name,
        columns=table.columns,
        rows=list(table.rows),
        null_bitmap=[list(bits) for bits in table.null_bitmap],
        histograms=dict(table.histograms),
    )


def _resize_table(table: Table, target_count: int) -> Table:
    current = table.row_count
    if target_count == current:
        return table
    if target_count < current:
        rows = list(table.rows[:target_count])
        bitmap = [list(bits) for bits in table.null_bitmap[:target_count]]
    else:
        if current == 0:
            raise schema_error(
                "空表无法按行数扩展生成确定性样本",
                table=table.name,
                field="change_set.table_row_counts",
            )
        rows = list(table.rows)
        bitmap = [list(bits) for bits in table.null_bitmap]
        for index in range(current, target_count):
            source = index % current
            rows.append(tuple(rows[source]))
            bitmap.append(list(bitmap[source]))
    histograms: dict[str, Histogram] = {}
    for column_index, column in enumerate(table.columns):
        old = table.histograms[column.name]
        counts = [0] * len(old.bins)
        outside = 0
        for row in rows:
            value = row[column_index]
            if value is None:
                continue
            marker = Decimal(str(value))
            matched = False
            for bin_index, bin_item in enumerate(old.bins):
                if bin_item.lo <= marker < bin_item.hi:
                    counts[bin_index] += 1
                    matched = True
                    break
            if not matched:
                outside += 1
        if outside:
            raise schema_error(
                "行数调整后出现直方图区间外数据，请同时提供直方图变更",
                table=table.name,
                column=column.name,
                field="change_set.histograms",
            )
        bins = tuple(
            HistogramBin(old_bin.lo, old_bin.hi, count)
            for old_bin, count in zip(old.bins, counts)
        )
        version = old.version + 1
        digest = _histogram_digest(version, bins)
        histograms[column.name] = Histogram(version, bins, digest)
    return Table(table.name, table.columns, rows, bitmap, histograms)


def _histogram_digest(version: int, bins: Iterable[HistogramBin]) -> str:
    data = {
        "version": version,
        "bins": [
            {"lo": str(bin.lo), "hi": str(bin.hi), "count": bin.count} for bin in bins
        ],
    }
    return hashlib.sha256(canonical_key(data).encode("utf-8")).hexdigest()


def _run_id(workload: Workload) -> str:
    material = (
        f"joinlab-run-v{RUNNER_FORMAT_VERSION}|{workload.digest}|"
        f"{canonical_key(workload.query.canonical())}"
    )
    return "R-" + hashlib.sha256(material.encode("utf-8")).hexdigest()[:24]


def _fingerprint(workload: Workload) -> str:
    return hashlib.sha256(
        f"runner-v{RUNNER_FORMAT_VERSION}|{workload.digest}".encode("utf-8")
    ).hexdigest()


def _batch_id(run_id: str, batch_index: int, candidate_id: str) -> str:
    return "B-" + hashlib.sha256(
        f"{run_id}:{batch_index}:{candidate_id}".encode("utf-8")
    ).hexdigest()[:24]


def _decimal_json(value: Decimal) -> int | float:
    if value == value.to_integral_value():
        return int(value)
    return float(value)


def _require_nonnegative_int(*, value: Any, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise schema_error("字段必须是非负整数", field=field, actual=value)
    return value
