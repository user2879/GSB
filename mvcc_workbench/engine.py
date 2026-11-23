"""End-to-end diagnostic round orchestration."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Callable

from .ledger import Ledger, LedgerRound
from .model import Spec, SpecIndex, build_index, conclusion_digest, spec_digest
from .planner import (
    MATERIALIZED,
    PER_QUERY,
    AlgorithmPlan,
    PlannerThresholds,
    select_plan,
)
from .scheduler import ScheduleReport, ScheduledTask, run_scheduled_tasks
from .visibility import (
    ConsistencyReport,
    QueryResult,
    build_query_result,
    compare_results,
    resolve_visibility,
    run_materialized,
)


@dataclass(frozen=True)
class DiagnosisRound:
    round_id: str
    plan: AlgorithmPlan
    results: tuple[QueryResult, ...]
    consistency: ConsistencyReport
    schedule: ScheduleReport
    alternate_schedule: ScheduleReport
    budget: dict[str, Any]
    progress: dict[str, Any]
    ledger: LedgerRound | None

    @property
    def conclusion_digest(self) -> str:
        return conclusion_digest(self.results)

    def summary(self) -> dict[str, Any]:
        return {
            "ok": True,
            "code": "OK",
            "round_id": self.round_id,
            "plan": self.plan.to_dict(),
            "consistency": self.consistency.to_dict(),
            "conclusion_digest": self.conclusion_digest,
            "result_count": len(self.results),
            "schedule": self.schedule.to_dict(),
            "budget": self.budget,
            "progress": self.progress,
            "ledger": self.ledger.summary() if self.ledger is not None else None,
        }

    def to_dict(self, *, include_results: bool = True) -> dict[str, Any]:
        payload = self.summary()
        if include_results:
            payload["results"] = [item.to_dict() for item in self.results]
        return payload


def _round_id(spec: Spec) -> str:
    digest_fragment = spec_digest(spec).split(":")[-1][:12]
    return f"round-{time.time_ns()}-{digest_fragment}"


def _per_query_tasks(index: SpecIndex) -> list[ScheduledTask]:
    tasks: list[ScheduledTask] = []
    for sequence, query in enumerate(sorted(index.spec.queries, key=lambda q: q.id), start=1):
        task_id = f"Q{sequence:06d}:{query.id}"

        def run(query=query) -> QueryResult:
            resolved = resolve_visibility(index, query.read_point_id, query.row_key)
            return build_query_result(query, resolved)

        tasks.append(
            ScheduledTask(
                task_id=task_id,
                read_point_id=query.read_point_id,
                fn=run,
            )
        )
    return tasks


def _materialized_tasks(index: SpecIndex) -> tuple[list[ScheduledTask], list[tuple[str, str]]]:
    pairs = sorted(index.query_pairs)
    tasks: list[ScheduledTask] = []
    for sequence, (read_point_id, row_key) in enumerate(pairs, start=1):
        task_id = f"M{sequence:06d}"

        def run(read_point_id=read_point_id, row_key=row_key):
            return resolve_visibility(index, read_point_id, row_key)

        tasks.append(
            ScheduledTask(
                task_id=task_id,
                read_point_id=read_point_id,
                fn=run,
            )
        )
    return tasks, pairs


def _run_path(
    index: SpecIndex,
    path: str,
) -> tuple[tuple[QueryResult, ...], ScheduleReport]:
    if path == PER_QUERY:
        tasks = _per_query_tasks(index)
        report = run_scheduled_tasks(
            tasks,
            global_concurrency=index.spec.budget.global_concurrency,
            per_read_point_concurrency=index.spec.budget.per_read_point_concurrency,
        )
        results = tuple(
            sorted(report.results.values(), key=lambda item: item.query_id)
        )
        return results, report
    if path == MATERIALIZED:
        tasks, pairs = _materialized_tasks(index)
        report = run_scheduled_tasks(
            tasks,
            global_concurrency=index.spec.budget.global_concurrency,
            per_read_point_concurrency=index.spec.budget.per_read_point_concurrency,
        )
        resolutions = [report.results[f"M{index:06d}"] for index in range(1, len(pairs) + 1)]
        batch = run_materialized(index, resolutions)
        return batch.results, report
    raise ValueError(f"unknown path: {path!r}")


def _build_budget_payload(plan: AlgorithmPlan, schedule: ScheduleReport) -> dict[str, Any]:
    return {
        "limits": dict(schedule.budget),
        "observed_peaks": dict(schedule.observed_peaks),
        "totals": dict(schedule.totals),
        "planner": {
            "selected_path": plan.selected_path,
            "threshold_influenced": plan.threshold_influenced,
            "work_estimates": plan.estimates,
        },
    }


class DiagnosticEngine:
    def __init__(
        self,
        spec: Spec,
        *,
        index: SpecIndex | None = None,
        ledger: Ledger | None = None,
        thresholds: PlannerThresholds | None = None,
    ) -> None:
        self.spec = spec
        self.index = index or build_index(spec)
        self.ledger = ledger
        self.thresholds = thresholds

    def plan(self, *, forced_path: str | None = None) -> AlgorithmPlan:
        return select_plan(
            self.spec,
            self.index,
            thresholds=self.thresholds,
            forced_path=forced_path,
        )

    def run(
        self,
        *,
        round_id: str | None = None,
        forced_path: str | None = None,
        verify_consistency: bool = True,
        persist: bool = True,
        crash_hook: Callable[[], None] | None = None,
    ) -> DiagnosisRound:
        plan = self.plan(forced_path=forced_path)
        selected_round_id = round_id or _round_id(self.spec)
        if persist and self.ledger is not None:
            self.ledger.begin_pending(
                round_id=selected_round_id,
                spec_digest=spec_digest(self.spec),
                query_ids=[query.id for query in self.spec.queries],
                started_wall_ms=int(time.time() * 1000),
            )
        if crash_hook is not None:
            crash_hook()

        selected_results, selected_schedule = _run_path(self.index, plan.selected_path)
        if verify_consistency:
            alternate_path = MATERIALIZED if plan.selected_path == PER_QUERY else PER_QUERY
            alternate_results, alternate_schedule = _run_path(self.index, alternate_path)
            consistency = compare_results(
                selected_results,
                alternate_results,
                left_path=plan.selected_path,
                right_path=alternate_path,
            )
        else:
            alternate_schedule = selected_schedule
            consistency = ConsistencyReport(
                ok=True,
                compared_queries=0,
                left_path=plan.selected_path,
                right_path="not_run",
                conclusion_digest=conclusion_digest(selected_results),
            )

        budget = _build_budget_payload(plan, selected_schedule)
        progress = {
            "cursor": len(selected_results),
            "completed_count": len(selected_results),
            "total_count": len(self.spec.queries),
            "last_query_id": selected_results[-1].query_id if selected_results else None,
            "completed_query_ids": [item.query_id for item in selected_results],
        }

        ledger_round: LedgerRound | None = None
        if persist and self.ledger is not None:
            ledger_round = LedgerRound(
                round_id=selected_round_id,
                spec_digest=spec_digest(self.spec),
                selected_path=plan.selected_path,
                conclusion_digest=conclusion_digest(selected_results),
                results=tuple(item.to_dict() for item in selected_results),
                plan=plan.to_dict(),
                schedule=selected_schedule.to_dict(),
                budget=budget,
                progress=progress,
                committed_wall_ms=int(time.time() * 1000),
            )
            self.ledger.commit_round(ledger_round)

        return DiagnosisRound(
            round_id=selected_round_id,
            plan=plan,
            results=selected_results,
            consistency=consistency,
            schedule=selected_schedule,
            alternate_schedule=alternate_schedule,
            budget=budget,
            progress=progress,
            ledger=ledger_round,
        )


def diagnose_spec(
    spec: Spec,
    *,
    forced_path: str | None = None,
    ledger: Ledger | None = None,
    round_id: str | None = None,
    thresholds: PlannerThresholds | None = None,
) -> DiagnosisRound:
    engine = DiagnosticEngine(spec, ledger=ledger, thresholds=thresholds)
    return engine.run(round_id=round_id, forced_path=forced_path)
