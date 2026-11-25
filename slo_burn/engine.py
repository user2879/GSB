"""端到端编排：输入加载 -> 双路径判定 -> 迟到重算 -> 租约调度 -> 账本提交。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .budget import choose_path, run_exact, run_incremental
from .divergence import verify_paths
from .late import recompute_with_late
from .ledger import Ledger
from .loader import load_batch_directory, load_param_edits, load_service_specs
from .models import PATH_EXACT, PathDecision, WindowConclusion
from .params import build_version_chain
from .scheduler import ScheduledTask, WindowScheduler
from .stream import merge_streams


@dataclass
class ServiceRunResult:
    service_id: str
    path_decision: PathDecision
    exact: list[WindowConclusion]
    incremental: list[WindowConclusion]
    divergence_records: list
    late_impacts: list
    coverage: list
    selected: list[WindowConclusion]


def run_pipeline(
    workspace: str | Path,
    ledger_path: str | Path,
    scheduler: WindowScheduler | None = None,
    use_edits: bool = True,
) -> dict[str, Any]:
    workspace = Path(workspace)
    specs = load_service_specs(workspace / "services.json")
    regular, late = load_batch_directory(workspace / "batches", specs)
    edits_path = workspace / "param_edits.json"
    edits = load_param_edits(edits_path, specs) if use_edits and edits_path.exists() else []
    chains = build_version_chain(specs, edits)
    merged, coverage = merge_streams(regular, late)

    results: list[ServiceRunResult] = []
    all_selected: list[WindowConclusion] = []
    lease_tasks: list[ScheduledTask] = []
    stale_minutes: dict[str, set[int]] = {}

    for spec in sorted(specs, key=lambda item: item.service_id):
        chain = chains[spec.service_id]
        points = merged.get(spec.service_id, [])
        horizon = points[-1].minute if points else 0
        decision = choose_path(spec.window_minutes, len(points))

        exact_all: list[WindowConclusion] = []
        inc_all: list[WindowConclusion] = []
        for versioned in chain:
            version_points = [point for point in points if versioned.covers(point.minute)]
            if not version_points:
                continue
            exact_all.extend(run_exact(versioned.spec, version_points))
            inc, _ = run_incremental(
                versioned.spec, version_points, decision.chunk_capacity
            )
            inc_all.extend(inc)
        exact_all.sort(key=lambda item: item.minute)
        inc_all.sort(key=lambda item: item.minute)

        divergence_records = verify_paths(
            spec.service_id, exact_all, inc_all, decision.chunk_capacity, points
        )
        _, _, late_coverage, late_impacts = recompute_with_late(
            spec=spec,
            regular=regular,
            late=late,
            path=PATH_EXACT,
        )
        selected = exact_all if decision.path == PATH_EXACT else inc_all
        all_selected.extend(selected)
        if len(chain) > 1:
            from .params import affected_minutes

            stale_minutes[spec.service_id] = set(affected_minutes(chain, horizon))
        results.append(
            ServiceRunResult(
                service_id=spec.service_id,
                path_decision=decision,
                exact=exact_all,
                incremental=inc_all,
                divergence_records=divergence_records,
                late_impacts=late_impacts,
                coverage=late_coverage,
                selected=selected,
            )
        )
        for item in selected[: min(8, len(selected))]:
            lease_tasks.append(
                ScheduledTask(
                    task_id=f"{spec.service_id}:{item.minute}",
                    service_id=spec.service_id,
                    minute=item.minute,
                    work=lambda lease, conclusion=item: {
                        "slot": lease["slot_id"],
                        "minute": conclusion.minute,
                        "alert": conclusion.alert,
                    },
                )
            )

    schedule_report = None
    if scheduler is not None and lease_tasks:
        schedule_report = scheduler.run(lease_tasks)

    all_points = []
    for service_id, service_points in merged.items():
        for point in service_points:
            all_points.append((service_id, point))
    with Ledger(ledger_path) as ledger:
        commit_seq = ledger.write_commit(
            kind="pipeline",
            conclusions=all_selected,
            points=all_points,
            coverage=coverage,
            adjudications=schedule_report.adjudications if schedule_report else None,
            version_chains=chains,
            stale_minutes=stale_minutes,
            detail={"services": [spec.service_id for spec in specs]},
        )
        latest = ledger.latest_commit()

    return {
        "commit_seq": commit_seq,
        "latest": latest,
        "results": [
            {
                "service_id": result.service_id,
                "path_decision": result.path_decision.to_dict(),
                "divergence_records": len(result.divergence_records),
                "late_impacts": [
                    {
                        "minute": impact.minute,
                        "changed_fields": list(impact.changed_fields),
                    }
                    for impact in result.late_impacts
                ],
                "coverage": [
                    {
                        "minute": record.minute,
                        "covered_line": record.covered_line,
                        "kept_line": record.kept_line,
                    }
                    for record in result.coverage
                ],
                "conclusions": len(result.selected),
            }
            for result in results
        ],
        "schedule": _schedule_summary(schedule_report),
    }


def _schedule_summary(report: object) -> dict[str, Any] | None:
    if report is None:
        return None
    return {
        "global_limit": report.global_limit,
        "per_service_limit": report.per_service_limit,
        "max_concurrent_observed": report.max_concurrent_observed,
        "tasks": [
            {
                "task_id": record.task_id,
                "wait_ms": record.wait_ms,
                "slot_id": record.slot_id,
                "status": record.status,
            }
            for record in report.records
        ],
        "adjudications": [
            {
                "event_seq": decision.event_seq,
                "slot_id": decision.slot_id,
                "service_id": decision.service_id,
                "decision": decision.decision,
                "basis": decision.basis,
                "version": decision.version,
            }
            for decision in report.adjudications
        ],
    }
