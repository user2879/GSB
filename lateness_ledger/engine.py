"""Deterministic event-time window finalization, lateness adjudication and displacement."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Protocol

from .canonical import (
    INT64_MAX,
    digest_object,
    sat_add,
)
from .errors import (
    BudgetError,
    DiagnosticError,
    OrderError,
    SchedulerError,
    WatermarkError,
)
from .fixedpoint import (
    FixedEventValue,
    ScaleContext,
    aggregate_fixed,
    describe_fixed_contract,
    error_attribution,
)
from .partitions import make_window_id, partition_dict, partition_label
from .scheduler import (
    BoundedFinalizationScheduler,
    ScheduledTask,
    SchedulerStats,
)
from .simulate import Event, apply_injection, generate_events
from .spec import LEDGER_SPEC_VERSION, RuleSpec, SessionBundle
from .watermark import WatermarkPoint, WatermarkTracker


@dataclass
class PlannedWindow:
    window_id: str
    rule: RuleSpec
    partition: dict[str, object]
    partition_label: str
    start_ms: int
    nominal_end_ms: int
    end_ms: int
    parent_window_id: str | None
    rule_depth: int
    event_seqs: list[int] = field(default_factory=list)

    def plan_dict(self) -> dict[str, Any]:
        return {
            "window_id": self.window_id,
            "rule_id": self.rule.rule_id,
            "partition_label": self.partition_label,
            "partition": self.partition,
            "start_ms": self.start_ms,
            "nominal_end_ms": self.nominal_end_ms,
            "actual_end_ms": self.end_ms,
            "parent_window_id": self.parent_window_id,
            "rule_depth": self.rule_depth,
        }


@dataclass
class DisplacementStep:
    window_id: str
    path: list[str]
    basis: list[dict[str, Any]]
    changed: bool
    version_before: int
    version_after: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "window_id": self.window_id,
            "path": self.path,
            "basis": self.basis,
            "changed": self.changed,
            "version_before": self.version_before,
            "version_after": self.version_after,
        }


@dataclass
class DisplacementRun:
    displacement_id: str
    root_window_id: str
    reason: str
    event_seq: int
    affected: list[DisplacementStep]
    stable_topology_order: list[str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "displacement_id": self.displacement_id,
            "root_window_id": self.root_window_id,
            "reason": self.reason,
            "event_seq": self.event_seq,
            "affected": [step.to_dict() for step in self.affected],
            "stable_topology_order": self.stable_topology_order,
        }


@dataclass
class RunResult:
    session_id: str
    strategy: str
    scale: int
    rules_version: str
    spec_digest: str
    events_generated: int
    event_cursor: int
    watermarks: list[WatermarkPoint]
    windows: list[dict[str, Any]]
    displacement_runs: list[DisplacementRun]
    scheduler_stats: SchedulerStats
    elapsed_ms: float
    finalization_flush: bool
    fixed_contract: dict[str, Any]
    recalculations: int = 0

    def summary(self) -> dict[str, Any]:
        accepted = sum(
            len(window.get("late_decisions", {}).get("accepted", []))
            for window in self.windows
        )
        discarded = sum(
            len(window.get("late_decisions", {}).get("discarded", []))
            for window in self.windows
        )
        return {
            "session_id": self.session_id,
            "strategy": self.strategy,
            "scale": self.scale,
            "rules_version": self.rules_version,
            "spec_digest": self.spec_digest,
            "events_generated": self.events_generated,
            "event_cursor": self.event_cursor,
            "window_count": len(self.windows),
            "late_accepted": accepted,
            "late_discarded": discarded,
            "displacement_runs": len(self.displacement_runs),
            "recalculations": self.recalculations,
            "scheduler": self.scheduler_stats.to_dict(),
            "elapsed_ms": round(self.elapsed_ms, 3),
            "finalization_flush": self.finalization_flush,
        }


class RunWriter(Protocol):
    def start_run(
        self,
        *,
        bundle: SessionBundle,
        run_id: str,
        events: list[Event],
        plans: list[PlannedWindow],
        initial_watermarks: list[WatermarkPoint],
    ) -> None: ...

    def record_watermarks(self, points: list[WatermarkPoint]) -> None: ...

    def commit_window(self, window: dict[str, Any]) -> None: ...

    def record_displacement(self, displacement: DisplacementRun) -> None: ...

    def complete_run(self, summary: dict[str, Any]) -> None: ...

    def fail_run(self, error: DiagnosticError) -> None: ...


def _window_sort_key(window: PlannedWindow) -> tuple[int, int, str, int, str]:
    return (
        window.end_ms,
        window.rule_depth,
        window.partition_label,
        window.start_ms,
        window.window_id,
    )


def _topology_key(window: PlannedWindow) -> tuple[int, int, str, str]:
    return (window.start_ms, window.rule_depth, window.rule.rule_id, window.window_id)


class WindowEngine:
    def __init__(self, bundle: SessionBundle) -> None:
        self.bundle = bundle
        self.events: list[Event] = []
        self.plans: list[PlannedWindow] = []
        self.plan_by_id: dict[str, PlannedWindow] = {}
        self.memberships_by_seq: dict[int, list[str]] = {}
        self.window_members: dict[str, list[int]] = {}
        self.tracker: WatermarkTracker | None = None
        self.finalized: set[str] = set()
        self.results: dict[str, dict[str, Any]] = {}
        self.finalization_order = 0
        self.cursor = -1
        self.scheduler_stats = SchedulerStats()
        self.displacements: list[DisplacementRun] = []
        self.edges: dict[str, list[str]] = {}
        self.edge_details: dict[tuple[str, str], dict[str, Any]] = {}
        self.recalculations = 0
        self.writer: RunWriter | None = None
        self.run_id = ""
        self.watermark_points: list[WatermarkPoint] = []

    def _plan_windows(self) -> None:
        plans: dict[str, PlannedWindow] = {}
        rule_depths = {
            rule.rule_id: index
            for index, rule in enumerate(self.bundle.topological_rules)
        }
        for rule in self.bundle.topological_rules:
            derivation = self.bundle.derivation_by_child.get(rule.rule_id)
            if derivation is None:
                for event in self.events:
                    partition = partition_dict(event.key_values, rule)
                    start = rule.boundary_origin_ms + (
                        (event.event_time_ms - rule.boundary_origin_ms)
                        // rule.width_ms
                    ) * rule.width_ms
                    window_id = make_window_id(rule.rule_id, partition, start)
                    if window_id not in plans:
                        plans[window_id] = PlannedWindow(
                            window_id=window_id,
                            rule=rule,
                            partition=partition,
                            partition_label=partition_label(partition),
                            start_ms=start,
                            nominal_end_ms=start + rule.width_ms,
                            end_ms=start + rule.width_ms,
                            parent_window_id=None,
                            rule_depth=rule_depths[rule.rule_id],
                        )
            else:
                parent_rule = self.bundle.rule_by_id(derivation.parent_rule_id)
                parent_windows = [
                    plan
                    for plan in plans.values()
                    if plan.rule.rule_id == parent_rule.rule_id
                ]
                for parent in sorted(
                    parent_windows,
                    key=lambda item: (item.partition_label, item.start_ms),
                ):
                    start = parent.start_ms
                    window_id = make_window_id(rule.rule_id, parent.partition, start)
                    if window_id in plans:
                        continue
                    plans[window_id] = PlannedWindow(
                        window_id=window_id,
                        rule=rule,
                        partition=dict(parent.partition),
                        partition_label=parent.partition_label,
                        start_ms=start,
                        nominal_end_ms=start + rule.width_ms,
                        end_ms=parent.end_ms + derivation.offset_ms,
                        parent_window_id=parent.window_id,
                        rule_depth=rule_depths[rule.rule_id],
                    )
        self.plans = sorted(plans.values(), key=lambda item: item.window_id)
        self.plan_by_id = {plan.window_id: plan for plan in self.plans}

    def _build_memberships(self) -> None:
        memberships: dict[int, list[str]] = {
            event.seq: [] for event in self.events
        }
        window_members: dict[str, list[int]] = {
            plan.window_id: [] for plan in self.plans
        }
        for plan in self.plans:
            for event in self.events:
                partition = partition_dict(event.key_values, plan.rule)
                if partition != plan.partition:
                    continue
                if plan.start_ms <= event.event_time_ms < plan.end_ms:
                    memberships[event.seq].append(plan.window_id)
                    window_members[plan.window_id].append(event.seq)
        for seq in memberships:
            memberships[seq].sort()
        for window_id in window_members:
            window_members[window_id].sort()
        self.memberships_by_seq = memberships
        self.window_members = window_members

    def _build_dependency_edges(self) -> None:
        edges: dict[str, list[str]] = {plan.window_id: [] for plan in self.plans}
        detail: dict[tuple[str, str], dict[str, Any]] = {}
        for child_id, derivation in sorted(
            self.bundle.derivation_by_child.items()
        ):
            for plan in self.plans:
                if plan.rule.rule_id != child_id or plan.parent_window_id is None:
                    continue
                parent_id = plan.parent_window_id
                edges[parent_id].append(plan.window_id)
                detail[(parent_id, plan.window_id)] = {
                    "kind": "derivation",
                    "derivation_id": derivation.derivation_id,
                    "reason": "child end boundary is derived from parent conclusion",
                }

        successors: dict[tuple[str, str], list[PlannedWindow]] = {}
        for plan in self.plans:
            successors.setdefault(
                (plan.rule.rule_id, plan.partition_label), []
            ).append(plan)
        for key in sorted(successors):
            same_series = sorted(successors[key], key=lambda item: item.start_ms)
            for left, right in zip(same_series, same_series[1:]):
                edges[left.window_id].append(right.window_id)
                detail[(left.window_id, right.window_id)] = {
                    "kind": "shared_partition_successor",
                    "reason": "later window shares the same rule partition key set",
                }
        for window_id in edges:
            edges[window_id] = sorted(set(edges[window_id]))
        self.edges = edges
        self.edge_details = detail

    def _register_tracker(self) -> None:
        partitions: dict[str, dict[str, object]] = {}
        for plan in self.plans:
            partitions[plan.partition_label] = plan.partition
        self.tracker = WatermarkTracker(self.bundle.strategy, partitions)

    def _partition_labels_for_event(self, event: Event) -> list[str]:
        labels = {
            partition_label(partition_dict(event.key_values, rule))
            for rule in self.bundle.rules
        }
        return sorted(labels)

    def _event_fixed_values(
        self, event: Event
    ) -> tuple[FixedEventValue, int]:
        context = ScaleContext(self.bundle.scale)
        measure_fixed = context.quantize_raw_integer(
            event.measure_raw, event.measure_decimals
        )
        weight_fixed = context.quantize_raw_integer(
            event.weight_raw, event.weight_decimals
        )
        return (
            FixedEventValue(
                event_seq=event.seq,
                event_time_ms=event.event_time_ms,
                measure_fixed=measure_fixed,
                weight_fixed=weight_fixed,
            ),
            measure_fixed,
        )

    def _compute_core(
        self,
        plan: PlannedWindow,
        *,
        trigger_seq: int,
        cursor: int,
    ) -> dict[str, Any]:
        context = ScaleContext(self.bundle.scale)
        budget = self.bundle.budgets.resolve(plan.rule.rule_id, plan.window_id)
        remaining_count = budget.event_count_limit
        remaining_abs = budget.absolute_measure_limit_fixed
        rejected_cumulative = 0
        included: list[FixedEventValue] = []
        accepted_late: list[FixedEventValue] = []
        accepted_evidence: list[dict[str, Any]] = []
        rejected_evidence: list[dict[str, Any]] = []
        event_by_seq = {event.seq: event for event in self.events}

        for seq in self.window_members.get(plan.window_id, []):
            if seq > cursor:
                continue
            event = event_by_seq[seq]
            fixed, measure_fixed = self._event_fixed_values(event)
            if seq <= trigger_seq:
                included.append(fixed)
                continue
            absolute = min(abs(measure_fixed), INT64_MAX)
            if remaining_count >= 1 and remaining_abs >= absolute:
                remaining_count -= 1
                remaining_abs -= absolute
                included.append(fixed)
                accepted_late.append(fixed)
                used_rule = "accepted_within_count_and_absolute_budget"
                decision = "accepted"
            else:
                if remaining_count <= 0:
                    used_rule = "discarded_after_event_count_budget"
                else:
                    used_rule = "discarded_after_absolute_measure_budget"
                decision = "discarded"
                rejected_cumulative = sat_add(rejected_cumulative, absolute)
            evidence = {
                "window_id": plan.window_id,
                "event_id": event.event_id,
                "event_seq": event.seq,
                "source_id": event.source_id,
                "source_event_index": event.source_event_index,
                "event_time_ms": event.event_time_ms,
                "measure_fixed": measure_fixed,
                "absolute_measure_fixed": absolute,
                "decision": decision,
                "rule": used_rule,
                "remaining_event_count": remaining_count,
                "remaining_absolute_measure_fixed": remaining_abs,
                "rejected_cumulative_absolute_fixed": rejected_cumulative,
            }
            if decision == "accepted":
                accepted_evidence.append(evidence)
            else:
                rejected_evidence.append(evidence)

        complete_stats = aggregate_fixed(included)
        stats_payload: dict[str, Any] = {
            "computed_functions": list(plan.rule.aggregate_functions)
        }
        if "count" in plan.rule.aggregate_functions:
            stats_payload["count"] = complete_stats.count
        if "sum" in plan.rule.aggregate_functions:
            stats_payload["sum_fixed"] = complete_stats.sum_fixed
        if "mean" in plan.rule.aggregate_functions:
            stats_payload["mean_fixed"] = complete_stats.mean_fixed
        if "weighted_mean" in plan.rule.aggregate_functions:
            stats_payload["weight_sum_fixed"] = complete_stats.weight_sum_fixed
            stats_payload["weighted_mean_fixed"] = complete_stats.weighted_mean_fixed
        errors = error_attribution(
            included=included,
            accepted_late=accepted_late,
            nominal_end_ms=plan.nominal_end_ms,
            scale=self.bundle.scale,
        )
        return {
            "stats": stats_payload,
            "late_decisions": {
                "budget_initial": budget.to_dict(),
                "budget_remaining": {
                    "event_count": remaining_count,
                    "absolute_measure_fixed": remaining_abs,
                },
                "accepted": accepted_evidence,
                "discarded": rejected_evidence,
                "accepted_count": len(accepted_evidence),
                "discarded_count": len(rejected_evidence),
            },
            "errors": errors,
            "included_event_count": len(included),
            "quantized_measure_sum_fixed": complete_stats.sum_fixed,
            "progress_cursor": cursor,
            "fixed_contract": describe_fixed_contract(self.bundle.scale),
            "quantum": str(context.quantum),
        }

    def _payload(
        self,
        plan: PlannedWindow,
        core: dict[str, Any],
        *,
        finalization_order: int,
        trigger_seq: int,
        trigger_event_id: str,
        version: int,
        previous: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        payload = {
            "format_version": LEDGER_SPEC_VERSION,
            "window_id": plan.window_id,
            "rule_id": plan.rule.rule_id,
            "partition": plan.partition,
            "partition_label": plan.partition_label,
            "start_ms": plan.start_ms,
            "nominal_end_ms": plan.nominal_end_ms,
            "end_ms": plan.end_ms,
            "parent_window_id": plan.parent_window_id,
            "finalization_order": finalization_order,
            "trigger_seq": trigger_seq,
            "trigger_event_id": trigger_event_id,
            "strategy": self.bundle.strategy.kind,
            "rules_version": self.bundle.rules_version,
            "scale": self.bundle.scale,
            "version": version,
            "status": "finalized",
            "displacement_trajectory": list(
                (previous or {}).get("displacement_trajectory", [])
            ),
            **core,
        }
        return payload

    @staticmethod
    def _conclusion_signature(payload: dict[str, Any]) -> str:
        comparable = {
            key: value
            for key, value in payload.items()
            if key not in {"version", "digest", "progress_cursor"}
        }
        return digest_object(comparable)

    def _store_payload(self, payload: dict[str, Any]) -> dict[str, Any]:
        payload = dict(payload)
        payload.pop("digest", None)
        payload["digest"] = digest_object(payload)
        self.results[payload["window_id"]] = payload
        return payload

    def _record_watermarks(self, points: list[WatermarkPoint]) -> None:
        self.watermark_points.extend(points)
        if self.writer is not None:
            self.writer.record_watermarks(points)

    def _commit_changed_window(self, payload: dict[str, Any]) -> None:
        if self.writer is not None:
            self.writer.commit_window(payload)

    def _finalize_due(
        self,
        *,
        trigger_seq: int,
        trigger_event_id: str,
        logical_global: int,
    ) -> int:
        assert self.tracker is not None
        finalized_any = 0
        while True:
            eligible: list[PlannedWindow] = []
            for plan in self.plans:
                if plan.window_id in self.finalized:
                    continue
                if plan.parent_window_id and plan.parent_window_id not in self.finalized:
                    continue
                if plan.end_ms <= logical_global - plan.rule.allowed_lateness_ms:
                    eligible.append(plan)
            if not eligible:
                break
            eligible.sort(key=_window_sort_key)
            scheduler = BoundedFinalizationScheduler(
                global_limit=self.bundle.scheduler.global_limit,
                per_partition_limit=self.bundle.scheduler.per_partition_limit,
                task_timeout_ms=self.bundle.scheduler.task_timeout_ms,
                retries=self.bundle.scheduler.retries,
            )
            tasks = [
                ScheduledTask(
                    task_id=plan.window_id,
                    partition_key=plan.partition_label,
                    function=lambda plan=plan: self._compute_core(
                        plan,
                        trigger_seq=trigger_seq,
                        cursor=trigger_seq,
                    ),
                )
                for plan in eligible
            ]
            outcomes, stats = scheduler.run(tasks)
            self.scheduler_stats.merge(stats)
            for outcome, plan in zip(outcomes, eligible):
                if outcome.error is not None or outcome.value is None:
                    raise SchedulerError(
                        "window finalization task failed",
                        details={**outcome.to_dict(), "window_id": plan.window_id},
                    )
                self.finalization_order += 1
                payload = self._payload(
                    plan,
                    outcome.value,
                    finalization_order=self.finalization_order,
                    trigger_seq=trigger_seq,
                    trigger_event_id=trigger_event_id,
                    version=1,
                )
                stored = self._store_payload(payload)
                self.finalized.add(plan.window_id)
                self._commit_changed_window(stored)
                finalized_any += 1
        return finalized_any

    def _dependency_closure(self, root_id: str) -> tuple[list[str], dict[str, list[str]]]:
        paths: dict[str, list[str]] = {root_id: [root_id]}
        queue = [root_id]
        seen = {root_id}
        order: list[str] = []
        while queue:
            current = queue.pop(0)
            for neighbor in self.edges.get(current, []):
                if neighbor in seen:
                    continue
                seen.add(neighbor)
                paths[neighbor] = paths[current] + [neighbor]
                queue.append(neighbor)
                order.append(neighbor)
        return order, paths

    def _basis_for_path(self, path: list[str]) -> list[dict[str, Any]]:
        basis: list[dict[str, Any]] = []
        for left, right in zip(path, path[1:]):
            detail = self.edge_details.get(
                (left, right),
                {"kind": "unknown", "reason": "dependency edge"},
            )
            basis.append(
                {
                    "from_window_id": left,
                    "to_window_id": right,
                    **detail,
                }
            )
        return basis

    def _propagate_displacement(
        self,
        *,
        root_id: str,
        reason: str,
        event_seq: int,
        trigger_event_id: str,
    ) -> DisplacementRun:
        descendants, paths = self._dependency_closure(root_id)
        plans = [self.plan_by_id[window_id] for window_id in descendants]
        plans.sort(key=_topology_key)
        affected: list[DisplacementStep] = []
        for plan in plans:
            previous = self.results.get(plan.window_id)
            if previous is None or plan.window_id not in self.finalized:
                affected.append(
                    DisplacementStep(
                        window_id=plan.window_id,
                        path=paths[plan.window_id],
                        basis=self._basis_for_path(paths[plan.window_id]),
                        changed=False,
                        version_before=0,
                        version_after=0,
                    )
                )
                continue
            core = self._compute_core(
                plan,
                trigger_seq=int(previous["trigger_seq"]),
                cursor=self.cursor,
            )
            candidate = self._payload(
                plan,
                core,
                finalization_order=int(previous["finalization_order"]),
                trigger_seq=int(previous["trigger_seq"]),
                trigger_event_id=str(previous["trigger_event_id"]),
                version=int(previous["version"]),
                previous=previous,
            )
            before_signature = self._conclusion_signature(previous)
            after_signature = self._conclusion_signature(candidate)
            changed = before_signature != after_signature
            version_before = int(previous["version"])
            if changed:
                version_after = version_before + 1
                candidate["version"] = version_after
                stored = self._store_payload(candidate)
                self.recalculations += 1
            else:
                version_after = version_before
            affected.append(
                DisplacementStep(
                    window_id=plan.window_id,
                    path=paths[plan.window_id],
                    basis=self._basis_for_path(paths[plan.window_id]),
                    changed=changed,
                    version_before=version_before,
                    version_after=version_after,
                )
            )
        displacement = DisplacementRun(
            displacement_id=(
                f"d{event_seq:08d}:{digest_object([root_id, reason])[:16]}"
            ),
            root_window_id=root_id,
            reason=reason,
            event_seq=event_seq,
            affected=affected,
            stable_topology_order=[step.window_id for step in affected],
        )
        self.displacements.append(displacement)
        return displacement

    def _attach_displacement_and_commit(
        self,
        displacement: DisplacementRun,
        root_step: DisplacementStep,
    ) -> None:
        """Attach the trajectory and commit each changed window atomically."""

        entries: list[tuple[DisplacementStep, dict[str, Any]]] = []
        root_payload = self.results[displacement.root_window_id]
        entries.append((root_step, root_payload))
        changed_steps = [step for step in displacement.affected if step.changed]
        for step in changed_steps:
            payload = self.results.get(step.window_id)
            if payload is not None:
                entries.append((step, payload))
        for step, payload in entries:
            trajectory_entry = {
                "displacement_id": displacement.displacement_id,
                "root_window_id": displacement.root_window_id,
                "reason": displacement.reason,
                "event_seq": displacement.event_seq,
                "path": step.path,
                "basis": step.basis,
                "changed": step.changed,
                "version_before": step.version_before,
                "version_after": step.version_after,
            }
            updated = dict(payload)
            updated["displacement_trajectory"] = list(
                updated.get("displacement_trajectory", [])
            ) + [trajectory_entry]
            stored = self._store_payload(updated)
            self._commit_changed_window(stored)
        if self.writer is not None:
            self.writer.record_displacement(displacement)

    def _apply_late_event(self, plan: PlannedWindow, event: Event) -> str | None:
        previous = self.results.get(plan.window_id)
        if previous is None:
            return None
        core = self._compute_core(
            plan,
            trigger_seq=int(previous["trigger_seq"]),
            cursor=event.seq,
        )
        candidate = self._payload(
            plan,
            core,
            finalization_order=int(previous["finalization_order"]),
            trigger_seq=int(previous["trigger_seq"]),
            trigger_event_id=str(previous["trigger_event_id"]),
            version=int(previous["version"]),
            previous=previous,
        )
        if self._conclusion_signature(candidate) == self._conclusion_signature(previous):
            return None
        version = int(previous["version"]) + 1
        candidate["version"] = version
        stored = self._store_payload(candidate)
        self.recalculations += 1
        decisions = core["late_decisions"]
        decision = next(
            (
                item
                for item in decisions["accepted"] + decisions["discarded"]
                if int(item["event_seq"]) == event.seq
            ),
            None,
        )
        if decision is None:
            return None
        if decision["decision"] == "accepted":
            displacement = self._propagate_displacement(
                root_id=plan.window_id,
                reason=f"late_accept:{event.event_id}",
                event_seq=event.seq,
                trigger_event_id=event.event_id,
            )
            root_step = DisplacementStep(
                window_id=plan.window_id,
                path=[plan.window_id],
                basis=[],
                changed=True,
                version_before=version - 1,
                version_after=version,
            )
            self._attach_displacement_and_commit(displacement, root_step)
            return "accepted"
        self._commit_changed_window(stored)
        return "discarded"

    def _handle_late_events(self, event: Event) -> None:
        for window_id in self.memberships_by_seq.get(event.seq, []):
            if window_id not in self.finalized:
                continue
            plan = self.plan_by_id[window_id]
            self._apply_late_event(plan, event)

    def _raise_injected_budget_if_needed(self, event: Event) -> None:
        if event.injection != "budget":
            return
        finalized_members = [
            window_id
            for window_id in self.memberships_by_seq.get(event.seq, [])
            if window_id in self.finalized
        ]
        target = finalized_members[0] if finalized_members else None
        if target is None and self.finalized:
            target = sorted(self.finalized)[0]
        if target is None:
            return
        payload = self.results[target]
        raise BudgetError(
            "injected late-budget overrun: remaining event budget forced below zero",
            details={
                "window_id": target,
                "event_id": event.event_id,
                "event_seq": event.seq,
                "injected": True,
                "forced_remaining_event_count": -1,
                "prior_remaining": payload["late_decisions"]["budget_remaining"],
                "window_end_ms": payload["end_ms"],
                "partition": payload["partition"],
            },
        )

    def _raise_injected_order_if_needed(self, event: Event) -> None:
        if event.injection != "order":
            return
        raise OrderError(
            "injected event stream timestamp regression",
            details={
                "source_id": event.source_id,
                "event_id": event.event_id,
                "event_seq": event.seq,
                "event_time_ms": event.event_time_ms,
                "injected": True,
                "declared_ppm_exceeded": True,
            },
        )

    def _flush_end_of_stream(self) -> bool:
        assert self.tracker is not None
        if not self.plans or not self.events:
            return False
        last_event = self.events[-1]
        target = max(
            plan.end_ms + plan.rule.allowed_lateness_ms for plan in self.plans
        )
        levels = self.tracker.flush_globals(target)
        for level in levels:
            points = self.tracker.snapshot(
                event_seq=last_event.seq,
                event_id=last_event.event_id,
                phase="eos",
                global_override=level,
            )
            self._record_watermarks(points)
            self._finalize_due(
                trigger_seq=last_event.seq,
                trigger_event_id=last_event.event_id,
                logical_global=level,
            )
        # An equality-offset derivation can require one more dependency pass.
        while any(plan_id not in self.finalized for plan_id in self.plan_by_id):
            due = [
                plan
                for plan in self.plans
                if plan.window_id not in self.finalized
                and (
                    plan.parent_window_id is None
                    or plan.parent_window_id in self.finalized
                )
            ]
            if not due:
                break
            self._finalize_due(
                trigger_seq=last_event.seq,
                trigger_event_id=last_event.event_id,
                logical_global=max(levels[-1], target + 1),
            )
            if not any(
                plan.window_id not in self.finalized for plan in self.plans
            ):
                break
        return True

    def run(
        self,
        *,
        writer: RunWriter | None = None,
        injection: str | None = None,
        cursor_limit: int | None = None,
        flush_eos: bool = True,
    ) -> RunResult:
        started = time.monotonic_ns()
        self.writer = writer
        self.events = apply_injection(generate_events(self.bundle), injection)
        events_generated = len(self.events)
        if cursor_limit is not None:
            self.events = self.events[: max(0, cursor_limit)]
            if cursor_limit != events_generated:
                flush_eos = False
        self.cursor = self.events[-1].seq if self.events else -1
        self._plan_windows()
        self._build_memberships()
        self._build_dependency_edges()
        self._register_tracker()
        assert self.tracker is not None
        initial = self.tracker.snapshot(
            event_seq=-1,
            event_id="<init>",
            phase="init",
        )
        self.watermark_points.extend(initial)
        self.run_id = digest_object(
            [self.bundle.session_id, self.bundle.spec_digest, events_generated]
        )[:24]
        if writer is not None:
            writer.start_run(
                bundle=self.bundle,
                run_id=self.run_id,
                events=self.events,
                plans=self.plans,
                initial_watermarks=initial,
            )

        for event in self.events:
            self.cursor = event.seq
            self._raise_injected_order_if_needed(event)
            self._raise_injected_budget_if_needed(event)
            self._handle_late_events(event)
            labels = self._partition_labels_for_event(event)
            for label in labels:
                self.tracker.advance(label, event.event_time_ms)
            if event.injection == "watermark":
                try:
                    self.tracker.inject_watermark_gap(labels[0] if labels else "")
                except WatermarkError as exc:
                    exc.details.update(
                        {
                            "event_id": event.event_id,
                            "event_seq": event.seq,
                            "source_id": event.source_id,
                        }
                    )
                    raise
            points = self.tracker.snapshot(
                event_seq=event.seq,
                event_id=event.event_id,
                phase="event",
            )
            self._record_watermarks(points)
            self._finalize_due(
                trigger_seq=event.seq,
                trigger_event_id=event.event_id,
                logical_global=self.tracker.global_watermark,
            )

        flushed = self._flush_end_of_stream() if flush_eos else False
        result = RunResult(
            session_id=self.bundle.session_id,
            strategy=self.bundle.strategy.kind,
            scale=self.bundle.scale,
            rules_version=self.bundle.rules_version,
            spec_digest=self.bundle.spec_digest,
            events_generated=events_generated,
            event_cursor=self.events[-1].seq if self.events else -1,
            watermarks=self.watermark_points,
            windows=[
                self.results[window_id]
                for window_id in sorted(
                    self.results,
                    key=lambda item: (
                        int(self.results[item]["finalization_order"]),
                        item,
                    ),
                )
            ],
            displacement_runs=self.displacements,
            scheduler_stats=self.scheduler_stats,
            elapsed_ms=(time.monotonic_ns() - started) / 1_000_000.0,
            finalization_flush=flushed,
            fixed_contract=describe_fixed_contract(self.bundle.scale),
            recalculations=self.recalculations,
        )
        if writer is not None:
            writer.complete_run(result.summary())
        return result


def run_engine(
    bundle: SessionBundle,
    *,
    writer: RunWriter | None = None,
    injection: str | None = None,
    cursor_limit: int | None = None,
    flush_eos: bool = True,
) -> RunResult:
    return WindowEngine(bundle).run(
        writer=writer,
        injection=injection,
        cursor_limit=cursor_limit,
        flush_eos=flush_eos,
    )
