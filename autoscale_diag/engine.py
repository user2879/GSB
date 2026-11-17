"""Deterministic minute-by-minute autoscaling decision engine."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, Iterable

from .aggregation import (
    Aggregator,
    PathName,
    WindowStats,
    choose_path,
    make_aggregator,
)
from .errors import BusinessError, E_LEDGER
from .spec import ClusterSpec, Observation, Rule, Workload


BUDGET_POLICY = "proportional-largest-remainder-v1"
DECISION_EXPAND = "expand"
DECISION_SHRINK = "shrink"
DECISION_HOLD = "hold"


def ceil_div(numerator: int, denominator: int) -> int:
    if denominator <= 0:
        raise ValueError("denominator must be positive")
    return (numerator + denominator - 1) // denominator


@dataclass(frozen=True)
class WorkloadState:
    replicas: int
    last_change_minute: int | None = None


@dataclass(frozen=True)
class RuleEvaluation:
    rule_id: str
    metric: str
    observed_value: int
    target_value: int
    per_replica_capacity: int
    demand_numerator: int
    demand_denominator: int
    unclamped_replicas: int
    clamped_replicas: int
    quantization_error_upper_bound_milli_replicas: int
    quantization_error_upper_bound_metric_units: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "rule_id": self.rule_id,
            "metric": self.metric,
            "observed_value": self.observed_value,
            "target_value": self.target_value,
            "per_replica_capacity": self.per_replica_capacity,
            "effective_capacity_numerator": self.per_replica_capacity
            * self.target_value,
            "effective_capacity_denominator": 100,
            "demand_numerator": self.demand_numerator,
            "demand_denominator": self.demand_denominator,
            "unclamped_replicas": self.unclamped_replicas,
            "clamped_replicas": self.clamped_replicas,
            "quantization_error_upper_bound_milli_replicas": (
                self.quantization_error_upper_bound_milli_replicas
            ),
            "quantization_error_upper_bound_metric_units": (
                self.quantization_error_upper_bound_metric_units
            ),
        }


@dataclass(frozen=True)
class BudgetAllocation:
    desired: dict[str, int]
    final: dict[str, int]
    clipped: dict[str, int]
    affected_workloads: tuple[str, ...]
    fixed_workloads: tuple[str, ...]
    requested_total: int
    allocated_total: int
    budget: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "policy": BUDGET_POLICY,
            "budget": self.budget,
            "requested_total": self.requested_total,
            "allocated_total": self.allocated_total,
            "clipped": dict(sorted(self.clipped.items())),
            "affected_workloads": list(self.affected_workloads),
            "fixed_cooldown_workloads": list(self.fixed_workloads),
            "desired": dict(sorted(self.desired.items())),
            "final": dict(sorted(self.final.items())),
        }


@dataclass(frozen=True)
class WorkloadDecision:
    minute_index: int
    minute: int
    workload_id: str
    observation: Observation
    window_stats: WindowStats
    rule_evaluations: tuple[RuleEvaluation, ...]
    selected_rule_id: str
    expected_replicas: int
    previous_replicas: int
    attempted_replicas: int
    budget_target_replicas: int
    final_replicas: int
    decision_type: str
    budget_clipped: bool
    budget_clip_amount: int
    budget_affected_workloads: tuple[str, ...]
    cooldown_active: bool
    cooldown_suppressed: bool
    cooldown_code: str | None
    cooldown_trigger_minute: int | None
    cooldown_until_minute: int | None
    cooldown_message: str | None
    change_took_effect: bool
    evidence: tuple[str, ...] = field(default_factory=tuple)

    def to_dict(self) -> dict[str, Any]:
        return {
            "minute_index": self.minute_index,
            "minute": self.minute,
            "workload_id": self.workload_id,
            "observation": self.observation.to_dict(),
            "window_stats": self.window_stats.to_dict(),
            "rule_evaluations": [item.to_dict() for item in self.rule_evaluations],
            "selected_rule_id": self.selected_rule_id,
            "expected_replicas": self.expected_replicas,
            "previous_replicas": self.previous_replicas,
            "attempted_replicas": self.attempted_replicas,
            "budget_target_replicas": self.budget_target_replicas,
            "final_replicas": self.final_replicas,
            "decision_type": self.decision_type,
            "budget_clipped": self.budget_clipped,
            "budget_clip_amount": self.budget_clip_amount,
            "budget_affected_workloads": list(self.budget_affected_workloads),
            "cooldown_active": self.cooldown_active,
            "cooldown_suppressed": self.cooldown_suppressed,
            "cooldown_code": self.cooldown_code,
            "cooldown_trigger_minute": self.cooldown_trigger_minute,
            "cooldown_until_minute": self.cooldown_until_minute,
            "cooldown_message": self.cooldown_message,
            "change_took_effect": self.change_took_effect,
            "evidence": list(self.evidence),
        }

    def replay_core(self) -> dict[str, Any]:
        return {
            "observation": self.observation.to_dict(),
            "expected_replicas": self.expected_replicas,
            "budget_target_replicas": self.budget_target_replicas,
            "attempted_replicas": self.attempted_replicas,
            "final_replicas": self.final_replicas,
            "decision_type": self.decision_type,
            "cooldown_suppressed": self.cooldown_suppressed,
            "cooldown_code": self.cooldown_code,
            "cooldown_trigger_minute": self.cooldown_trigger_minute,
            "cooldown_message": self.cooldown_message,
            "selected_rule_id": self.selected_rule_id,
            "window_stats": self.window_stats.to_dict(),
            "rule_evaluations": [item.to_dict() for item in self.rule_evaluations],
        }


@dataclass(frozen=True)
class MinuteResult:
    minute_index: int
    minute: int
    decisions: tuple[WorkloadDecision, ...]
    budget: BudgetAllocation
    path: PathName
    window_size: int
    peak_memory_estimate_bytes: int

    @property
    def budget_used(self) -> int:
        return sum(item.final_replicas for item in self.decisions)

    def to_dict(self) -> dict[str, Any]:
        return {
            "minute_index": self.minute_index,
            "minute": self.minute,
            "path": self.path,
            "window_capacity": self.window_size,
            "peak_memory_estimate_bytes": self.peak_memory_estimate_bytes,
            "budget_used": self.budget_used,
            "budget": self.budget.to_dict(),
            "decisions": [item.to_dict() for item in self.decisions],
        }

    def canonical_payload(self) -> dict[str, Any]:
        """Path-independent payload used to prove full/incremental parity."""

        return {
            "minute_index": self.minute_index,
            "minute": self.minute,
            "budget": self.budget.to_dict(),
            "budget_used": self.budget_used,
            "decisions": [item.to_dict() for item in self.decisions],
        }


@dataclass(frozen=True)
class RunResult:
    path: PathName
    window_size: int
    peak_memory_estimate_bytes: int
    minutes: tuple[MinuteResult, ...]
    initial_states: dict[str, WorkloadState]
    final_states: dict[str, WorkloadState]

    def canonical_json(self) -> str:
        return json.dumps(
            [item.canonical_payload() for item in self.minutes],
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )

    def decision_digest(self) -> str:
        return hashlib.sha256(self.canonical_json().encode("utf-8")).hexdigest()

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "window_capacity": self.window_size,
            "peak_memory_estimate_bytes": self.peak_memory_estimate_bytes,
            "decision_digest": self.decision_digest(),
            "minutes": [item.to_dict() for item in self.minutes],
        }


def _evaluate_rule(rule: Rule, stats: WindowStats, workload: Workload) -> RuleEvaluation:
    observed = stats.metric_value(rule.metric)
    demand_numerator = observed * 100
    demand_denominator = rule.per_replica_capacity * rule.target_value
    unclamped = ceil_div(demand_numerator, demand_denominator)
    clamped = min(max(unclamped, workload.min_replicas), workload.max_replicas)
    remainder = unclamped * demand_denominator - demand_numerator
    milli_error = ceil_div(remainder * 1000, demand_denominator)
    metric_error = ceil_div(remainder, 100)
    return RuleEvaluation(
        rule_id=rule.id,
        metric=rule.metric,
        observed_value=observed,
        target_value=rule.target_value,
        per_replica_capacity=rule.per_replica_capacity,
        demand_numerator=demand_numerator,
        demand_denominator=demand_denominator,
        unclamped_replicas=unclamped,
        clamped_replicas=clamped,
        quantization_error_upper_bound_milli_replicas=milli_error,
        quantization_error_upper_bound_metric_units=metric_error,
    )


def evaluate_workload(
    workload: Workload, stats: WindowStats
) -> tuple[tuple[RuleEvaluation, ...], str, int]:
    evaluations = tuple(
        _evaluate_rule(rule, stats, workload)
        for rule in sorted(workload.rules, key=lambda item: item.id)
    )
    expected = max(item.clamped_replicas for item in evaluations)
    selected = next(
        item.rule_id
        for item in sorted(evaluations, key=lambda item: (-item.clamped_replicas, item.rule_id))
        if item.clamped_replicas == expected
    )
    return evaluations, selected, expected


def allocate_budget(
    workload_ids: Iterable[str],
    desired: dict[str, int],
    minimums: dict[str, int],
    maximums: dict[str, int],
    budget: int,
    fixed: dict[str, int] | None = None,
) -> BudgetAllocation:
    """Allocate a hard global budget with a deterministic largest-remainder rule.

    Workload IDs are sorted before all arithmetic.  Fixed workloads are normally
    cooldown-locked at their previous replicas.  Variable workloads first get
    their minimum, then the remaining budget is shared in proportion to each
    workload's desired replicas above its minimum.  Unallocated whole replicas
    go to the largest remainders, with workload ID as the final tie-breaker.
    """

    ids = tuple(sorted(workload_ids))
    fixed = dict(fixed or {})
    fixed_total = sum(fixed.values())
    variable_ids = tuple(item for item in ids if item not in fixed)
    targets = {item: max(minimums[item], desired.get(item, minimums[item])) for item in ids}
    targets = {item: min(targets[item], maximums[item]) for item in ids}
    requested_total = fixed_total + sum(targets[item] for item in variable_ids)

    if requested_total <= budget:
        final = dict(fixed)
        final.update({item: targets[item] for item in variable_ids})
        final = {item: final[item] for item in ids}
        return BudgetAllocation(
            desired=targets,
            final=final,
            clipped={},
            affected_workloads=(),
            fixed_workloads=tuple(sorted(fixed)),
            requested_total=requested_total,
            allocated_total=sum(final.values()),
            budget=budget,
        )

    minimum_total = sum(minimums[item] for item in variable_ids)
    remaining = budget - fixed_total
    if remaining < minimum_total:
        raise BusinessError(
            E_LEDGER,
            "global budget cannot cover fixed workloads and variable minima",
            details={
                "budget": budget,
                "fixed_total": fixed_total,
                "minimum_total_for_variable_workloads": minimum_total,
            },
        )

    extra_budget = remaining - minimum_total
    weights = {item: targets[item] - minimums[item] for item in variable_ids}
    total_weight = sum(weights.values())
    if extra_budget >= total_weight:
        final = dict(fixed)
        final.update({item: targets[item] for item in variable_ids})
        final = {item: final[item] for item in ids}
        return BudgetAllocation(
            desired=targets,
            final=final,
            clipped={},
            affected_workloads=(),
            fixed_workloads=tuple(sorted(fixed)),
            requested_total=requested_total,
            allocated_total=sum(final.values()),
            budget=budget,
        )

    extras = {
        item: (extra_budget * weights[item]) // total_weight if total_weight else 0
        for item in variable_ids
    }
    remainders = {
        item: (extra_budget * weights[item]) % total_weight if total_weight else 0
        for item in variable_ids
    }
    leftover = extra_budget - sum(extras.values())
    order = sorted(
        (item for item in variable_ids if weights[item] > 0),
        key=lambda item: (-remainders[item], item),
    )
    for item in order[:leftover]:
        extras[item] += 1

    final = dict(fixed)
    for item in variable_ids:
        final[item] = minimums[item] + extras[item]
    final = {item: final[item] for item in ids}
    clipped = {
        item: targets[item] - final[item]
        for item in ids
        if final[item] < targets[item]
    }
    return BudgetAllocation(
        desired=targets,
        final=final,
        clipped=clipped,
        affected_workloads=tuple(sorted(clipped)),
        fixed_workloads=tuple(sorted(fixed)),
        requested_total=requested_total,
        allocated_total=sum(final.values()),
        budget=budget,
    )


class Simulation:
    """Stateful engine.  A resume primes the aggregator from committed history."""

    def __init__(
        self,
        spec: ClusterSpec,
        path: PathName | None = None,
        *,
        requested_path: str = "auto",
        states: dict[str, WorkloadState] | None = None,
    ):
        self.spec = spec
        self.path = path or choose_path(spec.minute_count, spec.aggregation_threshold, requested_path)
        self.workload_ids = tuple(sorted(item.id for item in spec.workloads))
        self.workloads = {item.id: item for item in spec.workloads}
        self.aggregator: Aggregator = make_aggregator(
            self.path, list(self.workload_ids), spec.window_size
        )
        self.states = {
            workload_id: (
                states[workload_id]
                if states and workload_id in states
                else WorkloadState(self.workloads[workload_id].min_replicas, None)
            )
            for workload_id in self.workload_ids
        }
        self.index = 0

    def prime(self, committed_minutes: int) -> None:
        """Consume observations needed for window state without making decisions."""

        if committed_minutes < 0 or committed_minutes > self.spec.minute_count:
            raise ValueError("committed_minutes out of range")
        for index in range(self.index, committed_minutes):
            for workload_id in self.workload_ids:
                self.aggregator.add(
                    workload_id, self.workloads[workload_id].observations[index]
                )
        self.index = committed_minutes

    def step(self) -> MinuteResult:
        if self.index >= self.spec.minute_count:
            raise StopIteration("simulation is complete")
        index = self.index
        observations: dict[str, Observation] = {
            workload_id: self.workloads[workload_id].observations[index]
            for workload_id in self.workload_ids
        }
        minute = observations[self.workload_ids[0]].minute
        stats: dict[str, WindowStats] = {
            workload_id: self.aggregator.add(workload_id, observations[workload_id])
            for workload_id in self.workload_ids
        }
        evaluations: dict[str, tuple[tuple[RuleEvaluation, ...], str, int]] = {
            workload_id: evaluate_workload(self.workloads[workload_id], stats[workload_id])
            for workload_id in self.workload_ids
        }
        expected = {workload_id: evaluations[workload_id][2] for workload_id in self.workload_ids}
        minimums = {item: self.workloads[item].min_replicas for item in self.workload_ids}
        maximums = {item: self.workloads[item].max_replicas for item in self.workload_ids}

        # A provisional allocation exposes what budget would have done before
        # the cooldown lock.  It is retained as direct suppression evidence.
        provisional = allocate_budget(
            self.workload_ids,
            expected,
            minimums,
            maximums,
            self.spec.global_replica_budget,
        )

        cooldown_active: dict[str, bool] = {}
        fixed: dict[str, int] = {}
        for workload_id in self.workload_ids:
            state = self.states[workload_id]
            workload = self.workloads[workload_id]
            active = (
                state.last_change_minute is not None
                and (minute - state.last_change_minute) * 60_000 < workload.cooldown_ms
            )
            cooldown_active[workload_id] = active
            if active:
                fixed[workload_id] = state.replicas

        # Cooldown workloads are fixed before the final allocation, so their
        # previous replicas can never be overridden by a budget-driven change
        # and the aggregate can never exceed the global budget.
        allocation = allocate_budget(
            self.workload_ids,
            expected,
            minimums,
            maximums,
            self.spec.global_replica_budget,
            fixed=fixed,
        )

        decisions: list[WorkloadDecision] = []
        new_states: dict[str, WorkloadState] = {}
        for workload_id in self.workload_ids:
            workload = self.workloads[workload_id]
            state = self.states[workload_id]
            final = allocation.final[workload_id]
            previous = state.replicas
            evaluation, selected_rule_id, expected_value = evaluations[workload_id]
            attempted = provisional.final[workload_id]
            suppressed = cooldown_active[workload_id] and attempted != previous
            if final > previous:
                decision_type = DECISION_EXPAND
            elif final < previous:
                decision_type = DECISION_SHRINK
            else:
                decision_type = DECISION_HOLD
            change_took_effect = (not cooldown_active[workload_id]) and final != previous

            trigger_minute: int | None = None
            until_minute: int | None = None
            message: str | None = None
            if cooldown_active[workload_id]:
                trigger_minute = state.last_change_minute
                until_minute = (state.last_change_minute or minute) + ceil_div(
                    workload.cooldown_ms, 60_000
                )
                if suppressed:
                    message = (
                        f"E_COOLDOWN: attempted {attempted} replicas but final remains "
                        f"{previous}; change cooldown from minute {trigger_minute} "
                        f"ends at minute {until_minute}"
                    )

            budget_clipped = workload_id not in fixed and final < expected_value
            clip_amount = expected_value - final if budget_clipped else 0
            evidence = [
                f"observation minute={minute} requests={observations[workload_id].requests} "
                f"latency_ms={observations[workload_id].avg_latency_ms} "
                f"errors={observations[workload_id].errors}",
                f"concurrency_per_replica={workload.concurrency_per_replica}",
                "window "
                f"samples={stats[workload_id].samples} "
                f"avg_requests={stats[workload_id].avg_requests} "
                f"avg_latency_ms={stats[workload_id].avg_latency_ms} "
                f"error_rate_ppm={stats[workload_id].error_rate_ppm}",
                f"rule={selected_rule_id} expected={expected_value} "
                f"range=[{workload.min_replicas},{workload.max_replicas}]",
                f"provisional_budget_target={attempted} budget={self.spec.global_replica_budget}",
            ]
            if cooldown_active[workload_id]:
                evidence.append(
                    f"cooldown_guard active trigger_minute={trigger_minute} "
                    f"until_minute={until_minute}"
                )
            if suppressed:
                evidence.append(message or "E_COOLDOWN")
            if budget_clipped:
                evidence.append(
                    f"budget_clip final={final} clip_amount={clip_amount} "
                    f"affected={list(allocation.affected_workloads)}"
                )
            evidence.append(f"final={final} decision={decision_type}")

            decisions.append(
                WorkloadDecision(
                    minute_index=index,
                    minute=minute,
                    workload_id=workload_id,
                    observation=observations[workload_id],
                    window_stats=stats[workload_id],
                    rule_evaluations=evaluation,
                    selected_rule_id=selected_rule_id,
                    expected_replicas=expected_value,
                    previous_replicas=previous,
                    attempted_replicas=attempted,
                    budget_target_replicas=allocation.final[workload_id],
                    final_replicas=final,
                    decision_type=decision_type,
                    budget_clipped=budget_clipped,
                    budget_clip_amount=clip_amount,
                    budget_affected_workloads=allocation.affected_workloads,
                    cooldown_active=cooldown_active[workload_id],
                    cooldown_suppressed=suppressed,
                    cooldown_code="E_COOLDOWN" if suppressed else None,
                    cooldown_trigger_minute=trigger_minute,
                    cooldown_until_minute=until_minute,
                    cooldown_message=message,
                    change_took_effect=change_took_effect,
                    evidence=tuple(evidence),
                )
            )
            new_states[workload_id] = WorkloadState(
                replicas=final,
                last_change_minute=(
                    minute if change_took_effect else state.last_change_minute
                ),
            )

        self.states = new_states
        self.index += 1
        return MinuteResult(
            minute_index=index,
            minute=minute,
            decisions=tuple(decisions),
            budget=allocation,
            path=self.path,
            window_size=self.spec.window_size,
            peak_memory_estimate_bytes=self.aggregator.peak_memory_estimate_bytes,
        )


def simulate(
    spec: ClusterSpec,
    *,
    requested_path: str = "auto",
    forced_path: PathName | None = None,
    states: dict[str, WorkloadState] | None = None,
    start_index: int = 0,
) -> RunResult:
    """Run a complete or resumed simulation.

    ``states`` and ``start_index`` resume from a fully committed ledger prefix.
    The aggregator is primed only from observations in that prefix.
    """

    engine = Simulation(spec, forced_path, requested_path=requested_path, states=states)
    if start_index:
        engine.prime(start_index)
    initial_states = dict(engine.states)
    minutes: list[MinuteResult] = []
    for _ in range(engine.index, spec.minute_count):
        minutes.append(engine.step())
    return RunResult(
        path=engine.path,
        window_size=spec.window_size,
        peak_memory_estimate_bytes=engine.aggregator.peak_memory_estimate_bytes,
        minutes=tuple(minutes),
        initial_states=initial_states,
        final_states=dict(engine.states),
    )
