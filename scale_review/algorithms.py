from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

from .estimator import estimate_minute, scaling_action
from .spec import Spec, ServiceSpec


@dataclass(frozen=True)
class MinuteResult:
    service_id: str
    minute: int
    request_count: int
    per_replica_capacity: int
    unclamped_replicas: int
    clamped_replicas: int
    min_replicas: int
    max_replicas: int
    baseline_replicas: int
    replica_delta: int
    scaling_action: str
    window_length: int
    window_size: int
    window_request_sum: int
    window_demand_sum: int
    ring_index: int
    estimation_rule_version: str

    def comparable_dict(self) -> dict[str, Any]:
        return {
            "baseline_replicas": self.baseline_replicas,
            "clamped_replicas": self.clamped_replicas,
            "estimation_rule_version": self.estimation_rule_version,
            "max_replicas": self.max_replicas,
            "minute": self.minute,
            "min_replicas": self.min_replicas,
            "per_replica_capacity": self.per_replica_capacity,
            "replica_delta": self.replica_delta,
            "request_count": self.request_count,
            "ring_index": self.ring_index,
            "scaling_action": self.scaling_action,
            "service_id": self.service_id,
            "unclamped_replicas": self.unclamped_replicas,
            "window_demand_sum": self.window_demand_sum,
            "window_length": self.window_length,
            "window_request_sum": self.window_request_sum,
            "window_size": self.window_size,
        }


@dataclass(frozen=True)
class PathResult:
    path: str
    minutes: dict[str, tuple[MinuteResult, ...]]

    def iter_minutes(self) -> Iterable[MinuteResult]:
        for service_id in sorted(self.minutes):
            yield from self.minutes[service_id]


@dataclass(frozen=True)
class PathSelection:
    selected_path: str
    switch_threshold: int
    sequence_length: int
    window_length: int
    full_work_units: int
    incremental_work_units: int
    work_delta: int
    effort_tie_boundary: int
    tie_break_path: str
    reason: str

    def ledger_basis(self) -> dict[str, Any]:
        return {
            "effort_tie_boundary": self.effort_tie_boundary,
            "full_work_units": self.full_work_units,
            "incremental_work_units": self.incremental_work_units,
            "reason": self.reason,
            "selected_path": self.selected_path,
            "sequence_length": self.sequence_length,
            "switch_threshold": self.switch_threshold,
            "tie_break_path": self.tie_break_path,
            "window_length": self.window_length,
            "work_delta": self.work_delta,
        }


def work_estimates(sequence_length: int, window_length: int) -> tuple[int, int]:
    """Transparent deterministic cost model.

    Full recalculation estimates every minute and scans each trailing window.
    The ring path initializes ``window_length`` slots then performs constant
    accounting work per minute.
    """

    window_scans = 0
    for minute in range(sequence_length):
        window_scans += min(window_length, minute + 1)
    full_work = sequence_length + window_scans
    incremental_work = window_length + 4 * sequence_length
    return full_work, incremental_work


def select_path(spec: Spec, sequence_length: int | None = None) -> PathSelection:
    length = spec.minute_count if sequence_length is None else sequence_length
    config = spec.algorithm
    full_work, incremental_work = work_estimates(length, config.window_length)
    delta = abs(full_work - incremental_work)

    if length < config.path_switch_length:
        selected = "full"
        reason = "length_below_switch_threshold"
    elif delta <= config.effort_tie_boundary:
        selected = config.tie_break_path
        reason = "estimated_effort_within_explicit_tie_boundary"
    elif full_work < incremental_work:
        selected = "full"
        reason = "full_path_has_lower_estimated_effort"
    else:
        selected = "incremental"
        reason = "incremental_path_has_lower_estimated_effort"

    return PathSelection(
        selected_path=selected,
        switch_threshold=config.path_switch_length,
        sequence_length=length,
        window_length=config.window_length,
        full_work_units=full_work,
        incremental_work_units=incremental_work,
        work_delta=delta,
        effort_tie_boundary=config.effort_tie_boundary,
        tie_break_path=config.tie_break_path,
        reason=reason,
    )


def _result_for_values(
    service: ServiceSpec,
    minute: int,
    request_count: int,
    raw_demand: int,
    clamped: int,
    baseline: int,
    window_length: int,
    window_size: int,
    window_request_sum: int,
    window_demand_sum: int,
    ring_index: int,
    spec: Spec,
) -> MinuteResult:
    delta = clamped - baseline
    action = scaling_action(
        delta,
        spec.decision_rules.scale_up_threshold,
        spec.decision_rules.scale_down_threshold,
    )
    return MinuteResult(
        service_id=service.service_id,
        minute=minute,
        request_count=request_count,
        per_replica_capacity=service.concurrency_per_replica,
        unclamped_replicas=raw_demand,
        clamped_replicas=clamped,
        min_replicas=service.min_replicas,
        max_replicas=service.max_replicas,
        baseline_replicas=baseline,
        replica_delta=delta,
        scaling_action=action,
        window_length=window_length,
        window_size=window_size,
        window_request_sum=window_request_sum,
        window_demand_sum=window_demand_sum,
        ring_index=ring_index,
        estimation_rule_version=spec.decision_rules.estimation_rule_version,
    )


def run_full_path(spec: Spec) -> PathResult:
    results: dict[str, tuple[MinuteResult, ...]] = {}
    factor = spec.scaling_factor_permille
    window_length = spec.algorithm.window_length

    for service in spec.services:
        items: list[MinuteResult] = []
        baseline = service.min_replicas
        counts = service.requests_per_minute
        for minute, request_count in enumerate(counts):
            raw_demand, clamped = estimate_minute(
                request_count,
                service.concurrency_per_replica,
                service.min_replicas,
                service.max_replicas,
                factor,
            )
            start = max(0, minute - window_length + 1)
            window_counts = counts[start : minute + 1]
            window_request_sum = sum(window_counts)
            window_demand_sum = sum(
                estimate_minute(
                    value,
                    service.concurrency_per_replica,
                    service.min_replicas,
                    service.max_replicas,
                    factor,
                )[0]
                for value in window_counts
            )
            items.append(
                _result_for_values(
                    service,
                    minute,
                    request_count,
                    raw_demand,
                    clamped,
                    baseline,
                    window_length,
                    len(window_counts),
                    window_request_sum,
                    window_demand_sum,
                    minute % window_length,
                    spec,
                )
            )
            baseline = clamped
        results[service.service_id] = tuple(items)
    return PathResult(path="full", minutes=results)


def run_incremental_path(spec: Spec) -> PathResult:
    results: dict[str, tuple[MinuteResult, ...]] = {}
    factor = spec.scaling_factor_permille
    window_length = spec.algorithm.window_length

    for service in spec.services:
        items: list[MinuteResult] = []
        baseline = service.min_replicas
        ring_counts = [0] * window_length
        ring_demands = [0] * window_length
        window_request_sum = 0
        window_demand_sum = 0

        for minute, request_count in enumerate(service.requests_per_minute):
            ring_index = minute % window_length
            if minute >= window_length:
                window_request_sum -= ring_counts[ring_index]
                window_demand_sum -= ring_demands[ring_index]

            raw_demand, clamped = estimate_minute(
                request_count,
                service.concurrency_per_replica,
                service.min_replicas,
                service.max_replicas,
                factor,
            )
            ring_counts[ring_index] = request_count
            ring_demands[ring_index] = raw_demand
            window_request_sum += request_count
            window_demand_sum += raw_demand
            window_size = min(window_length, minute + 1)

            items.append(
                _result_for_values(
                    service,
                    minute,
                    request_count,
                    raw_demand,
                    clamped,
                    baseline,
                    window_length,
                    window_size,
                    window_request_sum,
                    window_demand_sum,
                    ring_index,
                    spec,
                )
            )
            baseline = clamped
        results[service.service_id] = tuple(items)
    return PathResult(path="incremental", minutes=results)


def compute_full_minute(
    spec: Spec,
    service: ServiceSpec,
    minute: int,
    baseline_replicas: int | None = None,
) -> MinuteResult:
    request_count = service.requests_per_minute[minute]
    raw_demand, clamped = estimate_minute(
        request_count,
        service.concurrency_per_replica,
        service.min_replicas,
        service.max_replicas,
        spec.scaling_factor_permille,
    )
    window_length = spec.algorithm.window_length
    start = max(0, minute - window_length + 1)
    window_counts = service.requests_per_minute[start : minute + 1]
    window_request_sum = sum(window_counts)
    window_demand_sum = sum(
        estimate_minute(
            value,
            service.concurrency_per_replica,
            service.min_replicas,
            service.max_replicas,
            spec.scaling_factor_permille,
        )[0]
        for value in window_counts
    )
    if baseline_replicas is None:
        baseline_replicas = service.min_replicas if minute == 0 else estimate_minute(
            service.requests_per_minute[minute - 1],
            service.concurrency_per_replica,
            service.min_replicas,
            service.max_replicas,
            spec.scaling_factor_permille,
        )[1]
    return _result_for_values(
        service,
        minute,
        request_count,
        raw_demand,
        clamped,
        baseline_replicas,
        window_length,
        len(window_counts),
        window_request_sum,
        window_demand_sum,
        minute % window_length,
        spec,
    )


def append_incremental_minutes(
    spec: Spec,
    service: ServiceSpec,
    previous: tuple[MinuteResult, ...],
    new_counts: tuple[int, ...],
) -> tuple[MinuteResult, ...]:
    """Continue a ring-buffer cursor without rescanning prior minutes."""

    window_length = spec.algorithm.window_length
    old_length = len(previous)
    ring_counts = [0] * window_length
    ring_demands = [0] * window_length
    start = max(0, old_length - window_length)
    for index in range(start, old_length):
        ring_index = index % window_length
        ring_counts[ring_index] = service.requests_per_minute[index]
        ring_demands[ring_index] = previous[index].unclamped_replicas

    if previous:
        window_request_sum = previous[-1].window_request_sum
        window_demand_sum = previous[-1].window_demand_sum
        baseline = previous[-1].clamped_replicas
    else:
        window_request_sum = 0
        window_demand_sum = 0
        baseline = service.min_replicas

    added: list[MinuteResult] = []
    for offset, request_count in enumerate(new_counts):
        minute = old_length + offset
        ring_index = minute % window_length
        if minute >= window_length:
            window_request_sum -= ring_counts[ring_index]
            window_demand_sum -= ring_demands[ring_index]

        raw_demand, clamped = estimate_minute(
            request_count,
            service.concurrency_per_replica,
            service.min_replicas,
            service.max_replicas,
            spec.scaling_factor_permille,
        )
        ring_counts[ring_index] = request_count
        ring_demands[ring_index] = raw_demand
        window_request_sum += request_count
        window_demand_sum += raw_demand
        window_size = min(window_length, minute + 1)
        result = _result_for_values(
            service,
            minute,
            request_count,
            raw_demand,
            clamped,
            baseline,
            window_length,
            window_size,
            window_request_sum,
            window_demand_sum,
            ring_index,
            spec,
        )
        added.append(result)
        baseline = clamped
    return tuple(added)
