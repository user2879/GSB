"""Concurrent shard scheduling under global, cluster and task constraints."""

from __future__ import annotations

import threading
import time
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping

from .errors import E_MANIFEST, E_REPLY, PipelineError
from .planner import Plan, ShardPlan
from .util import digest_json, is_int, monotonic_ms


@dataclass(frozen=True)
class DispatchOutcome:
    aggregate_summary: dict[str, Any]
    aggregate_digest: str
    elapsed_ms: int
    transport_ms: int = 0
    ledger_entry_id: int | None = None


@dataclass
class ShardTrace:
    stable_id: str
    cluster_id: str
    task_id: str
    shard_index: int
    revision_version: int
    resource_units: int
    status: str = "pending"
    queued_at_ms: int = 0
    wait_start_ms: int = 0
    wait_end_ms: int = 0
    wait_ms: int = 0
    execution_start_ms: int = 0
    execution_end_ms: int = 0
    duration_ms: int = 0
    service_elapsed_ms: int = 0
    aggregate_summary: dict[str, Any] | None = None
    aggregate_digest: str | None = None
    error_code: str | None = None
    error_message: str | None = None
    thread_name: str | None = None
    ledger_entry_id: int | None = None

    def canonical(self) -> dict[str, Any]:
        return {
            "stable_id": self.stable_id,
            "cluster_id": self.cluster_id,
            "task_id": self.task_id,
            "shard_index": self.shard_index,
            "revision_version": self.revision_version,
            "resource_units": self.resource_units,
            "status": self.status,
            "queued_at_ms": self.queued_at_ms,
            "wait_start_ms": self.wait_start_ms,
            "wait_end_ms": self.wait_end_ms,
            "wait_ms": self.wait_ms,
            "execution_start_ms": self.execution_start_ms,
            "execution_end_ms": self.execution_end_ms,
            "duration_ms": self.duration_ms,
            "service_elapsed_ms": self.service_elapsed_ms,
            "aggregate_summary": self.aggregate_summary,
            "aggregate_digest": self.aggregate_digest,
            "error_code": self.error_code,
            "error_message": self.error_message,
            "thread_name": self.thread_name,
            "ledger_entry_id": self.ledger_entry_id,
        }


@dataclass
class ScheduleStats:
    global_budget: int
    total_resource_units: int = 0
    peak_resource_units: int = 0
    peak_cluster_running: dict[str, int] = field(default_factory=dict)
    peak_task_running: dict[str, int] = field(default_factory=dict)
    max_observed_workers: int = 0

    def canonical(self) -> dict[str, Any]:
        return {
            "global_budget": self.global_budget,
            "total_resource_units": self.total_resource_units,
            "peak_resource_units": self.peak_resource_units,
            "peak_cluster_running": {
                key: self.peak_cluster_running[key]
                for key in sorted(self.peak_cluster_running)
            },
            "peak_task_running": {
                key: self.peak_task_running[key] for key in sorted(self.peak_task_running)
            },
            "max_observed_workers": self.max_observed_workers,
        }


@dataclass
class ScheduleReport:
    traces: tuple[ShardTrace, ...]
    stats: ScheduleStats

    @property
    def failed(self) -> tuple[ShardTrace, ...]:
        return tuple(trace for trace in self.traces if trace.status == "failed")

    @property
    def succeeded(self) -> tuple[ShardTrace, ...]:
        return tuple(trace for trace in self.traces if trace.status == "success")

    def canonical(self) -> dict[str, Any]:
        return {
            "traces": [trace.canonical() for trace in self.traces],
            "stats": self.stats.canonical(),
        }


Dispatcher = Callable[[ShardPlan], DispatchOutcome | dict[str, Any]]


def _normalise_outcome(shard: ShardPlan, result: Any) -> DispatchOutcome:
    if isinstance(result, DispatchOutcome):
        outcome = result
    elif isinstance(result, dict):
        try:
            summary = result["aggregate_summary"]
            aggregate_digest = result["aggregate_digest"]
            elapsed_ms = result["elapsed_ms"]
            transport_ms = result.get("_transport_ms", result.get("transport_ms", 0))
            ledger_entry_id = result.get("ledger_entry_id")
        except KeyError as exc:
            raise PipelineError(
                E_REPLY,
                "dispatcher result is missing a required aggregate field",
                field=f"result.{exc.args[0]}",
                shard_id=shard.stable_id,
            ) from exc
        if not isinstance(summary, dict):
            raise PipelineError(
                E_REPLY,
                "dispatcher aggregate summary must be an object",
                field="result.aggregate_summary",
                shard_id=shard.stable_id,
            )
        computed = digest_json(summary)
        if not isinstance(aggregate_digest, str) or computed != aggregate_digest:
            raise PipelineError(
                E_REPLY,
                "dispatcher aggregate digest does not match its summary",
                field="result.aggregate_digest",
                shard_id=shard.stable_id,
            )
        if not is_int(elapsed_ms) and not isinstance(elapsed_ms, float):
            raise PipelineError(
                E_REPLY,
                "dispatcher elapsed_ms must be numeric",
                field="result.elapsed_ms",
                shard_id=shard.stable_id,
            )
        if not is_int(transport_ms) and not isinstance(transport_ms, float):
            transport_ms = 0
        outcome = DispatchOutcome(
            aggregate_summary=summary,
            aggregate_digest=aggregate_digest,
            elapsed_ms=int(round(float(elapsed_ms))),
            transport_ms=int(round(float(transport_ms))),
            ledger_entry_id=int(ledger_entry_id)
            if is_int(ledger_entry_id)
            else None,
        )
    else:
        raise PipelineError(
            E_REPLY,
            "dispatcher returned an unsupported result type",
            field="result",
            shard_id=shard.stable_id,
        )
    if outcome.aggregate_digest != digest_json(outcome.aggregate_summary):
        raise PipelineError(
            E_REPLY,
            "dispatch outcome digest mismatch",
            field="result.aggregate_digest",
            shard_id=shard.stable_id,
        )
    return outcome


def schedule_shards(
    plan: Plan,
    dispatch: Dispatcher,
    *,
    global_budget: int,
    cluster_quotas: Mapping[str, int],
    timeout_seconds: float = 120.0,
) -> ScheduleReport:
    """Execute all shards while enforcing the three independent limits.

    A shard is reserved against all three counters *before* it is submitted to a
    real ``ThreadPoolExecutor``.  Counters are released only after the worker
    has returned, so the accounting cannot be bypassed by executor queuing.
    """

    if not is_int(global_budget) or global_budget < 0:
        raise PipelineError(
            E_MANIFEST,
            "global budget must be a non-negative integer",
            field="global_resource_budget",
        )
    shards = sorted(plan.shards, key=lambda shard: shard.order_key)
    for shard in shards:
        quota = cluster_quotas.get(shard.cluster_id)
        if not is_int(quota) or int(quota) < 1:
            raise PipelineError(
                E_MANIFEST,
                "cluster concurrency quota must be a positive integer",
                field=f"clusters[{shard.cluster_id}].concurrency",
                shard_id=shard.stable_id,
            )
        if shard.resource_units > global_budget:
            raise PipelineError(
                E_MANIFEST,
                "shard resource requirement exceeds the global budget and can never run",
                field="global_resource_budget",
                shard_id=shard.stable_id,
                details={
                    "resource_units": shard.resource_units,
                    "global_budget": global_budget,
                },
            )
        if shard.task_concurrency < 1:
            raise PipelineError(
                E_MANIFEST,
                "task concurrency limit must be positive",
                field=f"tasks[{shard.task_id}].concurrency_limit",
                shard_id=shard.stable_id,
            )

    traces = {
        shard.stable_id: ShardTrace(
            stable_id=shard.stable_id,
            cluster_id=shard.cluster_id,
            task_id=shard.task_id,
            shard_index=shard.shard_index,
            revision_version=shard.revision_version,
            resource_units=shard.resource_units,
        )
        for shard in shards
    }
    pending = list(shards)
    queued_at = monotonic_ms()
    for shard in pending:
        trace = traces[shard.stable_id]
        trace.queued_at_ms = queued_at
        trace.wait_start_ms = queued_at

    stats = ScheduleStats(global_budget=global_budget)
    for cluster_id in sorted(cluster_quotas):
        stats.peak_cluster_running[cluster_id] = 0
    for shard in shards:
        stats.peak_task_running.setdefault(shard.task_id, 0)

    resource_in_use = 0
    cluster_running: dict[str, int] = {cluster_id: 0 for cluster_id in cluster_quotas}
    task_running: dict[str, int] = {task_id: 0 for task_id in stats.peak_task_running}

    cluster_capacity = sum(max(1, int(quota)) for quota in cluster_quotas.values())
    max_workers = min(len(shards), max(1, cluster_capacity, 1))
    max_workers = min(max_workers, 256)
    running: dict[Future[Any], ShardPlan] = {}
    deadline = time.monotonic() + timeout_seconds

    def worker(shard: ShardPlan) -> tuple[DispatchOutcome, str]:
        outcome = _normalise_outcome(shard, dispatch(shard))
        return outcome, threading.current_thread().name

    def can_start(shard: ShardPlan) -> bool:
        return (
            resource_in_use + shard.resource_units <= global_budget
            and cluster_running[shard.cluster_id] + 1
            <= int(cluster_quotas[shard.cluster_id])
            and task_running[shard.task_id] + 1 <= shard.task_concurrency
        )

    if not shards:
        return ScheduleReport(traces=(), stats=stats)

    with ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="distributor") as pool:
        while pending or running:
            if time.monotonic() > deadline:
                for future in running:
                    future.cancel()
                raise PipelineError(
                    E_REPLY,
                    "parallel dispatch exceeded the scheduler timeout",
                    field="scheduler.timeout",
                    details={"timeout_seconds": timeout_seconds},
                )

            launched = False
            for shard in list(pending):
                if not can_start(shard):
                    continue
                pending.remove(shard)
                resource_in_use += shard.resource_units
                cluster_running[shard.cluster_id] += 1
                task_running[shard.task_id] += 1
                stats.total_resource_units += shard.resource_units
                stats.peak_resource_units = max(stats.peak_resource_units, resource_in_use)
                stats.peak_cluster_running[shard.cluster_id] = max(
                    stats.peak_cluster_running[shard.cluster_id],
                    cluster_running[shard.cluster_id],
                )
                stats.peak_task_running[shard.task_id] = max(
                    stats.peak_task_running[shard.task_id], task_running[shard.task_id]
                )
                trace = traces[shard.stable_id]
                trace.status = "running"
                now = monotonic_ms()
                trace.wait_end_ms = now
                trace.wait_ms = max(0, now - trace.wait_start_ms)
                trace.execution_start_ms = now
                future = pool.submit(worker, shard)
                running[future] = shard
                stats.max_observed_workers = max(
                    stats.max_observed_workers, len(running)
                )
                launched = True

            if not running:
                if pending and not launched:
                    raise PipelineError(
                        E_MANIFEST,
                        "no pending shard can satisfy the combined concurrency constraints",
                        field="global_resource_budget",
                        details={
                            "pending": [item.stable_id for item in pending],
                            "global_budget": global_budget,
                            "cluster_quotas": dict(cluster_quotas),
                        },
                    )
                continue

            if not launched:
                done, _ = wait(
                    tuple(running), timeout=0.01, return_when=FIRST_COMPLETED
                )
            else:
                done, _ = wait(
                    tuple(running), timeout=0, return_when=FIRST_COMPLETED
                )

            for future in done:
                shard = running.pop(future)
                resource_in_use -= shard.resource_units
                cluster_running[shard.cluster_id] -= 1
                task_running[shard.task_id] -= 1
                trace = traces[shard.stable_id]
                trace.execution_end_ms = monotonic_ms()
                trace.duration_ms = max(
                    0, trace.execution_end_ms - trace.execution_start_ms
                )
                try:
                    outcome, thread_name = future.result()
                    trace.status = "success"
                    trace.aggregate_summary = outcome.aggregate_summary
                    trace.aggregate_digest = outcome.aggregate_digest
                    trace.service_elapsed_ms = outcome.elapsed_ms
                    trace.thread_name = thread_name
                    trace.ledger_entry_id = outcome.ledger_entry_id
                except Exception as exc:  # noqa: BLE001 - persisted as a failed outcome
                    trace.status = "failed"
                    if isinstance(exc, PipelineError):
                        trace.error_code = exc.code
                        trace.error_message = str(exc)
                    else:
                        trace.error_code = E_REPLY
                        trace.error_message = f"{type(exc).__name__}: {exc}"
                    trace.thread_name = f"worker-exception:{threading.current_thread().name}"

            active_workers = sum(
                1
                for future in running
                if future.running()
            )
            stats.max_observed_workers = max(stats.max_observed_workers, active_workers)

    return ScheduleReport(
        traces=tuple(traces[shard.stable_id] for shard in shards),
        stats=stats,
    )
