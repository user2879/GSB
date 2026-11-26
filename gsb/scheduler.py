"""Quota-gated parallel batch scheduling on a real thread pool."""

from __future__ import annotations

import concurrent.futures
import threading
from dataclasses import dataclass
from typing import Any, Callable, Iterable, Sequence

from .errors import DiagnosticError
from .leases import LeaseManager, LeaseRecord, monotonic_ms
from .model import BatchSpec, SessionSpec, ShardSpec


@dataclass(frozen=True)
class SchedulerJob:
    batch: BatchSpec
    shard: ShardSpec
    parameter_version: int
    parameter_digest: str
    error_bound: int

    @property
    def sort_key(self) -> tuple[str, int, str]:
        return (self.shard.shard_id, self.batch.order, self.batch.batch_id)


@dataclass
class ScheduledOutcome:
    job: SchedulerJob
    lease: LeaseRecord | None
    wait_start_ms: int
    wait_end_ms: int | None
    error_affected: bool
    tie_reasons: tuple[str, ...]
    thread_name: str | None
    response: dict[str, Any] | None = None
    error: DiagnosticError | None = None
    status: str = "pending"

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "batch_id": self.job.batch.batch_id,
            "shard_id": self.job.shard.shard_id,
            "batch_order": self.job.batch.order,
            "parameter_version": self.job.parameter_version,
            "wait_start_ms": self.wait_start_ms,
            "wait_end_ms": self.wait_end_ms,
            "error_affected": self.error_affected,
            "tie_reasons": list(self.tie_reasons),
            "thread_name": self.thread_name,
            "status": self.status,
        }
        if self.lease is not None:
            payload["lease"] = self.lease.to_dict()
        if self.response is not None:
            payload["loss"] = self.response.get("loss")
            payload["loss_digest"] = self.response.get("loss_digest")
            payload["parameter_digest"] = self.response.get("parameter_digest")
            payload["elapsed_ms"] = self.response.get("elapsed_ms")
        if self.error is not None:
            payload["error"] = self.error.to_dict()
        return payload


ExecuteJob = Callable[[SchedulerJob, LeaseRecord], dict[str, Any]]
IssueLease = Callable[[SchedulerJob], LeaseRecord]
OnSuccess = Callable[[ScheduledOutcome], None]
OnError = Callable[[ScheduledOutcome], None]
OnStart = Callable[[ScheduledOutcome], None]


class QuotaScheduler:
    """Deterministic admission plus a real ThreadPoolExecutor for execution."""

    def __init__(
        self,
        session: SessionSpec,
        lease_manager: LeaseManager,
        *,
        clock: Callable[[], int] = monotonic_ms,
        max_workers: int | None = None,
    ) -> None:
        self.session = session
        self.lease_manager = lease_manager
        self.clock = clock
        concurrency = sum(shard.concurrency_limit for shard in session.shards)
        if max_workers is None:
            max_workers = max(1, min(64, concurrency))
        self.max_workers = max_workers
        self._active_units = 0
        self._active_by_shard: dict[str, int] = {shard.shard_id: 0 for shard in session.shards}
        self._max_units_observed = 0
        self._max_leases_observed: dict[str, int] = {shard.shard_id: 0 for shard in session.shards}
        self._max_active_observed: dict[str, int] = {shard.shard_id: 0 for shard in session.shards}
        self._lock = threading.Lock()

    @property
    def quota_high_watermarks(self) -> dict[str, Any]:
        return {
            "max_global_units": self._max_units_observed,
            "global_budget": self.session.global_budget,
            "max_active_by_shard": dict(self._max_active_observed),
            "max_leases_by_shard": dict(self._max_leases_observed),
        }

    def _feasible(self, job: SchedulerJob, remaining: int) -> bool:
        shard_id = job.shard.shard_id
        if job.batch.resource_units > remaining:
            return False
        if self._active_by_shard[shard_id] >= job.shard.concurrency_limit:
            return False
        if self.lease_manager.active_count(shard_id) >= job.shard.lease_limit:
            return False
        return job.shard.concurrency_limit > 0 and job.shard.lease_limit > 0

    def _choose(
        self, candidates: Sequence[SchedulerJob]
    ) -> tuple[SchedulerJob, bool, tuple[str, ...]]:
        ordered = sorted(candidates, key=lambda job: job.sort_key)
        chosen = ordered[0]
        remaining = self.session.global_budget - self._active_units
        reasons: list[str] = []
        slack = remaining - chosen.batch.resource_units
        if slack <= chosen.error_bound:
            reasons.append("remaining_budget_within_quantization_error")
        for other in ordered[1:]:
            waterline_delta = abs(chosen.batch.waterline - other.batch.waterline)
            combined_bound = chosen.error_bound + other.error_bound
            if waterline_delta <= combined_bound:
                reasons.append(
                    f"waterline_delta({chosen.batch.batch_id},{other.batch.batch_id})"
                    f"={waterline_delta}<=bound:{combined_bound}"
                )
        return chosen, bool(reasons), tuple(reasons)

    def run(
        self,
        jobs: Iterable[SchedulerJob],
        execute: ExecuteJob,
        issue_lease: IssueLease,
        *,
        on_start: OnStart | None = None,
        on_success: OnSuccess | None = None,
        on_error: OnError | None = None,
    ) -> tuple[ScheduledOutcome, ...]:
        pending = sorted(jobs, key=lambda job: job.sort_key)
        queued_at = {job.batch.batch_id: self.clock() for job in pending}
        outcomes: dict[str, ScheduledOutcome] = {}
        active: dict[concurrent.futures.Future[dict[str, Any]], ScheduledOutcome] = {}
        fatal: DiagnosticError | None = None

        with concurrent.futures.ThreadPoolExecutor(
            max_workers=self.max_workers, thread_name_prefix="gsb-batch"
        ) as executor:
            while pending or active:
                # Complete callbacks first so their quotas are available immediately.
                done = [future for future in active if future.done()]
                for future in done:
                    outcome = active.pop(future)
                    lease = outcome.lease
                    try:
                        response = future.result()
                        outcome.response = response
                        outcome.thread_name = response.get("_thread_name")
                        outcome.status = "committed"
                    except DiagnosticError as exc:
                        outcome.error = exc
                        outcome.status = "unknown" if exc.code in {"E_TIMEOUT", "E_PROTOCOL", "E_PORT"} else "failed"
                    except Exception as exc:  # pragma: no cover - defensive boundary
                        outcome.error = DiagnosticError(
                            "E_PROTOCOL", f"batch worker raised unexpected exception: {exc}"
                        )
                        outcome.status = "unknown"
                    finally:
                        self._active_units -= outcome.job.batch.resource_units
                        shard_id = outcome.job.shard.shard_id
                        self._active_by_shard[shard_id] -= 1
                        if lease is not None:
                            try:
                                self.lease_manager.release(lease.lease_id)
                            except DiagnosticError:
                                pass
                        outcomes[outcome.job.batch.batch_id] = outcome
                    try:
                        if outcome.error is None:
                            if on_success is not None:
                                on_success(outcome)
                        elif on_error is not None:
                            on_error(outcome)
                    except DiagnosticError as exc:
                        fatal = exc

                if fatal is not None:
                    break

                self.lease_manager.prune()
                candidates = [
                    job
                    for job in pending
                    if self._feasible(job, self.session.global_budget - self._active_units)
                ]
                while candidates:
                    chosen, affected, reasons = self._choose(candidates)
                    outcome = ScheduledOutcome(
                        job=chosen,
                        lease=None,
                        wait_start_ms=queued_at[chosen.batch.batch_id],
                        wait_end_ms=None,
                        error_affected=affected,
                        tie_reasons=reasons,
                        thread_name=None,
                    )
                    try:
                        lease = issue_lease(chosen)
                    except DiagnosticError as exc:
                        outcome.error = exc
                        outcome.status = "failed"
                        outcome.wait_end_ms = self.clock()
                        outcomes[chosen.batch.batch_id] = outcome
                        pending.remove(chosen)
                        if on_error is not None:
                            on_error(outcome)
                        candidates = [
                            job
                            for job in pending
                            if self._feasible(
                                job,
                                self.session.global_budget - self._active_units,
                            )
                        ]
                        continue
                    outcome.lease = lease
                    self._active_units += chosen.batch.resource_units
                    shard_id = chosen.shard.shard_id
                    self._active_by_shard[shard_id] += 1
                    self._max_units_observed = max(self._max_units_observed, self._active_units)
                    self._max_active_observed[shard_id] = max(
                        self._max_active_observed[shard_id], self._active_by_shard[shard_id]
                    )
                    self._max_leases_observed[shard_id] = max(
                        self._max_leases_observed[shard_id],
                        self.lease_manager.active_count(shard_id),
                    )
                    if on_start is not None:
                        try:
                            on_start(outcome)
                        except DiagnosticError as exc:
                            self._active_units -= chosen.batch.resource_units
                            self._active_by_shard[shard_id] -= 1
                            try:
                                self.lease_manager.release(lease.lease_id)
                            except DiagnosticError:
                                pass
                            fatal = exc
                            break
                    outcome.wait_end_ms = self.clock()
                    pending.remove(chosen)
                    future = executor.submit(self._execute_wrapped, execute, chosen, lease)
                    active[future] = outcome
                    candidates = [
                        job
                        for job in pending
                        if self._feasible(
                            job, self.session.global_budget - self._active_units
                        )
                    ]
                if fatal is not None:
                    break
                if not active:
                    if pending:
                        # No runnable job can ever fit under the declared quotas.
                        for job in pending:
                            outcome = ScheduledOutcome(
                                job=job,
                                lease=None,
                                wait_start_ms=queued_at[job.batch.batch_id],
                                wait_end_ms=None,
                                error_affected=False,
                                tie_reasons=(),
                                thread_name=None,
                                error=DiagnosticError(
                                    "E_QUOTA",
                                    "batch cannot fit global budget or shard quotas",
                                    details={
                                        "resource_units": job.batch.resource_units,
                                        "global_budget": self.session.global_budget,
                                        "shard_id": job.shard.shard_id,
                                    },
                                ),
                                status="blocked",
                            )
                            outcomes[job.batch.batch_id] = outcome
                            if on_error is not None:
                                on_error(outcome)
                        pending.clear()
                    break
                concurrent.futures.wait(
                    list(active),
                    timeout=0.01,
                    return_when=concurrent.futures.FIRST_COMPLETED,
                )

            if fatal is not None:
                for future, outcome in list(active.items()):
                    future.cancel()
                    if outcome.lease is not None:
                        try:
                            self.lease_manager.release(outcome.lease.lease_id)
                        except DiagnosticError:
                            pass
                raise fatal

        ordered_outcomes = tuple(
            outcomes[job.batch.batch_id]
            for job in sorted(jobs, key=lambda item: item.sort_key)
            if job.batch.batch_id in outcomes
        )
        return ordered_outcomes

    @staticmethod
    def _execute_wrapped(
        execute: ExecuteJob, job: SchedulerJob, lease: LeaseRecord
    ) -> dict[str, Any]:
        # The callback is what proves execution really projected onto pool threads.
        response = execute(job, lease)
        response.setdefault("_thread_name", threading.current_thread().name)
        return response
