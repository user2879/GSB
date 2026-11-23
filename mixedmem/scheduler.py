"""Budget-constrained batch scheduling on real local worker threads."""

from __future__ import annotations

import os
import threading
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from .errors import DiagnosticError
from .graph import DependencyGraph
from .inference import LayerBatchResult, compute_layer_result
from .lease import Lease, LeasePool, LeaseRequest
from .ledger import Ledger
from .samples import BatchSpec, SampleFactory
from .schema import Plan, layer_memory_bytes
from .util import monotonic_ms


@dataclass
class BatchOutcome:
    batch_id: str
    sample_ids: tuple[str, ...]
    queue_start_ms: int
    queue_end_ms: int | None = None
    compute_start_ms: int | None = None
    end_ms: int | None = None
    worker_name: str | None = None
    leases: list[dict[str, Any]] = field(default_factory=list)
    results: dict[str, LayerBatchResult] = field(default_factory=dict)
    error: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "batch_id": self.batch_id,
            "sample_ids": list(self.sample_ids),
            "queue_start_ms": self.queue_start_ms,
            "queue_end_ms": self.queue_end_ms,
            "queue_wait_ms": (
                None if self.queue_end_ms is None else self.queue_end_ms - self.queue_start_ms
            ),
            "compute_start_ms": self.compute_start_ms,
            "end_ms": self.end_ms,
            "compute_ms": (
                None
                if self.compute_start_ms is None or self.end_ms is None
                else self.end_ms - self.compute_start_ms
            ),
            "worker_name": self.worker_name,
            "leases": self.leases,
            "lease_ids": [lease["lease_id"] for lease in self.leases],
            "block_ranges": [lease["block_range"] for lease in self.leases],
            "layers": {
                layer_id: result.ledger_payload()
                for layer_id, result in sorted(self.results.items())
            },
            "error": self.error,
        }


class CrashInjector:
    def __init__(self, after_committed_layers: int | None) -> None:
        self.after = after_committed_layers
        self.count = 0
        self.lock = threading.Lock()

    def committed_layer(self) -> None:
        if self.after is None:
            return
        with self.lock:
            self.count += 1
            if self.count == self.after:
                # Hard termination after the SQLite transaction has committed.
                os._exit(91)


def _lease_requests(plan: Plan, graph: DependencyGraph, batch_size: int) -> list[LeaseRequest]:
    return [
        LeaseRequest(
            layer_id=layer_id,
            blocks=layer_memory_bytes(plan, layer_id, batch_size)
            // plan.block_size_bytes,
        )
        for layer_id in graph.order
    ]


def _release_batch_leases(pool: LeasePool, leases: Sequence[Lease]) -> None:
    for lease in reversed(tuple(leases)):
        try:
            pool.release(lease.lease_id, batch_id=lease.batch_id, layer_id=lease.layer_id)
        except DiagnosticError:
            # A torn-down or expired lease is already recorded by the pool.  The
            # worker records failure; cleanup should not mask the first error.
            pass


def _run_worker(
    *,
    plan: Plan,
    graph: DependencyGraph,
    factory: SampleFactory,
    batch: BatchSpec,
    leases: Sequence[Lease],
    layer_versions: Mapping[str, int],
    ledger: Ledger,
    pool: LeasePool,
    outcome: BatchOutcome,
    crash: CrashInjector,
) -> BatchOutcome:
    started = monotonic_ms()
    outcome.queue_end_ms = started
    outcome.compute_start_ms = started
    outcome.worker_name = threading.current_thread().name
    ledger.mark_worker_started(
        plan.version, batch.batch_id, worker_start_ms=started, worker_name=outcome.worker_name
    )
    lease_by_layer = {lease.layer_id: lease for lease in leases}
    results: dict[str, LayerBatchResult] = {}
    try:
        for lease in leases:
            pool.assert_active(lease.lease_id, batch_id=batch.batch_id)
        for layer_id in graph.order:
            lease = lease_by_layer[layer_id]
            pool.assert_active(lease.lease_id, batch_id=batch.batch_id)
            result = compute_layer_result(
                plan,
                factory,
                plan.layer_by_id[layer_id],
                batch.sample_ids,
                batch.batch_id,
                layer_versions[layer_id],
                results,
            ).clone_with_lease(lease)
            # The layer commit and progress cursor share one SQLite transaction.
            ledger.commit_layer_result(result)
            crash.committed_layer()
            results[layer_id] = result
        _release_batch_leases(pool, leases)
        ended = monotonic_ms()
        outcome.end_ms = ended
        outcome.results = results
        ledger.finish_batch(plan.version, batch.batch_id, leases, end_ms=ended)
    except BaseException as exc:
        _release_batch_leases(pool, leases)
        ended = monotonic_ms()
        outcome.end_ms = ended
        outcome.error = (
            exc.to_dict()
            if isinstance(exc, DiagnosticError)
            else {"ok": False, "code": "E_LEASE", "message": f"{type(exc).__name__}: {exc}"}
        )
        try:
            ledger.finish_batch(
                plan.version,
                batch.batch_id,
                leases,
                end_ms=ended,
                error=outcome.error["message"],
            )
        except BaseException:
            pass
    outcome.leases = [lease.to_dict() for lease in leases]
    return outcome


def schedule_batches(
    *,
    plan: Plan,
    graph: DependencyGraph,
    factory: SampleFactory,
    batches: Sequence[BatchSpec],
    layer_versions: Mapping[str, int],
    ledger: Ledger,
    pool: LeasePool,
    worker_count: int = 4,
    crash_after_committed_layers: int | None = None,
) -> dict[str, Any]:
    if worker_count <= 0:
        raise ValueError("worker_count must be positive")
    crash = CrashInjector(crash_after_committed_layers)
    outcomes: list[BatchOutcome] = []
    executor = ThreadPoolExecutor(
        max_workers=worker_count, thread_name_prefix="mixedmem-worker"
    )
    active: dict[Future[BatchOutcome], BatchOutcome] = {}
    next_index = 0
    pool.start_reaper(interval_ms=2)
    try:
        while next_index < len(batches) or active:
            while next_index < len(batches) and len(active) < worker_count:
                batch = batches[next_index]
                queue_start = monotonic_ms()
                outcome = BatchOutcome(
                    batch_id=batch.batch_id,
                    sample_ids=batch.sample_ids,
                    queue_start_ms=queue_start,
                )
                requests = _lease_requests(plan, graph, batch.size)
                leases: tuple[Lease, ...] = ()
                try:
                    leases = pool.acquire_bundle(
                        batch.batch_id,
                        requests,
                        ttl_ms=plan.lease_ttl_ms,
                        plan_version=plan.version,
                    )
                    ledger.begin_batch(
                        plan.version,
                        batch.batch_id,
                        batch.sample_ids,
                        leases,
                        queue_start_ms=queue_start,
                    )
                except BaseException as exc:
                    for lease in reversed(leases):
                        try:
                            pool.release(
                                lease.lease_id,
                                batch_id=lease.batch_id,
                                layer_id=lease.layer_id,
                            )
                        except BaseException:
                            pass
                    outcome.end_ms = monotonic_ms()
                    outcome.error = (
                        exc.to_dict()
                        if isinstance(exc, DiagnosticError)
                        else {"ok": False, "code": "E_BUDGET", "message": str(exc)}
                    )
                    outcomes.append(outcome)
                    next_index += 1
                    continue
                future = executor.submit(
                    _run_worker,
                    plan=plan,
                    graph=graph,
                    factory=factory,
                    batch=batch,
                    leases=leases,
                    layer_versions=layer_versions,
                    ledger=ledger,
                    pool=pool,
                    outcome=outcome,
                    crash=crash,
                )
                active[future] = outcome
                next_index += 1
            if not active:
                continue
            completed, _ = wait(tuple(active), return_when=FIRST_COMPLETED)
            for future in completed:
                outcome = future.result()
                active.pop(future, None)
                outcomes.append(outcome)
    finally:
        executor.shutdown(wait=True, cancel_futures=False)
        pool.stop_reaper()
    outcomes.sort(key=lambda item: item.batch_id)
    return {
        "plan_version": plan.version,
        "worker_count": worker_count,
        "batches": [outcome.to_dict() for outcome in outcomes],
        "failed_batches": [outcome.batch_id for outcome in outcomes if outcome.error],
        "pool": pool.current_view(),
        "invariants": pool.verify_invariants(),
        "_outcome_objects": outcomes,
    }
