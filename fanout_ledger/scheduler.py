"""Dual-budget fan-out scheduler with real timeouts and deterministic retries.

Concurrency model:

* one bounded worker pool sized to ``global_concurrency`` (global budget);
* one bounded semaphore per node sized to ``node_concurrency`` (node budget);
* delivery jobs are submitted in a stable round-robin order (scene node order
  outer, delivery sequence inner) so scheduling never depends on dict/hash
  iteration;
* every attempt records monotonic-clock queue wait start/end, run start/end
  and the consumed time budget.

Retry model: failure/timeout/unreachable attempts wait on the backoff series
(backoff index = attempt index - 1, clamped to the last entry) and retry with
the same per-attempt socket timeout.  Exhausting ``max_attempts`` finalizes
the delivery with E_RETRY semantics and records attempts used and cumulative
wait milliseconds.
"""
from __future__ import annotations

import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Callable

from .config import Scene
from .errors import RetryExhaustedError
from .http_client import post_attempt
from .ledger import AttemptRecord, Ledger
from .snapshot import Snapshot, build_snapshot, write_snapshot

TERMINAL_FAILURES = ("failure", "timeout", "unreachable")


def now_ms() -> int:
    return int(time.monotonic_ns() // 1_000_000)


@dataclass
class RuntimeCounters:
    running: int = 0
    queued: int = 0
    outcome_counts: dict[str, int] = field(default_factory=lambda: {
        "success": 0, "failure": 0, "timeout": 0, "unreachable": 0, "unknown": 0})
    node_running: dict[str, int] = field(default_factory=dict)
    node_queued: dict[str, int] = field(default_factory=dict)
    budget_global_used_ms: int = 0
    budget_node_used_ms: dict[str, int] = field(default_factory=dict)
    attempts_used: dict[str, int] = field(default_factory=dict)
    trajectory: list[dict[str, Any]] = field(default_factory=list)
    lock: threading.Lock = field(default_factory=threading.Lock)


def trajectory_from_ledger(records: list) -> list[dict[str, Any]]:
    """Rebuild the ordered scheduling trajectory from committed attempt rows.

    Used on resume/crash recovery where the in-memory counters no longer exist.
    Ordering follows attempt sequence id (commit order), never dict iteration.
    """
    entries: list[dict[str, Any]] = []
    for record in records:
        entries.append({
            "node_id": record.node_id,
            "delivery_seq": record.delivery_seq,
            "attempt_seq": record.attempt_seq,
            "wait_start_ms": record.wait_start_ms,
            "wait_end_ms": record.wait_end_ms,
            "wait_ms": max(0, record.wait_end_ms - record.wait_start_ms),
            "started_ms": record.started_ms,
            "ended_ms": record.ended_ms,
            "elapsed_ms": record.elapsed_ms,
            "outcome": record.outcome,
            "budget_consumed_ms": record.budget_consumed_ms,
        })
    entries.sort(key=lambda item: (item["started_ms"], item["node_id"],
                                   item["delivery_seq"], item["attempt_seq"]))
    return entries


class FanoutScheduler:
    def __init__(self, scene: Scene, ledger: Ledger, snapshot_dir: str | None = None,
                 snapshot_every: int = 0, progress_cb: Callable[[RuntimeCounters], None] | None = None,
                 only_jobs: list[tuple[str, int]] | None = None):
        self.scene = scene
        self.ledger = ledger
        self.snapshot_dir = snapshot_dir
        self.snapshot_every = snapshot_every
        self.progress_cb = progress_cb
        self.only_jobs = only_jobs
        self.counters = RuntimeCounters()
        for node in scene.nodes:
            self.counters.node_running[node.node_id] = 0
            self.counters.node_queued[node.node_id] = 0
            self.counters.budget_node_used_ms[node.node_id] = 0
            self.counters.attempts_used[node.node_id] = 0
        self._global_slots = threading.BoundedSemaphore(scene.scheduling.global_concurrency)
        self._node_slots = {
            node.node_id: threading.BoundedSemaphore(scene.scheduling.node_concurrency)
            for node in scene.nodes
        }
        self._committed_events = 0
        self._snapshot_lock = threading.Lock()

    # ------------------------------------------------------------------ jobs
    def _stable_jobs(self) -> list[tuple[str, int]]:
        if self.only_jobs is not None:
            # Preserve the canonical round-robin ordering for any subset.
            canonical: list[tuple[str, int]] = []
            for delivery_seq in range(self.scene.task.deliveries_per_node):
                for node in self.scene.nodes:
                    canonical.append((node.node_id, delivery_seq))
            allowed = set(self.only_jobs)
            return [job for job in canonical if job in allowed]
        jobs: list[tuple[str, int]] = []
        for delivery_seq in range(self.scene.task.deliveries_per_node):
            for node in self.scene.nodes:
                jobs.append((node.node_id, delivery_seq))
        return jobs

    def run(self) -> RuntimeCounters:
        jobs = self._stable_jobs()
        with ThreadPoolExecutor(max_workers=self.scene.scheduling.global_concurrency,
                                thread_name_prefix="fanout") as pool:
            futures = [pool.submit(self._run_delivery, node_id, seq) for node_id, seq in jobs]
            for future in futures:
                future.result()  # surface unexpected exceptions
        return self.counters

    def _run_delivery(self, node_id: str, delivery_seq: int) -> None:
        node = self.scene.node(node_id)
        max_attempts = self.scene.task.max_attempts
        total_backoff = 0
        attempts_used = 0
        final_outcome = "failure"
        for attempt_seq in range(max_attempts):
            record = self._run_one_attempt(node, delivery_seq, attempt_seq)
            attempts_used = attempt_seq + 1
            if record.outcome == "success":
                final_outcome = "success"
                break
            final_outcome = record.outcome
            if attempt_seq < max_attempts - 1:
                backoff = self.scene.task.backoff_ms[
                    min(attempt_seq, len(self.scene.task.backoff_ms) - 1)
                ]
                total_backoff += backoff
                time.sleep(backoff / 1000.0)
        with self.counters.lock:
            self.counters.attempts_used[node_id] += attempts_used
        self.ledger.commit_delivery(node_id, delivery_seq, final_outcome,
                                    attempts_used, total_backoff)
        if final_outcome != "success":
            # E_RETRY terminal marker in its own ledger table; it is not a
            # network attempt and therefore never enters metric aggregation.
            self.ledger.mark_retry_exhaustion(
                node_id=node_id, delivery_seq=delivery_seq,
                attempts_used=attempts_used, total_backoff_ms=total_backoff,
                last_outcome=final_outcome, marked_at_ms=now_ms(),
            )
            RetryExhaustedError(
                f"node {node_id} delivery {delivery_seq} exhausted {attempts_used} attempt(s)",
                {
                    "node_id": node_id,
                    "delivery_seq": delivery_seq,
                    "attempts_used": attempts_used,
                    "total_backoff_ms": total_backoff,
                    "last_outcome": final_outcome,
                },
            )

    def _run_one_attempt(self, node, delivery_seq: int, attempt_seq: int) -> AttemptRecord:
        with self.counters.lock:
            self.counters.queued += 1
            self.counters.node_queued[node.node_id] += 1
        wait_start = now_ms()
        # global budget first, then node budget: both limits always enforced
        self._global_slots.acquire()
        try:
            self._node_slots[node.node_id].acquire()
            try:
                wait_end = now_ms()
                with self.counters.lock:
                    self.counters.queued -= 1
                    self.counters.node_queued[node.node_id] -= 1
                    self.counters.running += 1
                    self.counters.node_running[node.node_id] += 1
                started = now_ms()
                response = post_attempt(
                    node, attempt_seq, self.scene.task.task_id,
                    self.scene.task.payload_bytes, self.scene.task.attempt_timeout_ms,
                )
                ended = now_ms()
                elapsed = max(0, ended - started)
                budget = elapsed
                record = AttemptRecord(
                    node_id=node.node_id, delivery_seq=delivery_seq,
                    attempt_seq=attempt_seq, wait_start_ms=wait_start,
                    wait_end_ms=wait_end, started_ms=started, ended_ms=ended,
                    elapsed_ms=elapsed, outcome=response.outcome,
                    budget_consumed_ms=budget, backoff_ms=0,
                    http_status=response.http_status, note=response.note,
                )
                self.ledger.commit_attempt(record)  # single atomic transaction
                with self.counters.lock:
                    self.counters.running -= 1
                    self.counters.node_running[node.node_id] -= 1
                    self.counters.outcome_counts[record.outcome] += 1
                    self.counters.budget_global_used_ms += budget
                    self.counters.budget_node_used_ms[node.node_id] += budget
                    self.counters.trajectory.append(self._trajectory_entry(record, wait_start, wait_end))
                self._maybe_snapshot()
                if self.progress_cb:
                    self.progress_cb(self.counters)
                return record
            finally:
                self._node_slots[node.node_id].release()
        finally:
            self._global_slots.release()

    @staticmethod
    def _trajectory_entry(record: AttemptRecord, wait_start: int, wait_end: int) -> dict[str, Any]:
        return {
            "node_id": record.node_id,
            "delivery_seq": record.delivery_seq,
            "attempt_seq": record.attempt_seq,
            "wait_start_ms": wait_start,
            "wait_end_ms": wait_end,
            "wait_ms": max(0, wait_end - wait_start),
            "started_ms": record.started_ms,
            "ended_ms": record.ended_ms,
            "elapsed_ms": record.elapsed_ms,
            "outcome": record.outcome,
            "budget_consumed_ms": record.budget_consumed_ms,
        }

    def _maybe_snapshot(self) -> None:
        if not self.snapshot_dir or self.snapshot_every <= 0:
            return
        with self._snapshot_lock:
            self._committed_events += 1
            if self._committed_events % self.snapshot_every != 0:
                return
        snapshot = self.snapshot("running")
        write_snapshot(snapshot, self.snapshot_dir)

    def snapshot(self, status: str, metrics: dict[str, Any] | None = None,
                 judgment: dict[str, Any] | None = None,
                 trajectory: list[dict[str, Any]] | None = None,
                 node_attempts_used: dict[str, int] | None = None,
                 budget_global_ms: int | None = None,
                 budget_per_node_ms: dict[str, int] | None = None) -> Snapshot:
        with self.counters.lock:
            if trajectory is None:
                trajectory = list(self.counters.trajectory)
            if node_attempts_used is None:
                node_attempts_used = dict(self.counters.attempts_used)
            if budget_global_ms is None:
                budget_global_ms = self.counters.budget_global_used_ms
            if budget_per_node_ms is None:
                budget_per_node_ms = dict(self.counters.budget_node_used_ms)
            return build_snapshot(
                session_id=self.ledger.session_id,
                scene_version=self.scene.scene_version,
                payload_digest=self.scene.payload_digest,
                mock_config=[
                    {
                        "node_id": n.node_id,
                        "failure_rate": n.failure_rate,
                        "slow_rate": n.slow_rate,
                        "process_ms": n.process_ms,
                        "slow_extra_ms": n.slow_extra_ms,
                        "seed": n.seed,
                    }
                    for n in self.scene.nodes
                ],
                node_attempts_used=node_attempts_used,
                budget_global_ms=budget_global_ms,
                budget_per_node_ms=budget_per_node_ms,
                trajectory=trajectory,
                metrics=metrics or {},
                judgment=judgment or {},
                status=status,
            )
