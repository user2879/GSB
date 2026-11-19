"""Budgeted parallel packing scheduler with real monotonic timeouts."""

from __future__ import annotations

import heapq
import threading
import time
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import dataclass
from typing import Any, Iterable

from .audit import utc_now
from .packing import PackStore, PackTask, PreparedArtifact


@dataclass(frozen=True)
class ScheduleEntry:
    task_id: str
    ledger_task_id: str
    attempt: int
    node_id: str
    logical_id: str
    seq: int
    bytes: int
    queued_at_ms: float
    started_at_ms: float | None
    ended_at_ms: float | None
    queue_ms: float
    duration_ms: float
    outcome: str
    completed_bytes: int
    content_digest: str | None
    version: int | None
    budget_consumed: int
    skip_reason: str
    retry_of: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "ledger_task_id": self.ledger_task_id,
            "attempt": self.attempt,
            "node_id": self.node_id,
            "logical_id": self.logical_id,
            "seq": self.seq,
            "bytes": self.bytes,
            "queued_at_monotonic_ms": round(self.queued_at_ms, 3),
            "started_at_monotonic_ms": None if self.started_at_ms is None else round(self.started_at_ms, 3),
            "ended_at_monotonic_ms": None if self.ended_at_ms is None else round(self.ended_at_ms, 3),
            "queue_ms": round(self.queue_ms, 3),
            "duration_ms": round(self.duration_ms, 3),
            "outcome": self.outcome,
            "completed_bytes": self.completed_bytes,
            "content_digest": self.content_digest,
            "version": self.version,
            "budget_consumed": self.budget_consumed,
            "skip_reason": self.skip_reason,
            "retry_of": self.retry_of,
        }


@dataclass(frozen=True)
class ScheduleResult:
    run_id: str
    entries: tuple[ScheduleEntry, ...]
    global_slots: int
    node_quotas: dict[str, int]
    max_global_observed: int
    max_per_node_observed: dict[str, int]
    started_at: str
    ended_at: str
    elapsed_ms: float

    @property
    def successful(self) -> tuple[ScheduleEntry, ...]:
        return tuple(entry for entry in self.entries if entry.outcome == "success")

    def as_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "global_slots": self.global_slots,
            "node_quotas": dict(sorted(self.node_quotas.items())),
            "max_global_observed": self.max_global_observed,
            "max_per_node_observed": dict(sorted(self.max_per_node_observed.items())),
            "started_at": self.started_at,
            "ended_at": self.ended_at,
            "elapsed_ms": round(self.elapsed_ms, 3),
            "entries": [entry.as_dict() for entry in self.entries],
        }


@dataclass(frozen=True)
class _Attempt:
    task: PackTask
    attempt: int
    queued_at_ms: float
    retry_of: str | None


class _Observation:
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.global_active = 0
        self.global_max = 0
        self.node_active: dict[str, int] = {}
        self.node_max: dict[str, int] = {}

    def enter(self, node_id: str) -> None:
        with self.lock:
            self.global_active += 1
            self.global_max = max(self.global_max, self.global_active)
            self.node_active[node_id] = self.node_active.get(node_id, 0) + 1
            self.node_max[node_id] = max(self.node_max.get(node_id, 0), self.node_active[node_id])

    def leave(self, node_id: str) -> None:
        with self.lock:
            self.global_active -= 1
            self.node_active[node_id] -= 1


class PackScheduler:
    def __init__(
        self,
        *,
        store: PackStore,
        global_slots: int,
        node_quotas: dict[str, int],
        logger: Any | None = None,
        ledger: Any | None = None,
        run_id: str = "run-0001",
        crash_after_commits: int | None = None,
    ) -> None:
        if global_slots < 1:
            raise ValueError("global_slots must be >= 1")
        self.store = store
        self.global_slots = global_slots
        self.node_quotas = {key: int(value) for key, value in node_quotas.items()}
        self.logger = logger
        self.ledger = ledger
        self.run_id = run_id
        self.crash_after_commits = crash_after_commits
        self._commits = 0
        self._commit_lock = threading.Lock()

    def run(
        self,
        tasks: Iterable[PackTask],
        *,
        resume_completed: bool = False,
    ) -> ScheduleResult:
        started_at = utc_now()
        start_mono = time.monotonic()
        observation = _Observation()
        pending: list[tuple[tuple[int, str, int, int], _Attempt]] = []
        for task in tasks:
            attempt = _Attempt(task=task, attempt=0, queued_at_ms=time.monotonic() * 1000.0, retry_of=None)
            heapq.heappush(pending, ((task.seq, task.logical_id, 0, 0), attempt))

        completed_rows: dict[str, dict[str, Any]] = {}
        if resume_completed and self.ledger is not None:
            completed_rows = self.ledger.completed_success_rows(self.run_id)
        entries: list[ScheduleEntry] = []
        active_by_node: dict[str, int] = {}
        inflight: dict[Future[ScheduleEntry], _Attempt] = {}

        with ThreadPoolExecutor(max_workers=self.global_slots, thread_name_prefix="shardrep-pack") as executor:
            while pending or inflight:
                submitted_any = False
                remaining: list[tuple[tuple[int, str, int, int], _Attempt]] = []
                while pending and len(inflight) < self.global_slots:
                    key, attempt = heapq.heappop(pending)
                    task = attempt.task
                    if task.task_id in completed_rows:
                        self.store.promote_staged(task.task_id, task.logical_id)
                        if self.store.artifact_info(task.logical_id) is not None:
                            entries.append(
                                self._recovered_entry(attempt, completed_rows[task.task_id])
                            )
                            continue
                        completed_rows.pop(task.task_id, None)
                    quota = self.node_quotas.get(task.node_id, 0)
                    if quota <= 0:
                        entries.append(
                            self._skipped_entry(
                                attempt,
                                outcome="skipped",
                                reason="node_quota_zero_or_missing",
                                budget_consumed=0,
                            )
                        )
                        continue
                    if active_by_node.get(task.node_id, 0) >= quota:
                        remaining.append((key, attempt))
                        continue
                    active_by_node[task.node_id] = active_by_node.get(task.node_id, 0) + 1
                    future = executor.submit(self._execute, attempt, observation)
                    inflight[future] = attempt
                    submitted_any = True
                for item in remaining:
                    heapq.heappush(pending, item)

                if inflight:
                    done, _not_done = wait(tuple(inflight), timeout=0.02, return_when=FIRST_COMPLETED)
                    for future in done:
                        attempt = inflight.pop(future)
                        active_by_node[attempt.task.node_id] -= 1
                        entry = future.result()
                        entries.append(entry)
                        if entry.outcome == "timeout" and entry.skip_reason != "timeout_retries_exhausted":
                            retry = _Attempt(
                                task=attempt.task,
                                attempt=attempt.attempt + 1,
                                queued_at_ms=time.monotonic() * 1000.0,
                                retry_of=entry.ledger_task_id,
                            )
                            heapq.heappush(
                                pending,
                                (
                                    (
                                        attempt.task.seq,
                                        attempt.task.logical_id,
                                        retry.attempt,
                                        0,
                                    ),
                                    retry,
                                ),
                            )
                elif not submitted_any and pending:
                    # No node currently has quota available and nothing is running.
                    key, attempt = heapq.heappop(pending)
                    entries.append(
                        self._skipped_entry(
                            attempt,
                            outcome="skipped",
                            reason="node_quota_exhausted",
                            budget_consumed=0,
                        )
                    )

        entries.sort(
            key=lambda entry: (
                entry.seq,
                entry.logical_id,
                entry.attempt,
                entry.ledger_task_id,
            )
        )
        elapsed_ms = (time.monotonic() - start_mono) * 1000.0
        return ScheduleResult(
            run_id=self.run_id,
            entries=tuple(entries),
            global_slots=self.global_slots,
            node_quotas=dict(self.node_quotas),
            max_global_observed=observation.global_max,
            max_per_node_observed=dict(observation.node_max),
            started_at=started_at,
            ended_at=utc_now(),
            elapsed_ms=elapsed_ms,
        )

    def _execute(self, attempt: _Attempt, observation: _Observation) -> ScheduleEntry:
        task = attempt.task
        observation.enter(task.node_id)
        started_ms = time.monotonic() * 1000.0
        ledger_task_id = f"{self.run_id}:{task.task_id}:a{attempt.attempt}"
        try:
            if self.ledger is not None:
                self.ledger.claim(
                    ledger_task_id=ledger_task_id,
                    run_id=self.run_id,
                    base_task_id=task.task_id,
                    node_id=task.node_id,
                    logical_id=task.logical_id,
                    attempt=attempt.attempt,
                )
            outcome, completed_bytes, prepared = self._simulate_work(task)
            ended_ms = time.monotonic() * 1000.0
            if outcome == "success" and prepared is not None:
                if self.ledger is not None:
                    self.ledger.commit(
                        ledger_task_id=ledger_task_id,
                        run_id=self.run_id,
                        base_task_id=task.task_id,
                        node_id=task.node_id,
                        logical_id=task.logical_id,
                        attempt=attempt.attempt,
                        outcome="success",
                        bytes_completed=completed_bytes,
                        budget_consumed=1,
                        result=prepared.as_dict(),
                    )
                self._crash_if_due()
                self.store.publish(prepared)
                version = prepared.version
                digest = prepared.content_digest
                skip_reason = ""
            else:
                final_timeout = outcome == "timeout" and attempt.attempt >= task.max_retries
                skip_reason = "timeout_retries_exhausted" if final_timeout else ""
                version = None
                digest = None
                if self.ledger is not None:
                    self.ledger.commit(
                        ledger_task_id=ledger_task_id,
                        run_id=self.run_id,
                        base_task_id=task.task_id,
                        node_id=task.node_id,
                        logical_id=task.logical_id,
                        attempt=attempt.attempt,
                        outcome=outcome,
                        bytes_completed=completed_bytes,
                        budget_consumed=1,
                        result={
                            "task_id": task.task_id,
                            "logical_id": task.logical_id,
                            "node_id": task.node_id,
                            "outcome": outcome,
                            "completed_bytes": completed_bytes,
                            "retry_scheduled": outcome == "timeout" and not final_timeout,
                        },
                    )
                self._crash_if_due()
            if self.logger is not None and outcome != "success":
                self.logger.emit(
                    "E_TIMEOUT" if outcome == "timeout" else "E_PACK",
                    shard_id=task.logical_id,
                    version=self.store.spec.group_version,
                    task_id=task.task_id,
                    node_id=task.node_id,
                    completed_bytes=completed_bytes,
                )
            return ScheduleEntry(
                task_id=task.task_id,
                ledger_task_id=ledger_task_id,
                attempt=attempt.attempt,
                node_id=task.node_id,
                logical_id=task.logical_id,
                seq=task.seq,
                bytes=task.bytes,
                queued_at_ms=attempt.queued_at_ms,
                started_at_ms=started_ms,
                ended_at_ms=ended_ms,
                queue_ms=max(0.0, started_ms - attempt.queued_at_ms),
                duration_ms=max(0.0, ended_ms - started_ms),
                outcome=outcome,
                completed_bytes=completed_bytes,
                content_digest=digest,
                version=version,
                budget_consumed=1,
                skip_reason=skip_reason,
                retry_of=attempt.retry_of,
            )
        except Exception as exc:  # Keep a scheduler failure attributable and visible.
            ended_ms = time.monotonic() * 1000.0
            if self.logger is not None:
                self.logger.emit(
                    "E_PACK",
                    shard_id=task.logical_id,
                    version=self.store.spec.group_version,
                    task_id=task.task_id,
                    node_id=task.node_id,
                    error=str(exc),
                )
            return ScheduleEntry(
                task_id=task.task_id,
                ledger_task_id=ledger_task_id,
                attempt=attempt.attempt,
                node_id=task.node_id,
                logical_id=task.logical_id,
                seq=task.seq,
                bytes=task.bytes,
                queued_at_ms=attempt.queued_at_ms,
                started_at_ms=started_ms,
                ended_at_ms=ended_ms,
                queue_ms=max(0.0, started_ms - attempt.queued_at_ms),
                duration_ms=max(0.0, ended_ms - started_ms),
                outcome="error",
                completed_bytes=0,
                content_digest=None,
                version=None,
                budget_consumed=1,
                skip_reason=f"exception:{type(exc).__name__}",
                retry_of=attempt.retry_of,
            )
        finally:
            observation.leave(task.node_id)

    def _crash_if_due(self) -> None:
        if self.crash_after_commits is None:
            return
        with self._commit_lock:
            self._commits += 1
            should_crash = self._commits >= self.crash_after_commits
        if should_crash:
            import os

            os._exit(17)

    def _simulate_work(self, task: PackTask) -> tuple[str, int, PreparedArtifact | None]:
        deadline = time.monotonic() + max(0, task.timeout_ms) / 1000.0
        processed = 0
        chunks = 8
        per_chunk_seconds = max(0.0, task.simulate_ms) / 1000.0 / chunks
        for chunk in range(chunks):
            if time.monotonic() >= deadline:
                return "timeout", processed, None
            if per_chunk_seconds > 0:
                remaining_seconds = max(0.0, deadline - time.monotonic())
                time.sleep(min(per_chunk_seconds, remaining_seconds))
            processed = int(task.bytes * (chunk + 1) / chunks)
            if time.monotonic() >= deadline and chunk < chunks - 1:
                return "timeout", processed, None
        prepared = self.store.prepare(task)
        return "success", task.bytes, prepared

    @staticmethod
    def _skipped_entry(
        attempt: _Attempt,
        *,
        outcome: str,
        reason: str,
        budget_consumed: int,
    ) -> ScheduleEntry:
        task = attempt.task
        ledger_task_id = f"skipped:{task.task_id}:a{attempt.attempt}"
        return ScheduleEntry(
            task_id=task.task_id,
            ledger_task_id=ledger_task_id,
            attempt=attempt.attempt,
            node_id=task.node_id,
            logical_id=task.logical_id,
            seq=task.seq,
            bytes=task.bytes,
            queued_at_ms=attempt.queued_at_ms,
            started_at_ms=None,
            ended_at_ms=None,
            queue_ms=0.0,
            duration_ms=0.0,
            outcome=outcome,
            completed_bytes=0,
            content_digest=None,
            version=None,
            budget_consumed=budget_consumed,
            skip_reason=reason,
            retry_of=attempt.retry_of,
        )

    @staticmethod
    def _recovered_entry(attempt: _Attempt, row: dict[str, Any]) -> ScheduleEntry:
        task = attempt.task
        import json

        try:
            result = json.loads(row.get("result_json") or "{}")
        except json.JSONDecodeError:
            result = {}
        return ScheduleEntry(
            task_id=task.task_id,
            ledger_task_id=str(row["ledger_task_id"]),
            attempt=attempt.attempt,
            node_id=task.node_id,
            logical_id=task.logical_id,
            seq=task.seq,
            bytes=task.bytes,
            queued_at_ms=attempt.queued_at_ms,
            started_at_ms=None,
            ended_at_ms=None,
            queue_ms=0.0,
            duration_ms=0.0,
            outcome="recovered",
            completed_bytes=int(row.get("bytes_completed", 0)),
            content_digest=result.get("content_digest"),
            version=result.get("version"),
            budget_consumed=0,
            skip_reason="already_committed",
            retry_of=attempt.retry_of,
        )
