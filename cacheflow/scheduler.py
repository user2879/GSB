from __future__ import annotations

import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Callable


def monotonic_ms() -> int:
    return time.monotonic_ns() // 1_000_000


@dataclass(frozen=True)
class JobContext:
    seq: int
    node_id: str | None
    wait_started_ms: int
    wait_ended_ms: int
    scheduler: "Scheduler"

    def inflight(self) -> tuple[int, dict[str, int]]:
        return self.scheduler.current_inflight()


@dataclass
class Job:
    seq: int
    fn: Callable[[JobContext], Any]
    node_id: str | None = None
    group: str | None = None
    exclusive: bool = False


@dataclass(frozen=True)
class WaitRecord:
    seq: int
    node_id: str | None
    group: str | None
    wait_started_ms: int
    wait_ended_ms: int
    wait_ms: int
    dispatched_as_exclusive: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "seq": self.seq,
            "node_id": self.node_id,
            "group": self.group,
            "wait_started_ms": self.wait_started_ms,
            "wait_ended_ms": self.wait_ended_ms,
            "wait_ms": self.wait_ms,
            "exclusive": self.dispatched_as_exclusive,
        }


@dataclass
class JobResult:
    seq: int
    ok: bool
    value: Any = None
    error: BaseException | None = None


class Scheduler:
    """Dispatch jobs to a real thread pool under global/node in-flight caps."""

    def __init__(
        self,
        *,
        global_limit: int,
        node_limits: dict[str, int],
        max_workers: int | None = None,
    ) -> None:
        if global_limit <= 0:
            raise ValueError("global_limit must be positive")
        self.global_limit = int(global_limit)
        self.node_limits = dict(node_limits)
        self._lock = threading.RLock()
        self._cond = threading.Condition(self._lock)
        self._inflight_global = 0
        self._inflight_node: dict[str, int] = {node: 0 for node in node_limits}
        self._peak_global = 0
        self._peak_node: dict[str, int] = {node: 0 for node in node_limits}
        self.observed_violations: list[dict[str, Any]] = []
        self.wait_records: list[WaitRecord] = []
        self.results: dict[int, JobResult] = {}
        self._active_groups: set[str] = set()
        self._exclusive_active = False

    def current_inflight(self) -> tuple[int, dict[str, int]]:
        with self._lock:
            return self._inflight_global, dict(self._inflight_node)

    def peaks(self) -> dict[str, Any]:
        with self._lock:
            return {
                "global": self._peak_global,
                "nodes": dict(sorted(self._peak_node.items())),
            }

    def _node_limit(self, node_id: str | None) -> int:
        if node_id is None:
            return self.global_limit
        return int(self.node_limits.get(node_id, self.global_limit))

    def _can_start(self, job: Job) -> bool:
        if job.group and job.group in self._active_groups:
            return False
        if job.exclusive:
            return self._inflight_global == 0 and not self._exclusive_active
        if self._exclusive_active:
            return False
        if self._inflight_global >= self.global_limit:
            return False
        if job.node_id is not None:
            node_inflight = self._inflight_node.get(job.node_id, 0)
            if node_inflight >= self._node_limit(job.node_id):
                return False
        return True

    def _start_locked(self, job: Job, started_ms: int) -> JobContext:
        ended_ms = monotonic_ms()
        record = WaitRecord(
            seq=job.seq,
            node_id=job.node_id,
            group=job.group,
            wait_started_ms=started_ms,
            wait_ended_ms=ended_ms,
            wait_ms=max(0, ended_ms - started_ms),
            dispatched_as_exclusive=job.exclusive,
        )
        self.wait_records.append(record)
        self._inflight_global += 1
        self._peak_global = max(self._peak_global, self._inflight_global)
        if job.node_id is not None:
            self._inflight_node[job.node_id] = self._inflight_node.get(job.node_id, 0) + 1
            self._peak_node[job.node_id] = max(
                self._peak_node.get(job.node_id, 0), self._inflight_node[job.node_id]
            )
            if self._inflight_node[job.node_id] > self._node_limit(job.node_id):
                self.observed_violations.append(
                    {"seq": job.seq, "node_id": job.node_id, "kind": "node_limit"}
                )
        if self._inflight_global > self.global_limit:
            self.observed_violations.append({"seq": job.seq, "kind": "global_limit"})
        if job.group:
            self._active_groups.add(job.group)
        if job.exclusive:
            self._exclusive_active = True
        return JobContext(
            seq=job.seq,
            node_id=job.node_id,
            wait_started_ms=started_ms,
            wait_ended_ms=ended_ms,
            scheduler=self,
        )

    def _finish(self, job: Job, future: Future[Any]) -> None:
        with self._cond:
            error: BaseException | None = None
            value: Any = None
            ok = True
            try:
                value = future.result()
            except BaseException as exc:  # captured and returned to caller
                error = exc
                ok = False
            self.results[job.seq] = JobResult(job.seq, ok, value, error)
            self._inflight_global = max(0, self._inflight_global - 1)
            if job.node_id is not None:
                self._inflight_node[job.node_id] = max(0, self._inflight_node.get(job.node_id, 1) - 1)
            if job.group:
                self._active_groups.discard(job.group)
            if job.exclusive:
                self._exclusive_active = False
            self._cond.notify_all()

    def run_all(self, jobs: list[Job]) -> dict[int, JobResult]:
        ordered = sorted(jobs, key=lambda job: job.seq)
        enqueued_ms = monotonic_ms()
        start_times = {job.seq: enqueued_ms for job in ordered}
        pending = list(ordered)
        active_count = 0
        pool = ThreadPoolExecutor(max_workers=max(1, self.global_limit), thread_name_prefix="cacheflow-job")
        try:
            with self._cond:
                while pending or self._inflight_global:
                    launched_any = False
                    index = 0
                    while index < len(pending):
                        job = pending[index]
                        if job.exclusive:
                            if self._can_start(job):
                                pending.pop(index)
                                context = self._start_locked(job, start_times[job.seq])
                                active_count += 1
                                launched_any = True
                                future = pool.submit(self._invoke, job, context)
                                future.add_done_callback(lambda done, current=job: self._finish(current, done))
                            # A quota/control barrier must order before every later event.
                            break
                        if self._can_start(job):
                            pending.pop(index)
                            context = self._start_locked(job, start_times[job.seq])
                            active_count += 1
                            launched_any = True
                            future = pool.submit(self._invoke, job, context)
                            future.add_done_callback(lambda done, current=job: self._finish(current, done))
                        else:
                            index += 1
                    if not launched_any and (pending or self._inflight_global):
                        self._cond.wait(timeout=0.01)
        finally:
            pool.shutdown(wait=True)
        if self.observed_violations:
            raise RuntimeError(f"scheduler limit violations: {self.observed_violations}")
        return dict(sorted(self.results.items()))

    @staticmethod
    def _invoke(job: Job, context: JobContext) -> Any:
        return job.fn(context)

    def ordered_wait_records(self) -> list[dict[str, Any]]:
        return [record.to_dict() for record in sorted(self.wait_records, key=lambda item: item.seq)]
