"""Bounded parallel collection scheduler with retry isolation."""

from __future__ import annotations

import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Callable

from .errors import E_EXHAUST
from .transport import AttemptRecord, send_request


def now_ms() -> int:
    """Monotonic clock in integer milliseconds (never goes backwards)."""

    return int(time.monotonic() * 1000)


@dataclass
class CollectionJob:
    job_id: str
    node_id: str
    host: str
    port: int
    manifest_version: int
    payload: dict


@dataclass
class JobResult:
    job_id: str
    node_id: str
    manifest_version: int
    status: str
    response: dict | None
    attempts: list[AttemptRecord] = field(default_factory=list)
    enqueued_monotonic_ms: int = 0
    wait_started_monotonic_ms: int = 0
    wait_ended_monotonic_ms: int = 0
    queue_wait_ms: int = 0
    error_code: str | None = None
    error: str | None = None


class Scheduler:
    """Run jobs under a global and a per-node concurrency ceiling.

    Invariants enforced at every instant:

    * in-flight jobs <= ``global_concurrency`` (global semaphore);
    * in-flight jobs for one node <= ``node_concurrency`` (per-node semaphore).

    Jobs that cannot acquire a slot block in FIFO order and record both the
    monotonic timestamps at which waiting started and ended.
    """

    def __init__(self, global_concurrency: int, node_concurrency: int,
                 socket_timeout_ms: int, max_attempts: int,
                 retry_backoff_ms: int):
        self.global_limit = global_concurrency
        self.node_limit = node_concurrency
        self.socket_timeout_ms = socket_timeout_ms
        self.max_attempts = max_attempts
        self.retry_backoff_ms = retry_backoff_ms
        self._global_slots = threading.BoundedSemaphore(global_concurrency)
        self._node_slots: dict[str, threading.BoundedSemaphore] = {}
        self._node_locks_guard = threading.Lock()

    def _node_semaphore(self, node_id: str) -> threading.BoundedSemaphore:
        with self._node_locks_guard:
            sem = self._node_slots.get(node_id)
            if sem is None:
                sem = threading.BoundedSemaphore(self.node_limit)
                self._node_slots[node_id] = sem
            return sem

    def run_all(self, jobs: list[CollectionJob]) -> list[JobResult]:
        ordered = sorted(jobs, key=lambda job: (job.node_id,
                                                job.manifest_version,
                                                job.job_id))
        for job in ordered:
            job.enqueued_monotonic_ms = now_ms()
        results: list[JobResult | None] = [None] * len(ordered)
        with ThreadPoolExecutor(max_workers=self.global_limit) as pool:
            futures = {pool.submit(self._run_job, job): index
                       for index, job in enumerate(ordered)}
            for future in futures:
                index = futures[future]
                results[index] = future.result()
        return [result for result in results if result is not None]

    def _run_job(self, job: CollectionJob) -> JobResult:
        wait_started = now_ms()
        node_sem = self._node_semaphore(job.node_id)
        node_sem.acquire()
        try:
            self._global_slots.acquire()
            try:
                wait_ended = now_ms()
                return self._execute(job, wait_started, wait_ended)
            finally:
                self._global_slots.release()
        finally:
            node_sem.release()

    def _execute(self, job: CollectionJob, wait_started: int,
                 wait_ended: int) -> JobResult:
        attempts: list[AttemptRecord] = []
        last_error: str | None = None
        for attempt in range(1, self.max_attempts + 1):
            started = now_ms()
            try:
                # Same payload/request_id on every attempt: retries never
                # alter the semantics of a single collection.
                response = send_request(job.host, job.port, job.payload,
                                        self.socket_timeout_ms)
                ended = now_ms()
                attempts.append(AttemptRecord(attempt, started, ended, "ok",
                                              None))
                return JobResult(
                    job_id=job.job_id, node_id=job.node_id,
                    manifest_version=job.manifest_version, status="ok",
                    response=response, attempts=attempts,
                    enqueued_monotonic_ms=job.enqueued_monotonic_ms,
                    wait_started_monotonic_ms=wait_started,
                    wait_ended_monotonic_ms=wait_ended,
                    queue_wait_ms=wait_ended - wait_started)
            except (OSError, TimeoutError, ValueError) as exc:
                ended = now_ms()
                last_error = f"{type(exc).__name__}: {exc}"
                attempts.append(AttemptRecord(attempt, started, ended,
                                              "failed", last_error))
                if attempt < self.max_attempts:
                    time.sleep(self.retry_backoff_ms / 1000.0)
        return JobResult(
            job_id=job.job_id, node_id=job.node_id,
            manifest_version=job.manifest_version, status=E_EXHAUST,
            response=None, attempts=attempts,
            enqueued_monotonic_ms=job.enqueued_monotonic_ms,
            wait_started_monotonic_ms=wait_started,
            wait_ended_monotonic_ms=wait_ended,
            queue_wait_ms=wait_ended - wait_started,
            error_code=E_EXHAUST, error=last_error)


def build_jobs(manifest, registry, quantized: dict[str, int],
               session) -> list[CollectionJob]:
    """Construct one collection job per (target node, manifest version)."""

    jobs: list[CollectionJob] = []
    targets_by_node: dict[str, list[str]] = {}
    for key in manifest.order:
        for node_id in manifest.items[key].targets:
            targets_by_node.setdefault(node_id, []).append(key)
    for node_id in sorted(targets_by_node):
        entry = registry.nodes[node_id]
        item_payload = []
        for key in sorted(targets_by_node[node_id]):
            item = manifest.items[key]
            item_payload.append({
                "key": key,
                "value": item.value,
                "scale_value": item.value,
                "quantized_value": quantized.get(key),
            })
        payload = {
            "request_id": f"req-{manifest.version}-{node_id}",
            "node_id": node_id,
            "manifest_version": manifest.version,
            "base_seed": session.base_seed,
            "sample_count": session.sample_count,
            "group_count": session.group_count,
            "aggregations": list(session.aggregations),
            "items": item_payload,
        }
        jobs.append(CollectionJob(
            job_id=f"job-{manifest.version}-{node_id}",
            node_id=node_id, host=entry.host,
            port=entry.port, manifest_version=manifest.version,
            payload=payload))
    return jobs
