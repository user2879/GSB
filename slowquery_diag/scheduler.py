"""Subprocess-managed loopback service and quota-aware replay scheduler.

Concurrency is enforced by real synchronization primitives (bounded
thread pool plus per-source condition variables); the only waiting a job
performs is genuinely waiting for a free slot, and the wait interval is
measured with ``time.monotonic_ns``.  Socket timeout is enforced inside
:class:`ReplayClient`, never by artificial scheduler sleeps.
"""

from __future__ import annotations

import os
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

from .errors import PortError
from .replay_server import ReplayClient

READY_MARKER = "REPLAY_SERVICE_READY"


def _monotonic_ms() -> int:
    return time.monotonic_ns() // 1_000_000


class LoopbackServiceProcess:
    """Owns the replay-service child process lifecycle."""

    def __init__(self, catalog_path: str, startup_timeout_ms: int = 5000) -> None:
        self.catalog_path = os.path.abspath(catalog_path)
        self.startup_timeout_ms = startup_timeout_ms
        self._process: subprocess.Popen | None = None
        self.host = "127.0.0.1"
        self.port: int | None = None

    def start(self) -> int:
        command = [
            sys.executable,
            "-m",
            "slowquery_diag.replay_server",
            "--catalog",
            self.catalog_path,
            "--host",
            self.host,
            "--port",
            "0",
        ]
        try:
            self._process = subprocess.Popen(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                # New process group so terminate() stays scoped on Windows.
                creationflags=self._creation_flags(),
            )
        except OSError as exc:
            raise PortError(
                f"failed to spawn replay service: {exc}",
                ref=self.host, field="spawn",
            )

        assert self._process.stdout is not None
        deadline = _monotonic_ms() + self.startup_timeout_ms
        line = self._process.stdout.readline()
        if not line:
            error = self._drain_stderr()
            self._cleanup()
            raise PortError(
                f"replay service exited before ready: {error}",
                ref=self.host, field="startup",
            )
        if _monotonic_ms() > deadline or READY_MARKER not in line:
            self._cleanup()
            raise PortError(
                f"replay service did not announce readiness: {line.strip()}",
                ref=self.host, field="startup",
            )
        try:
            self.port = int(line.split("port=")[1].split()[0])
        except (IndexError, ValueError):
            self._cleanup()
            raise PortError(
                f"malformed readiness line: {line.strip()}",
                ref=self.host, field="startup",
            )
        return self.port

    def stop(self) -> None:
        if self._process is None:
            return
        self._cleanup()

    def _cleanup(self) -> None:
        process = self._process
        if process is None:
            return
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=3.0)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=3.0)
        for stream in (process.stdout, process.stderr):
            if stream is not None:
                stream.close()
        self._process = None

    def _drain_stderr(self) -> str:
        if self._process is None or self._process.stderr is None:
            return ""
        try:
            return self._process.stderr.read()[:2000]
        except OSError:
            return ""

    @staticmethod
    def _creation_flags() -> int:
        # CREATE_NEW_PROCESS_GROUP on Windows; 0 elsewhere.
        return getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)


@dataclass
class ReplayJob:
    candidate_id: str
    source_id: str
    rewritten_sql: str
    queued_at_ms: int = 0
    wait_start_ms: int = 0
    wait_end_ms: int = 0
    started_at_ms: int = 0
    finished_at_ms: int = 0


@dataclass
class ReplayResult:
    job: ReplayJob
    ack: dict | None = None
    error_code: str | None = None
    error_message: str | None = None

    @property
    def ok(self) -> bool:
        return self.error_code is None


@dataclass
class SchedulerStats:
    total_jobs: int = 0
    peak_global_inflight: int = 0
    peak_per_source: dict[str, int] = field(default_factory=dict)
    queued_wait_count: int = 0


class ReplayScheduler:
    """Bounded parallel replay with global and per-source quotas."""

    def __init__(self, port: int, *, global_concurrency: int,
                 per_source_quota: dict[str, int], socket_timeout_ms: int,
                 host: str = "127.0.0.1") -> None:
        if global_concurrency < 1:
            raise ValueError("global_concurrency must be >= 1")
        self.client = ReplayClient(host=host, port=port,
                                   timeout_ms=socket_timeout_ms)
        self.global_limit = global_concurrency
        self.per_source_quota = dict(per_source_quota)
        self._global_slots = threading.BoundedSemaphore(global_concurrency)
        self._source_conditions = {
            source: threading.Condition() for source in per_source_quota
        }
        self._source_available = dict(per_source_quota)
        self._stats_lock = threading.Lock()
        self.stats = SchedulerStats(peak_per_source={})
        self._global_inflight = 0
        self._source_inflight: dict[str, int] = {
            source: 0 for source in per_source_quota
        }

    def run_all(self, jobs: list[ReplayJob]) -> list[ReplayResult]:
        for job in jobs:
            job.queued_at_ms = _monotonic_ms()
        self.stats.total_jobs = len(jobs)
        ordered = sorted(jobs, key=lambda item: (item.source_id, item.candidate_id))
        with ThreadPoolExecutor(max_workers=self.global_limit) as pool:
            futures = [pool.submit(self._run_job, job) for job in ordered]
            results = [future.result() for future in futures]
        # Stable output order regardless of thread completion order.
        results.sort(key=lambda result: (result.job.source_id,
                                         result.job.candidate_id))
        return results

    def _run_job(self, job: ReplayJob) -> ReplayResult:
        condition = self._source_conditions[job.source_id]

        job.wait_start_ms = _monotonic_ms()
        # Real blocking wait on the per-source condition until a quota slot
        # frees; no sleep/poll loop and no artificial delay is involved.
        with condition:
            while self._source_available[job.source_id] == 0:
                condition.wait()
            self._source_available[job.source_id] -= 1
        self._global_slots.acquire()
        job.wait_end_ms = _monotonic_ms()
        if job.wait_end_ms - job.wait_start_ms > 0:
            with self._stats_lock:
                self.stats.queued_wait_count += 1
        self._enter_inflight(job.source_id)
        job.started_at_ms = _monotonic_ms()
        try:
            ack = self.client.replay(
                job.candidate_id, job.source_id, job.rewritten_sql
            )
            result = ReplayResult(job=job, ack=ack)
        except Exception as exc:  # normalized into coded ReplayResult
            result = ReplayResult(
                job=job,
                error_code=getattr(exc, "code", "E_INTERNAL"),
                error_message=str(exc),
            )
        finally:
            job.finished_at_ms = _monotonic_ms()
            self._exit_inflight(job.source_id)
            self._global_slots.release()
            with condition:
                self._source_available[job.source_id] += 1
                condition.notify()
        return result

    def _enter_inflight(self, source_id: str) -> None:
        with self._stats_lock:
            self._global_inflight += 1
            self.stats.peak_global_inflight = max(
                self.stats.peak_global_inflight, self._global_inflight
            )
            current = self._source_inflight[source_id] + 1
            self._source_inflight[source_id] = current
            previous = self.stats.peak_per_source.get(source_id, 0)
            self.stats.peak_per_source[source_id] = max(previous, current)

    def _exit_inflight(self, source_id: str) -> None:
        with self._stats_lock:
            self._global_inflight -= 1
            self._source_inflight[source_id] -= 1
