"""Bounded concurrent prefetch execution over the deterministic download plan."""

from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor, Future
from typing import Any, Callable

from .canon import ceil_fraction, fraction_from_str, sha256_canonical
from .errors import BusinessError


class _Cancelled(Exception):
    pass


def run_prefetch_validation(
    planned_jobs: list[dict],
    *,
    concurrency: int,
    memory_budget_bytes: int,
    task_hook: Callable[[dict], Any] | None = None,
) -> dict:
    """Execute the deterministic plan in bounded concurrent batches.

    The executor only validates fragment delivery. It never mutates the ABR
    timeline, which is the invariant that makes concurrency independent from
    buffer/rate decisions.
    """
    if concurrency < 1:
        raise BusinessError("E_PREFETCH", "concurrency must be at least one")
    if memory_budget_bytes < 1:
        raise BusinessError("E_PREFETCH", "memory budget must be positive")
    if not planned_jobs:
        return {
            "concurrency": concurrency,
            "memory_budget_bytes": memory_budget_bytes,
            "job_count": 0,
            "batch_count": 0,
            "executed_jobs": 0,
            "cancelled_jobs": 0,
            "peak_active_tasks": 0,
            "peak_active_bytes": 0,
            "digests": [],
        }

    stop_event = threading.Event()
    lock = threading.Lock()
    active_tasks = 0
    active_bytes = 0
    peak_tasks = 0
    peak_bytes = 0
    errors: list[tuple[int, BaseException]] = []
    cancelled = 0
    executed = 0
    digests: list[str] = []
    batch_count = 0

    def work(job: dict, release: threading.Event) -> str:
        nonlocal active_tasks, active_bytes, peak_tasks, peak_bytes, cancelled, executed
        release.wait()
        if stop_event.is_set():
            with lock:
                cancelled += 1
            raise _Cancelled()
        size = ceil_fraction(fraction_from_str(job["size_bytes"], "job.size_bytes"))
        with lock:
            active_tasks += 1
            active_bytes += size
            peak_tasks = max(peak_tasks, active_tasks)
            peak_bytes = max(peak_bytes, active_bytes)
        try:
            if task_hook is not None:
                task_hook(job)
            digest = sha256_canonical(
                {
                    "job_seq": job["job_seq"],
                    "segment_index": job["segment_index"],
                    "rep_index": job["rep_index"],
                    "bitrate_bps": job["bitrate_bps"],
                    "size_bytes": job["size_bytes"],
                    "duration_ms": job["duration_ms"],
                }
            )
            with lock:
                executed += 1
            return digest
        except _Cancelled:
            raise
        except BaseException:
            stop_event.set()
            raise
        finally:
            with lock:
                active_tasks -= 1
                active_bytes -= size

    index = 0
    executor = ThreadPoolExecutor(max_workers=concurrency, thread_name_prefix="streamsim-prefetch")
    try:
        while index < len(planned_jobs):
            batch: list[dict] = []
            batch_bytes = 0
            while index < len(planned_jobs) and len(batch) < concurrency:
                job = planned_jobs[index]
                job_bytes = ceil_fraction(fraction_from_str(job["size_bytes"], "job.size_bytes"))
                if job_bytes > memory_budget_bytes:
                    raise BusinessError(
                        "E_PREFETCH",
                        "a fragment is larger than the configured memory budget",
                        "client.memory_budget_bytes",
                        {"job_seq": job.get("job_seq"), "size_bytes": job_bytes},
                    )
                if batch and batch_bytes + job_bytes > memory_budget_bytes:
                    break
                batch.append(job)
                batch_bytes += job_bytes
                index += 1
            if not batch:
                raise BusinessError("E_PREFETCH", "failed to form a prefetch batch")
            batch_count += 1
            peak_tasks = max(peak_tasks, len(batch))
            peak_bytes = max(peak_bytes, batch_bytes)
            release = threading.Event()
            futures: list[tuple[int, Future]] = []
            for job in batch:
                future = executor.submit(work, job, release)
                futures.append((int(job["job_seq"]), future))
            release.set()
            for job_seq, future in futures:
                try:
                    digests.append(future.result())
                except _Cancelled:
                    pass
                except BaseException as exc:
                    errors.append((job_seq, exc))
                    stop_event.set()
                    for _, other in futures:
                        if not other.done():
                            other.cancel()
            if errors:
                break
    finally:
        stop_event.set()
        executor.shutdown(wait=True, cancel_futures=True)

    if errors:
        job_seq, exc = min(errors, key=lambda item: item[0])
        if isinstance(exc, BusinessError):
            raise exc
        raise BusinessError(
            "E_PREFETCH",
            f"prefetch job {job_seq} failed: {exc}",
            details={"first_error_job_seq": job_seq, "error_type": type(exc).__name__},
        ) from exc

    return {
        "concurrency": concurrency,
        "memory_budget_bytes": memory_budget_bytes,
        "job_count": len(planned_jobs),
        "batch_count": batch_count,
        "executed_jobs": executed,
        "cancelled_jobs": cancelled,
        "peak_active_tasks": peak_tasks,
        "peak_active_bytes": peak_bytes,
        "digests": digests,
    }
