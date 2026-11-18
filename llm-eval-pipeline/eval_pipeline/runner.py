"""Deterministic concurrent execution.

Tasks carry a stable integer sort key. Results are always returned in key
order regardless of the worker count, so concurrency changes nothing beyond
wall-clock time. Per-task exceptions are captured instead of aborting the
batch, which keeps single-sample faults isolated.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor


def run_tordered(tasks: list, worker_count: int) -> list:
    """Run (key, callable) tasks and return (key, result_or_exception) in key order."""
    # Keys are stable integer indices; sort numerically, never lexicographically.
    ordered = sorted(tasks, key=lambda item: int(item[0]))
    if worker_count <= 1:
        outcomes = []
        for key, callable_task in ordered:
            try:
                outcomes.append((key, callable_task(), None))
            except Exception as exc:  # isolated per-sample failure
                outcomes.append((key, None, exc))
        return outcomes

    with ThreadPoolExecutor(max_workers=worker_count) as pool:
        futures = {pool.submit(callable_task): key for key, callable_task in ordered}
        collected = {}
        for future, key in futures.items():
            try:
                collected[key] = (key, future.result(), None)
            except Exception as exc:
                collected[key] = (key, None, exc)
    return [collected[key] for key, _task in ordered]
