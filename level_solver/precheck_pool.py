"""Bounded concurrent feasibility prechecks.

Candidate constraint prechecks run in a fixed-size thread pool.  A second
semaphore bounds how many candidate states may be materialised at the same
time, enforcing the memory budget.  On the first failure the remaining tasks
are cancelled and the earliest failure (by specification order) is reported.

Prechecks are pure functions over independent grid copies, so thread-safety
only requires that no candidate mutates a shared grid.
"""

from __future__ import annotations

import threading
from concurrent.futures import FIRST_EXCEPTION, ThreadPoolExecutor, wait
from dataclasses import dataclass

from .constraints import precheck
from .errors import E_CONSTRAINT, SolverError
from .grid import Grid


@dataclass
class PrecheckJob:
    order: int
    cid: str
    spec: object
    grid: Grid
    rooms: list[dict[str, int]]


class BoundedPrecheckPool:
    """Run precheck jobs with bounded worker and in-flight candidate counts."""

    def __init__(self, concurrency: int, memory_budget: int) -> None:
        if concurrency < 1 or memory_budget < concurrency:
            raise SolverError(E_CONSTRAINT, "invalid precheck pool sizing")
        self.concurrency = concurrency
        self.memory_budget = memory_budget
        self._slots = threading.BoundedSemaphore(memory_budget)

    def _run_job(self, job: PrecheckJob) -> None:
        with self._slots:
            precheck(job.spec, job.grid, list(job.rooms))

    def run(self, jobs: list[PrecheckJob]) -> None:
        """Execute all jobs; raise the first failure in specification order."""
        if not jobs:
            return
        ordered = sorted(jobs, key=lambda job: job.order)
        with ThreadPoolExecutor(max_workers=self.concurrency) as pool:
            futures = {pool.submit(self._run_job, job): job for job in ordered}
            done, _ = wait(futures, return_when=FIRST_EXCEPTION)
            failure = None
            for future in done:
                if future.exception() is not None:
                    failure = future
                    break
            if failure is not None:
                for future in futures:
                    future.cancel()
                pool.shutdown(wait=True, cancel_futures=True)
                cause = failure.exception()
                job = futures[failure]
                if isinstance(cause, SolverError):
                    raise cause
                raise SolverError(
                    E_CONSTRAINT,
                    f"precheck failed for '{job.cid}': {cause}",
                )
            # Drain the rest so a late failure cannot be silently swallowed.
            for future in futures:
                future.result()
