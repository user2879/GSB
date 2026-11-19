"""Deterministic scheduler honoring concurrency and in-flight buffer budgets.

Admission order is fixed (device id, then block index). A block reserves its
command count against the global in-flight budget before synthesis and frees
the reservation once its frame block is durably persisted. When the budget is
exhausted, higher device ids wait — quantization, durations and commands are
never altered to make room.
"""

from __future__ import annotations

import threading
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import dataclass
from typing import Callable


@dataclass(frozen=True)
class ScheduledBlock:
    device_id: str
    block_index: int
    command_count: int


class InterruptRequested(Exception):
    """Raised at a frame-block boundary after a handled termination signal."""


class Scheduler:
    def __init__(
        self,
        device_ids: list[str],
        plans: dict[str, list[ScheduledBlock]],
        concurrency: int,
        inflight_limit: int,
        signal_flag: Callable[[], bool],
    ) -> None:
        self.device_ids = sorted(device_ids)
        self.plans = plans
        self.concurrency = max(1, concurrency)
        self.inflight_limit = inflight_limit
        self.signal_flag = signal_flag
        self._lock = threading.Lock()
        self.inflight_usage = 0
        self.peak_inflight = 0
        self.waited_events = 0

    def run_serial(self, worker: Callable[[ScheduledBlock], None]) -> None:
        """Concurrency = 1 reference path, sharing the same admission rules."""
        cursors = {device_id: 0 for device_id in self.device_ids}
        remaining = sum(len(plan) for plan in self.plans.values())
        while remaining:
            candidate = self._next_admissible(cursors)
            if candidate is None:
                break
            if self.signal_flag():
                raise InterruptRequested()
            self._reserve(candidate)
            worker(candidate)
            self._release(candidate)
            cursors[candidate.device_id] += 1
            remaining -= 1

    def run_parallel(self, worker: Callable[[ScheduledBlock], None]) -> None:
        cursors = {device_id: 0 for device_id in self.device_ids}
        remaining = sum(len(plan) for plan in self.plans.values())
        active: dict[Future, ScheduledBlock] = {}
        with ThreadPoolExecutor(max_workers=self.concurrency) as pool:
            while remaining:
                candidate = self._next_admissible(cursors)
                if candidate is not None and not self.signal_flag():
                    self._reserve(candidate)
                    future = pool.submit(worker, candidate)
                    active[future] = candidate
                    cursors[candidate.device_id] += 1
                    continue
                if not active:
                    if self.signal_flag():
                        raise InterruptRequested()
                    break
                done, _ = wait(active.keys(), return_when=FIRST_COMPLETED)
                for future in done:
                    block = active.pop(future)
                    future.result()
                    self._release(block)
                    remaining -= 1

    def _next_admissible(self, cursors: dict[str, int]) -> ScheduledBlock | None:
        """Deterministically choose the next block allowed to start.

        Device ids are probed in fixed sorted order; the first device whose next
        block fits the remaining budget wins, so budget starvation makes later
        devices wait rather than reordering commands.
        """
        with self._lock:
            for device_id in self.device_ids:
                cursor = cursors[device_id]
                plan = self.plans[device_id]
                if cursor >= len(plan):
                    continue
                candidate = plan[cursor]
                if self.inflight_usage + candidate.command_count <= self.inflight_limit:
                    return candidate
            self.waited_events += 1
            return None

    def _reserve(self, block: ScheduledBlock) -> None:
        with self._lock:
            self.inflight_usage += block.command_count
            if self.inflight_usage > self.inflight_limit:
                raise AssertionError("in-flight budget exceeded")
            self.peak_inflight = max(self.peak_inflight, self.inflight_usage)

    def _release(self, block: ScheduledBlock) -> None:
        with self._lock:
            self.inflight_usage -= block.command_count
            if self.inflight_usage < 0:
                raise AssertionError("in-flight accounting underflow")
