"""Batch parallel scheduler under dual constraints:
  * global memory-unit budget: sum of memory units of RUNNING batches
    must never exceed the global budget;
  * per-task concurrency quota: running batches per task <= its quota.

Batches whose quota/budget is exhausted wait in a queue; wait start/end and
start/end are recorded as monotonic-clock milliseconds. Every batch reports
its memory-unit footprint (peak), start/end times and queue wait.
"""

import threading
import time

MEM_UNIT_BYTES = 1024


def monotonic_ms():
    return int(time.monotonic() * 1000)


def batch_mem_units(rows, n_columns):
    """Deterministic memory footprint estimate, in 1 KiB units."""
    need = rows * max(1, n_columns) * 8
    return max(1, -(-need // MEM_UNIT_BYTES))


class BatchTiming:
    def __init__(self, task_id, batch_index, mem_units):
        self.task_id = task_id
        self.batch_index = batch_index
        self.mem_units = mem_units
        self.queued_ms = monotonic_ms()
        self.start_ms = None
        self.end_ms = None

    @property
    def queue_wait_ms(self):
        if self.start_ms is None:
            return None
        return self.start_ms - self.queued_ms

    def as_dict(self):
        return {
            "task_id": self.task_id,
            "batch_index": self.batch_index,
            "mem_units": self.mem_units,
            "peak_mem_bytes": self.mem_units * MEM_UNIT_BYTES,
            "queued_ms": self.queued_ms,
            "start_ms": self.start_ms,
            "end_ms": self.end_ms,
            "queue_wait_ms": self.queue_wait_ms,
        }


class Scheduler:
    def __init__(self, global_budget_units, quotas):
        if global_budget_units <= 0:
            raise ValueError("global budget must be positive")
        self.global_budget = global_budget_units
        self.quotas = dict(quotas)
        self._cond = threading.Condition()
        self._running_units = 0
        self._running_per_task = {t: 0 for t in self.quotas}
        self.timings = []
        self.max_running_units = 0
        self.max_running_per_task = {t: 0 for t in self.quotas}

    def acquire(self, task_id, batch_index, mem_units):
        if mem_units > self.global_budget:
            raise ValueError(
                f"batch needs {mem_units} units > global budget {self.global_budget}")
        timing = BatchTiming(task_id, batch_index, mem_units)
        with self._cond:
            while (self._running_units + mem_units > self.global_budget
                   or self._running_per_task[task_id] >= self.quotas[task_id]):
                self._cond.wait()
            self._running_units += mem_units
            self._running_per_task[task_id] += 1
            self.max_running_units = max(self.max_running_units, self._running_units)
            self.max_running_per_task[task_id] = max(
                self.max_running_per_task[task_id], self._running_per_task[task_id])
            timing.start_ms = monotonic_ms()
            self.timings.append(timing)
        return timing

    def release(self, timing):
        with self._cond:
            timing.end_ms = monotonic_ms()
            self._running_units -= timing.mem_units
            self._running_per_task[timing.task_id] -= 1
            self._cond.notify_all()

    def run_batch(self, task_id, batch_index, mem_units, fn):
        """Acquire a slot, run fn(), release; return (fn_result, timing)."""
        timing = self.acquire(task_id, batch_index, mem_units)
        try:
            result = fn()
        finally:
            self.release(timing)
        return result, timing
