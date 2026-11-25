"""Budget/quota constrained parallel recompute scheduler (requirement 7).

Tasks are (stage, window) units with a point cost equal to the window length.
At any instant the sum of in-flight points must not exceed the global budget,
and per-stage in-flight points must not exceed that stage's quota. Tasks that
cannot be admitted queue; queue wait start/end are measured with a monotonic
clock in milliseconds. Work is genuinely dispatched to a worker thread pool.

Dispatch order is deterministic: (stage_order, window_order). The report
carries the effective quotas for the round, the actual peak occupancy and
per-task queue durations.
"""

from __future__ import annotations

import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field


def _now_ms():
    return time.monotonic_ns() // 1_000_000


@dataclass(frozen=True)
class Task:
    stage_id: str
    stage_order: int
    window_id: str
    window_order: int
    points: int
    payload: object = None  # opaque, passed through to the worker


@dataclass
class TaskTiming:
    stage_id: str
    window_id: str
    points: int
    enqueue_ms: int
    wait_start_ms: int | None = None
    wait_end_ms: int | None = None
    run_start_ms: int | None = None
    run_end_ms: int | None = None

    @property
    def queue_wait_ms(self):
        if self.wait_start_ms is None or self.wait_end_ms is None:
            return 0
        return self.wait_end_ms - self.wait_start_ms


@dataclass
class ScheduleReport:
    global_budget: int
    stage_quotas: dict
    peak_global_points: int = 0
    peak_stage_points: dict = field(default_factory=dict)
    timings: list = field(default_factory=list)
    results: dict = field(default_factory=dict)  # (stage_id, window_id) -> worker result

    @property
    def total_queue_wait_ms(self):
        return sum(t.queue_wait_ms for t in self.timings)


class Scheduler:
    def __init__(self, global_budget, stage_quotas, workers=4):
        self.global_budget = global_budget
        self.stage_quotas = dict(stage_quotas)
        self.workers = workers

    def run(self, tasks, worker_fn):
        """Run tasks under budget/quota constraints; worker_fn(task) -> result."""
        ordered = sorted(tasks, key=lambda t: (t.stage_order, t.window_order))
        for task in ordered:
            quota = self.stage_quotas.get(task.stage_id, self.global_budget)
            if task.points > self.global_budget or task.points > quota:
                raise ValueError(
                    "task (%s, %s) needs %d points, can never be admitted"
                    " (budget=%d, quota=%d)"
                    % (task.stage_id, task.window_id, task.points,
                       self.global_budget, quota)
                )
        condition = threading.Condition()
        in_flight_global = [0]
        in_flight_stage = {sid: 0 for sid in self.stage_quotas}
        peaks = {"global": 0, "stage": {sid: 0 for sid in self.stage_quotas}}
        timings = []
        report = ScheduleReport(self.global_budget, dict(self.stage_quotas))

        def release(task):
            with condition:
                in_flight_global[0] -= task.points
                in_flight_stage[task.stage_id] -= task.points
                condition.notify_all()

        def execute(task, timing):
            timing.run_start_ms = _now_ms()
            try:
                result = worker_fn(task)
                report.results[(task.stage_id, task.window_id)] = result
            finally:
                timing.run_end_ms = _now_ms()
                release(task)

        with ThreadPoolExecutor(max_workers=self.workers) as pool:
            futures = []
            for task in ordered:
                timing = TaskTiming(task.stage_id, task.window_id, task.points, _now_ms())
                timings.append(timing)
                quota = self.stage_quotas.get(task.stage_id, self.global_budget)
                waiting = False
                with condition:
                    while (
                        in_flight_global[0] + task.points > self.global_budget
                        or in_flight_stage[task.stage_id] + task.points > quota
                    ):
                        if not waiting:
                            timing.wait_start_ms = _now_ms()
                            waiting = True
                        condition.wait()
                    if waiting:
                        timing.wait_end_ms = _now_ms()
                    in_flight_global[0] += task.points
                    in_flight_stage[task.stage_id] += task.points
                    peaks["global"] = max(peaks["global"], in_flight_global[0])
                    peaks["stage"][task.stage_id] = max(
                        peaks["stage"][task.stage_id], in_flight_stage[task.stage_id]
                    )
                futures.append(pool.submit(execute, task, timing))
            for future in futures:
                future.result()

        report.peak_global_points = peaks["global"]
        report.peak_stage_points = peaks["stage"]
        report.timings = timings
        return report
