"""Requirement 7: global budget + per-stage quota scheduling on a real worker
pool, with monotonic-ms queue waits and peak occupancy reporting."""

import time

from obs_pipeline.scheduler import Scheduler, Task


def _tasks():
    return [
        Task("s1", 0, "w0", 0, 30),
        Task("s1", 0, "w1", 1, 30),
        Task("s1", 0, "w2", 2, 30),
        Task("s2", 1, "w0", 0, 40),
        Task("s2", 1, "w1", 1, 40),
    ]


def _worker(task):
    time.sleep(0.02)
    return task.points * 2


def test_budget_and_quota_never_exceeded():
    scheduler = Scheduler(global_budget=50, stage_quotas={"s1": 30, "s2": 40}, workers=4)
    report = scheduler.run(_tasks(), _worker)
    assert report.peak_global_points <= 50
    assert report.peak_stage_points["s1"] <= 30
    assert report.peak_stage_points["s2"] <= 40
    assert report.peak_global_points > 0
    assert set(report.results) == {("s1", "w0"), ("s1", "w1"), ("s1", "w2"),
                                   ("s2", "w0"), ("s2", "w1")}


def test_queue_wait_recorded_in_monotonic_ms():
    scheduler = Scheduler(global_budget=40, stage_quotas={"s1": 30, "s2": 40}, workers=2)
    report = scheduler.run(_tasks(), _worker)
    waits = [t for t in report.timings if t.queue_wait_ms > 0]
    assert waits, "tight budget must force queuing"
    for timing in report.timings:
        if timing.wait_start_ms is not None:
            assert timing.wait_end_ms >= timing.wait_start_ms
        assert timing.run_end_ms >= timing.run_start_ms
    assert report.total_queue_wait_ms == sum(t.queue_wait_ms for t in report.timings)


def test_report_carries_effective_quotas():
    scheduler = Scheduler(global_budget=1000, stage_quotas={"s1": 60, "s2": 60}, workers=4)
    report = scheduler.run(_tasks(), _worker)
    assert report.global_budget == 1000
    assert report.stage_quotas == {"s1": 60, "s2": 60}
    assert report.total_queue_wait_ms >= 0
