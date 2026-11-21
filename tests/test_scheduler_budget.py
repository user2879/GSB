from __future__ import annotations

import random
import time

from layerpack.scheduler import PointBudgetScheduler, ScheduledTask


def _tasks(points: list[int]):
    tasks = []
    for index, points_value in enumerate(points):
        def action(value=index):
            time.sleep((5 - value % 5) * 0.002)
            return value

        tasks.append(
            ScheduledTask(
                layer=f"layer-{chr(97 + index)}",
                stage="quantization",
                points=points_value,
                action=action,
            )
        )
    random.shuffle(tasks)
    return tasks


def test_concurrency_does_not_change_order_or_budget_deductions():
    serial = PointBudgetScheduler(
        max_workers=1, point_budget_per_batch=100
    ).run(_tasks([20, 30, 40]))
    concurrent = PointBudgetScheduler(
        max_workers=4, point_budget_per_batch=100
    ).run(_tasks([20, 30, 40]))
    assert [key for key, _ in serial.results] == [key for key, _ in concurrent.results]
    assert [value for _, value in serial.results] == [value for _, value in concurrent.results]
    assert [event.to_dict() for event in serial.events] == [
        event.to_dict() for event in concurrent.events
    ]
    assert concurrent.max_observed_concurrency <= 4


def test_point_budget_starvation_keeps_stable_pending_queue():
    result = PointBudgetScheduler(
        max_workers=3, point_budget_per_batch=50
    ).run(_tasks([60, 10, 20]))
    assert not result.results
    assert [item.layer for item in result.pending] == [
        "layer-a",
        "layer-b",
        "layer-c",
    ]
    assert result.pending[0].reason == "task_exceeds_batch_budget"


def test_budget_resets_across_batches_in_serial_order():
    result = PointBudgetScheduler(
        max_workers=3, point_budget_per_batch=100
    ).run(_tasks([40, 70, 30]))
    assert [point for _, point in result.results] == [0, 1, 2]
    assert [event.points for event in result.events] == [40, 70, 30]
    assert result.batch_count == 2


def test_task_equal_to_batch_budget_is_scheduled():
    result = PointBudgetScheduler(
        max_workers=2, point_budget_per_batch=8
    ).run(_tasks([8]))
    assert [point for _, point in result.results] == [0]
    assert not result.pending
    assert result.events[0].points == 8


def test_pending_reasons_distinguish_head_from_blocked_tail():
    result = PointBudgetScheduler(
        max_workers=2, point_budget_per_batch=8
    ).run(_tasks([9, 8]))
    assert [item.layer for item in result.pending] == ["layer-a", "layer-b"]
    assert [item.reason for item in result.pending] == [
        "task_exceeds_batch_budget",
        "point_budget_exhausted",
    ]
