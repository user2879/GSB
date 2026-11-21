from __future__ import annotations

import threading
import time
import unittest

from alert_pipeline.errors import PipelineError
from alert_pipeline.scheduler import ScheduledTask, run_tasks


class SchedulerTests(unittest.TestCase):
    def test_dual_quota_bounds_and_queue_timings(self) -> None:
        lock = threading.Lock()
        active = 0
        active_by_service: dict[str, int] = {}
        max_active = 0
        max_by_service: dict[str, int] = {}

        def make_task(service: str, value: int):
            def work():
                nonlocal active, max_active
                with lock:
                    active += 1
                    active_by_service[service] = active_by_service.get(service, 0) + 1
                    max_active = max(max_active, active)
                    max_by_service[service] = max(
                        max_by_service.get(service, 0), active_by_service[service]
                    )
                try:
                    time.sleep(0.02)
                    return value
                finally:
                    with lock:
                        active -= 1
                        active_by_service[service] -= 1

            return work

        tasks = [
            ScheduledTask(
                ordinal=index,
                task_id=f"t{index}",
                stage="normalize",
                service_id=("a", "b", "c")[index % 3],
                function=make_task(("a", "b", "c")[index % 3], index),
            )
            for index in range(12)
        ]
        result = run_tasks(tasks, mode="parallel", global_limit=3, per_service_limit=1)
        self.assertIsNone(result.error)
        self.assertEqual(result.values, tuple(range(12)))
        self.assertLessEqual(max_active, 3)
        self.assertLessEqual(max(max_by_service.values()), 1)
        self.assertGreaterEqual(max_active, 2)
        self.assertLessEqual(result.report.peak_global, 3)
        self.assertTrue(all(value <= 1 for value in result.report.peak_by_service.values()))
        self.assertEqual([item.ordinal for item in result.report.timeline], list(range(12)))
        self.assertTrue(all(item.wait_ms >= 0 and item.run_ms >= 0 for item in result.report.timeline))

    def test_per_service_two_allows_parallelism_within_service(self) -> None:
        tasks = [
            ScheduledTask(
                ordinal=index,
                task_id=f"t{index}",
                stage="correlate",
                service_id="same",
                function=lambda value=index: (time.sleep(0.01), value)[1],
            )
            for index in range(6)
        ]
        result = run_tasks(tasks, mode="parallel", global_limit=3, per_service_limit=2)
        self.assertEqual(result.values, tuple(range(6)))
        self.assertLessEqual(result.report.peak_by_service["same"], 2)
        self.assertGreaterEqual(result.report.peak_by_service["same"], 2)

    def test_serial_and_parallel_have_same_value_order(self) -> None:
        tasks = [
            ScheduledTask(
                ordinal=index,
                task_id=f"t{index}",
                stage="deduplicate",
                service_id=("a", "b")[index % 2],
                function=lambda value=index: value * value,
            )
            for index in range(10)
        ]
        parallel = run_tasks(tasks, mode="parallel", global_limit=4, per_service_limit=2)
        serial = run_tasks(tasks, mode="serial", global_limit=4, per_service_limit=2)
        self.assertEqual(parallel.values, serial.values)

    def test_business_code_from_task_error(self) -> None:
        def fail():
            raise PipelineError("E_FIELD", "bad field", line=3, field="fingerprint")

        result = run_tasks(
            [ScheduledTask(0, "t0", "normalize", "a", fail)],
            mode="parallel",
            global_limit=2,
            per_service_limit=1,
        )
        self.assertIsInstance(result.error, PipelineError)
        self.assertEqual(result.report.business_code, "E_FIELD")
        self.assertEqual(result.report.timeline[0].outcome, "E_FIELD")


if __name__ == "__main__":
    unittest.main()
