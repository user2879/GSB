from __future__ import annotations

import threading
import time
import unittest
from types import MappingProxyType

from config_arbiter.scheduler import BoundedScheduler, SliceTask
from config_arbiter.spec import SliceSpec

from .util import session_spec


class SchedulerTests(unittest.TestCase):
    def test_global_and_per_environment_limits_use_worker_threads(self) -> None:
        scheduler = BoundedScheduler(
            session_spec(global_concurrency=2, per_environment_concurrency=1)
        )
        slice_spec = SliceSpec(version=1, values=MappingProxyType({"x": "1"}))
        tasks = [
            SliceTask(task_id=f"a-{index}", view_id="v", env_id="a", slice_spec=slice_spec)
            for index in range(4)
        ] + [
            SliceTask(task_id=f"b-{index}", view_id="v", env_id="b", slice_spec=slice_spec)
            for index in range(2)
        ]
        lock = threading.Lock()
        active_global = 0
        active_env: dict[str, int] = {}
        observed_global_peak = 0
        observed_env_peak: dict[str, int] = {}

        def execute(task: SliceTask) -> str:
            nonlocal active_global, observed_global_peak
            with lock:
                active_global += 1
                active_env[task.env_id] = active_env.get(task.env_id, 0) + 1
                observed_global_peak = max(observed_global_peak, active_global)
                observed_env_peak[task.env_id] = max(
                    observed_env_peak.get(task.env_id, 0), active_env[task.env_id]
                )
            time.sleep(0.03)
            with lock:
                active_global -= 1
                active_env[task.env_id] -= 1
            return task.task_id

        results = scheduler.run_batch(tasks, execute)
        self.assertEqual(len(results), len(tasks))
        self.assertTrue(all(result.error is None for result in results))
        self.assertLessEqual(observed_global_peak, 2)
        self.assertLessEqual(observed_env_peak["a"], 1)
        self.assertLessEqual(observed_env_peak["b"], 1)
        for result in results:
            self.assertTrue(result.metrics.worker_thread.startswith("arbiter-dispatch"))
            self.assertLessEqual(
                result.metrics.quota_wait_start_ms, result.metrics.quota_wait_end_ms
            )
            self.assertLessEqual(
                result.metrics.socket_started_ms, result.metrics.socket_ended_ms
            )
        self.assertGreater(max(result.metrics.quota_wait_ms for result in results), 0)
        snapshot = scheduler.gate.snapshot()
        self.assertLessEqual(snapshot["peak_global_in_flight"], 2)
        self.assertLessEqual(snapshot["peak_environment_in_flight"]["a"], 1)
        self.assertLessEqual(snapshot["peak_environment_in_flight"]["b"], 1)


if __name__ == "__main__":
    unittest.main()

