from __future__ import annotations

import threading
import time
import unittest

from lateness_ledger.probe import LoopbackSourceSubprocess
from lateness_ledger.scheduler import BoundedFinalizationScheduler, ScheduledTask


class SchedulerTests(unittest.TestCase):
    def test_dual_quota_and_real_wait_accounting(self) -> None:
        scheduler = BoundedFinalizationScheduler(
            global_limit=2,
            per_partition_limit=1,
            task_timeout_ms=1000,
            retries=0,
        )
        active = 0
        active_by_partition: dict[str, int] = {}
        maximum = 0
        maximum_by_partition = 0
        lock = threading.Lock()

        def make_task(index: int, partition: str):
            def task():
                nonlocal active, maximum, maximum_by_partition
                with lock:
                    active += 1
                    active_by_partition[partition] = active_by_partition.get(partition, 0) + 1
                    maximum = max(maximum, active)
                    maximum_by_partition = max(maximum_by_partition, active_by_partition[partition])
                time.sleep(0.025)
                with lock:
                    active -= 1
                    active_by_partition[partition] -= 1
                return index

            return task

        tasks = [
            ScheduledTask(
                task_id=f"t{index}",
                partition_key="p0" if index < 4 else "p1",
                function=make_task(index, "p0" if index < 4 else "p1"),
            )
            for index in range(8)
        ]
        outcomes, stats = scheduler.run(tasks)
        self.assertEqual([outcome.value for outcome in outcomes], list(range(8)))
        self.assertEqual(maximum, 2)
        self.assertEqual(maximum_by_partition, 1)
        self.assertEqual(stats.max_global_active, 2)
        self.assertEqual(stats.max_partition_active, 1)
        self.assertGreater(stats.max_wait_ms, 0)
        self.assertTrue(all(outcome.wait_ms >= 0 for outcome in outcomes))

    def test_real_timeout_is_retried_with_monotonic_durations(self) -> None:
        scheduler = BoundedFinalizationScheduler(
            global_limit=1,
            per_partition_limit=1,
            task_timeout_ms=10,
            retries=1,
        )
        calls = {"count": 0}

        def slow_task():
            calls["count"] += 1
            time.sleep(0.03)
            return "late"

        outcomes, stats = scheduler.run(
            [ScheduledTask(task_id="timeout", partition_key="p0", function=slow_task)]
        )
        self.assertEqual(calls["count"], 2)
        self.assertEqual(outcomes[0].attempts, 2)
        self.assertTrue(outcomes[0].timed_out)
        self.assertEqual(stats.total_timeout_attempts, 2)
        self.assertEqual(stats.retries, 1)
        self.assertTrue(all(duration >= 20 for duration in outcomes[0].attempt_durations_ms))


class ProbeTests(unittest.TestCase):
    def test_loopback_sampling_success(self) -> None:
        with LoopbackSourceSubprocess(server_delay_ms=0) as source:
            result = source.probe(timeout_ms=1000, attempts=1, backoff_ms=0)
        self.assertTrue(result.ok)
        self.assertEqual(result.attempts[0].response, "OK")
        self.assertFalse(result.timed_out)

    def test_real_socket_timeout_and_retry_are_measured(self) -> None:
        with LoopbackSourceSubprocess(server_delay_ms=250) as source:
            started = time.monotonic()
            result = source.probe(timeout_ms=30, attempts=2, backoff_ms=5)
            elapsed = time.monotonic() - started
        self.assertFalse(result.ok)
        self.assertTrue(result.timed_out)
        self.assertEqual(result.retries, 1)
        self.assertEqual(len(result.attempts), 2)
        self.assertTrue(all(attempt.timed_out for attempt in result.attempts))
        self.assertGreaterEqual(result.total_elapsed_ms, 60)
        self.assertGreater(elapsed, 0.055)


if __name__ == "__main__":
    unittest.main()
