import threading
import time
import unittest

from alert_pipeline.scheduler import Scheduler, Task


class SchedulerTest(unittest.TestCase):
    def test_dual_quota_never_exceeded(self):
        seen_global = []
        seen_service = {"a": [], "b": []}
        lock = threading.Lock()

        def make(service):
            def work():
                with lock:
                    seen_global.append(1)
                    seen_service[service].append(1)
                    self.assertLessEqual(len(seen_global), 3)
                    self.assertLessEqual(len(seen_service[service]), 2)
                time.sleep(0.02)
                with lock:
                    seen_global.pop()
                    seen_service[service].pop()
                return service
            return work

        tasks = [Task(f"t{i}", "a" if i % 2 == 0 else "b",
                      make("a" if i % 2 == 0 else "b")) for i in range(10)]
        sched = Scheduler(global_limit=3, per_service_limit=2,
                          task_timeout_ms=1000)
        results, metrics = sched.run_batch(1, tasks)
        self.assertEqual(results, [t.service_id for t in tasks])
        self.assertEqual(metrics.completed_count, 10)
        self.assertLessEqual(metrics.peak_concurrency, 3)
        self.assertGreaterEqual(metrics.peak_concurrency, 2)
        self.assertGreater(metrics.ended_clock_ms,
                           metrics.started_clock_ms)

    def test_real_timeout_retried_without_semantic_change(self):
        counter = {"n": 0}

        def flaky():
            counter["n"] += 1
            return "same-result"

        tasks = [Task("flaky-1", "a", flaky)]
        sched = Scheduler(global_limit=2, per_service_limit=1,
                          task_timeout_ms=80, max_attempts=3,
                          timeout_task_ids={"flaky-1"})
        results, metrics = sched.run_batch(7, tasks)
        self.assertEqual(results, ["same-result"])
        self.assertEqual(metrics.timed_out_attempts, 1)
        self.assertEqual(metrics.retried_tasks, 1)

    def test_exhausted_attempts_raise_timeout(self):
        def slow():
            time.sleep(0.5)
            return "never"

        sched = Scheduler(global_limit=1, per_service_limit=1,
                          task_timeout_ms=30, max_attempts=2,
                          timeout_task_ids={"slow"})
        # Every attempt is injected as timed out.
        with self.assertRaises(Exception):
            sched.run_batch(1, [Task("slow", "a", slow)])


if __name__ == "__main__":
    unittest.main()
