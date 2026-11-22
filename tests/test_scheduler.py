"""Dual-constraint scheduler tests (global budget + per-task quota)."""
import threading
import time
import unittest

from colval.scheduler import Scheduler, batch_mem_units


class TestScheduler(unittest.TestCase):
    def test_mem_units(self):
        self.assertEqual(batch_mem_units(1, 1), 1)
        self.assertEqual(batch_mem_units(64, 2), 1)
        self.assertEqual(batch_mem_units(128, 2), 2)

    def test_budget_and_quota_respected(self):
        quotas = {"ta": 2, "tb": 1}
        sched = Scheduler(global_budget_units=3, quotas=quotas)
        running_units = []
        running_task = {"ta": [], "tb": []}
        stop = threading.Event()

        def observer():
            while not stop.is_set():
                with sched._cond:
                    running_units.append(sched._running_units)
                    for t in quotas:
                        running_task[t].append(sched._running_per_task[t])
                time.sleep(0.001)

        obs = threading.Thread(target=observer)
        obs.start()

        def work():
            time.sleep(0.02)

        threads = []
        for i in range(6):
            for tid in ("ta", "tb"):
                threads.append(threading.Thread(
                    target=sched.run_batch, args=(tid, i, 1, work)))
        for th in threads:
            th.start()
        for th in threads:
            th.join()
        stop.set()
        obs.join()

        self.assertTrue(running_units)
        self.assertLessEqual(max(running_units), 3)
        self.assertLessEqual(max(running_task["ta"]), 2)
        self.assertLessEqual(max(running_task["tb"]), 1)
        for tm in sched.timings:
            self.assertIsNotNone(tm.start_ms)
            self.assertIsNotNone(tm.end_ms)
            self.assertGreaterEqual(tm.end_ms, tm.start_ms)
            self.assertGreaterEqual(tm.queue_wait_ms, 0)

    def test_oversized_batch_rejected(self):
        sched = Scheduler(global_budget_units=2, quotas={"t": 1})
        with self.assertRaises(ValueError):
            sched.acquire("t", 0, 3)


if __name__ == "__main__":
    unittest.main()
