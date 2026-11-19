from __future__ import annotations

import threading
import time
import unittest

from meshdiag.scheduler import QuotaScheduler


class SchedulerTests(unittest.TestCase):
    def test_service_quota_and_global_budget_are_never_exceeded(self):
        active = {"A": 0, "B": 0}
        maximum = {"A": 0, "B": 0}
        lock = threading.Lock()

        def worker(item):
            service_id = item[0]
            with lock:
                active[service_id] += 1
                maximum[service_id] = max(maximum[service_id], active[service_id])
            time.sleep(0.03)
            with lock:
                active[service_id] -= 1
            return item

        items = [("A", str(index)) for index in range(3)] + [
            ("B", str(index)) for index in range(4)
        ]
        scheduler = QuotaScheduler(2, {"A": 1, "B": 2})
        results = scheduler.run(items, lambda item: item[0], worker)
        self.assertEqual(results, tuple(items))
        self.assertLessEqual(maximum["A"], 1)
        self.assertLessEqual(maximum["B"], 2)
        self.assertEqual(scheduler.stats.max_global_active, 2)
        self.assertEqual(scheduler.stats.max_service_active["A"], 1)
        self.assertEqual(scheduler.stats.max_service_active["B"], 2)
        self.assertGreaterEqual(scheduler.stats.queue_wait_ms.get("A", 0), 1)

    def test_one_slot_service_admits_in_request_order(self):
        starts = []
        lock = threading.Lock()

        def worker(item):
            with lock:
                starts.append(item)
            time.sleep(0.005)
            return item

        items = list(range(10))
        scheduler = QuotaScheduler(4, {"S": 1})
        scheduler.run(items, lambda item: "S", worker)
        self.assertEqual(starts, items)


if __name__ == "__main__":
    unittest.main()
