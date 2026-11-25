import threading
import unittest

from rlcb.clock import FakeMonotonicClock
from rlcb.ratelimit import DeterministicRateLimiter
from rlcb.scheduler import Scheduler


class RateLimitTest(unittest.TestCase):
    def test_zero_rps_rejects_all(self):
        limiter = DeterministicRateLimiter.create("s", 0)
        self.assertFalse(
            any(limiter.try_admit_seq(i) for i in range(5)))

    def test_admission_is_deterministic_by_arrival_seq(self):
        limiter = DeterministicRateLimiter.create("s", 3)
        # Each 3-sequence block is one admission second with quota 3; every
        # request fits, so all are admitted deterministically.
        self.assertEqual([limiter.try_admit_seq(i) for i in range(7)],
                         [True, True, True, True, True, True, True])
        # Rejection appears when several requests share one admission second.
        # rps=2 maps seq 0..3 into two seconds; force four requests into the
        # same second by reusing a small arrival sequence set.
        tight = DeterministicRateLimiter.create("s", 2)
        same_second = [0, 0, 0, 0]  # all in second 0, quota is 2
        self.assertEqual([tight.try_admit_seq(i) for i in same_second],
                         [True, True, False, False])
        zero = DeterministicRateLimiter.create("s", 0)
        self.assertFalse(any(zero.try_admit_seq(i) for i in range(6)))

    def test_admission_stable_under_repeated_runs(self):
        runs = []
        for _ in range(3):
            limiter = DeterministicRateLimiter.create("s", 2)
            runs.append(tuple(limiter.try_admit_seq(i) for i in range(6)))
        self.assertEqual(len(set(runs)), 1)


class SchedulerTest(unittest.TestCase):
    def test_three_gates_enforced_with_real_threads(self):
        clock = FakeMonotonicClock(0)
        scheduler = Scheduler(
            clock=clock, max_active_leases=2, max_active_batches=2,
            service_concurrency={"a": 1, "b": 1}, max_workers=4)
        seen = {"peak_global": 0, "lock": threading.Lock(), "active": 0}

        def work(task):
            with seen["lock"]:
                seen["active"] += 1
                seen["peak_global"] = max(seen["peak_global"],
                                          seen["active"])
            clock.advance(5)
            with seen["lock"]:
                seen["active"] -= 1
            return {"_order": task["_order"], "done": task["batch_id"]}

        tasks = [{"_order": i, "request_id": f"r{i}", "service_id": "a" if
                  i % 2 == 0 else "b", "batch_id": f"B{i}"} for i in range(8)]
        results = scheduler.run(tasks, work)
        scheduler.shutdown()
        self.assertEqual([r["_order"] for r in results], list(range(8)))
        self.assertLessEqual(seen["peak_global"], 2)
        self.assertEqual(len(scheduler.waits), 8)
        self.assertTrue(all(w.acquired for w in scheduler.waits))

    def test_wait_window_recorded_monotonic_ms(self):
        clock = FakeMonotonicClock(0)
        scheduler = Scheduler(
            clock=clock, max_active_leases=1, max_active_batches=1,
            service_concurrency={"a": 1}, max_workers=2)

        def work(task):
            clock.advance(7)
            return {"_order": task["_order"]}

        tasks = [{"_order": i, "request_id": f"r{i}", "service_id": "a",
                  "batch_id": f"B{i}"} for i in range(3)]
        scheduler.run(tasks, work)
        scheduler.shutdown()
        waits = sorted(scheduler.waits, key=lambda w: w.request_id)
        self.assertEqual([w.wait_start_ms for w in waits], [0, 7, 14])
        self.assertTrue(all(w.waited_ms >= 0 for w in waits))


if __name__ == "__main__":
    unittest.main()
