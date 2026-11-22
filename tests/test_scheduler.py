import threading
import time
import unittest

from distlock.clock import FakeClock
from distlock.errors import ExhaustError
from distlock.scheduler import BoundedScheduler


class SchedulerTests(unittest.TestCase):
    def test_global_and_per_holder_limits_never_exceeded(self):
        clock = FakeClock()
        sched = BoundedScheduler(global_limit=3, per_holder_limit=2,
                                 backoff_ms=[], clock=clock)
        peak = {"total": 0, "h1": 0}
        lock = threading.Lock()
        current = {"total": 0, "h1": 0}

        def call(holder):
            with lock:
                current["total"] += 1
                current[holder] = current.get(holder, 0) + 1
                peak["total"] = max(peak["total"], current["total"])
                peak["h1"] = max(peak["h1"], current["h1"])
            time.sleep(0.02)
            with lock:
                current["total"] -= 1
                current[holder] -= 1
            return {"seq": 0, "verdict": "OK"}

        events = [("h1", i) for i in range(6)] + [("h2", i) for i in range(4)]
        threads = [
            threading.Thread(target=sched.submit,
                             args=(i, h, "d", lambda h=h: call(h)))
            for i, (h, _) in enumerate(events, start=1)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        self.assertLessEqual(peak["total"], 3)
        self.assertLessEqual(peak["h1"], 2)
        sched.shutdown()

    def test_backoff_retry_then_exhaust(self):
        clock = FakeClock()
        sched = BoundedScheduler(global_limit=2, per_holder_limit=2,
                                 backoff_ms=[5, 10, 20], clock=clock)
        attempts = {"n": 0}

        def always_fail():
            attempts["n"] += 1
            raise ConnectionError("simulated transport reset")

        with self.assertRaises(ExhaustError) as ctx:
            sched.submit(1, "h1", "d", always_fail)
        self.assertEqual(ctx.exception.code, "E_EXHAUST")
        # 1 initial attempt + 3 retries = 4 transmissions, same semantics.
        self.assertEqual(attempts["n"], 4)
        self.assertEqual(clock.sleeps, [5, 10, 20])
        sched.shutdown()

    def test_successful_call_records_quota_wait(self):
        clock = FakeClock()
        sched = BoundedScheduler(global_limit=1, per_holder_limit=1,
                                 backoff_ms=[], clock=clock)
        result = sched.submit(1, "h1", "d", lambda: {"verdict": "GRANTED"})
        self.assertTrue(result.ok)
        self.assertEqual(len(result.attempts), 1)
        self.assertGreaterEqual(result.total_wait_quota_ms, 0)
        sched.shutdown()


if __name__ == "__main__":
    unittest.main()
