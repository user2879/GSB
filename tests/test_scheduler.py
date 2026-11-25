import threading
import time
import unittest

from arbplatform.scheduler import QuotaScheduler
from tests.test_engine import manifest


class SchedulerTest(unittest.TestCase):
    def test_global_and_per_env_quotas_never_exceeded(self):
        m = manifest("latest_revision")
        scheduler = QuotaScheduler(m)
        lock = threading.Lock()
        inflight = {"total": 0}
        env_inflight: dict[str, int] = {}
        peaks = {"total": 0, "env": {}}

        def work(env_id):
            with lock:
                inflight["total"] += 1
                env_inflight[env_id] = env_inflight.get(env_id, 0) + 1
                peaks["total"] = max(peaks["total"], inflight["total"])
                peaks["env"][env_id] = max(
                    peaks["env"].get(env_id, 0), env_inflight[env_id])
            # Real contention window; not used as a timeout or quota mechanism.
            time.sleep(0.02)
            with lock:
                inflight["total"] -= 1
                env_inflight[env_id] -= 1
            return env_id

        tasks = []
        for i in range(24):
            env_id = m.environments[i % len(m.environments)].env_id
            tasks.append((f"t{i:02d}", env_id, env_id,
                          lambda eid=env_id: work(eid)))
        records = scheduler.run(tasks)

        self.assertEqual(len(records), 24)
        self.assertLessEqual(peaks["total"], m.global_concurrency)
        for env in m.environments:
            self.assertLessEqual(
                peaks["env"][env.env_id], env.lease_capacity)
        self.assertIsNone(records[0].error)
        self.assertIsNotNone(records[0].wait_start_ms)
        self.assertIsNotNone(records[0].wait_end_ms)
        self.assertGreaterEqual(records[0].wait_ms, 0)

    def test_records_are_stable_ordered(self):
        m = manifest("latest_revision")
        records = QuotaScheduler(m).run([
            (f"t{i}", "e1", "e1", lambda i=i: i) for i in range(5)
        ])
        self.assertEqual([r.task_id for r in records],
                         [f"t{i}" for i in range(5)])


if __name__ == "__main__":
    unittest.main()
