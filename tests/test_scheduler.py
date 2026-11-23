from __future__ import annotations

import tempfile
import threading
import time
import unittest
from pathlib import Path

from imv.errors import BusinessError
from imv.lease import LeaseManager
from imv.logging_utils import StructuredLogger
from imv.scheduler import JobSpec, RefreshScheduler


class SchedulerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.logger = StructuredLogger(Path(self.temp.name) / "events.ndjson")
        self.lease_manager = LeaseManager()

    def tearDown(self) -> None:
        self.temp.cleanup()

    def test_global_and_per_view_quotas_use_real_threads(self) -> None:
        scheduler = RefreshScheduler(
            self.lease_manager,
            self.logger,
            global_limit=2,
            per_view_quotas={"a": 1, "b": 1, "c": 1},
        )
        active = 0
        max_active = 0
        per_view: dict[str, int] = {}
        per_view_max: dict[str, int] = {}
        workers: set[str] = set()
        lock = threading.Lock()

        def work(view_id: str) -> str:
            nonlocal active, max_active
            with lock:
                active += 1
                max_active = max(max_active, active)
                per_view[view_id] = per_view.get(view_id, 0) + 1
                per_view_max[view_id] = max(per_view_max.get(view_id, 0), per_view[view_id])
                workers.add(threading.current_thread().name)
            time.sleep(0.045)
            with lock:
                active -= 1
                per_view[view_id] -= 1
            return view_id

        try:
            handles = []
            for index, view_id in enumerate(["a", "a", "b", "b", "c"]):
                handles.append(
                    scheduler.submit(
                        JobSpec(
                            view_id=view_id,
                            kind="test",
                            param_version=1,
                            refresh_version=index + 1,
                            ttl_ms=5_000,
                            renew_ms=1_000,
                            fn=lambda view_id=view_id: work(view_id),
                        )
                    )
                )
            values, errors, pending = scheduler.wait_all(handles)
            self.assertEqual(errors, [])
            self.assertEqual(pending, [])
            self.assertEqual(values, ["a", "a", "b", "b", "c"])
            self.assertLessEqual(max_active, 2)
            self.assertEqual(max_active, 2)
            self.assertTrue(all(limit <= 1 for limit in per_view_max.values()))
            self.assertGreaterEqual(len(workers), 2)
            payload = scheduler.payload()
            self.assertLessEqual(payload["max_active_total_observed"], 2)
            self.assertTrue(all(value <= 1 for value in payload["max_active_by_view_observed"].values()))
            self.assertTrue(any(job["wait_ms"] and job["wait_ms"] > 0 for job in payload["recent_jobs"]))
            for trace in payload["recent_jobs"]:
                self.assertIsNotNone(trace["started_mono_ms"])
                self.assertIsNotNone(trace["finished_mono_ms"])
        finally:
            scheduler.shutdown()

    def test_long_job_renews_lease_and_failure_is_retryable(self) -> None:
        scheduler = RefreshScheduler(self.lease_manager, self.logger, global_limit=1, per_view_quotas={"x": 1})
        try:
            handle = scheduler.submit(
                JobSpec("x", "slow", 1, 1, ttl_ms=90, renew_ms=20, fn=lambda: time.sleep(0.3))
            )
            scheduler.wait(handle)
            lease = self.lease_manager.get("x")
            self.assertIsNotNone(lease)
            self.assertEqual(lease.state, "completed")
            self.assertGreater(lease.renew_count, 0)

            failed = scheduler.submit(
                JobSpec(
                    "x",
                    "failure",
                    1,
                    2,
                    ttl_ms=5_000,
                    renew_ms=1_000,
                    fn=lambda: (_ for _ in ()).throw(BusinessError("E_DATA", "injected row damage")),
                )
            )
            with self.assertRaises(BusinessError) as caught:
                scheduler.wait(failed)
            self.assertEqual(caught.exception.code, "E_DATA")
            retried = scheduler.submit(JobSpec("x", "retry", 1, 3, ttl_ms=5_000, renew_ms=1_000, fn=lambda: "ok"))
            self.assertEqual(scheduler.wait(retried), "ok")
            traces = {trace["job_id"]: trace for trace in scheduler.payload()["recent_jobs"]}
            self.assertEqual(traces[failed.job_id]["error_code"], "E_DATA")
            self.assertEqual(traces[retried.job_id]["status"], "succeeded")
        finally:
            scheduler.shutdown()


if __name__ == "__main__":
    unittest.main()
