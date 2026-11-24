from __future__ import annotations

import sys
import time
import unittest
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from colpage_recovery.models import Budget
from colpage_recovery.scheduler import ScheduledTask, run_quota_schedule


@dataclass
class Result:
    patch_digest: str
    thread_name: str


class SchedulerQuotaTest(unittest.TestCase):
    def test_global_and_transaction_limits_hold_for_real_threads(self) -> None:
        budget = Budget(global_slots=2, per_tx_quota=1, work_delay_ms=0)
        tasks = [
            ScheduledTask(key=index, tx_id=f"T{(index % 3) + 1}", label=f"P{index}", byte_count=8)
            for index in range(6)
        ]

        def worker(task: ScheduledTask) -> Result:
            time.sleep(0.025)
            import threading

            return Result(patch_digest=f"d{task.key}", thread_name=threading.current_thread().name)

        result = run_quota_schedule(tasks, worker, budget)
        self.assertLessEqual(result.slot_peak, 2)
        self.assertGreaterEqual(result.slot_peak, 1)
        events = result.timeline_dicts()
        self.assertEqual(6, len(events))

        raw_events = result.timeline

        def max_overlap(filter_tx: str | None = None) -> int:
            points: list[tuple[float, int]] = []
            for event in raw_events:
                if filter_tx is not None and event.tx_id != filter_tx:
                    continue
                assert event.worker_started_ms is not None
                assert event.worker_ended_ms is not None
                points.append((event.worker_started_ms, 1))
                points.append((event.worker_ended_ms, -1))
            points.sort(key=lambda item: (item[0], item[1]))
            active = maximum = 0
            for _when, delta in points:
                active += delta
                maximum = max(maximum, active)
            return maximum

        self.assertLessEqual(max_overlap(), 2)
        for tx_id in {event.tx_id for event in raw_events}:
            self.assertLessEqual(max_overlap(tx_id), 1)

        for event in events:
            self.assertIsNotNone(event["wait_started_ms"])
            self.assertIsNotNone(event["wait_ended_ms"])
            self.assertGreaterEqual(event["wait_ms"], 0.0)
            self.assertTrue(event["thread_name"].startswith("colpage-replay"))


if __name__ == "__main__":
    unittest.main()
