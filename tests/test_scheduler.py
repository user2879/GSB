import os

from tests._base import BaseCase
from vdu.ledger import Ledger
from vdu.scheduler import BatchScheduler


class SchedulerTests(BaseCase):
    def setUp(self):
        super().setUp()
        self.ledger = Ledger(os.path.join(self.root, "ledger.json"))
        self.config["scheduler"]["batch_budget"] = 6
        self.config["scheduler"]["segment_quota"] = 5
        self.scheduler = BatchScheduler(self.config, self.ledger)
        self.meta = {seg["segment_id"]: seg
                     for seg in self.transcript["segments"]}

    def test_quota_exceeded_marker(self):
        points = {seg["segment_id"]: len(seg["words"])
                  for seg in self.transcript["segments"]}
        # Each segment has 4 words; inflate seg02 above the quota.
        points["seg02"] = 9
        self.scheduler.enqueue(list(points), points)
        runnable, report = self.scheduler.next_batch(self.meta)
        self.assertIn("seg02", report["quota_blocked"])
        self.assertNotIn("seg02", runnable)
        errors = self.scheduler.quota_errors()
        self.assertTrue(any(e.code == "E_EVIDENCE_QUOTA_EXCEEDED"
                            for e in errors))

    def test_budget_pending_retained_and_prioritized(self):
        points = {seg_id: 4 for seg_id in self.meta}
        self.scheduler.enqueue(list(points), points)
        first, report_one = self.scheduler.next_batch(self.meta)
        # Budget 6 only covers one 4-word segment; rest stay pending.
        self.assertEqual(len(first), 1)
        self.assertTrue(report_one["budget_pending"])
        self.scheduler.mark_done(first)
        second, report_two = self.scheduler.next_batch(self.meta)
        self.assertEqual(len(second), 1)
        self.assertEqual(second[0], report_one["budget_pending"][0])
        pending = self.scheduler.budget_errors()
        self.assertTrue(any(e.code == "E_EVIDENCE_BUDGET_PENDING"
                            for e in pending))

    def test_stable_ordering_by_start_and_speaker(self):
        points = {seg_id: 1 for seg_id in self.meta}
        self.scheduler.enqueue(list(points), points)
        runnable, _ = self.scheduler.next_batch(self.meta)
        starts = [self.meta[seg_id]["start"] for seg_id in runnable]
        self.assertEqual(starts, sorted(starts))
