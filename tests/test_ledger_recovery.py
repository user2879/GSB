import os
import tempfile
import unittest

from batch_receipt_ledger.ledger import Ledger
from batch_receipt_ledger.plan import build_plan, plan_fingerprint, stable_hash
from batch_receipt_ledger.recovery import recover
from batch_receipt_ledger.spec import parse_spec

from tests._support import make_spec


def _batch_result(plan, seq):
    return {
        "batch_id": plan.batch_id,
        "batch_seq": plan.batch_seq,
        "worker_id": plan.worker_id,
        "request_ids": list(plan.request_ids),
        "arrival_seqs": list(plan.arrival_seqs),
        "param_version": 1,
        "elapsed_ms": seq,
        "results": [
            {"request_id": rid, "output_digest": f"d-{rid}-{seq}",
             "output": seq}
            for rid in plan.request_ids],
    }


class LedgerTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.db_path = os.path.join(self.tmp.name, "ledger.sqlite3")
        self.raw = make_spec(deliveries=6, batch_size=2)
        self.spec = parse_spec(self.raw)
        self.plans = build_plan(self.spec)

    def tearDown(self):
        self.tmp.cleanup()

    def _receipt_records(self, plan, result):
        from batch_receipt_ledger.service import make_receipt_records
        records, _ = make_receipt_records(self.spec, plan, result)
        return records

    def test_atomic_commit_and_snapshot_versions(self):
        ledger = Ledger(self.db_path)
        v1 = ledger.begin_snapshot(stable_hash(self.raw), 1,
                                   plan_fingerprint(self.plans), "v1")
        result = _batch_result(self.plans[0], 0)
        ledger.commit_batch(
            v1, result,
            {"lease-1": {"batch_id": result["batch_id"], "state": "released_done"}},
            self._receipt_records(self.plans[0], result))
        v2 = ledger.begin_snapshot(stable_hash(self.raw), 2,
                                   plan_fingerprint(self.plans), "v2")
        self.assertEqual(v2, v1 + 1)
        snapshot = ledger.load_snapshot(v1)
        self.assertEqual(len(snapshot["batches"]), 1)
        self.assertEqual(len(snapshot["receipts"]),
                         len(self.plans[0].request_ids) * 3)
        self.assertTrue(ledger.replay_verify(v1)["ok"])
        ledger.close()

    def test_replay_detects_first_difference(self):
        ledger = Ledger(self.db_path)
        version = ledger.begin_snapshot(stable_hash(self.raw), 1,
                                        plan_fingerprint(self.plans), "v")
        result = _batch_result(self.plans[0], 0)
        ledger.commit_batch(
            version, result, {},
            self._receipt_records(self.plans[0], result))
        ledger.conn.execute(
            "UPDATE receipts SET payload=? WHERE snapshot_version=? "
            "AND fmt=1",
            ('{"tampered": true}', version))
        ledger.conn.commit()
        report = ledger.replay_verify(version)
        self.assertFalse(report["ok"])
        self.assertEqual(report["first_diff"]["table"], "receipts")
        ledger.close()

    def test_recovery_marks_uncommitted_unknown_and_reclaims(self):
        ledger = Ledger(self.db_path)
        version = ledger.begin_snapshot(stable_hash(self.raw), 1,
                                        plan_fingerprint(self.plans), "crash")
        committed_plan = self.plans[0]
        result = _batch_result(committed_plan, 0)
        ledger.commit_batch(
            version, result,
            {"lease-1": {"batch_id": committed_plan.batch_id,
                         "state": "released_done"}},
            self._receipt_records(committed_plan, result))
        ledger.close()

        report = recover(self.db_path, self.spec)
        self.assertEqual(report["committed_batches"],
                         [committed_plan.batch_id])
        unknown_ids = set(report["unknown_batches"])
        expected_unknown = {plan.batch_id for plan in self.plans[1:]}
        self.assertEqual(unknown_ids, expected_unknown)
        reclaimed_workers = {item["worker_id"]
                             for item in report["quota_reclaimed"]}
        self.assertTrue(reclaimed_workers)
        self.assertTrue(report["replay"]["ok"])


if __name__ == "__main__":
    unittest.main()
