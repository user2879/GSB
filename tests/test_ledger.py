import json
import os
import stat
import tempfile
import unittest

from convd.ledger import (Ledger, backup_path_for, reconcile_snapshot,
                          snapshot_path_for)
from convd.scene import Scenario


def scenario(budget=1):
    return Scenario({"scenario_id": "sc", "max_rounds": 2,
                     "window_ms": 300, "pulse_interval_ms": 50,
                     "lease_ms": 250, "receipt_timeout_ms": 100,
                     "budget_version": budget, "executor_set_version": 1})


def entry(executor_id, round_no, conclusion="converged"):
    return {"executor_id": executor_id,
            "logical": {"round": round_no, "executor_id": executor_id,
                        "version_before": round_no - 1,
                        "version_after": round_no, "digest_before": None,
                        "digest_after": "d%d" % round_no,
                        "obs_seq": round_no, "advance_ms": 1,
                        "attempts": 1, "in_lease": True,
                        "conclusion": conclusion,
                        "reason": "ok"},
            "timing": {"send_mono_ms": 1}}


class TestLedger(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.db = os.path.join(self.tmp.name, "l.db")

    def tearDown(self):
        self.tmp.cleanup()

    def test_atomic_round_commit_and_cursor(self):
        ledger = Ledger(self.db)
        ledger.commit_round(1, [entry("a", 1), entry("b", 1)],
                            [{"executor_id": "a", "kind": "obs_seq_gap",
                              "payload": {"from": 0, "to": 1}}],
                            [{"executor_id": "b", "attempt": 1,
                              "payload": {"kind": "timeout"}}])
        self.assertEqual(ledger.cursor(), 1)
        self.assertEqual(len(ledger.round_entries()), 2)
        self.assertEqual(len(ledger.anomaly_entries()), 1)
        self.assertEqual(len(ledger.retry_entries()), 1)
        self.assertEqual(ledger.get_meta("ledger_digest"),
                         ledger.content_digest())
        ledger.close()

    def test_digest_excludes_timing(self):
        ledger = Ledger(self.db)
        ledger.commit_round(1, [entry("a", 1)], [], [])
        digest1 = ledger.content_digest()
        ledger.close()
        os.remove(self.db)
        ledger = Ledger(self.db)
        different_timing = entry("a", 1)
        different_timing["timing"] = {"send_mono_ms": 999999}
        ledger.commit_round(1, [different_timing], [], [])
        self.assertEqual(digest1, ledger.content_digest())
        ledger.close()

    def test_summary_counts(self):
        ledger = Ledger(self.db)
        ledger.commit_round(1, [entry("a", 1)], [], [])
        ledger.commit_round(2, [entry("a", 2, "not_converged")], [], [])
        summary = ledger.summary()
        self.assertEqual(summary["a"]["converged_rounds"], 1)
        self.assertEqual(summary["a"]["not_converged"],
                         [{"round": 2, "reason": "ok"}])
        ledger.close()


class TestSnapshot(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.db = os.path.join(self.tmp.name, "l.db")

    def tearDown(self):
        self.tmp.cleanup()

    def test_snapshot_written_then_matches(self):
        sc = scenario()
        snap, diffs = reconcile_snapshot(self.db, sc, {"1": {"a": "c"}},
                                         "digest")
        self.assertEqual(diffs, [])
        snap2, diffs2 = reconcile_snapshot(self.db, sc, {"1": {"a": "c"}},
                                           "digest")
        self.assertEqual(diffs2, [])

    def test_mismatch_replaces_wholesale_and_keeps_readonly_backup(self):
        sc_v1 = scenario(budget=1)
        reconcile_snapshot(self.db, sc_v1, {"1": {"a": "converged"}},
                           "digest-1")
        sc_v2 = scenario(budget=2)
        snap, diffs = reconcile_snapshot(self.db, sc_v2,
                                         {"1": {"a": "not_converged"}},
                                         "digest-2")
        fields = [d["field"] for d in diffs]
        self.assertEqual(fields, sorted(fields))
        self.assertIn("budget_version", fields)
        self.assertIn("round_conclusions", fields)
        self.assertIn("ledger_digest", fields)
        # Wholesale replace: new snapshot reflects only the new state.
        self.assertEqual(snap["budget_version"], 2)
        self.assertEqual(snap["ledger_digest"], "digest-2")
        backup = backup_path_for(self.db)
        self.assertTrue(os.path.exists(backup))
        mode = stat.S_IMODE(os.stat(backup).st_mode)
        self.assertFalse(mode & stat.S_IWUSR)
        with open(backup) as fh:
            old = json.load(fh)
        self.assertEqual(old["budget_version"], 1)
        self.assertEqual(old["ledger_digest"], "digest-1")


if __name__ == "__main__":
    unittest.main()
