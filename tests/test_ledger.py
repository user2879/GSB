from __future__ import annotations

import json
import stat
import tempfile
import unittest
from pathlib import Path

from drift_diag.declarations import normalize_declaration
from drift_diag.drift import aggregate_all
from drift_diag.errors import DiagnosticError
from drift_diag.ledger import LedgerStore, make_batch_key

from .helpers import complete_actual, config, decisions_for, sample_path, target_v3


class LedgerTests(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        self.addCleanup(self.tempdir.cleanup)
        self.db = Path(self.tempdir.name) / "ledger.sqlite3"

    def target_v1(self):
        raw = json.loads(sample_path("declaration-v1.json").read_text(encoding="utf-8"))
        return normalize_declaration(raw, defaults_version="defaults-v1")

    def target_v2(self):
        raw = json.loads(sample_path("declaration-v2.json").read_text(encoding="utf-8"))
        return normalize_declaration(raw, defaults_version="defaults-v1")

    def commit_one(self, ledger, target, *, cursor=0):
        cfg = config()
        actual = complete_actual(target)
        decisions = decisions_for(target, actual, scale_exp=cfg.scale_exp)
        metrics = aggregate_all(target, decisions, cfg)
        snapshot = ledger.read_snapshot()
        assert snapshot is not None
        attempt = ledger.begin_attempt(
            batch_key=make_batch_key(snapshot["snapshot_digest"], cursor),
            workloads=sorted(target.workloads),
            plan={"cursor": cursor},
        )
        trace = {
            "rounds": [],
            "global_active_peak": 0,
            "per_workload_active_peak": {},
            "main_thread_ident": 0,
            "global_limit": 1,
            "per_workload_quota": 1,
            "task_delay_ms": 0,
            "concurrency_trajectory": [],
            "assignments_all_on_workers": True,
            "quota_respected": True,
            "global_limit_respected": True,
        }
        return ledger.commit_batch(
            event_cursor=cursor,
            first_event_seq=0,
            batch_kind="initial",
            decisions=decisions,
            workload_metrics=metrics,
            trace=trace,
            attempt_id=attempt,
        )

    def test_snapshot_replacement_backup_and_stable_diff(self):
        with LedgerStore(self.db) as ledger:
            first = ledger.install_snapshot(
                self.target_v1(), json.loads(sample_path("declaration-v1.json").read_text()),
                scale_version="fixed-point-v1", scale_exp=3,
            )
            self.assertTrue(first["replaced"])
            self.assertEqual(first["reasons"], ["initial"])
            second = ledger.install_snapshot(
                self.target_v2(), json.loads(sample_path("declaration-v2.json").read_text()),
                scale_version="fixed-point-v1", scale_exp=3,
            )
            self.assertTrue(second["replaced"])
            self.assertIn("declaration_format_version_changed", second["reasons"])
            self.assertIn("image", " ".join(second["diff"]["upgraded"]))
            self.assertIn("resource_limit", " ".join(second["diff"]["upgraded"]))
            self.assertIsNotNone(second["backup_path"])
            backup = Path(second["backup_path"])
            self.assertTrue(backup.exists())
            self.assertEqual(stat.S_IMODE(backup.stat().st_mode) & 0o222, 0)
            # The backup is a complete old declaration, not an incremental patch.
            self.assertEqual(json.loads(backup.read_text(encoding="utf-8"))["format_version"], 1)

            third = ledger.install_snapshot(
                self.target_v2(), json.loads(sample_path("declaration-v2.json").read_text()),
                scale_version="fixed-point-v1", scale_exp=3,
            )
            self.assertFalse(third["replaced"])
            self.assertEqual(third["diff"]["added"], [])

    def test_workload_removal_is_a_whole_target_replacement(self):
        target = target_v3()
        target.workloads.pop("cache")
        with LedgerStore(self.db) as ledger:
            ledger.install_snapshot(
                target_v3(), json.loads(sample_path("declaration-v3.json").read_text()),
                scale_version="fixed-point-v1", scale_exp=3,
            )
            result = ledger.install_snapshot(
                target, json.loads(sample_path("declaration-v3.json").read_text()),
                scale_version="fixed-point-v1", scale_exp=3,
            )
            self.assertTrue(result["replaced"])
            self.assertEqual(len(result["diff"]["removed"]), 6)
            self.assertTrue(all(path.startswith("cache/") for path in result["diff"]["removed"]))
            self.assertNotIn("cache", ledger.read_snapshot()["target"]["workloads"])

    def test_batch_is_atomic_visible_and_idempotent(self):
        target = target_v3()
        with LedgerStore(self.db) as ledger:
            ledger.install_snapshot(
                target, json.loads(sample_path("declaration-v3.json").read_text()),
                scale_version="fixed-point-v1", scale_exp=3,
            )
            batch = self.commit_one(ledger, target)
            self.assertEqual(len(batch["decisions"]), 18)
            self.assertEqual(batch["event_cursor"], 0)
            self.assertEqual(len(ledger.read_batches()), 1)
            self.assertEqual(ledger.verify_integrity()["decisions_checked"], 18)
            again = self.commit_one(ledger, target)
            self.assertTrue(again["idempotent"])
            self.assertEqual(len(ledger.read_batches()), 1)

    def test_recover_inflight_marks_unknown_until_successful_rerun(self):
        target = target_v3()
        with LedgerStore(self.db) as ledger:
            ledger.install_snapshot(
                target, json.loads(sample_path("declaration-v3.json").read_text()),
                scale_version="fixed-point-v1", scale_exp=3,
            )
            snapshot = ledger.read_snapshot()
            ledger.begin_attempt(
                batch_key=make_batch_key(snapshot["snapshot_digest"], 4),
                workloads=["worker", "cache"],
                plan={"event_cursor": 4},
            )
        with LedgerStore(self.db) as ledger:
            recovered = ledger.recover_inflight()
            self.assertEqual({item["workload"] for item in recovered}, {"worker", "cache"})
            self.assertEqual(len(ledger.list_unknown_workloads()), 2)
            self.assertEqual(ledger.read_batches(), [])
        with LedgerStore(self.db) as ledger:
            self.assertEqual(len(ledger.list_unknown_workloads()), 2)
            self.commit_one(ledger, target)
            self.assertEqual(ledger.list_unknown_workloads(), [])

    def test_snapshot_mismatch_clears_old_batches_and_conclusions(self):
        with LedgerStore(self.db) as ledger:
            ledger.install_snapshot(
                target_v3(), json.loads(sample_path("declaration-v3.json").read_text()),
                scale_version="fixed-point-v1", scale_exp=3,
            )
            self.commit_one(ledger, target_v3())
            self.assertEqual(len(ledger.read_batches()), 1)
            replaced = ledger.install_snapshot(
                self.target_v2(), json.loads(sample_path("declaration-v2.json").read_text()),
                scale_version="fixed-point-v1", scale_exp=3,
            )
            self.assertTrue(replaced["replaced"])
            self.assertEqual(ledger.read_batches(), [])
            self.assertEqual(ledger.replacement_history()[0]["new_snapshot_digest"], replaced["snapshot_digest"])

    def test_tolerance_change_replaces_snapshot_and_clears_conclusions(self):
        target = target_v3()
        with LedgerStore(self.db) as ledger:
            ledger.install_snapshot(
                target, json.loads(sample_path("declaration-v3.json").read_text()),
                scale_version="fixed-point-v1", scale_exp=3,
                tolerance_num=0, tolerance_den=1,
            )
            self.commit_one(ledger, target)
            replaced = ledger.install_snapshot(
                target, json.loads(sample_path("declaration-v3.json").read_text()),
                scale_version="fixed-point-v1", scale_exp=3,
                tolerance_num=1, tolerance_den=6,
            )
            self.assertTrue(replaced["replaced"])
            self.assertIn("tolerance_changed", replaced["reasons"])
            self.assertEqual(ledger.read_batches(), [])
            self.assertEqual(ledger.read_snapshot()["tolerance_num"], 1)

    def test_tampered_decision_digest_returns_e_snapshot(self):
        target = target_v3()
        with LedgerStore(self.db) as ledger:
            ledger.install_snapshot(
                target, json.loads(sample_path("declaration-v3.json").read_text()),
                scale_version="fixed-point-v1", scale_exp=3,
            )
            self.commit_one(ledger, target)
            ledger.conn.execute(
                "UPDATE decisions SET reason = 'tampered' WHERE rowid = (SELECT MIN(rowid) FROM decisions)"
            )
            with self.assertRaises(DiagnosticError) as context:
                ledger.verify_integrity()
        self.assertEqual(context.exception.code, "E_SNAPSHOT")
        self.assertEqual(context.exception.field, "content_digest")

    def test_tampered_snapshot_digest_returns_e_snapshot(self):
        target = target_v3()
        with LedgerStore(self.db) as ledger:
            ledger.install_snapshot(
                target, json.loads(sample_path("declaration-v3.json").read_text()),
                scale_version="fixed-point-v1", scale_exp=3,
            )
            ledger.conn.execute("UPDATE snapshot SET target_json = '{}' WHERE id = 1")
            with self.assertRaises(DiagnosticError) as context:
                ledger.read_snapshot()
        self.assertEqual(context.exception.code, "E_SNAPSHOT")


if __name__ == "__main__":
    unittest.main()
