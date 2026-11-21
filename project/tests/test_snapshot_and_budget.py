from __future__ import annotations

import json
import tempfile
import threading
import time
import unittest
from pathlib import Path

from arbitration_workbench.numeric import read_json
from arbitration_workbench.workbench import Workbench

from .helpers import copy_sample_workspace, rewrite_config


class SnapshotAndBudgetTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = copy_sample_workspace(Path(self.temp.name))

    def tearDown(self):
        self.temp.cleanup()

    def test_resume_reuses_every_cached_stage(self):
        first = Workbench(root=self.root, persist_edits=True)
        initial = first.run(use_cache=False, mode="full")
        self.assertFalse(initial.interrupted)
        version = initial.global_version
        second = Workbench(
            root=self.root,
            persist_edits=True,
            expected_snapshot_version=version,
        )
        self.assertTrue(second.startup_status["ok"], second.startup_status)
        self.assertEqual(version, second.global_version)
        resumed = second.run(use_cache=True, mode="incremental")
        stats = resumed.diagnostics["invalidation_and_reuse"]["stages"]
        for stage in ("score", "relations", "decision", "convert"):
            self.assertEqual(48, stats[stage]["reused"])
            self.assertEqual(0, stats[stage]["computed"])
        self.assertTrue(second.compare_incremental_with_full(resumed)["equal"])
        self.assertTrue(second.recovery_info["available"])
        self.assertGreaterEqual(resumed.global_version, version)

    def test_config_hash_mismatch_reports_stale_and_does_not_resume_cache(self):
        first = Workbench(root=self.root, persist_edits=True)
        first.run(use_cache=False, mode="full")
        rewrite_config(self.root, lambda raw: raw.__setitem__("decision_threshold", "1.25"))
        stale = Workbench(root=self.root, persist_edits=True)
        self.assertEqual("E_SNAPSHOT_STALE", stale.startup_code)
        self.assertIn("config_hash", stale.startup_status["stale_reasons"])
        self.assertEqual(0, stale.global_version)
        self.assertTrue(all(not entries for entries in stale.cache_store.stages.values()))

    def test_expected_version_mismatch_reports_stale(self):
        first = Workbench(root=self.root, persist_edits=True)
        initial = first.run(use_cache=False, mode="full")
        stale = Workbench(
            root=self.root,
            persist_edits=True,
            expected_snapshot_version=initial.global_version + 99,
        )
        self.assertEqual("E_SNAPSHOT_STALE", stale.startup_code)
        self.assertIn("global_version", stale.startup_status["stale_reasons"])

    def test_orphan_generation_and_staging_are_ignored(self):
        first = Workbench(root=self.root, persist_edits=True)
        initial = first.run(use_cache=False, mode="full")
        snapshots = self.root / "snapshots"
        pointer_before = read_json(snapshots / "CURRENT")
        orphan = snapshots / "generations" / "gen-99999999-deadbeef"
        orphan.mkdir(parents=True, exist_ok=True)
        (orphan / "cache.json").write_text("{partial", encoding="utf-8")
        staging = self.root / "tmp" / "snapshot-stage-interrupted"
        staging.mkdir(parents=True, exist_ok=True)
        (staging / "manifest.json").write_text('{"schema":"broken"}', encoding="utf-8")
        resumed = Workbench(root=self.root, persist_edits=True)
        self.assertTrue(resumed.startup_status["ok"], resumed.startup_status)
        self.assertEqual(initial.global_version, resumed.global_version)
        self.assertEqual(pointer_before, read_json(snapshots / "CURRENT"))
        self.assertTrue(resumed.recovery_info["available"])

    def test_interaction_budget_returns_previous_summary_then_background_finishes(self):
        workbench = Workbench(root=self.root, persist_edits=True)
        initial = workbench.run(use_cache=False, mode="full")
        before = workbench.latest_summary()
        workbench.engine.pre_run_delay_ms = 90
        started = time.monotonic()
        result = workbench.apply_edit_and_run({"op": "set_threshold", "value": "1.15"}, budget_ms=1)
        elapsed = time.monotonic() - started
        self.assertLess(elapsed, 0.25)
        self.assertEqual("pending", result["status"])
        self.assertEqual("E_INTERACT_BUDGET_PENDING", result["code"])
        self.assertEqual(before, result["latest_summary"])
        self.assertEqual(48, len(result["affected_tickets"]))
        self.assertIsNotNone(workbench._background_thread)
        workbench._background_thread.join(timeout=5)
        self.assertFalse(workbench._background_thread.is_alive())
        self.assertGreater(workbench.latest_summary()["snapshot_version"], initial.global_version)
        self.assertTrue(workbench.compare_incremental_with_full(workbench.last_result)["equal"])

    def test_stop_request_completes_inflight_and_publishes_partial_snapshot(self):
        workbench = Workbench(root=self.root, persist_edits=True)
        workbench.run(use_cache=False, mode="full")
        workbench.engine.exception_injections = [
            {
                "active": True,
                "ticket_id": "*",
                "stage": "score",
                "action": "delay",
                "delay_ms": 120,
                "exception_type": "DelayOnly",
                "message": "simulate long ticket work",
            }
        ]
        holder = {}

        def run_engine():
            holder["result"] = workbench.engine.run(use_cache=True, mode="incremental")

        thread = threading.Thread(target=run_engine)
        thread.start()
        time.sleep(0.02)
        workbench.request_stop()
        thread.join(timeout=5)
        self.assertFalse(thread.is_alive())
        result = holder["result"]
        self.assertTrue(result.interrupted)
        self.assertLess(len(result.tickets), 48)
        self.assertIsNotNone(result.manifest)
        self.assertTrue(result.manifest["partial"])
        self.assertLess(len(result.manifest["completed_ticket_ids"]), 48)
        self.assertTrue(result.manifest["artifacts"])
        self.assertTrue((self.root / "snapshots" / "CURRENT").exists())


if __name__ == "__main__":
    unittest.main()
