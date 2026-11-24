from __future__ import annotations

import json
import sqlite3
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from quota_ledger.client import NodePool
from quota_ledger.engine import Engine
from quota_ledger.errors import BusinessError
from quota_ledger.ledger import Ledger, compare_replay
from quota_ledger.spec import load_spec

from .helpers import make_sample, wait_ports_free


class EngineEndToEndTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.temp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temp.name)
        cls.sample, cls.base_port = make_sample(cls.root)
        cls.spec = load_spec(cls.sample)
        cls.baseline_dir = cls.root / "baseline"
        cls.baseline_result = Engine(cls.spec, cls.baseline_dir).run()

    @classmethod
    def tearDownClass(cls) -> None:
        cls.temp.cleanup()

    def test_baseline_completes_and_cancellation_is_final(self) -> None:
        result = self.baseline_result
        self.assertTrue(result.success)
        self.assertEqual(result.exit_code, 0)
        self.assertEqual(result.report["completed_count"], 9)
        self.assertNotIn("s10", result.report["completed_shards"])
        self.assertEqual(result.report["cancelled_shards"], ["s10"])
        ledger = Ledger(self.baseline_dir / "ledger.sqlite3")
        try:
            self.assertNotIn("s10", ledger.results())
            canceled = ledger.cancelled_shards()
            self.assertEqual(canceled["s10"]["reason"], "sample_cancel_leaf")
            self.assertFalse(any(row["shard_id"] == "s10" for row in ledger.attempts()))
            self.assertEqual(ledger.verify_snapshots()["snapshot_count"], 19)
        finally:
            ledger.close()

    def test_three_way_quota_constraints_and_wait_accounting(self) -> None:
        report = self.baseline_result.report
        scheduler = report["scheduler"]
        self.assertLessEqual(scheduler["max_units_in_flight"], scheduler["global_budget"])
        self.assertEqual(scheduler["max_units_in_flight"], 4)
        for node_id, limits in scheduler["node_limits"].items():
            self.assertLessEqual(scheduler["max_node_active"][node_id], limits["concurrency"], node_id)
            self.assertLessEqual(scheduler["leases"]["max_node_leases"][node_id], limits["lease_limit"], node_id)
        self.assertTrue(scheduler["wait_events"])
        for event in scheduler["wait_events"]:
            self.assertLessEqual(event["wait_start_ms"], event["wait_end_ms"])
        ledger = Ledger(self.baseline_dir / "ledger.sqlite3")
        try:
            budget = ledger.budget_state()
            self.assertEqual(budget["used_units"], 0)
            for row in ledger.node_quotas():
                self.assertEqual(row["active_shards"], 0)
                self.assertEqual(row["active_leases"], 0)
        finally:
            ledger.close()

    def test_lease_ownership_is_unique_and_bound_to_declared_node(self) -> None:
        ledger = Ledger(self.baseline_dir / "ledger.sqlite3")
        try:
            results = ledger.results()
            lease_ids = [row["lease_id"] for row in results.values()]
            self.assertEqual(len(lease_ids), len(set(lease_ids)))
            for shard_id, row in results.items():
                self.assertEqual(row["node_id"], self.spec.shards[shard_id].node_id)
                self.assertEqual(row["lease_id"], f"lease:{shard_id}:g{row['generation']}")
                self.assertEqual(row["outcome"], "completed")
                summary = json.loads(row["summary_json"])
                self.assertIn("digest", summary)
                self.assertGreater(summary["sample_count"], 0)
        finally:
            ledger.close()

    def test_incremental_invalidation_is_stable_and_preserves_unaffected_rows(self) -> None:
        report = self.baseline_result.report
        invalidations = report["invalidations"]
        revision_rows = [row for row in invalidations if row["event_seq"] == 12]
        revision_shards = [row["shard_id"] for row in revision_rows]
        self.assertEqual(revision_shards, [shard for shard in self.spec.topo_order if shard in revision_shards])
        self.assertIn("revision:2:node:node-b", {row["reason"] for row in revision_rows})
        self.assertTrue(any(str(row["reason"]).startswith("dependency:") for row in revision_rows))
        self.assertTrue(all(row["unchanged"] for row in report["incremental_checks"]))
        ledger = Ledger(self.baseline_dir / "ledger.sqlite3")
        try:
            history = ledger.result_history()
            counts: dict[str, int] = {}
            for row in history:
                counts[row["shard_id"]] = counts.get(row["shard_id"], 0) + 1
            self.assertEqual(counts["s01"], 1)
            self.assertEqual(counts["s04"], 1)
            self.assertEqual(counts["s02"], 2)
            self.assertEqual(counts["s09"], 3)
        finally:
            ledger.close()

    def test_same_input_same_ledger_digest(self) -> None:
        second_dir = self.root / "deterministic-second"
        second = Engine(self.spec, second_dir).run()
        self.assertTrue(second.success)
        self.assertEqual(second.ledger_digest, self.baseline_result.ledger_digest)
        comparison = compare_replay(
            self.baseline_dir / "ledger.sqlite3",
            second_dir / "ledger.sqlite3",
        )
        self.assertTrue(comparison["ok"])
        self.assertEqual(comparison["shard_count"], 9)

    def test_forced_termination_has_no_exit_report_and_recovery_replays(self) -> None:
        crash_dir = self.root / "crash"
        command = [
            sys.executable,
            "-m",
            "quota_ledger.cli",
            "run",
            "--sample",
            str(self.sample),
            "--workdir",
            str(crash_dir),
            "--crash-after-commits",
            "2",
        ]
        crashed = subprocess.run(command, capture_output=True, text=True, timeout=30)
        self.assertEqual(crashed.returncode, 70, crashed.stderr)
        ports = [self.base_port, self.base_port + 1, self.base_port + 2]
        self.assertTrue(wait_ports_free(ports))
        ledger = Ledger(crash_dir / "ledger.sqlite3")
        try:
            self.assertEqual(ledger._meta(key="clean_shutdown"), "0")
            self.assertEqual(ledger.exit_reports(), [])
            self.assertGreaterEqual(len(ledger.results()), 2)
            self.assertTrue(any(row["status"] in {"in_flight", "unknown"} for row in ledger.attempts()))
        finally:
            ledger.close()
        resumed = subprocess.run(
            [
                sys.executable,
                "-m",
                "quota_ledger.cli",
                "run",
                "--sample",
                str(self.sample),
                "--workdir",
                str(crash_dir),
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )
        self.assertEqual(resumed.returncode, 0, resumed.stderr)
        payload = json.loads(resumed.stdout.strip().splitlines()[-1])
        self.assertTrue(payload["unknown_shards_on_recovery"])
        comparison = compare_replay(
            self.baseline_dir / "ledger.sqlite3",
            crash_dir / "ledger.sqlite3",
        )
        self.assertTrue(comparison["ok"])
        self.assertEqual(comparison["ledger_digest"], self.baseline_result.ledger_digest)
        ledger = Ledger(crash_dir / "ledger.sqlite3")
        try:
            self.assertEqual(len(ledger.exit_reports()), 1)
            self.assertEqual(ledger.verify_snapshots()["content_digest"], self.baseline_result.ledger_digest)
        finally:
            ledger.close()

    def test_graceful_stop_writes_complete_exit_report_then_resumes(self) -> None:
        graceful_dir = self.root / "graceful"
        stopped = Engine(self.spec, graceful_dir, graceful_after_commits=2).run()
        self.assertTrue(stopped.interrupted)
        self.assertFalse(stopped.success)
        ledger = Ledger(graceful_dir / "ledger.sqlite3")
        try:
            reports = ledger.exit_reports()
            self.assertEqual(len(reports), 1)
            self.assertEqual(reports[0]["clean_exit"], 1)
            body = json.loads(reports[0]["report_json"])
            self.assertEqual(body["reason"], "graceful-after-2-commits")
            self.assertIn("unfinished_shards", body)
        finally:
            ledger.close()
        resumed = Engine(self.spec, graceful_dir).run()
        self.assertTrue(resumed.success)
        comparison = compare_replay(
            self.baseline_dir / "ledger.sqlite3",
            graceful_dir / "ledger.sqlite3",
        )
        self.assertEqual(comparison["ledger_digest"], self.baseline_result.ledger_digest)

    def test_replay_mismatch_reports_first_shard_and_field(self) -> None:
        candidate = self.root / "candidate-mismatch.sqlite3"
        source = sqlite3.connect(self.baseline_dir / "ledger.sqlite3")
        target = sqlite3.connect(candidate)
        try:
            source.backup(target)
        finally:
            source.close()
            target.close()
        ledger = Ledger(candidate)
        try:
            with ledger._lock, ledger._transaction() as conn:
                conn.execute("UPDATE results SET outcome='corrupted' WHERE shard_id='s01'")
                ledger._update_digest(conn)
        finally:
            ledger.close()
        with self.assertRaises(BusinessError) as ctx:
            compare_replay(self.baseline_dir / "ledger.sqlite3", candidate)
        self.assertEqual(ctx.exception.code, "E_REPLAY")
        self.assertEqual(ctx.exception.shard_id, "s01")
        self.assertEqual(ctx.exception.field, "outcome")

    def test_snapshot_digest_tamper_is_detected(self) -> None:
        path = self.root / "snapshot-tamper.sqlite3"
        ledger = Ledger(path)
        ledger.inject_digest_corruption()
        ledger.close()
        with self.assertRaises(BusinessError) as ctx:
            Ledger(path)
        self.assertEqual(ctx.exception.code, "E_SNAPSHOT")
        self.assertEqual(ctx.exception.field, "content_digest")


if __name__ == "__main__":
    unittest.main()
