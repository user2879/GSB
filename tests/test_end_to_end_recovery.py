from __future__ import annotations

import copy
import json
import os
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path

from sharddiag.cluster import NodeCluster, is_port_free
from sharddiag.engine import Engine
from sharddiag.ledger import (
    Ledger,
    replace_sqlite_with_snapshot,
    verify_snapshot_file,
    write_full_snapshot,
)
from sharddiag.spec import load_spec, validate_spec


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SAMPLE_SPEC = PROJECT_ROOT / "config" / "spec.sample.json"


class EngineAcceptanceTests(unittest.TestCase):
    def _run_engine(self, root: Path, *, spec=None, batch_size=None):
        spec = spec or load_spec(SAMPLE_SPEC)
        ledger = Ledger.open(root / "ledger.sqlite3", expected_spec_digest=spec.spec_digest)
        cluster = NodeCluster(spec, log_dir=root / "logs")
        cluster.start()
        try:
            engine = Engine(spec, ledger, cluster, batch_size=batch_size)
            report = engine.run()
            return spec, ledger, cluster, engine, report
        except Exception:
            cluster.stop()
            ledger.close()
            raise

    def test_real_tcp_scheduler_wait_renewal_and_constraints(self) -> None:
        with tempfile.TemporaryDirectory(prefix="sharddiag-test-") as tmp:
            root = Path(tmp)
            spec, ledger, cluster, engine, report = self._run_engine(root)
            try:
                self.assertTrue(report["ok"], json.dumps(report, indent=2, default=str)[:6000])
                self.assertEqual(8, sum(1 for row in report["shards"].values() if row["status"] == "succeeded"))
                self.assertLessEqual(report["water_report"]["peak_used_units"], spec.raw["global_budget_units"])
                queued = report["shards"]["s00_a_side"]
                self.assertEqual("succeeded", queued["status"])
                self.assertIsNotNone(queued["wait_started_monotonic_ms"])
                self.assertIsNotNone(queued["wait_ended_monotonic_ms"])
                self.assertGreaterEqual(queued["wait_ms"], 0)
                ingest = report["shards"]["s01_a_ingest"]
                self.assertEqual(2, len(ingest["renewals"]))
                self.assertEqual(3, ingest["renewals"][-1]["lease_version"])
                ledger.verify_chain(expected_spec_digest=spec.spec_digest)
                digest_a = ledger.canonical_accounting_summary()["ledger_summary_digest"]
            finally:
                cluster.stop()
                ledger.close()

        with tempfile.TemporaryDirectory(prefix="sharddiag-test-b-") as tmp:
            root_b = Path(tmp)
            spec_b, ledger_b, cluster_b, engine_b, report_b = self._run_engine(root_b, batch_size=7)
            try:
                self.assertTrue(report_b["ok"])
                digest_b = ledger_b.canonical_accounting_summary()["ledger_summary_digest"]
                summaries_a = None
            finally:
                cluster_b.stop()
                ledger_b.close()

        with tempfile.TemporaryDirectory(prefix="sharddiag-test-a-") as tmp:
            root_a = Path(tmp)
            spec_a, ledger_a, cluster_a, engine_a, report_a = self._run_engine(root_a, batch_size=16)
            try:
                self.assertTrue(report_a["ok"])
                digest_a = ledger_a.canonical_accounting_summary()["ledger_summary_digest"]
            finally:
                cluster_a.stop()
                ledger_a.close()

        self.assertEqual(digest_a, digest_b)

    def test_cancel_freezes_only_downstream_closure(self) -> None:
        raw = copy.deepcopy(load_spec(SAMPLE_SPEC).raw)
        raw["events"] = [
            {"seq": index + 1, "type": "shard_arrived", "shard_id": shard_id}
            for index, shard_id in enumerate(
                ["s00_a_side", "s01_a_ingest", "s02_b_ingest", "s03_a_transform", "s04_b_transform", "s05_c_join", "s06_c_aggregate", "s07_sink"]
            )
        ] + [{"seq": 100, "type": "cancel", "shard_id": "s04_b_transform"}]
        spec = validate_spec(raw)
        with tempfile.TemporaryDirectory(prefix="sharddiag-cancel-") as tmp:
            _, ledger, cluster, engine, report = self._run_engine(Path(tmp), spec=spec)
            try:
                self.assertEqual("cancelled", report["shards"]["s04_b_transform"]["status"])
                self.assertEqual(0, report["shards"]["s04_b_transform"]["dispatch_count"])
                self.assertEqual("frozen", report["shards"]["s06_c_aggregate"]["status"])
                self.assertIn("s04_b_transform", report["shards"]["s06_c_aggregate"]["missing_upstream_ids"])
                self.assertEqual("frozen", report["shards"]["s07_sink"]["status"])
                self.assertIn("s06_c_aggregate", report["shards"]["s07_sink"]["missing_upstream_ids"])
                self.assertEqual("succeeded", report["shards"]["s05_c_join"]["status"])
                self.assertEqual(0, engine.lease_manager.used_units())
            finally:
                cluster.stop()
                ledger.close()


class SnapshotReplacementTests(unittest.TestCase):
    def test_snapshot_digest_mismatch_replaces_whole_state(self) -> None:
        with tempfile.TemporaryDirectory(prefix="sharddiag-snapshot-") as tmp:
            root = Path(tmp)
            db = root / "ledger.sqlite3"
            snapshot = root / "snapshot.json"
            ledger = Ledger.open(db)
            ledger.commit_batch("unit_test", {"answer": 42}, shard_states=[{"shard_id": "s1", "status": "succeeded"}])
            envelope = ledger.export_snapshot(snapshot)
            ledger.close()

            broken = json.loads(snapshot.read_text(encoding="utf-8"))
            broken["content_digest"] = "0" * 64
            snapshot.write_text(json.dumps(broken), encoding="utf-8")
            with self.assertRaises(Exception):
                verify_snapshot_file(snapshot)

            replacement = write_full_snapshot(
                snapshot,
                {
                    "spec_digest": envelope["spec_digest"],
                    "head_digest": "REPLACEMENT",
                    "state": {"shards": [{"shard_id": "whole", "status": "succeeded"}], "leases": []},
                },
            )
            self.assertEqual("REPLACEMENT", replacement["head_digest"])
            verified = verify_snapshot_file(snapshot)
            self.assertEqual("REPLACEMENT", verified["head_digest"])

            ledger = Ledger.open(db)
            ledger.close()
            conn = __import__("sqlite3").connect(db)
            conn.execute("UPDATE meta SET value='bad-head' WHERE key='head_digest'")
            conn.commit()
            conn.close()
            with self.assertRaises(Exception):
                Ledger.open(db)

            valid_snapshot = verify_snapshot_file(snapshot)
            restored = replace_sqlite_with_snapshot(db, valid_snapshot)
            states = restored.shard_states()
            self.assertIn("whole", states)
            self.assertNotIn("s1", states)
            restored.verify_chain()
            restored.close()


class CrashReplaySubprocessTests(unittest.TestCase):
    def test_sigkill_then_restart_recovers_committed_batches(self) -> None:
        with tempfile.TemporaryDirectory(prefix="sharddiag-crash-") as tmp:
            root = Path(tmp)
            db = root / "crash.sqlite3"
            report1 = root / "report1.json"
            snapshot1 = root / "snapshot1.json"
            logs = root / "logs"
            env = os.environ.copy()
            env["PYTHONPATH"] = str(PROJECT_ROOT) + os.pathsep + env.get("PYTHONPATH", "")
            first = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "sharddiag.cli",
                    "run",
                    "--spec",
                    str(SAMPLE_SPEC),
                    "--db",
                    str(db),
                    "--report",
                    str(report1),
                    "--snapshot",
                    str(snapshot1),
                    "--log-dir",
                    str(logs),
                    "--fresh",
                    "--kill-after-terminal",
                    "1",
                ],
                cwd=PROJECT_ROOT,
                env=env,
                capture_output=True,
                text=True,
                timeout=30,
            )
            self.assertNotEqual(0, first.returncode)
            for port in (39731, 39732, 39733):
                for _ in range(100):
                    if is_port_free("127.0.0.1", port):
                        break
                    time.sleep(0.05)
                self.assertTrue(is_port_free("127.0.0.1", port), f"port {port} remained occupied")

            restarted = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "sharddiag.cli",
                    "run",
                    "--spec",
                    str(SAMPLE_SPEC),
                    "--db",
                    str(db),
                    "--report",
                    str(root / "report2.json"),
                    "--snapshot",
                    str(root / "snapshot2.json"),
                    "--log-dir",
                    str(logs),
                ],
                cwd=PROJECT_ROOT,
                env=env,
                capture_output=True,
                text=True,
                timeout=40,
            )
            self.assertEqual(0, restarted.returncode, restarted.stderr + restarted.stdout)
            report2 = json.loads((root / "report2.json").read_text(encoding="utf-8"))
            self.assertTrue(report2["ok"], json.dumps(report2, indent=2)[:6000])

            ledger = Ledger.open(db, expected_spec_digest=load_spec(SAMPLE_SPEC).spec_digest)
            try:
                ledger.verify_chain()
                recovery_payloads = [
                    batch["payload"] for batch in ledger.batches() if batch["kind"] == "recovery_reconcile"
                ]
                self.assertTrue(recovery_payloads)
                self.assertTrue(recovery_payloads[0]["unknown_shard_ids"])
                self.assertTrue(recovery_payloads[0]["reclaimed_lease_ids"])
                final_digest = ledger.canonical_accounting_summary()["ledger_summary_digest"]
            finally:
                ledger.close()

            clean_db = root / "clean.sqlite3"
            clean = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "sharddiag.cli",
                    "run",
                    "--spec",
                    str(SAMPLE_SPEC),
                    "--db",
                    str(clean_db),
                    "--report",
                    str(root / "clean-report.json"),
                    "--log-dir",
                    str(root / "clean-logs"),
                    "--fresh",
                ],
                cwd=PROJECT_ROOT,
                env=env,
                capture_output=True,
                text=True,
                timeout=40,
            )
            self.assertEqual(0, clean.returncode, clean.stderr + clean.stdout)
            clean_ledger = Ledger.open(clean_db, expected_spec_digest=load_spec(SAMPLE_SPEC).spec_digest)
            try:
                clean_digest = clean_ledger.canonical_accounting_summary()["ledger_summary_digest"]
            finally:
                clean_ledger.close()
            self.assertEqual(clean_digest, final_digest)


if __name__ == "__main__":
    unittest.main()
