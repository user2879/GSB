from __future__ import annotations

import sqlite3
import tempfile
import unittest
from pathlib import Path

from replica_repair.engine import run_pipeline
from replica_repair.errors import ReplayError
from replica_repair.ledger import Ledger
from replica_repair.manifest_repo import compute_state_digest
from replica_repair.model import Manifest
from replica_repair.planner import RepairBudget
from replica_repair.replay import ReplayVerifier
from replica_repair.sample import init_sample, sample_ledger_path, sample_manifest_path
from replica_repair.util import atomic_write_bytes
from tests.helpers import group_servers, run_cli


class LedgerReplayTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        init_sample(self.root)
        self.ledger_path = sample_ledger_path(self.root, "interop-v3")
        self.manifest_path = sample_manifest_path(self.root, "interop-v3")

    def tearDown(self) -> None:
        self.temp.cleanup()

    def test_complete_run_ledger_before_after_replay_and_tamper_detection(self) -> None:
        with group_servers(self.root) as (manifest, clients, _servers):
            result = run_pipeline(
                root=self.root,
                manifest_path=self.manifest_path,
                ledger_path=self.ledger_path,
                clients=clients,
                budget=RepairBudget(3, 3),
            )
            self.assertTrue(result.replay["ok"])
            self.assertEqual(len(result.ledger_report["baseline"]["mismatches"]), 4)
            self.assertEqual(result.ledger_report["current"]["mismatches"], [])
            self.assertEqual(result.ledger_report["baseline"]["repair_stats"]["completed_tasks"], 4)
            self.assertEqual(result.ledger_report["baseline"]["repair_stats"]["total_repair_kb"], 10)
            self.assertEqual(result.ledger_report["baseline"]["repair_stats"]["total_cost_units"], 10)

            with sqlite3.connect(self.ledger_path) as connection:
                active_id = connection.execute(
                    "SELECT snapshot_id FROM snapshots WHERE is_active=1 AND is_stale=0"
                ).fetchone()[0]
                original = connection.execute(
                    "SELECT digest FROM distributions WHERE snapshot_id=? AND block_id='blk-0002' AND node_id='v3-a'",
                    (active_id,),
                ).fetchone()[0]
                connection.execute(
                    "UPDATE distributions SET digest=? WHERE snapshot_id=? AND block_id='blk-0002' AND node_id='v3-a'",
                    ("sha256:" + "0" * 64, active_id),
                )
                connection.commit()
            with Ledger(self.ledger_path) as ledger:
                with self.assertRaises(ReplayError) as caught:
                    ReplayVerifier(ledger, self.root, clients).verify()
            self.assertEqual(caught.exception.code, "E_REPLAY")
            mismatch = caught.exception.details["first_mismatch"]
            self.assertEqual(mismatch["block_id"], "blk-0002")
            self.assertEqual(mismatch["node_id"], "v3-a")
            self.assertEqual(mismatch["field"], "digest")

            with sqlite3.connect(self.ledger_path) as connection:
                connection.execute(
                    "UPDATE distributions SET digest=? WHERE snapshot_id=? AND block_id='blk-0002' AND node_id='v3-a'",
                    (original, active_id),
                )
                baseline_id = connection.execute(
                    "SELECT parent_snapshot_id FROM snapshots WHERE snapshot_id=?", (active_id,)
                ).fetchone()[0]
                connection.execute(
                    "UPDATE repair_tasks SET source_node='v3-b' WHERE snapshot_id=? AND task_id='rep-0001'",
                    (baseline_id,),
                )
                connection.commit()
            with Ledger(self.ledger_path) as ledger:
                with self.assertRaises(ReplayError) as caught:
                    ReplayVerifier(ledger, self.root, clients).verify()
            mismatch = caught.exception.details["first_mismatch"]
            self.assertEqual(mismatch["block_id"], "blk-0002")
            self.assertEqual(mismatch["field"], "source_node")
            self.assertEqual(caught.exception.code, "E_REPLAY")

    def test_current_block_bytes_tamper_is_located_as_replay_difference(self) -> None:
        with group_servers(self.root) as (manifest, clients, _servers):
            run_pipeline(
                root=self.root,
                manifest_path=self.manifest_path,
                ledger_path=self.ledger_path,
                clients=clients,
                budget=RepairBudget(3, 3),
            )
            path = self.root / "nodes" / "v3-a" / "blocks" / "blk-0001.bin"
            original = path.read_bytes()
            atomic_write_bytes(path, bytes([original[0] ^ 0xFF]) + original[1:])
            with Ledger(self.ledger_path) as ledger:
                with self.assertRaises(ReplayError) as caught:
                    ReplayVerifier(ledger, self.root, clients).verify()
            mismatch = caught.exception.details["first_mismatch"]
            self.assertEqual(mismatch["block_id"], "blk-0001")
            self.assertEqual(mismatch["node_id"], "v3-a")
            self.assertEqual(mismatch["field"], "digest")
            self.assertEqual(caught.exception.code, "E_REPLAY")

    def test_forced_termination_marks_unfinished_tasks_unknown(self) -> None:
        result = run_cli(
            [
                "run",
                "--root",
                str(self.root),
                "--manifest",
                str(self.manifest_path),
                "--ledger",
                str(self.ledger_path),
                "--max-parallel",
                "3",
                "--per-node-kb",
                "3",
                "--crash-after-task",
                "1",
            ]
        )
        self.assertEqual(result.returncode, 86, result.stderr)
        manifest = Manifest.from_file(self.manifest_path)
        state_digest = compute_state_digest(self.root, manifest)
        with Ledger(self.ledger_path) as ledger:
            recovery = ledger.open_for_state(manifest, state_digest)
            self.assertTrue(recovery.replaced)
            self.assertGreaterEqual(recovery.unknown_tasks, 1)
            report = ledger.current_report()
            self.assertIsNone(report["current"])
            self.assertEqual(report["progress"]["status"], "recovered")
            self.assertGreater(report["stale"]["snapshot_count"], 0)


    def test_retry_after_forced_termination_reconciles_and_completes(self) -> None:
        first = run_cli(
            [
                "run",
                "--root",
                str(self.root),
                "--manifest",
                str(self.manifest_path),
                "--ledger",
                str(self.ledger_path),
                "--max-parallel",
                "3",
                "--per-node-kb",
                "3",
                "--crash-after-task",
                "1",
            ]
        )
        self.assertEqual(first.returncode, 86, first.stderr)
        with group_servers(self.root) as (manifest, clients, _servers):
            result = run_pipeline(
                root=self.root,
                manifest_path=self.manifest_path,
                ledger_path=self.ledger_path,
                clients=clients,
                budget=RepairBudget(3, 3),
            )
            self.assertTrue(result.recovery["replaced"])
            self.assertGreaterEqual(result.recovery["unknown_tasks"], 1)
            self.assertTrue(result.reconciliation["performed"])
            self.assertGreaterEqual(len(result.reconciliation["reconciled_tasks"]), 1)
            self.assertEqual(result.ledger_report["current"]["mismatches"], [])
            self.assertTrue(result.replay["ok"])
        with sqlite3.connect(self.ledger_path) as connection:
            unknown = connection.execute(
                "SELECT count(*) FROM repair_tasks WHERE status='unknown'"
            ).fetchone()[0]
        self.assertGreaterEqual(unknown, 1)

    def test_ledger_format_mismatch_replaces_current_view(self) -> None:
        with group_servers(self.root) as (manifest, clients, _servers):
            run_pipeline(
                root=self.root,
                manifest_path=self.manifest_path,
                ledger_path=self.ledger_path,
                clients=clients,
                budget=RepairBudget(3, 3),
            )
        with sqlite3.connect(self.ledger_path) as connection:
            connection.execute("UPDATE meta SET value='999' WHERE key='format_version'")
            connection.commit()
        manifest = Manifest.from_file(self.manifest_path)
        state_digest = compute_state_digest(self.root, manifest)
        with Ledger(self.ledger_path) as ledger:
            recovery = ledger.open_for_state(manifest, state_digest)
            self.assertTrue(recovery.replaced)
            self.assertIn("ledger_format_version", recovery.reason or "")
            self.assertIsNone(ledger.current_report()["current"])
            self.assertGreaterEqual(ledger.current_report()["stale"]["snapshot_count"], 1)


if __name__ == "__main__":
    unittest.main()
