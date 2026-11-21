from __future__ import annotations

import json
import sqlite3
import subprocess
import sys
import unittest

from mvcc_workbench.errors import DiagnosticError
from mvcc_workbench.ledger import Ledger
from mvcc_workbench.loader import load_dataset
from mvcc_workbench.workbench import Workbench

from tests.helpers import ROOT, temp_db


class LedgerRecoveryTests(unittest.TestCase):
    def test_rounds_snapshot_and_clean_replay(self) -> None:
        db = temp_db()
        dataset = load_dataset(ROOT / "samples" / "base.json")
        with Ledger(db) as ledger:
            first = Workbench(dataset, ledger).execute()
            self.assertEqual(first["status"], "COLD_START_COMPLETE")
            self.assertEqual(first["ledger_version"]["ledger_cursor"], 3)
            self.assertEqual(len(first["rounds"]), 3)
            snapshot = ledger.latest_snapshot()
            self.assertEqual(snapshot["metadata"]["manifest"]["txn_set_version"], 1)
            self.assertEqual(snapshot["cursor"]["ledger_cursor"], 3)
            self.assertTrue(snapshot["content_digest"].startswith("sha256:"))
        with Ledger(db) as ledger:
            replay = Workbench(dataset, ledger).execute()
            self.assertEqual(replay["status"], "REPLAY_OK")
            self.assertEqual(replay["code"], "OK")
            self.assertEqual(len(ledger.completed_rounds(1)), 3)

    def test_tampered_snapshot_digest_raises_e_snapshot(self) -> None:
        db = temp_db()
        dataset = load_dataset(ROOT / "samples" / "base.json")
        with Ledger(db) as ledger:
            Workbench(dataset, ledger).execute()
        connection = sqlite3.connect(db)
        try:
            connection.execute("UPDATE snapshots SET content_digest = 'sha256:" + "0" * 64 + "'")
            connection.commit()
        finally:
            connection.close()
        with self.assertRaises(DiagnosticError) as caught:
            Ledger(db)
        self.assertEqual(caught.exception.code, "E_SNAPSHOT")

    def test_forced_termination_unknown_task_and_resume(self) -> None:
        db = temp_db("mvcc-crash-test-")
        base = str(ROOT / "samples" / "base.json")
        crash = subprocess.run(
            [
                sys.executable,
                "-m",
                "mvcc_workbench.cli",
                "run",
                "--input",
                base,
                "--db",
                str(db),
                "--json",
                "--crash-in-round",
                "2",
                "--crash-after-submit",
                "1",
            ],
            cwd=ROOT,
            capture_output=True,
            text=True,
        )
        self.assertEqual(crash.returncode, 87)
        with Ledger(db) as ledger:
            unknown = ledger.unknown_tasks()
            self.assertEqual({item["query_id"] for item in unknown}, {"Q02", "Q07"})
            self.assertEqual([row["round_no"] for row in ledger.completed_rounds(1)], [1])
            self.assertIsNotNone(ledger.latest_snapshot())
            resumed = Workbench(load_dataset(base), ledger).execute()
            self.assertEqual(resumed["status"], "RESUMED_COMPLETE")
            self.assertEqual(resumed["generation"], 1)
            self.assertEqual(len(resumed["decisions"]), 8)
            self.assertEqual(ledger.unknown_tasks(), [])
            self.assertEqual(
                {item["status"] for item in ledger.all_unknown_task_history()},
                {"RERUN_COMPLETED"},
            )

    def test_recover_cli_never_claims_a_complete_report(self) -> None:
        db = temp_db("mvcc-recover-test-")
        base = str(ROOT / "samples" / "base.json")
        subprocess.run(
            [
                sys.executable,
                "-m",
                "mvcc_workbench.cli",
                "run",
                "--input",
                base,
                "--db",
                str(db),
                "--json",
                "--crash-in-round",
                "2",
                "--crash-after-submit",
                "1",
            ],
            cwd=ROOT,
            check=False,
            capture_output=True,
            text=True,
        )
        recovered = subprocess.run(
            [
                sys.executable,
                "-m",
                "mvcc_workbench.cli",
                "recover",
                "--input",
                base,
                "--db",
                str(db),
                "--json",
            ],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
        payload = json.loads(recovered.stdout)
        self.assertEqual(payload["status"], "RECOVERY_REQUIRED")
        self.assertFalse(payload["report_available"])
        self.assertEqual(payload["next_action"], "RERUN_UNKNOWN_TASKS")


if __name__ == "__main__":
    unittest.main()
