from __future__ import annotations

import json
import subprocess
import sys
import unittest

from mvcc_workbench.ledger import Ledger
from mvcc_workbench.loader import load_dataset
from mvcc_workbench.renderer import render_full_report
from mvcc_workbench.workbench import Workbench

from tests.helpers import ROOT, temp_db


class CliAndUiTests(unittest.TestCase):
    def test_exception_injections_return_expected_codes(self) -> None:
        cases = (
            ("chain_cycle", "invalid_chain_cycle.json", "E_CHAIN", 3),
            ("txn_commit_regress", "invalid_txn_commit_regression.json", "E_TXN", 2),
            ("snapshot_digest", "invalid_snapshot_digest.json", "E_SNAPSHOT", 6),
        )
        for kind, filename, code, exit_code in cases:
            db = temp_db(f"mvcc-inject-test-{kind}-")
            command = [
                sys.executable,
                "-m",
                "mvcc_workbench.cli",
                "inject",
                "--kind",
                kind,
                "--input",
                str(ROOT / "samples" / filename),
                "--db",
                str(db),
                "--json",
            ]
            completed = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
            self.assertEqual(completed.returncode, exit_code, completed.stderr)
            payload = json.loads(completed.stdout)
            self.assertEqual(payload["code"], code)

    def test_render_is_stable_and_refresh_does_not_advance_ledger(self) -> None:
        db = temp_db("mvcc-view-test-")
        dataset = load_dataset(ROOT / "samples" / "base.json")
        with Ledger(db) as ledger:
            workbench = Workbench(dataset, ledger)
            workbench.execute()
            cursor = ledger.ledger_seq()
            snapshot_digest = ledger.latest_snapshot()["content_digest"]
            first = render_full_report(workbench.refresh_view(), dataset)
            second = render_full_report(workbench.refresh_view(), dataset)
            self.assertEqual(first, second)
            self.assertEqual(ledger.ledger_seq(), cursor)
            self.assertEqual(ledger.latest_snapshot()["content_digest"], snapshot_digest)
            for heading in (
                "Version chains",
                "Read-point snapshot lines",
                "Visibility decisions",
                "Scheduling timeline",
                "Budget and ledger",
                "Affected set and replay checks",
            ):
                self.assertIn(heading, first)
            self.assertLess(first.index("[acct:1]"), first.index("[acct:2]"))


if __name__ == "__main__":
    unittest.main()
