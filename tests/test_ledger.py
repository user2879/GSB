from __future__ import annotations

import dataclasses
import sqlite3
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path
from types import MappingProxyType

from config_arbiter.arbitration import arbitrate_key, summarize_view
from config_arbiter.errors import ReplayError
from config_arbiter.ledger import Ledger
from config_arbiter.spec import KeyRule, ViewSpec

from .util import candidate


class LedgerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tempdir = tempfile.TemporaryDirectory()
        self.directory = Path(self.tempdir.name)
        self.database = self.directory / "ledger.sqlite3"
        self.view = ViewSpec(
            id="v",
            environments=("a", "b"),
            keys=MappingProxyType({"k": KeyRule("latest", 0)}),
        )
        self.result = arbitrate_key(
            view=self.view,
            key="k",
            rule=self.view.keys["k"],
            candidates=[
                candidate("a", 100, version=2),
                candidate("b", 99, version=1),
            ],
            scale_exponent=3,
        )
        self.summary = summarize_view("v", 1, [self.result])

    def tearDown(self) -> None:
        self.tempdir.cleanup()

    def _commit_ledger(self) -> Ledger:
        ledger = Ledger(self.database)
        ledger.commit_view(self.summary, config_digest="config-v1")
        return ledger

    def test_reopen_preserves_digest_and_replay(self) -> None:
        ledger = self._commit_ledger()
        first_digest = ledger.get_content_digest()
        ledger.close()
        reopened = Ledger(self.database)
        try:
            self.assertFalse(reopened.open_report.replaced)
            self.assertEqual(reopened.get_content_digest(), first_digest)
            self.assertEqual(len(reopened.current_results()), 1)
            report = reopened.replay(
                {("v", "k"): self.result}, {"v": 1}
            )
            self.assertEqual(report["status"], "OK")
        finally:
            reopened.close()

    def test_content_digest_mismatch_replaces_whole_snapshot(self) -> None:
        ledger = self._commit_ledger()
        ledger.close()
        connection = sqlite3.connect(self.database)
        connection.execute("UPDATE results SET summary_units=777 WHERE view_id='v' AND key='k'")
        connection.commit()
        connection.close()
        replaced = Ledger(self.database)
        try:
            self.assertTrue(replaced.open_report.replaced)
            self.assertEqual(
                replaced.open_report.differences[0]["field"], "content_digest"
            )
            self.assertEqual(replaced.all_results(), [])
        finally:
            replaced.close()
        self.assertTrue((self.directory / "ledger.sqlite3.replaced.1").exists())
        stable_reopen = Ledger(self.database)
        try:
            self.assertFalse(stable_reopen.open_report.replaced)
            self.assertEqual(stable_reopen.snapshot_report()["counts"]["replacements"], 1)
        finally:
            stable_reopen.close()

    def test_integer_format_version_mismatch_replaces_whole_snapshot(self) -> None:
        ledger = self._commit_ledger()
        ledger.close()
        connection = sqlite3.connect(self.database)
        connection.execute("PRAGMA user_version=99")
        connection.close()
        replaced = Ledger(self.database)
        try:
            self.assertTrue(replaced.open_report.replaced)
            self.assertEqual(
                replaced.open_report.differences[0]["field"], "format_version"
            )
            self.assertEqual(replaced.get_format_version(), 1)
        finally:
            replaced.close()

    def test_sigkill_recovers_only_committed_rows(self) -> None:
        ledger = self._commit_ledger()
        committed_digest = ledger.get_content_digest()
        ledger.close()
        marker = self.directory / "crash-ready"
        helper = Path(__file__).resolve().parent / "helpers" / "crash_writer.py"
        process = subprocess.Popen(
            [
                sys.executable,
                str(helper),
                str(self.database),
                str(marker),
                "v",
                "k",
                "99",
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline and not marker.exists():
            time.sleep(0.02)
        process.wait(timeout=5.0)
        if process.stdout is not None:
            process.stdout.close()
        if process.stderr is not None:
            process.stderr.close()
        self.assertTrue(marker.exists())
        self.assertNotEqual(process.returncode, 0)

        recovered = Ledger(self.database)
        try:
            self.assertFalse(recovered.open_report.replaced)
            self.assertEqual(recovered.get_content_digest(), committed_digest)
            versions = [int(row["view_version"]) for row in recovered.all_results()]
            self.assertEqual(versions, [1])
            self.assertEqual(
                recovered.replay({("v", "k"): self.result}, {"v": 1})["status"],
                "OK",
            )
        finally:
            recovered.close()

    def test_replay_locates_first_mismatching_key_and_field(self) -> None:
        ledger = self._commit_ledger()
        try:
            with self.assertRaises(ReplayError) as wrong_version:
                ledger.replay({("v", "k"): self.result}, {"v": 2})
            self.assertEqual(wrong_version.exception.context["key"], "v/k")
            self.assertEqual(wrong_version.exception.field, "view_version")

            altered = dataclasses.replace(self.result, summary_units=999)
            with self.assertRaises(ReplayError) as wrong_summary:
                ledger.replay({("v", "k"): altered}, {"v": 1})
            self.assertEqual(wrong_summary.exception.context["key"], "v/k")
            self.assertEqual(wrong_summary.exception.field, "summary_units")
            self.assertEqual(wrong_summary.exception.code, "E_REPLAY")
        finally:
            ledger.close()


if __name__ == "__main__":
    unittest.main()
