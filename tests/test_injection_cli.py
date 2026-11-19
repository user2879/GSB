from __future__ import annotations

import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path

from srebudget.cli import main
from srebudget.samples import write_samples


class InjectionCliTests(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        self.data_dir = Path(self.tempdir.name) / "data"
        write_samples(self.data_dir)
        self.log_path = self.data_dir / "logs" / "events.ndjson"

    def tearDown(self):
        self.tempdir.cleanup()

    def _run(self, *args):
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            code = main([*args, "--data-dir", str(self.data_dir)])
        return code, json.loads(output.getvalue())

    def test_three_injections_report_expected_business_codes(self):
        cases = [
            ("chain_corrupt", "E_CHAIN", 20),
            ("timestamp_regression", "E_TRACE", 21),
            ("snapshot_digest_mismatch", "E_SNAPSHOT", 22),
        ]
        for kind, code, exit_code in cases:
            with self.subTest(kind=kind):
                actual_exit, payload = self._run("inject", kind)
                self.assertEqual(actual_exit, exit_code)
                self.assertEqual(payload["reported"]["code"], code)

        log_records = [
            json.loads(line)
            for line in self.log_path.read_text(encoding="utf-8").splitlines()
        ]
        codes = [record["code"] for record in log_records]
        self.assertEqual(codes, ["E_CHAIN", "E_TRACE", "E_SNAPSHOT"])

    def test_deterministic_cli_run_and_replay(self):
        db_path = self.data_dir / "run.sqlite3"
        exit_code, payload = self._run(
            "run",
            "--db",
            str(db_path),
            "--session-id",
            "cli",
            "--batch-size",
            "4",
            "--clock",
            "trace",
            "--snapshot-at",
            "5",
        )
        self.assertEqual(exit_code, 0)
        self.assertEqual(payload["processed_calls"], 12)
        self.assertTrue(any(item["budget_exhausted"] for item in payload["services"]))
        self.assertTrue(payload["snapshot"]["snapshot_id"].startswith("snap-"))
        self.assertEqual(payload["snapshot"]["cursor"], 5)
        replay_exit, replay = self._run(
            "replay",
            "--db",
            str(db_path),
            "--session-id",
            "cli",
        )
        self.assertEqual(replay_exit, 0)
        self.assertEqual(replay["compared_calls"], 7)


if __name__ == "__main__":
    unittest.main()
