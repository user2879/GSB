from __future__ import annotations

import json
import os
import sqlite3
import stat
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from affinity_scheduler.migration import migrate_records
from affinity_scheduler.records import NormalizationParams, load_records
from affinity_scheduler.samples import write_sample
from affinity_scheduler.spec import load_spec

from tests.helpers import REPO_ROOT, parse_stdout_json, run_cli


class MigrationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.sample = write_sample(self.root, port_base=40400)
        self.spec = load_spec(self.sample["spec"])

    def tearDown(self) -> None:
        self.temp.cleanup()

    def test_migration_is_atomic_verified_and_backed_up(self) -> None:
        source_batch = load_records(self.sample["records"], params=None)
        self.assertEqual(source_batch.format_versions, (1, 2, 3))
        result = migrate_records(
            self.sample["records"],
            self.spec,
            self.root / "data" / "migration.sqlite3",
            batch_size=1,
        )
        self.assertEqual(result["status"], "migrated")
        backup = Path(result["backup"])
        self.assertTrue(backup.exists())
        self.assertEqual(stat.S_IMODE(backup.stat().st_mode), 0o444)
        params = NormalizationParams.from_spec(self.spec)
        after = load_records(self.sample["records"], params=params)
        self.assertEqual(after.format_versions, (3,))
        before = load_records(backup, params=params)
        ordered_before = sorted(
            before.records,
            key=lambda record: (record.source_format_version, record.source_line),
        )
        for old, new in zip(ordered_before, after.records):
            self.assertEqual(old.conclusion_bytes(), new.conclusion_bytes())
        second = migrate_records(
            self.sample["records"],
            self.spec,
            self.root / "data" / "migration.sqlite3",
            batch_size=1,
        )
        self.assertEqual(second["status"], "already-current")

    def test_forced_termination_resumes_from_committed_checkpoint(self) -> None:
        db_path = self.root / "data" / "forced.sqlite3"
        args = [
            "--log-file",
            str(self.root / "logs" / "forced.jsonl"),
            "migrate",
            "--records",
            self.sample["records"],
            "--spec",
            self.sample["spec"],
            "--db",
            str(db_path),
            "--batch-size",
            "1",
            "--abort-after-batches",
            "1",
        ]
        killed = run_cli(args)
        self.assertEqual(killed.returncode, 137, killed.stderr)
        self.assertEqual(load_records(self.sample["records"]).format_versions, (1, 2, 3))
        connection = sqlite3.connect(db_path)
        try:
            checkpoint = connection.execute(
                "SELECT checkpoint,status FROM migration_batches"
            ).fetchone()
        finally:
            connection.close()
        self.assertIsNotNone(checkpoint)
        self.assertEqual(checkpoint[0], 1)
        self.assertEqual(checkpoint[1], "running")

        resumed = run_cli(
            [
                "--log-file",
                str(self.root / "logs" / "forced.jsonl"),
                "migrate",
                "--records",
                self.sample["records"],
                "--spec",
                self.sample["spec"],
                "--db",
                str(db_path),
                "--batch-size",
                "1",
            ]
        )
        self.assertEqual(resumed.returncode, 0, resumed.stderr)
        payload = parse_stdout_json(resumed)
        self.assertEqual(payload["status"], "migrated")
        self.assertTrue(payload["migration_verification"]["conclusion_bytes_equal"])
        self.assertEqual(payload["checkpoint"], 3)


if __name__ == "__main__":
    unittest.main()
