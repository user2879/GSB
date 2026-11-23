from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path

from config_pipeline.manifest import manifest_from_dict
from config_pipeline.errors import E_REPLAY, PipelineError
from config_pipeline.session import PipelineSession
from config_pipeline.store import Store
from tests.helpers import free_port, manifest


class RecoveryTests(unittest.TestCase):
    def test_ledger_content_tamper_is_e_replay(self) -> None:
        data = manifest(
            ports=(free_port(),),
            budget=1,
            task_shards=1,
            task_limit=1,
            cluster_quotas=(1,),
        )
        with tempfile.TemporaryDirectory() as directory:
            db = Path(directory) / "state.db"
            session = PipelineSession(manifest_from_dict(data), str(db))
            try:
                session.run_current()
            finally:
                session.close()
            import sqlite3

            with sqlite3.connect(db) as connection:
                connection.execute(
                    "UPDATE ledger_entries SET outcome = 'failed' WHERE entry_id = 1"
                )
            with self.assertRaises(PipelineError) as caught:
                Store(str(db))
            self.assertEqual(caught.exception.code, E_REPLAY)
            self.assertIn("ledger", caught.exception.field or "")

    def test_forced_exit_keeps_committed_prefix_and_marks_running_unknown(self) -> None:
        data = manifest(
            ports=(free_port(),),
            budget=1,
            task_shards=3,
            task_limit=1,
            cluster_quotas=(1,),
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest_path = root / "manifest.json"
            db = root / "state.db"
            snapshot = root / "snapshot.json"
            child = root / "child.py"
            manifest_path.write_text(json.dumps(data), encoding="utf-8")
            child.write_text(
                textwrap.dedent(
                    """
                    import json, sys
                    from config_pipeline.manifest import manifest_from_dict
                    from config_pipeline.session import PipelineSession
                    with open(sys.argv[1], encoding='utf-8') as handle:
                        data = json.load(handle)
                    session = PipelineSession(manifest_from_dict(data), sys.argv[2], snapshot_path=sys.argv[3])
                    session.run_current(crash_after_completed=1)
                    raise SystemExit(99)
                    """
                ),
                encoding="utf-8",
            )
            environment = os.environ.copy()
            environment["PYTHONPATH"] = str(Path.cwd())
            result = subprocess.run(
                [sys.executable, str(child), str(manifest_path), str(db), str(snapshot)],
                cwd=str(Path.cwd()),
                env=environment,
                capture_output=True,
                text=True,
                timeout=20,
            )
            self.assertEqual(result.returncode, 137)
            self.assertEqual(result.stdout, "")

            store = Store(str(db))
            try:
                self.assertEqual(store.ledger_cursor, 1)
                entries = store.ledger_entries()
                self.assertEqual(len(entries), 1)
                self.assertEqual(entries[0]["outcome"], "success")
                running = store.connection.execute(
                    "SELECT shard_id, state FROM inflight WHERE state = 'running'"
                ).fetchall()
                self.assertEqual(len(running), 1)
                self.assertEqual(running[0]["state"], "running")
                prepared = store.latest_snapshot()
                self.assertEqual(prepared["payload"]["phase"], "prepared")
                self.assertEqual(prepared["ledger_cursor"], 0)
            finally:
                store.close()

            session = PipelineSession(
                manifest_from_dict(data), str(db), snapshot_path=str(snapshot)
            )
            try:
                report = session.recover()
                self.assertTrue(report.recovery.snapshot_replaced)
                self.assertEqual(len(report.recovery.unknown_shards), 1)
                self.assertEqual(len(report.recovery.replayed_shards), 1)
                self.assertEqual(session.store.ledger_cursor, 3)
                self.assertTrue(all(item["status"] == "success" for item in report.shards))
                self.assertEqual(len({item["ledger_entry_id"] for item in report.shards}), 3)
            finally:
                session.close()


if __name__ == "__main__":
    unittest.main()
