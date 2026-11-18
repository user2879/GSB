import sqlite3
import tempfile
import unittest
from pathlib import Path

from pool_workbench.app import WorkbenchSession
from pool_workbench.errors import BusinessError, ErrorCode
from pool_workbench.sample_db import (
    ALLOWED_TABLES,
    build_sample_db,
    version_sidecar_path,
)
from pool_workbench.snapshot import SnapshotStore
import json

COLD = Path(__file__).resolve().parent.parent / "samples" / "tasks_cold.json"
ADJUSTED = Path(__file__).resolve().parent.parent / "samples" / "tasks_adjusted.json"


class SnapshotReplayTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.workspace = Path(self.tmp.name)
        self.db = self.workspace / "sample.sqlite"
        build_sample_db(self.db)

    def tearDown(self):
        self.tmp.cleanup()

    def _checksums(self):
        conn = sqlite3.connect(str(self.db))
        try:
            out = {}
            for table in ALLOWED_TABLES:
                out[table] = conn.execute(
                    f"SELECT COUNT(*), COALESCE(SUM(rowid),0) FROM {table}"
                ).fetchone()
            return out
        finally:
            conn.close()

    def test_snapshot_roundtrip_and_reproducible_replay(self):
        session = WorkbenchSession(self.workspace)
        session.load(COLD, self.db)
        session.execute()
        path = session.save_snapshot()
        self.assertTrue(path.exists())
        before = self._checksums()
        report = session.replay(session.store.load(path))
        after = self._checksums()
        self.assertEqual(report["status"], "REPRODUCIBLE")
        self.assertTrue(report["sample_tables_unchanged"])
        self.assertEqual(before, after)
        session.close()

    def test_missing_param_version_rejected_as_e_snapshot(self):
        session = WorkbenchSession(self.workspace)
        session.load(COLD, self.db)
        session.execute()
        path = session.save_snapshot()
        document = json.loads(path.read_text(encoding="utf-8"))
        document["pool_config"].pop("param_version")
        broken = self.workspace / "broken_snapshot.json"
        broken.write_text(json.dumps(document), encoding="utf-8")
        with self.assertRaises(BusinessError) as ctx:
            SnapshotStore(self.workspace / "snapshots").load(broken)
        self.assertEqual(ctx.exception.code, ErrorCode.E_SNAPSHOT)
        session.close()

    def test_version_drift_triggers_whole_replacement_and_e_replay(self):
        first = WorkbenchSession(self.workspace)
        first.load(COLD, self.db)
        first.execute()
        snapshot_path = first.save_snapshot()
        first.close()

        second = WorkbenchSession(self.workspace)
        second.load(ADJUSTED, self.db)
        second.execute()
        restored = second.restore_snapshot(snapshot_path)
        self.assertTrue(restored["replaced"])
        changed = {item["param"] for item in restored["diff"]["changed_params"]}
        self.assertIn("param_version", changed)
        self.assertIn("workload_quotas", changed)
        # Whole replacement: pool reflects snapshot (v1), session stays v2.
        self.assertEqual(second.pool.config.param_version, 1)
        self.assertEqual(second.session_param_version, 2)
        self.assertEqual(
            restored["diff"]["affected_queued_tasks"],
            ["t09"],
        )
        with self.assertRaises(BusinessError) as ctx:
            second.replay(restored["document"])
        self.assertEqual(ctx.exception.code, ErrorCode.E_REPLAY)
        self.assertEqual(ctx.exception.details["field"], "pool_config.param_version")
        second.close()

    def test_sample_data_version_drift_is_e_replay(self):
        session = WorkbenchSession(self.workspace)
        session.load(COLD, self.db)
        session.execute()
        path = session.save_snapshot()
        document = session.store.load(path)
        sidecar = version_sidecar_path(self.db)
        meta = json.loads(sidecar.read_text(encoding="utf-8"))
        meta["sample_data_version"] = "999"
        sidecar.write_text(json.dumps(meta), encoding="utf-8")
        with self.assertRaises(BusinessError) as ctx:
            session.replay(document)
        self.assertEqual(ctx.exception.code, ErrorCode.E_REPLAY)
        self.assertEqual(ctx.exception.details["field"], "sample_data_version")
        session.close()


if __name__ == "__main__":
    unittest.main()
