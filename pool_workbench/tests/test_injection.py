import tempfile
import unittest
from pathlib import Path

from pool_workbench.app import WorkbenchSession
from pool_workbench.errors import BusinessError, ErrorCode
from pool_workbench.fault_injection import (
    inject_config_quota_exceed,
    inject_schema_missing_table,
    inject_snapshot_missing_version,
)
from pool_workbench.sample_db import INVENTORY_TABLE, build_sample_db

COLD = Path(__file__).resolve().parent.parent / "samples" / "tasks_cold.json"


class FaultInjectionTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.workspace = Path(self.tmp.name)
        self.db = self.workspace / "sample.sqlite"
        build_sample_db(self.db)

    def tearDown(self):
        self.tmp.cleanup()

    def test_schema_missing_table_is_e_schema(self):
        inject_schema_missing_table(self.db, INVENTORY_TABLE)
        session = WorkbenchSession(self.workspace)
        with self.assertRaises(BusinessError) as ctx:
            session.load(COLD, self.db)
        self.assertEqual(ctx.exception.code, ErrorCode.E_SCHEMA)
        self.assertTrue(any("inventory" in path for path in ctx.exception.field_paths))

    def test_quota_exceed_is_e_config(self):
        injected = inject_config_quota_exceed(COLD)
        session = WorkbenchSession(self.workspace)
        with self.assertRaises(BusinessError) as ctx:
            session.load(injected, self.db)
        self.assertEqual(ctx.exception.code, ErrorCode.E_CONFIG)
        self.assertIn("pool.workload_quotas.sum", ctx.exception.field_paths)

    def test_snapshot_missing_version_is_e_snapshot(self):
        session = WorkbenchSession(self.workspace)
        session.load(COLD, self.db)
        session.execute()
        path = session.save_snapshot()
        injected = inject_snapshot_missing_version(path)
        with self.assertRaises(BusinessError) as ctx:
            session.store.load(injected)
        self.assertEqual(ctx.exception.code, ErrorCode.E_SNAPSHOT)
        session.close()


if __name__ == "__main__":
    unittest.main()
