import json
import tempfile
import unittest
from pathlib import Path

from pool_workbench.config import config_from_dict, load_config, validate_sql
from pool_workbench.errors import BusinessError, ErrorCode
from pool_workbench.sample_db import ALLOWED_TABLES


class SqlValidationTests(unittest.TestCase):
    def test_allows_select_insert_update(self):
        for sql in (
            "SELECT * FROM inventory",
            "INSERT INTO ops_log (workload, task_id, note) SELECT 'a', 'b', 'c'",
            "UPDATE inventory SET qty = 1 WHERE item_id = 1",
        ):
            kind, error, _ = validate_sql(sql, ALLOWED_TABLES)
            self.assertIsNone(error)
            self.assertIn(kind, ("SELECT", "INSERT", "UPDATE"))

    def test_rejects_dangerous_and_unknown_table(self):
        for sql in (
            "DELETE FROM inventory",
            "DROP TABLE inventory",
            "SELECT * FROM secret_table",
            "ATTACH DATABASE 'x' AS y",
            "PRAGMA table_info(inventory)",
            "SELECT 1; SELECT 2",
            "",
        ):
            _kind, error, _ = validate_sql(sql, ALLOWED_TABLES)
            self.assertIsNotNone(error, sql)


class ConfigValidationTests(unittest.TestCase):
    def _base(self, **overrides):
        doc = {
            "sample_data_version": "1",
            "pool": {
                "max_connections": 3,
                "workload_quotas": {"wl_a": 2, "wl_b": 1},
                "hold_timeout_ms": 100,
                "acquire_timeout_ms": 50,
            },
            "tasks": [
                {"task_id": "a", "workload": "wl_a", "conn_type": "READ_WRITE",
                 "estimated_ms": 10, "sql": "UPDATE inventory SET qty = 1 WHERE item_id = 1"},
            ],
        }
        if overrides:
            doc["pool"].update(overrides.get("pool", {}))
            if "tasks" in overrides:
                doc["tasks"] = overrides["tasks"]
        return doc

    def test_valid_document_loads(self):
        config = config_from_dict(self._base(), ALLOWED_TABLES)
        self.assertEqual(config.pool.max_connections, 3)
        self.assertEqual(config.tasks[0].task_id, "a")

    def test_quota_sum_exceeding_cap_rejected_with_field_path(self):
        doc = self._base()
        doc["pool"]["workload_quotas"] = {"wl_a": 3, "wl_b": 2}
        with self.assertRaises(BusinessError) as ctx:
            config_from_dict(doc, ALLOWED_TABLES)
        self.assertEqual(ctx.exception.code, ErrorCode.E_CONFIG)
        self.assertIn("pool.workload_quotas.sum", ctx.exception.field_paths)

    def test_non_positive_and_duplicate_task_ids_rejected(self):
        doc = self._base()
        doc["pool"]["max_connections"] = 0
        doc["tasks"].append(dict(doc["tasks"][0]))
        with self.assertRaises(BusinessError) as ctx:
            config_from_dict(doc, ALLOWED_TABLES)
        paths = ctx.exception.field_paths
        self.assertIn("pool.max_connections", paths)
        self.assertTrue(any("task_id" in path for path in paths))

    def test_unknown_workload_and_readonly_write_rejected(self):
        doc = self._base()
        doc["tasks"][0]["workload"] = "wl_unknown"
        doc["tasks"][0]["conn_type"] = "READ_ONLY"
        with self.assertRaises(BusinessError) as ctx:
            config_from_dict(doc, ALLOWED_TABLES)
        paths = " ".join(ctx.exception.field_paths)
        self.assertIn("workload", paths)
        self.assertIn("conn_type", paths)

    def test_invalid_json_file_reports_e_config(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "broken.json"
            path.write_text("{not json", encoding="utf-8")
            with self.assertRaises(BusinessError) as ctx:
                load_config(path, ALLOWED_TABLES)
            self.assertEqual(ctx.exception.code, ErrorCode.E_CONFIG)


if __name__ == "__main__":
    unittest.main()
