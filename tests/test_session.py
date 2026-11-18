from __future__ import annotations

import sqlite3
import tempfile
import unittest
from pathlib import Path

from slowquery_workbench import Workbench, WorkbenchError

from tests.helpers import SAMPLE_QUERY, new_sample_workbench


class SessionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)

    def tearDown(self) -> None:
        self.temp.cleanup()

    def test_close_reopen_and_monotonic_versions(self) -> None:
        workbench = new_sample_workbench(self.root)
        initial = workbench.versions()
        self.assertEqual(initial["data_version"], 1)
        self.assertEqual(initial["index_version"], 1)
        self.assertEqual(initial["stats_version"], 1)
        appended = workbench.append_rows(
            "orders",
            ["id", "customer_id", "status", "amount", "created_at"],
            [(999991, 1, "paid", 12.5, "2024-12-31")],
        )
        self.assertGreater(appended["versions"]["data_version"], initial["data_version"])
        indexed = workbench.add_index("orders", ["amount"], name="idx_orders_amount_test")
        self.assertGreater(indexed["versions"]["index_version"], initial["index_version"])
        statistics = workbench.refresh_statistics("orders")
        self.assertGreater(statistics["stats_version"], initial["stats_version"])
        path = workbench.path
        final = workbench.versions()
        workbench.close()

        reopened = Workbench.open(path)
        self.assertEqual(reopened.versions()["data_version"], final["data_version"])
        self.assertEqual(reopened.versions()["index_version"], final["index_version"])
        self.assertEqual(reopened.versions()["stats_version"], final["stats_version"])
        reopened.close()

    def test_rollback_and_missing_versions_reject_open_with_e_session(self) -> None:
        workbench = new_sample_workbench(self.root)
        workbench.append_rows(
            "orders",
            ["id", "customer_id", "status", "amount", "created_at"],
            [(999992, 1, "paid", 12.5, "2024-12-31")],
        )
        path = workbench.path
        workbench.close()
        connection = sqlite3.connect(path)
        with connection:
            connection.execute("UPDATE __sw_meta SET value='0' WHERE key='data_version'")
        connection.close()
        with self.assertRaises(WorkbenchError) as captured:
            Workbench.open(path)
        self.assertEqual(captured.exception.code, "E_SESSION")
        self.assertIn("rollback", captured.exception.message)

        connection = sqlite3.connect(path)
        with connection:
            connection.execute("UPDATE __sw_meta SET value='3' WHERE key='data_version'")
            connection.execute("DELETE FROM __sw_meta WHERE key='index_version'")
        connection.close()
        with self.assertRaises(WorkbenchError) as captured:
            Workbench.open(path)
        self.assertEqual(captured.exception.code, "E_SESSION")
        self.assertIn("missing", captured.exception.message)

    def test_observation_ledger_and_threshold_recalculation(self) -> None:
        workbench = new_sample_workbench(self.root)
        execution = workbench.execute_select(SAMPLE_QUERY, timeout_ms=500)
        self.assertGreaterEqual(execution["elapsed_ms"], 0.0)
        self.assertGreater(execution["scanned_rows"], 0)
        self.assertEqual(execution["scanned_rows"], execution["result_rows"])
        raw_before = workbench.execution_history(execution["query_id"])
        self.assertEqual(len(raw_before), 1)

        workbench.set_thresholds(0, 0)
        self.assertEqual(len(workbench.slow_queries()), 1)
        workbench.set_thresholds(2_000_000_000, 2_000_000_000)
        self.assertEqual(workbench.slow_queries(), [])
        raw_after = workbench.execution_history(execution["query_id"])
        self.assertEqual(raw_before[0]["elapsed_ms"], raw_after[0]["elapsed_ms"])
        self.assertEqual(raw_before[0]["scanned_rows"], raw_after[0]["scanned_rows"])
        workbench.close()

    def test_create_table_append_and_clock_declaration(self) -> None:
        workbench = Workbench.create(self.root / "custom.sqlite", name="custom")
        workbench.create_table(
            "CREATE TABLE notes (id INTEGER PRIMARY KEY, body TEXT NOT NULL, priority INTEGER NOT NULL)"
        )
        workbench.append_rows("notes", ["id", "body", "priority"], [(1, "hello", 2), (2, "world", 1)])
        info = workbench.info()
        self.assertIn("monotonic", info["clock_basis"])
        self.assertEqual([item["name"] for item in workbench.schema()], ["notes"])
        self.assertGreaterEqual(workbench.versions()["data_version"], 1)
        workbench.close()


if __name__ == "__main__":
    unittest.main()
