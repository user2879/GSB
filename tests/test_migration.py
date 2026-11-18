from __future__ import annotations

import copy
import json
import tempfile
import unittest
from pathlib import Path

from tests.helpers import (
    MANIFEST_PATH,
    build_base,
    migrate_sample,
    sample_payload,
    working_copy,
    write_manifest,
)

from dbml.analysis import compute_error_report
from dbml.database import connect, table_content_digest
from dbml.errors import MigrationError
from dbml.manifest import load_manifest
from dbml.migrations import MigrationRunner, perturb_scale
from dbml.verify import verify_drift


class MigrationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.directory = Path(self.temp.name)

    def tearDown(self) -> None:
        self.temp.cleanup()

    def test_full_migration_is_batch_independent(self) -> None:
        directory1 = self.directory / "one"
        directory1.mkdir()
        base1, db1, runner1, result1 = migrate_sample(directory1, batch_size=1)
        directory2 = self.directory / "two"
        directory2.mkdir(exist_ok=True)
        base2, db2, runner2, result2 = migrate_sample(directory2, batch_size=19)
        self.assertTrue(result1.complete)
        self.assertEqual(result1.committed_steps, result2.committed_steps)
        self.assertEqual(
            [(step["chain_digest"], step["content_digest"]) for step in result1.ledger["steps"]],
            [(step["chain_digest"], step["content_digest"]) for step in result2.ledger["steps"]],
        )
        drift1 = verify_drift(
            base1,
            db1,
            table="orders",
            columns=["discount_rate", "net_cents", "tax_cents", "total_cents", "unit_price_cents"],
            before_calibrations=runner1.manifest,
            row_key=["id"],
            batch_size=1,
            raise_on_drift=False,
        )
        drift2 = verify_drift(
            base2,
            db2,
            table="orders",
            columns=["discount_rate", "net_cents", "tax_cents", "total_cents", "unit_price_cents"],
            before_calibrations=runner2.manifest,
            row_key=["id"],
            batch_size=48,
            raise_on_drift=False,
        )
        self.assertEqual(drift1["conclusion"], "precision_equivalent")
        self.assertEqual(drift1["report_digest"], drift2["report_digest"])
        error_report = compute_error_report(db1, table="orders", columns=["net_cents", "total_cents"])
        self.assertEqual([item["column"] for item in error_report["columns"]], ["net_cents", "total_cents"])
        self.assertTrue(
            all(item["max_error_bound_units"] == 3 for item in error_report["columns"])
        )
        paths = error_report["columns"][0]["source_paths"]
        self.assertEqual(paths, sorted(paths, key=lambda item: tuple(item["path"])))

    def test_scale_perturbation_recomputes_dependent_chain(self) -> None:
        base, db, runner, _ = migrate_sample(self.directory)
        result = perturb_scale(
            db,
            runner.manifest,
            table="orders",
            column="discount_rate",
            new_scale=2,
        )
        self.assertEqual(result.affected_steps, ["backfill-total", "backfill-net"])
        self.assertIn("orders.net_cents", result.affected_columns)
        self.assertEqual([item["target"] for item in result.report["backfills"]], [
            "orders.total_cents",
            "orders.net_cents",
        ])
        drift = verify_drift(
            base,
            db,
            table="orders",
            columns=["net_cents", "total_cents"],
            before_calibrations=runner.manifest,
            row_key=["id"],
            raise_on_drift=False,
        )
        self.assertEqual(drift["conclusion"], "precision_equivalent")
        self.assertTrue(any(item["nonzero_deviation_rows"] > 0 for item in drift["columns"]))

    def test_integer_to_real_widening(self) -> None:
        base = build_base(self.directory)
        db = working_copy(base, self.directory)
        payload = copy.deepcopy(sample_payload())
        payload["steps"][0]["to_type"] = "REAL"
        manifest_path = write_manifest(self.directory, payload)
        manifest = load_manifest(manifest_path, db)
        result = MigrationRunner(db, manifest, batch_size=11).run()
        self.assertTrue(result.complete)
        connection = connect(db, readonly=True)
        try:
            sql_type = connection.execute(
                "SELECT type FROM pragma_table_info('orders') WHERE name='unit_price_cents'"
            ).fetchone()[0]
            stored = connection.execute(
                "SELECT unit_price_cents FROM orders WHERE id=1"
            ).fetchone()[0]
        finally:
            connection.close()
        self.assertEqual(sql_type, "REAL")
        self.assertIsInstance(stored, float)
        drift = verify_drift(
            base,
            db,
            table="orders",
            columns=["unit_price_cents", "total_cents", "net_cents"],
            before_calibrations=manifest,
            row_key=["id"],
            raise_on_drift=False,
        )
        self.assertEqual(drift["conclusion"], "precision_equivalent")

    def test_create_rename_index_and_constraints(self) -> None:
        base = build_base(self.directory)
        db = working_copy(base, self.directory)
        payload = copy.deepcopy(sample_payload())
        payload["steps"] = [
            {
                "step_id": "create-margin",
                "version": 1,
                "depends_on": [],
                "operation": "create_column",
                "table": "orders",
                "column": "margin_cents",
                "type": "DECIMAL",
                "nullable": True,
                "quantization": {"scale": 2, "truncation": "toward_zero", "saturation": "reject"},
            },
            {
                "step_id": "backfill-margin",
                "version": 2,
                "depends_on": ["create-margin"],
                "operation": "backfill",
                "table": "orders",
                "column": "margin_cents",
                "expression": "total_cents - net_cents",
                "quantization": {"scale": 2, "truncation": "toward_zero", "saturation": "reject"},
            },
            {
                "step_id": "rename-margin",
                "version": 3,
                "depends_on": ["backfill-margin"],
                "operation": "rename_column",
                "table": "orders",
                "column": "margin_cents",
                "new_name": "margin_value",
            },
            {
                "step_id": "index-note",
                "version": 4,
                "depends_on": ["rename-margin"],
                "operation": "create_index",
                "table": "orders",
                "index": {"name": "ix_orders_note", "columns": ["note"], "unique": False},
            },
            {
                "step_id": "unique-note-id",
                "version": 5,
                "depends_on": ["index-note"],
                "operation": "add_constraint",
                "table": "orders",
                "constraint": {
                    "name": "uq_orders_note_id",
                    "kind": "unique",
                    "columns": ["id", "note"],
                },
            },
            {
                "step_id": "check-quantity",
                "version": 6,
                "depends_on": ["unique-note-id"],
                "operation": "add_constraint",
                "table": "orders",
                "constraint": {
                    "name": "ck_orders_margin_nonnegative",
                    "kind": "check",
                    "expression": "margin_value >= 0",
                },
            },
        ]
        manifest_path = write_manifest(self.directory, payload)
        manifest = load_manifest(manifest_path, db)
        result = MigrationRunner(db, manifest).run()
        self.assertTrue(result.complete)
        connection = connect(db, readonly=True)
        try:
            columns = {
                row["name"]
                for row in connection.execute("PRAGMA table_info(orders)")
            }
            index_sql = {
                row["name"]: row["sql"]
                for row in connection.execute(
                    "SELECT name, sql FROM sqlite_master WHERE type='index'"
                )
            }
            create_sql = connection.execute(
                "SELECT sql FROM sqlite_master WHERE type='table' AND name='orders'"
            ).fetchone()[0]
            margin_errors = connection.execute(
                "SELECT COUNT(*) FROM _mig_row_errors WHERE column_name='margin_value'"
            ).fetchone()[0]
        finally:
            connection.close()
        self.assertIn("margin_value", columns)
        self.assertNotIn("margin_cents", columns)
        self.assertIn("ix_orders_note", index_sql)
        self.assertIn("uq_orders_note_id", index_sql)
        self.assertIn("CHECK (margin_value >= 0)", create_sql)
        self.assertEqual(margin_errors, 48)

    def test_out_of_bound_drift_raises_e_drift(self) -> None:
        base, db, runner, _ = migrate_sample(self.directory)
        connection = connect(db)
        try:
            connection.execute("UPDATE orders SET total_cents = total_cents + 1000 WHERE id = 1")
        finally:
            connection.close()
        with self.assertRaises(MigrationError) as raised:
            verify_drift(
                base,
                db,
                table="orders",
                columns=["total_cents"],
                before_calibrations=runner.manifest,
                row_key=["id"],
            )
        self.assertEqual(raised.exception.code, "E_DRIFT")
        self.assertEqual(raised.exception.details["columns"][0]["first_out_of_bound"]["row_key"], "1")

    def test_invalid_runner_manifest_does_not_execute_partially(self) -> None:
        base = build_base(self.directory)
        db = working_copy(base, self.directory)
        payload = copy.deepcopy(sample_payload())
        payload["steps"][1]["expression"] = "missing * quantity"
        manifest_path = write_manifest(self.directory, payload)
        with self.assertRaises(MigrationError) as raised:
            MigrationRunner(db, manifest_path).run()
        self.assertEqual(raised.exception.code, "E_MIGRATE")
        connection = connect(db, readonly=True)
        try:
            has_run = connection.execute(
                "SELECT 1 FROM sqlite_master WHERE type='table' AND name='_mig_run'"
            ).fetchone()
            columns = {
                row["name"] for row in connection.execute("PRAGMA table_info(orders)")
            }
        finally:
            connection.close()
        self.assertIsNone(has_run)
        self.assertNotIn("margin_cents", columns)


if __name__ == "__main__":
    unittest.main()
