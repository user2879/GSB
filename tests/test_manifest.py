from __future__ import annotations

import copy
import tempfile
import unittest
from pathlib import Path

from tests.helpers import build_base, sample_payload

from dbml.errors import MigrationError
from dbml.manifest import validate_manifest


class ManifestValidationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.directory = Path(self.temp.name)
        self.base = build_base(self.directory)
        self.payload = sample_payload()

    def tearDown(self) -> None:
        self.temp.cleanup()

    def assert_invalid(self, payload: dict, *, step_id: str, field_prefix: str) -> None:
        with self.assertRaises(MigrationError) as raised:
            validate_manifest(payload, db_path=self.base)
        error = raised.exception
        self.assertEqual(error.code, "E_MIGRATE")
        self.assertEqual(error.step_id, step_id)
        self.assertTrue(error.field_path and error.field_path.startswith(field_prefix), error.field_path)

    def test_valid_manifest_and_stable_chain(self) -> None:
        manifest = validate_manifest(self.payload, db_path=self.base)
        self.assertEqual(
            manifest.execution_order,
            (
                "widen-unit-price",
                "backfill-total",
                "backfill-net",
                "index-customer",
                "constraint-quantity",
            ),
        )
        self.assertEqual(
            manifest.derivation_edges,
            (("orders.total_cents", "orders.net_cents"),),
        )

    def test_duplicate_step_id(self) -> None:
        payload = copy.deepcopy(self.payload)
        payload["steps"][1]["step_id"] = payload["steps"][0]["step_id"]
        self.assert_invalid(payload, step_id="widen-unit-price", field_prefix="steps[1].step_id")

    def test_versions_strictly_increase(self) -> None:
        payload = copy.deepcopy(self.payload)
        payload["steps"][1]["version"] = payload["steps"][0]["version"]
        self.assert_invalid(payload, step_id="backfill-total", field_prefix="steps[1].version")

    def test_missing_predecessor(self) -> None:
        payload = copy.deepcopy(self.payload)
        payload["steps"][1]["depends_on"] = ["does-not-exist"]
        self.assert_invalid(payload, step_id="backfill-total", field_prefix="steps[1].depends_on")

    def test_dependency_cycle(self) -> None:
        payload = copy.deepcopy(self.payload)
        payload["steps"][0]["depends_on"] = ["backfill-net"]
        self.assert_invalid(payload, step_id="widen-unit-price", field_prefix="steps[0].depends_on")

    def test_unsupported_type(self) -> None:
        payload = copy.deepcopy(self.payload)
        payload["steps"][0]["to_type"] = "BLOB"
        self.assert_invalid(payload, step_id="widen-unit-price", field_prefix="steps[0].to_type")

    def test_scale_out_of_range(self) -> None:
        payload = copy.deepcopy(self.payload)
        payload["steps"][1]["quantization"]["scale"] = 10
        self.assert_invalid(payload, step_id="backfill-total", field_prefix="steps[1].quantization.scale")

    def test_backfill_reference_must_exist_at_execution(self) -> None:
        payload = copy.deepcopy(self.payload)
        payload["steps"][1]["expression"] = "missing_column * quantity"
        self.assert_invalid(payload, step_id="backfill-total", field_prefix="steps[1].expression")

    def test_invalid_manifest_never_creates_ledger_tables(self) -> None:
        payload = copy.deepcopy(self.payload)
        payload["steps"][1]["quantization"]["scale"] = 99
        with self.assertRaises(MigrationError):
            validate_manifest(payload, db_path=self.base)
        import sqlite3

        connection = sqlite3.connect(self.base)
        try:
            names = {
                row[0]
                for row in connection.execute(
                    "SELECT name FROM sqlite_master WHERE type='table'"
                )
            }
        finally:
            connection.close()
        self.assertFalse(any(name.startswith("_mig_") for name in names))


if __name__ == "__main__":
    unittest.main()

