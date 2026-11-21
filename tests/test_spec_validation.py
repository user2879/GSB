import copy
import json
import os
import tempfile
import unittest

from hashlot.errors import SpecError
from hashlot.spec import load_spec
from hashlot.samplegen import generate


class SpecValidationTest(unittest.TestCase):
    def setUp(self):
        self.doc = generate()
        self.tmp = tempfile.mkdtemp()
        self.path = os.path.join(self.tmp, "spec.json")

    def _write(self, doc):
        with open(self.path, "w", encoding="utf-8") as handle:
            json.dump(doc, handle)
        return self.path

    def test_valid_spec_loads(self):
        spec = load_spec(self._write(self.doc))
        self.assertEqual(spec.topology.partition_count, 8)
        self.assertEqual(len(spec.tables), 2)

    def test_duplicate_table_name(self):
        self.doc["tables"][1]["name"] = self.doc["tables"][0]["name"]
        with self.assertRaises(SpecError) as ctx:
            load_spec(self._write(self.doc))
        self.assertEqual(ctx.exception.code, "E_SPEC")
        self.assertIn("tables[1].name", ctx.exception.location)

    def test_duplicate_column_name(self):
        col = copy.deepcopy(self.doc["tables"][0]["columns"][0])
        self.doc["tables"][0]["columns"].append(col)
        with self.assertRaises(SpecError) as ctx:
            load_spec(self._write(self.doc))
        self.assertEqual(ctx.exception.code, "E_SPEC")
        self.assertIn(".name", ctx.exception.location)

    def test_bitmap_length_mismatch(self):
        self.doc["tables"][0]["columns"][0]["null_bitmap"].pop()
        with self.assertRaises(SpecError) as ctx:
            load_spec(self._write(self.doc))
        self.assertEqual(ctx.exception.code, "E_SPEC")
        self.assertIn("null_bitmap", ctx.exception.location)

    def test_missing_group_key_column(self):
        self.doc["query"]["group_keys"] = ["not_exists"]
        with self.assertRaises(SpecError) as ctx:
            load_spec(self._write(self.doc))
        self.assertEqual(ctx.exception.code, "E_SPEC")
        self.assertIn("group_keys[0]", ctx.exception.location)

    def test_missing_aggregate_column(self):
        self.doc["query"]["aggregates"][1]["column"] = "ghost"
        with self.assertRaises(SpecError) as ctx:
            load_spec(self._write(self.doc))
        self.assertEqual(ctx.exception.code, "E_SPEC")
        self.assertIn("aggregates[1].column", ctx.exception.location)

    def test_self_borrow_rejected(self):
        self.doc["topology"]["borrows"][0]["lender"] = 0
        with self.assertRaises(SpecError) as ctx:
            load_spec(self._write(self.doc))
        self.assertEqual(ctx.exception.code, "E_SPEC")
        self.assertIn("borrower", ctx.exception.location)

    def test_borrow_cycle_rejected(self):
        self.doc["topology"]["borrows"].append(
            {"lender": 0, "borrower": 6, "borrowed_key_cap": 2, "deadline_ms": 10})
        with self.assertRaises(SpecError) as ctx:
            load_spec(self._write(self.doc))
        self.assertEqual(ctx.exception.code, "E_SPEC")
        self.assertEqual(ctx.exception.location, "topology.borrows")

    def test_unsupported_partition_count(self):
        self.doc["topology"]["partition_count"] = 5
        with self.assertRaises(SpecError) as ctx:
            load_spec(self._write(self.doc))
        self.assertEqual(ctx.exception.code, "E_SPEC")
        self.assertIn("partition_count", ctx.exception.location)

    def test_bad_scale(self):
        self.doc["scale"] = 7
        with self.assertRaises(SpecError) as ctx:
            load_spec(self._write(self.doc))
        self.assertEqual(ctx.exception.code, "E_SPEC")
        self.assertEqual(ctx.exception.location, "scale")


if __name__ == "__main__":
    unittest.main()
