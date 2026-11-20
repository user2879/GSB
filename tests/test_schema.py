import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dhj.errors import SchemaError
from dhj.jsonio import read_json
from dhj.schema import load_spec

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def base_spec():
    return read_json(os.path.join(ROOT, "data", "sample.json"))


class SchemaTests(unittest.TestCase):
    def test_valid_spec_loads(self):
        spec = load_spec(base_spec())
        self.assertEqual(set(spec.tables), {"orders", "customers"})
        self.assertEqual(spec.query.join_key, "cust_key")
        self.assertTrue(spec.source_digest)

    def test_bad_identifier_collected(self):
        doc = base_spec()
        doc["tables"]["orders"]["columns"][0]["name"] = "bad-name"
        with self.assertRaises(SchemaError) as ctx:
            load_spec(doc)
        self.assertEqual(ctx.exception.code, "E_SCHEMA")
        self.assertTrue(
            any("bad-name" in path for path in ctx.exception.field_paths)
        )

    def test_duplicate_table_column(self):
        doc = base_spec()
        doc["tables"]["orders"]["columns"][1]["name"] = "order_id"
        with self.assertRaises(SchemaError) as ctx:
            load_spec(doc)
        self.assertTrue(any("duplicate" in p for p in ctx.exception.field_paths))

    def test_null_bitmap_length_mismatch(self):
        doc = base_spec()
        doc["tables"]["orders"]["null_bitmap"].append(1)
        with self.assertRaises(SchemaError) as ctx:
            load_spec(doc)
        self.assertTrue(
            any("null_bitmap" in p for p in ctx.exception.field_paths)
        )

    def test_join_key_missing_on_side(self):
        doc = base_spec()
        doc["query"]["join_key"] = "missing"
        with self.assertRaises(SchemaError) as ctx:
            load_spec(doc)
        self.assertTrue(any("join_key" in p for p in ctx.exception.field_paths))

    def test_unknown_projection_and_predicate(self):
        doc = base_spec()
        doc["query"]["projections"].append("orders.nope")
        doc["query"]["predicates"]["left"][0]["ref"] = "orders.nope"
        with self.assertRaises(SchemaError) as ctx:
            load_spec(doc)
        paths = " ".join(ctx.exception.field_paths)
        self.assertIn("nope", paths)

    def test_type_mismatch_value(self):
        doc = base_spec()
        doc["query"]["predicates"]["right"][0]["value"] = "x"
        with self.assertRaises(SchemaError):
            load_spec(doc)

    def test_multiple_errors_reported_together(self):
        doc = base_spec()
        doc["tables"]["orders"]["columns"][0]["name"] = "1bad"
        doc["tables"]["customers"]["null_bitmap"] = [1]
        with self.assertRaises(SchemaError) as ctx:
            load_spec(doc)
        self.assertGreaterEqual(len(ctx.exception.field_paths), 2)


if __name__ == "__main__":
    unittest.main()
