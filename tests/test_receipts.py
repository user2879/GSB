import json
import os
import tempfile
import unittest

from coljoin.errors import FormatError
from coljoin.receipts import encode_receipt, load_receipts

COST_VECTOR = {
    "total_cost": 1000,
    "access_rows": 10,
    "compares": 20,
    "peak_memory": 30,
}
AGG = {"g1": {"count": 2, "sum": 5}}


def write_lines(lines):
    handle = tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False)
    handle.write("\n".join(lines))
    handle.close()
    return handle.name


class ReceiptNormalizationTest(unittest.TestCase):
    def test_q1_defaults_filled_with_basis(self):
        line = encode_receipt("q1", {"shard_id": "s1", "aggregate": AGG})
        path = write_lines([line])
        records = load_receipts(path, COST_VECTOR)
        record = records[0]
        self.assertEqual(record["_source_format"], "q1")
        filled = {item["field"] for item in record["_filled_fields"]}
        self.assertEqual(
            filled,
            {
                "plan_version",
                "elapsed_ms",
                "executor_id",
                "cost_vector",
                "clock_start_ms",
                "clock_end_ms",
            },
        )
        self.assertTrue(all(item["basis"] for item in record["_filled_fields"]))
        self.assertEqual(record["cost_vector"], COST_VECTOR)
        os.unlink(path)

    def test_q2_partial_fill(self):
        line = encode_receipt(
            "q2",
            {
                "shard_id": "s1",
                "aggregate": AGG,
                "plan_version": 3,
                "elapsed_ms": 12,
            },
        )
        path = write_lines([line])
        record = load_receipts(path, COST_VECTOR)[0]
        self.assertEqual(record["_source_format"], "q2")
        self.assertEqual(record["plan_version"], 3)
        self.assertEqual(record["elapsed_ms"], 12)
        filled = {item["field"] for item in record["_filled_fields"]}
        self.assertEqual(
            filled,
            {"executor_id", "cost_vector", "clock_start_ms", "clock_end_ms"},
        )
        os.unlink(path)

    def test_unknown_version_reports_line(self):
        path = write_lines([json.dumps({"version": "q9", "shard_id": "s"})])
        with self.assertRaises(FormatError) as ctx:
            load_receipts(path)
        self.assertEqual(ctx.exception.code, "E_FORMAT")
        self.assertEqual(ctx.exception.line, 1)
        self.assertEqual(ctx.exception.field, "version")
        os.unlink(path)

    def test_negative_elapsed_rejected(self):
        line = json.dumps(
            {
                "version": "q2",
                "shard_id": "s1",
                "aggregate": AGG,
                "plan_version": 1,
                "elapsed_ms": -5,
            }
        )
        path = write_lines([line])
        with self.assertRaises(FormatError) as ctx:
            load_receipts(path)
        self.assertEqual(ctx.exception.field, "elapsed_ms")
        self.assertEqual(ctx.exception.line, 1)
        os.unlink(path)

    def test_cost_vector_length_mismatch(self):
        line = json.dumps(
            {
                "version": "q3",
                "shard_id": "s1",
                "aggregate": AGG,
                "plan_version": 1,
                "elapsed_ms": 1,
                "executor_id": "e1",
                "cost_vector": {"only_one": 1},
                "clock_start_ms": 1,
                "clock_end_ms": 2,
            }
        )
        path = write_lines([line])
        with self.assertRaises(FormatError) as ctx:
            load_receipts(path, COST_VECTOR)
        self.assertEqual(ctx.exception.field, "cost_vector")
        os.unlink(path)

    def test_missing_required_field_line_number(self):
        good = encode_receipt(
            "q2",
            {
                "shard_id": "s1",
                "aggregate": AGG,
                "plan_version": 1,
                "elapsed_ms": 1,
            },
        )
        bad = json.dumps({"version": "q2", "shard_id": "s2"})
        path = write_lines([good, bad])
        with self.assertRaises(FormatError) as ctx:
            load_receipts(path)
        self.assertEqual(ctx.exception.line, 2)
        os.unlink(path)


if __name__ == "__main__":
    unittest.main()
