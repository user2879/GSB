import copy
import json
import os
import tempfile
import unittest

from coljoin.errors import SpecError
from coljoin.loader import load_block_groups, load_query, load_session, load_weights

VALID_GROUP = {
    "bg_id": "t1",
    "rows": 4,
    "seed": 1,
    "null_bitmap": [0, 0, 0, 0],
    "columns": [
        {
            "name": "id",
            "dtype": "integer",
            "encoding": "plain",
            "null_count": 0,
            "ndv": 4,
            "histogram": [
                {"low": 0, "high": 2, "count": 2},
                {"low": 2, "high": 4, "count": 2},
            ],
        }
    ],
}


def write_json(payload):
    handle = tempfile.NamedTemporaryFile("w", suffix=".json", delete=False)
    json.dump(payload, handle)
    handle.close()
    return handle.name


class SpecValidationTest(unittest.TestCase):
    def test_duplicate_bg_id_rejected(self):
        payload = {"block_groups": [VALID_GROUP, copy.deepcopy(VALID_GROUP)]}
        path = write_json(payload)
        with self.assertRaises(SpecError) as ctx:
            load_block_groups(path)
        self.assertEqual(ctx.exception.code, "E_SPEC")
        self.assertEqual(ctx.exception.field, "bg_id")
        os.unlink(path)

    def test_null_bitmap_length_mismatch(self):
        bad = copy.deepcopy(VALID_GROUP)
        bad["null_bitmap"] = [0, 1]
        path = write_json({"block_groups": [bad]})
        with self.assertRaises(SpecError) as ctx:
            load_block_groups(path)
        self.assertEqual(ctx.exception.field, "null_bitmap")
        os.unlink(path)

    def test_histogram_gap_and_overlap(self):
        bad = copy.deepcopy(VALID_GROUP)
        bad["columns"][0]["histogram"][1]["low"] = 3
        path = write_json({"block_groups": [bad]})
        with self.assertRaises(SpecError) as ctx:
            load_block_groups(path)
        self.assertIn("histogram", ctx.exception.field)
        os.unlink(path)

    def test_histogram_count_sum(self):
        bad = copy.deepcopy(VALID_GROUP)
        bad["columns"][0]["histogram"][0]["count"] = 1
        path = write_json({"block_groups": [bad]})
        with self.assertRaises(SpecError) as ctx:
            load_block_groups(path)
        self.assertEqual(ctx.exception.field, "columns.id.histogram")
        os.unlink(path)

    def test_bad_dtype_and_encoding(self):
        bad = copy.deepcopy(VALID_GROUP)
        bad["columns"][0]["dtype"] = "string"
        path = write_json({"block_groups": [bad]})
        with self.assertRaises(SpecError) as ctx:
            load_block_groups(path)
        self.assertIn("dtype", ctx.exception.field)
        os.unlink(path)

    def test_weight_sum_window(self):
        path = write_json(
            {
                "scan": 1,
                "compare": 1,
                "build": 1,
                "probe": 1,
                "memory": 1,
                "weight_sum_min": 100,
                "weight_sum_max": 200,
            }
        )
        with self.assertRaises(SpecError) as ctx:
            load_weights(path)
        self.assertEqual(ctx.exception.code, "E_SPEC")
        os.unlink(path)

    def test_query_unknown_column(self):
        groups_path = write_json({"block_groups": [VALID_GROUP]})
        groups = load_block_groups(groups_path)
        query = {
            "query_id": "q",
            "joins": [],
            "filters": [{"bg": "t1", "column": "missing", "op": "eq", "value": 1}],
        }
        query_path = write_json(query)
        with self.assertRaises(SpecError) as ctx:
            load_query(query_path, groups)
        self.assertEqual(ctx.exception.code, "E_SPEC")
        os.unlink(groups_path)
        os.unlink(query_path)

    def test_session_scale_positive(self):
        path = write_json({"scale": 0, "block_granularity": 128})
        with self.assertRaises(SpecError):
            load_session(path)
        os.unlink(path)


if __name__ == "__main__":
    unittest.main()
