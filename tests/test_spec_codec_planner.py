import struct
import unittest

from gsb.codec import encode_block, encode_dict, generate_column_values
from gsb.errors import E_SPEC, SpecError
from gsb.hashing import canonical_bytes, canonical_hash
from gsb.planner import compile_query_plan
from gsb.spec import load_spec, normalize_spec


def group(group_id="BG-1", rows=4, encoding="raw", compression="none"):
    return {
        "id": group_id,
        "row_count": rows,
        "columns": [
            {"name": "a", "seed": 7, "modulus": 100, "encoding": encoding, "compression": compression},
            {"name": "b", "seed": 9, "modulus": 4, "encoding": "dict", "compression": "none"},
        ],
        "skip_index": {
            "a": {"min": 0, "max": 99, "null_count": 0},
            "b": {"min": 0, "max": 3, "null_count": 0},
        },
    }


class SpecValidationTests(unittest.TestCase):
    def test_common_field_aliases_normalize(self):
        raw = {
            "blockGroups": [
                {
                    "blockGroupId": "BG-1",
                    "rows": 3,
                    "cols": [{"name": "a", "encoding": "dictionary", "compression": "zlib"}],
                    "statistics": {"a": {"min": 1, "max": 2, "nullCount": 0}},
                }
            ],
            "querySet": [{"id": "q", "where": [{"column": "a", "op": "eq", "value": 1}]}],
        }
        spec = normalize_spec(raw)
        self.assertEqual(spec["block_groups"][0]["id"], "BG-1")
        self.assertEqual(spec["block_groups"][0]["row_count"], 3)
        self.assertEqual(spec["block_groups"][0]["columns"][0]["encoding"], "dict")
        self.assertEqual(spec["block_groups"][0]["columns"][0]["compression"], "zlib-6")
        self.assertEqual(spec["queries"][0]["predicates"][0]["op"], "=")

    def test_duplicate_block_group_is_first_structural_error(self):
        raw = {"block_groups": [group("BG-DUP"), group("BG-DUP")], "queries": []}
        with self.assertRaises(SpecError) as ctx:
            load_spec(raw)
        self.assertEqual(ctx.exception.code, E_SPEC)
        self.assertEqual(ctx.exception.block_group_id, "BG-DUP")
        self.assertIn("id", ctx.exception.field)

    def test_rejects_invalid_rows_encoding_compression_and_index(self):
        cases = [
            ({**group("BG-ROWS"), "row_count": 0}, "row_count"),
            ({**group("BG-ENC"), "columns": [{"name": "a", "encoding": "nope", "compression": "none"}]}, "encoding"),
            (
                {**group("BG-ZIP"), "columns": [{"name": "a", "encoding": "raw", "compression": "bzip-9"}]},
                "compression",
            ),
            (
                {
                    **group("BG-IDX"),
                    "skip_index": {"a": {"min": 9, "max": 1, "null_count": 0}, "b": {"min": 0, "max": 1, "null_count": 0}},
                },
                "skip_index.a.min",
            ),
            (
                {
                    **group("BG-NULL"),
                    "skip_index": {"a": {"min": 0, "max": 1, "null_count": 99}, "b": {"min": 0, "max": 1, "null_count": 0}},
                },
                "null_count",
            ),
        ]
        for raw_group, expected_field in cases:
            with self.subTest(field=expected_field):
                with self.assertRaises(SpecError) as ctx:
                    load_spec({"block_groups": [raw_group], "queries": []})
                self.assertEqual(ctx.exception.code, E_SPEC)
                self.assertEqual(ctx.exception.block_group_id, raw_group["id"])
                self.assertIn(expected_field, ctx.exception.field)

    def test_rejects_missing_predicate_column_in_first_bg(self):
        first = group("BG-FIRST")
        second = group("BG-SECOND")
        second["columns"].append({"name": "missing", "seed": 1, "modulus": 3})
        second["skip_index"]["missing"] = {"min": 0, "max": 2, "null_count": 0}
        raw = {
            "block_groups": [first, second],
            "queries": [{"id": "q", "predicates": [{"column": "missing", "op": "=", "value": 1}]}],
        }
        with self.assertRaises(SpecError) as ctx:
            load_spec(raw)
        self.assertEqual(ctx.exception.block_group_id, "BG-FIRST")
        self.assertIn("predicates", ctx.exception.field)

    def test_rejects_sampling_verification(self):
        raw = {"block_groups": [group()], "queries": [], "verification": {"mode": "sample"}}
        with self.assertRaises(SpecError) as ctx:
            load_spec(raw)
        self.assertEqual(ctx.exception.code, E_SPEC)
        self.assertIn("verification.mode", ctx.exception.field)


class CodecTests(unittest.TestCase):
    def test_block_bytes_are_independent_of_declaration_order(self):
        first = normalize_spec({"block_groups": [group("BG-X", encoding="dict", compression="zlib-1")], "queries": []})["block_groups"][0]
        second = normalize_spec({"block_groups": [group("BG-X", encoding="dict", compression="zlib-1")], "queries": []})["block_groups"][0]
        block_a = encode_block(first)
        block_b = encode_block(second)
        self.assertEqual(block_a["bytes"], block_b["bytes"])
        self.assertEqual(block_a["content_digest"], block_b["content_digest"])
        self.assertEqual(block_a["logical_digest"], block_b["logical_digest"])

    def test_compression_changes_bytes_not_decoded_rows(self):
        values = {"a": [1, 1, 2, None], "b": [0, 1, 2, 3]}
        uncompressed_group = normalize_spec({"block_groups": [group("BG-C", rows=4, compression="none")], "queries": []})["block_groups"][0]
        compressed_group = normalize_spec({"block_groups": [group("BG-C", rows=4, compression="zlib-6")], "queries": []})["block_groups"][0]
        raw_block = encode_block(uncompressed_group, values)
        zlib_block = encode_block(compressed_group, values)
        self.assertNotEqual(raw_block["bytes"], zlib_block["bytes"])
        self.assertNotEqual(raw_block["content_digest"], zlib_block["content_digest"])
        self.assertEqual(raw_block["logical_digest"], zlib_block["logical_digest"])
        self.assertEqual(raw_block["columns"], zlib_block["columns"])

    def test_dictionary_entries_are_sorted(self):
        payload = encode_dict([3, 1, 2, None, 1])
        size = struct.unpack_from(">I", payload, 0)[0]
        dictionary = [struct.unpack_from(">q", payload, 4 + i * 8)[0] for i in range(size)]
        self.assertEqual(size, 3)
        self.assertEqual(dictionary, [1, 2, 3])

    def test_seed_generation_is_stable(self):
        column = {"name": "a", "seed": 12345, "modulus": 97, "offset": 3, "null_every": 5}
        first = generate_column_values(column, 20)
        second = generate_column_values(column, 20)
        self.assertEqual(first, second)
        self.assertEqual(canonical_hash(first), canonical_hash(second))
        self.assertEqual(first[4], None)
        self.assertTrue(all(value is None or 3 <= value < 100 for value in first))


class PlannerTests(unittest.TestCase):
    def test_plan_digest_is_independent_of_predicate_and_group_order(self):
        groups = [
            {
                "id": "BG-1",
                "row_count": 4,
                "columns": [{"name": "a"}, {"name": "b"}],
                "skip_index": {"a": {"min": 0, "max": 10, "null_count": 0}, "b": {"min": 0, "max": 3, "null_count": 0}},
            },
            {
                "id": "BG-2",
                "row_count": 4,
                "columns": [{"name": "a"}, {"name": "b"}],
                "skip_index": {"a": {"min": 20, "max": 30, "null_count": 0}, "b": {"min": 0, "max": 3, "null_count": 0}},
            },
        ]
        query_a = {"id": "q", "predicates": [{"column": "b", "op": "=", "value": 2}, {"column": "a", "op": ">=", "value": 15}]}
        query_b = {"id": "q", "predicates": list(reversed(query_a["predicates"]))}
        plan_a = compile_query_plan(query_a, groups)
        plan_b = compile_query_plan(query_b, list(reversed(groups)))
        self.assertEqual(plan_a["digest"], plan_b["digest"])
        self.assertEqual(canonical_bytes(plan_a), canonical_bytes(plan_b))
        self.assertEqual(plan_a["skip_block_groups"], ["BG-1"])
        self.assertEqual(plan_a["verify_block_groups"], ["BG-2"])


if __name__ == "__main__":
    unittest.main()
