from __future__ import annotations

import copy
import tempfile
import unittest
from pathlib import Path

from bucketpack.canonical import dumps
from bucketpack.errors import E_SPEC, BusinessError
from bucketpack.spec import load_spec, validate_spec

from .common import ROOT, sample_raw, tiny_raw


class SpecValidationTests(unittest.TestCase):
    def test_sample_validates(self) -> None:
        spec = load_spec(ROOT / "samples" / "spec.json")
        self.assertEqual([source.id for source in spec.sources], ["events", "orders"])
        self.assertEqual(len(spec.rules), 9)
        self.assertTrue(spec.digest)

    def test_bitmap_length_is_first_stable_error(self) -> None:
        raw = sample_raw()
        raw["sources"][0]["null_bitmaps"]["spend"] = "0"
        raw["rules"][0]["scale"] = 99
        with self.assertRaises(BusinessError) as caught:
            validate_spec(raw)
        error = caught.exception
        self.assertEqual(error.code, E_SPEC)
        self.assertEqual(error.path, "$.sources[0].null_bitmaps.spend")
        self.assertIn("length", error.message)

    def test_boundaries_must_be_strictly_increasing(self) -> None:
        raw = tiny_raw()
        raw["rules"][0]["boundaries"] = [1, 1]
        with self.assertRaises(BusinessError) as caught:
            validate_spec(raw)
        self.assertEqual(caught.exception.path, "$.rules[0].boundaries[1]")

    def test_scale_bool_is_rejected(self) -> None:
        raw = tiny_raw()
        raw["rules"][0]["scale"] = True
        with self.assertRaises(BusinessError) as caught:
            validate_spec(raw)
        self.assertEqual(caught.exception.path, "$.rules[0].scale")

    def test_duplicate_source_id_rejected(self) -> None:
        raw = tiny_raw()
        duplicate = copy.deepcopy(raw["sources"][0])
        raw["sources"].append(duplicate)
        with self.assertRaises(BusinessError) as caught:
            validate_spec(raw)
        self.assertEqual(caught.exception.path, "$.sources[1].id")

    def test_spec_digest_is_order_independent(self) -> None:
        raw_a = sample_raw()
        raw_b = sample_raw()
        raw_b["rules"].reverse()
        raw_b["sources"].reverse()
        raw_b["statistics"].reverse()
        spec_a = validate_spec(raw_a)
        spec_b = validate_spec(raw_b)
        self.assertEqual(spec_a.digest, spec_b.digest)
        self.assertEqual(dumps(spec_a.to_dict()), dumps(spec_b.to_dict()))

    def test_duplicate_json_key_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "dup.json"
            path.write_text('{"format_version":1,"format_version":1}', encoding="utf-8")
            with self.assertRaises(BusinessError) as caught:
                load_spec(path)
        self.assertEqual(caught.exception.code, E_SPEC)
        self.assertIn("duplicate object key", caught.exception.message)

    def test_unknown_field_rejected(self) -> None:
        raw = tiny_raw()
        raw["unexpected"] = 1
        with self.assertRaises(BusinessError) as caught:
            validate_spec(raw)
        self.assertEqual(caught.exception.path, "$.unexpected")

    def test_overflowing_float_bound_is_spec_error(self) -> None:
        raw = tiny_raw()
        raw["sources"][0]["columns"].append(
            {"name": "f", "type": "float", "min": 10**400, "max": 20.0}
        )
        with self.assertRaises(BusinessError) as caught:
            validate_spec(raw)
        self.assertEqual(caught.exception.code, E_SPEC)
        self.assertEqual(caught.exception.path, "$.sources[0].columns[2].min")

    def test_width_bits_must_cover_bucket_ids(self) -> None:
        raw = tiny_raw()
        raw["rules"][0]["width_bits"] = 1
        with self.assertRaises(BusinessError) as caught:
            validate_spec(raw)
        self.assertEqual(caught.exception.path, "$.rules[0].width_bits")


if __name__ == "__main__":
    unittest.main()
