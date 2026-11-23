from __future__ import annotations

import copy
import json
import tempfile
import unittest
from pathlib import Path

from near_dup.config import validate_config_payload
from near_dup.errors import E_CONFIG_INVALID, E_PLAN_INVALID, E_TEXT_INVALID
from near_dup.plan import load_edit_plan, load_injections
from near_dup.sample_data import BASE_CONFIG, build_injections
from near_dup.text_store import load_text_jsonl
from near_dup.yaml_subset import loads as yaml_loads

from tests.helpers import SampleData


class ValidationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.data = SampleData()

    @classmethod
    def tearDownClass(cls) -> None:
        cls.data.close()

    def test_config_validation_accepts_base(self) -> None:
        config, failures = validate_config_payload(BASE_CONFIG)
        self.assertIsNotNone(config)
        self.assertEqual(failures, [])
        self.assertEqual(config.signature_dimension, 32)
        self.assertEqual(config.threshold_units, 7500)

    def test_config_injection_codes(self) -> None:
        injections, failures = load_injections(str(self.data.root / "samples" / "injections.json"))
        self.assertEqual(failures, [])
        self.assertEqual(len(injections), 4)
        for injection in injections[:3]:
            raw = copy.deepcopy(BASE_CONFIG)
            if injection.kind == "zero_signature_dimension":
                raw["signature_dimension"] = 0
            elif injection.kind == "non_finite_similarity":
                raw["similarity_threshold"] = float("nan")
            elif injection.kind == "bucket_bits_out_of_range":
                raw["bucket_bits"] = 33
            config, failures = validate_config_payload(raw)
            self.assertIsNone(config)
            self.assertTrue(failures)
            self.assertTrue(all(item.code == E_CONFIG_INVALID for item in failures))

    def test_invalid_text_library_keeps_every_failure(self) -> None:
        records, failures = load_text_jsonl(
            str(self.data.root / "samples" / "invalid_texts.jsonl"), self.data.config
        )
        self.assertEqual([record.text_id for record in records], ["T001"])
        self.assertEqual(len(failures), 8)
        self.assertTrue(all(item.code == E_TEXT_INVALID for item in failures))
        messages = " ".join(item.message for item in failures)
        self.assertIn("duplicate text_id", messages)
        self.assertIn("non-empty list", messages)
        self.assertIn("controlled vocabulary", messages)
        self.assertIn("source", messages)
        self.assertIn("monotonically", messages)
        self.assertIn("invalid JSON", messages)

    def test_version_monotonic_across_invalid_entries(self) -> None:
        lines = [
            {"text_id": "A", "terms": ["甲"], "source": "forum", "version": 3},
            {"text_id": "B", "terms": ["未收录"], "source": "forum", "version": 9},
            {"text_id": "C", "terms": ["乙"], "source": "forum", "version": 8},
        ]
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "texts.jsonl"
            path.write_text("\n".join(json.dumps(item, ensure_ascii=False) for item in lines) + "\n", encoding="utf-8")
            records, failures = load_text_jsonl(str(path), self.data.config)
        self.assertEqual([record.text_id for record in records], ["A"])
        self.assertTrue(any("monotonically" in item.message for item in failures))

    def test_invalid_plan_codes(self) -> None:
        _, failures = load_edit_plan(str(self.data.root / "samples" / "invalid_plan.json"), self.data.config)
        codes = {item.code for item in failures}
        self.assertIn(E_PLAN_INVALID, codes)
        self.assertIn(E_TEXT_INVALID, codes)
        self.assertGreaterEqual(sum(1 for item in failures if item.code == E_PLAN_INVALID), 3)

    def test_yaml_subset_rejects_malformed_lines(self) -> None:
        with self.assertRaises(Exception):
            yaml_loads("key: value\n  - dangling\n")

    def test_injection_manifest_loads_all_ids(self) -> None:
        injections, failures = load_injections(str(self.data.root / "samples" / "injections.json"))
        self.assertEqual(failures, [])
        self.assertEqual(
            [item.injection_id for item in injections],
            [item["injection_id"] for item in build_injections()["injections"]],
        )


if __name__ == "__main__":
    unittest.main()
