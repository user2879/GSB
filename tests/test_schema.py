import os
import tempfile
import unittest

from sft_diag.errors import E_CONFIG_INVALID
from sft_diag.loader import load_config, load_seeds
from sft_diag.schemas import schema_errors, CONFIG_SCHEMA
from tests._helpers import write_json, write_jsonl


class SchemaTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()

    def tearDown(self) -> None:
        self.temp.cleanup()

    def test_schema_reports_type_errors(self) -> None:
        problems = schema_errors(
            CONFIG_SCHEMA,
            {"task_types": "not-a-list", "max_workers": "four"},
        )
        self.assertTrue(problems)
        self.assertTrue(any("task_types" in p for p in problems))
        self.assertTrue(any("max_workers" in p for p in problems))

    def test_loader_uses_schema_and_keeps_failure_code(self) -> None:
        path = os.path.join(self.temp.name, "config.json")
        write_json(path, {"task_types": [], "max_workers": 1,
                          "snapshot_interval_batches": 1})
        failures = []
        self.assertIsNone(load_config(path, failures))
        self.assertTrue(all(f.code == E_CONFIG_INVALID for f in failures))

    def test_seed_schema_wrong_slot_type(self) -> None:
        path = os.path.join(self.temp.name, "seeds.jsonl")
        write_jsonl(path, [
            {"seed_id": "s1", "task_type": "alpha", "template_id": "t1",
             "template": "x", "slots": "not-a-list", "answer_template": "y"},
        ])
        failures = []
        seeds = load_seeds(path, failures)
        self.assertEqual(seeds, {})
        self.assertTrue(failures)


if __name__ == "__main__":
    unittest.main()
