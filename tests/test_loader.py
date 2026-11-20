import json
import os
import tempfile
import unittest

from sft_diag.errors import (
    E_CONFIG_INVALID,
    E_PARAMS_INVALID,
    E_SEED_INVALID,
)
from sft_diag.loader import (
    load_config,
    load_score_params,
    load_seeds,
    load_slot_sources,
    load_vocab,
)
from tests._helpers import write_json, write_jsonl


class LoaderTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.root = self.temp.name

    def tearDown(self) -> None:
        self.temp.cleanup()

    def test_seed_errors_do_not_stop_loading(self) -> None:
        path = os.path.join(self.root, "seeds.jsonl")
        write_jsonl(path, [
            {
                "seed_id": "ok-1", "task_type": "alpha", "template_id": "t1",
                "template": "hi {subject}", "slots": ["subject"],
                "answer_template": "ans {subject}",
            },
        ])
        with open(path, "a", encoding="utf-8") as handle:
            handle.write("{not json\n")
            handle.write(json.dumps({
                "seed_id": "bad-1", "task_type": "alpha", "template_id": "t2",
                "template": "no slot marker", "slots": ["subject"],
                "answer_template": "{subject}",
            }) + "\n")
            handle.write(json.dumps({
                "seed_id": "ok-1", "task_type": "alpha", "template_id": "t3",
                "template": "dup id {x}", "slots": ["x"],
                "answer_template": "{x}",
            }) + "\n")
        failures = []
        seeds = load_seeds(path, failures)
        self.assertEqual(list(seeds), ["ok-1"])
        self.assertTrue(all(item.code == E_SEED_INVALID for item in failures))
        self.assertEqual(len(failures), 3)
        self.assertEqual(failures[0].ref.split(":")[-1], "L2")

    def test_config_weight_and_target_validation(self) -> None:
        path = os.path.join(self.root, "config.json")
        write_json(path, {
            "task_types": ["alpha"],
            "target_counts": {"alpha": -1},
            "template_weights": {"alpha": {"t1": 0.0, "t2": 0.0}},
            "slot_weights": {"alpha": {"s1": 1.0}},
            "dedup_threshold": 1.5,
            "boundary_rule": "merge",
            "max_workers": 0,
            "snapshot_interval_batches": 1,
        })
        failures = []
        self.assertIsNone(load_config(path, failures))
        messages = " | ".join(item.message for item in failures)
        self.assertTrue(all(item.code == E_CONFIG_INVALID for item in failures))
        self.assertIn("non-negative integer", messages)
        self.assertIn("sum to > 0", messages)
        self.assertIn("[0.0, 1.0]", messages)
        self.assertIn("max_workers", messages)

    def test_params_validation(self) -> None:
        path = os.path.join(self.root, "params.json")
        write_json(path, {
            "dimensions": ["clarity", "correctness", "format", "length"],
            "specs": {
                "clarity": {"weight": 0.0, "baseline": 0.5},
                "correctness": {"weight": 0.0, "baseline": 0.5},
                "format": {"weight": 0.0, "baseline": 0.5},
                "length": {"weight": 0.0, "baseline": 0.5},
            },
            "length_lower": 20,
            "length_upper": 5,
        })
        failures = []
        self.assertIsNone(load_score_params(path, failures))
        self.assertTrue(all(item.code == E_PARAMS_INVALID for item in failures))
        messages = " | ".join(item.message for item in failures)
        self.assertIn("weights must sum to > 0", messages)
        self.assertIn("length_lower", messages)

    def test_vocab_duplicates_rejected(self) -> None:
        path = os.path.join(self.root, "vocab.json")
        write_json(path, {"tokens": ["a", "a"], "rules": {}})
        failures = []
        vocab = load_vocab(path, failures)
        self.assertEqual(vocab.tokens, ())
        self.assertTrue(failures)

    def test_slot_source_duplicate_ids(self) -> None:
        path = os.path.join(self.root, "slots.jsonl")
        write_jsonl(path, [
            {"source_id": "s1", "task_type": "alpha", "slot_name": "x",
             "values": ["v1"]},
            {"source_id": "s1", "task_type": "alpha", "slot_name": "y",
             "values": ["v2"]},
        ])
        failures = []
        sources = load_slot_sources(path, failures)
        self.assertEqual(len(sources), 1)
        self.assertEqual(failures[0].code, E_SEED_INVALID)


if __name__ == "__main__":
    unittest.main()
