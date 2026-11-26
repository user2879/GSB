import unittest

from intent_router.errors import FailureCollector
from intent_router.loader import (
    load_config,
    load_corpus,
    load_revision_chain,
)


class ValidationIsolationTest(unittest.TestCase):
    def test_invalid_texts_do_not_stop_valid_ones(self):
        failures = FailureCollector()
        payload = {
            "vocab": ["a", "b", "c"],
            "texts": [
                {"text_id": "OK1", "tokens": ["a"], "channel": "C", "gold_label": "I"},
                {"text_id": "BAD1", "tokens": ["z"], "channel": "C", "gold_label": "I"},
                {"text_id": "OK2", "tokens": ["b", "c"], "channel": "C", "gold_label": "I"},
                {"text_id": "OK2", "tokens": ["a"], "channel": "C", "gold_label": "I"},
            ],
        }
        corpus = load_corpus(payload, failures)
        self.assertEqual([t.text_id for t in corpus.texts], ["OK1", "OK2"])
        codes = failures.count_by_code()
        self.assertIn("E_TOKEN_OUT_OF_VOCAB", codes)
        self.assertIn("E_DUPLICATE_TEXT_ID", codes)

    def test_missing_revision_parent_reported(self):
        failures = FailureCollector()
        payload = {
            "root_id": "ROOT",
            "head_id": "R1",
            "revisions": [
                {"revision_id": "R1", "parent_id": "GHOST", "sequence": 1, "ops": []}
            ],
        }
        self.assertIsNone(load_revision_chain(payload, failures))
        self.assertIn("E_BAD_REVISION_PARENT", failures.count_by_code())

    def test_bad_budget_and_gear_reported(self):
        failures = FailureCollector()
        cfg = load_config(
            {
                "fraction_bits": 8,
                "gear": 9,
                "max_workers": -1,
                "global_match_budget": 0,
                "per_batch_recompute_limit": 5,
                "channel_quotas": {"C": 5},
                "dedup_threshold_permille": 500,
                "intents": ["IA"],
                "intent_thresholds": {"IA": 1},
                "fallback_intent": "FB",
            },
            failures,
        )
        self.assertIsNone(cfg)
        codes = failures.count_by_code()
        self.assertIn("E_BAD_GEAR", codes)
        self.assertIn("E_BAD_CONCURRENCY", codes)
        self.assertIn("E_BAD_BUDGET", codes)


if __name__ == "__main__":
    unittest.main()
