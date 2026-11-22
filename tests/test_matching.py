from __future__ import annotations

import unittest
from pathlib import Path

from search_intent_workbench.entities import find_candidates, resolve_candidates
from search_intent_workbench.intent import classify_intent
from search_intent_workbench.loader import load_dataset
from search_intent_workbench.pipeline import Workbench


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "data" / "base"


class MatchingAndScoringTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.dataset = load_dataset(
            BASE / "queries.jsonl",
            BASE / "intent_config.json",
            BASE / "normalization_rules.json",
        )
        cls.pipeline_run = Workbench(
            cls.dataset.config, cls.dataset.rules, max_workers=4
        ).run(
            cls.dataset.queries,
            input_failures=cls.dataset.failures,
        )
        cls.by_id = {
            query["query_id"]: query for query in cls.pipeline_run["queries"]
        }

    def test_case_width_and_whitespace_normalization(self) -> None:
        self.assertEqual(
            self.by_id["q002"]["normalization"]["normalized_text"],
            "usb type-c 快充 数据线",
        )
        self.assertEqual(
            self.by_id["q015"]["normalization"]["normalized_text"], "无线 耳机"
        )
        self.assertEqual(
            self.by_id["q001"]["normalization"]["normalized_text"],
            "apple iphone 15 手机 壳",
        )

    def test_longest_match_suppresses_overlaps_with_positions(self) -> None:
        record = self.by_id["q001"]
        accepted = [(slot["entry_id"], slot["token_start"]) for slot in record["slots"]["slots"]]
        self.assertEqual(
            accepted,
            [("e_apple_iphone15", 0), ("e_phone_case", 3)],
        )
        suppressed = record["slots"]["suppressed"]
        by_entry = {slot["entry_id"]: slot for slot in suppressed}
        self.assertEqual(by_entry["e_iphone15"]["token_start"], 1)
        self.assertEqual(by_entry["e_iphone15"]["token_length"], 2)
        self.assertEqual(by_entry["e_apple"]["token_start"], 0)
        self.assertEqual(
            by_entry["e_iphone15"]["overlaps_slot_entry_ids"], ["e_apple_iphone15"]
        )

    def test_equal_length_priority_then_start_resolution(self) -> None:
        patterns = [
            {"entry_index": 0, "entry_id": "low", "dictionary_term": "a b", "normalized_term": "a b", "pattern_tokens": ["a", "b"], "entity_type": "x", "priority": 10.0},
            {"entry_index": 1, "entry_id": "high", "dictionary_term": "a b", "normalized_term": "a b", "pattern_tokens": ["a", "b"], "entity_type": "x", "priority": 20.0},
        ]
        candidates = find_candidates(["a", "b"], patterns)
        accepted, suppressed = resolve_candidates(candidates)
        self.assertEqual([slot["entry_id"] for slot in accepted], ["high"])
        self.assertEqual([slot["entry_id"] for slot in suppressed], ["low"])

    def test_softmax_quantization_metadata_and_lexicographic_tie(self) -> None:
        result = classify_intent([], self.dataset.config)
        self.assertEqual(result["primary_intent"], "browsing")
        self.assertEqual(result["normalization"]["decimal_places"], 6)
        self.assertEqual(result["normalization"]["quantum"], "0.000001")
        self.assertTrue(all(len(item["score_formatted"].split(".")[1]) == 6 for item in result["intent_scores"]))
        self.assertEqual(
            [item["intent"] for item in result["intent_scores"]],
            sorted(item["intent"] for item in result["intent_scores"]),
        )

    def test_position_decay_changes_contribution(self) -> None:
        result = classify_intent([], self.dataset.config)
        self.assertIn("position_decay", result["scoring_formula"]["raw_score"])
        q003 = self.by_id["q003"]["intent"]
        contributions = q003["contributions"]
        self.assertGreater(len(contributions), 1)
        self.assertTrue(all(item["position_decay_factor"] <= 1 for item in contributions))

    def test_intent_score_order_is_deterministic(self) -> None:
        for record in self.pipeline_run["queries"]:
            scores = record["intent"]["intent_scores"]
            self.assertEqual(
                scores,
                sorted(scores, key=lambda item: (-item["score_units"], item["intent"])),
            )


if __name__ == "__main__":
    unittest.main()
