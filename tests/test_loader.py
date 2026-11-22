from __future__ import annotations

import copy
import unittest
from pathlib import Path

from search_intent_workbench.loader import (
    load_config,
    load_dataset,
    load_rules,
    validate_config,
    validate_query_records,
)


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "data" / "base"
INJECTED = ROOT / "data" / "injected"


class LoaderValidationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.dataset = load_dataset(
            BASE / "queries.jsonl",
            BASE / "intent_config.json",
            BASE / "normalization_rules.json",
        )

    def test_base_dataset_is_valid(self) -> None:
        self.assertTrue(self.dataset.config_ok)
        self.assertEqual(len(self.dataset.queries), 16)
        self.assertEqual(self.dataset.failures, [])
        self.assertEqual(len({query["query_id"] for query in self.dataset.queries}), 16)
        sessions: dict[str, list[int]] = {}
        for query in self.dataset.queries:
            sessions.setdefault(query["session_id"], []).append(query["turn"])
        for turns in sessions.values():
            self.assertEqual(sorted(turns), list(range(1, len(turns) + 1)))

    def test_out_of_vocabulary_query_fails_but_valid_rows_continue(self) -> None:
        fault = load_dataset(
            INJECTED / "I04_oov_queries.jsonl",
            BASE / "intent_config.json",
            BASE / "normalization_rules.json",
        )
        self.assertTrue(fault.config_ok)
        self.assertEqual(len(fault.queries), 16)
        self.assertEqual(len(fault.failures), 1)
        self.assertEqual(fault.failures[0]["code"], "E_QUERY_INVALID")
        self.assertEqual(fault.failures[0]["query_id"], "q_fault_oov")
        self.assertIn("outside the controlled vocabulary", fault.failures[0]["message"])

    def test_turn_gap_fails_but_valid_rows_continue(self) -> None:
        fault = load_dataset(
            INJECTED / "I03_turn_gap_queries.jsonl",
            BASE / "intent_config.json",
            BASE / "normalization_rules.json",
        )
        self.assertEqual(len(fault.queries), 16)
        self.assertEqual(len(fault.failures), 1)
        self.assertEqual(fault.failures[0]["query_id"], "q_fault_turn")
        self.assertIn("turn gap", fault.failures[0]["message"])

    def test_non_finite_weight_is_rejected(self) -> None:
        config, issues = load_config(
            INJECTED / "I01_non_finite_weight_config.json",
            load_rules(BASE / "normalization_rules.json")[0],
        )
        self.assertTrue(config)
        self.assertTrue(any(issue["code"] == "E_CONFIG_INVALID" for issue in issues))
        self.assertTrue(any("positive and finite" in issue["message"] for issue in issues))

    def test_duplicate_entity_dictionary_entry_is_rejected(self) -> None:
        rules = load_rules(BASE / "normalization_rules.json")[0]
        config, issues = load_config(
            INJECTED / "I02_duplicate_entity_config.json", rules
        )
        self.assertTrue(config)
        self.assertTrue(
            any("duplicate entity dictionary entry" in issue["message"] for issue in issues)
        )

    def test_duplicate_intent_and_entity_types_and_bad_rule_reference(self) -> None:
        rules = load_rules(BASE / "normalization_rules.json")[0]
        config = load_config(BASE / "intent_config.json", rules)[0]
        broken = copy.deepcopy(config)
        broken["intent_categories"].append(copy.deepcopy(broken["intent_categories"][0]))
        broken["entity_types"].append(broken["entity_types"][0])
        broken["normalization"]["rule_ids"].append("missing_rule")
        _, issues = validate_config(broken, rules)
        messages = "\n".join(issue["message"] for issue in issues)
        self.assertIn("duplicate intent category", messages)
        self.assertIn("duplicate entity type", messages)
        self.assertIn("unknown normalization rule reference", messages)

    def test_duplicate_query_id_and_gap_are_reported(self) -> None:
        vocabulary = set(self.dataset.config["vocabulary"])
        intents = {item["name"] for item in self.dataset.config["intent_categories"]}
        rows = [
            {"query_id": "a", "terms": ["iPhone"], "session_id": "s", "turn": 1, "intent_label": "product_search"},
            {"query_id": "a", "terms": ["iPhone"], "session_id": "s", "turn": 2, "intent_label": "product_search"},
            {"query_id": "b", "terms": ["iPhone"], "session_id": "t", "turn": 2, "intent_label": "product_search"},
        ]
        valid, failures = validate_query_records(
            rows, vocabulary=vocabulary, intent_names=intents
        )
        self.assertEqual([query["query_id"] for query in valid], ["a"])
        messages = "\n".join(item["message"] for item in failures)
        self.assertIn("query_id must be unique", messages)
        self.assertIn("turn gap", messages)

    def test_rejects_non_finite_feature_weight(self) -> None:
        rules = load_rules(BASE / "normalization_rules.json")[0]
        config = load_config(BASE / "intent_config.json", rules)[0]
        broken = copy.deepcopy(config)
        broken["intent_categories"][0]["features"][0]["weight"] = float("nan")
        _, issues = validate_config(broken, rules)
        self.assertTrue(any("feature" in issue["message"] and "finite" in issue["message"] for issue in issues))


if __name__ == "__main__":
    unittest.main()
