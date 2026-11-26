from __future__ import annotations

import unittest

from intentlab.errors import Codes, IssueCollector
from intentlab.validation import (
    apply_edits,
    load_inputs,
    parse_config,
    parse_queries,
    parse_rules,
    parse_vocabulary,
)

from tests.helpers import INPUT, read_json


class ValidationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.issues = IssueCollector()
        self.config = parse_config(read_json(INPUT / "config.json"), self.issues)
        self.vocabulary = parse_vocabulary(read_json(INPUT / "vocabulary.json"), self.issues)

    def test_invalid_queries_are_rejected_individually_and_valid_part_is_retained(self) -> None:
        issues = IssueCollector()
        queries = parse_queries(read_json(INPUT / "invalid_queries.json"), self.vocabulary, issues)
        self.assertEqual([query.query_id for query in queries], ["good-1", "bad-slot"])
        self.assertEqual(queries[0].terms, ("北京", "天气"))
        self.assertEqual(queries[1].slots, ())
        self.assertEqual(issues.counts()[Codes.QUERY_ID_DUPLICATE], 1)
        self.assertEqual(issues.counts()[Codes.QUERY_TERM_UNKNOWN], 1)
        self.assertEqual(issues.counts()[Codes.SLOT_RANGE], 1)

    def test_invalid_rule_graph_keeps_only_acyclic_reachable_valid_parts(self) -> None:
        issues = IssueCollector()
        revisions, active = parse_rules(
            read_json(INPUT / "invalid_rules.json"),
            self.vocabulary,
            self.config,
            issues,
        )
        self.assertEqual([revision.revision_id for revision in revisions], ["bad-weight"])
        self.assertEqual(active, ("bad-weight",))
        self.assertEqual(revisions[0].ops, ())
        codes = issues.counts()
        self.assertGreaterEqual(codes[Codes.PARENT_MISSING], 1)
        self.assertGreaterEqual(codes[Codes.REVISION_CYCLE], 2)
        self.assertEqual(codes[Codes.TRIGGER_WEIGHT_ENUM], 1)
        self.assertEqual(codes[Codes.ACTIVE_REVISION_MISSING], 1)
        self.assertEqual(codes[Codes.ACTIVE_REVISION_INVALID], 1)

    def test_invalid_config_enums_fall_back_without_silent_drop(self) -> None:
        raw = read_json(INPUT / "config.json")
        raw["fixed_point"] = {"fraction_bits": 7, "quantum": 3}
        raw["allowed"]["weights"] = [1000, 123]
        issues = IssueCollector()
        config = parse_config(raw, issues)
        self.assertEqual(config.fraction_bits, 8)
        self.assertEqual(config.quantum, 1)
        self.assertIn(8, config.allowed_fraction_bits)
        self.assertIn(1, config.allowed_quantums)
        self.assertEqual(issues.counts()[Codes.FRACTION_BITS_ENUM], 1)
        self.assertEqual(issues.counts()[Codes.QUANTUM_ENUM], 1)
        self.assertIn(123, config.allowed_weights)

    def test_edit_stream_rejects_bad_commands_and_preserves_prior_good_edits(self) -> None:
        loaded, issues = load_inputs(
            config_path=INPUT / "config.json",
            vocabulary_path=INPUT / "vocabulary.json",
            queries_path=INPUT / "queries.json",
            rules_path=INPUT / "rules.json",
            edits_path=None,
        )
        self.assertEqual(len(issues), 0)
        commands = {
            "commands": [
                {
                    "op": "replace_query",
                    "query": {
                        **loaded.queries[0].to_dict(),
                        "terms": ["北京", "不存在"],
                    },
                },
                {"op": "set_fixed_point", "fraction_bits": 13, "quantum": 16},
                {"op": "set_fixed_point", "fraction_bits": 16, "quantum": 32},
                {"op": "set_threshold", "intent_id": "default", "value": -1},
                {"op": "set_threshold", "intent_id": "default", "value": 99},
                {"op": "unknown-operation"},
            ]
        }
        issues = IssueCollector()
        queries, revisions, active, config = apply_edits(
            commands,
            queries=loaded.queries,
            revisions=loaded.revisions,
            active_revision_ids=loaded.active_revision_ids,
            config=loaded.config,
            vocabulary=loaded.vocabulary,
            issues=issues,
        )
        self.assertEqual(config.fraction_bits, 16)
        self.assertEqual(config.quantum, 32)
        self.assertEqual(config.default_threshold, 99)
        self.assertEqual(queries[0].terms, loaded.queries[0].terms)
        self.assertEqual(revisions, loaded.revisions)
        self.assertEqual(active, loaded.active_revision_ids)
        codes = issues.counts()
        self.assertEqual(codes[Codes.QUERY_TERM_UNKNOWN], 1)
        self.assertEqual(codes[Codes.FRACTION_BITS_ENUM], 1)
        self.assertEqual(codes[Codes.THRESHOLD_INVALID], 1)
        self.assertEqual(codes[Codes.EDIT_OP_UNKNOWN], 1)


if __name__ == "__main__":
    unittest.main()
