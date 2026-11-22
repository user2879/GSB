from __future__ import annotations

import json
import unittest
from pathlib import Path

from mwlab.canonical import canonical_lines, read_json
from mwlab.config import declarations_from_document, load_declarations
from mwlab.records import load_normalized_records
from mwlab.semantics import analyze

from .helpers import SAMPLES, load_config


class SemanticsStatsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.declarations = load_declarations(SAMPLES / "session.json")
        cls.records = load_normalized_records(SAMPLES / "records.jsonl", cls.declarations)
        cls.analysis = analyze(cls.records, cls.declarations)
        cls.artifact = read_json(SAMPLES / "expected_summary.json")

    def test_expected_verdicts(self) -> None:
        expected = {row["message_id"]: row["verdict"] for row in self.artifact["conclusions"]}
        actual = {row["message_id"]: row["verdict"] for row in self.analysis["conclusions"]}
        self.assertEqual(actual, expected)
        self.assertEqual(self.analysis["conclusions"][0]["message_id"], "archive:1")
        self.assertEqual(
            [row["topic"] for row in self.analysis["conclusions"]],
            ["archive", "audit", "audit", "orders", "orders", "orders", "orders", "payments", "payments", "payments"],
        )
        self.assertEqual(self.analysis["verdict_counts"], {"at_least_once": 8, "at_most_once": 1, "exactly_once": 1})

    def test_conflicts_sorted_and_attributed(self) -> None:
        rows = self.analysis["conflicts"]
        keys = [(row["message_seq"], row["conflict_type"]) for row in rows]
        self.assertEqual(keys, sorted(keys))
        self.assertEqual(
            {row["conflict_type"] for row in rows},
            {"ACK_WITH_UNCOVERED_RANGE", "DUP_IDEMPOTENCY_ACROSS_GROUPS", "SEMANTIC_MARKER_MISMATCH"},
        )
        duplicate = next(row for row in rows if row["conflict_type"] == "DUP_IDEMPOTENCY_ACROSS_GROUPS")
        self.assertEqual(duplicate["message_id"], "payments:1")
        self.assertEqual(duplicate["consumer_groups"], ["analytics", "warehouse"])
        uncovered = [row for row in rows if row["conflict_type"] == "ACK_WITH_UNCOVERED_RANGE"]
        self.assertEqual([row["message_id"] for row in uncovered], ["payments:3", "orders:4"])
        self.assertEqual(uncovered[1]["basis"]["causes"][0]["uncovered_intervals"], [[2, 2]])
        marker = next(row for row in rows if row["conflict_type"] == "SEMANTIC_MARKER_MISMATCH")
        self.assertEqual(marker["message_id"], "audit:2")
        self.assertEqual(marker["basis"]["causes"][0]["semantic_marker"], "at_most_once")

    def test_fixed_point_error_envelope_and_tie(self) -> None:
        stats = self.analysis["stats"]
        self.assertEqual(stats["scale_factor"], 1_000_000)
        self.assertEqual(stats["truncation"], "floor_toward_negative_infinity")
        self.assertEqual(stats["error_envelope"]["single_value"]["fixed_units_strictly_less_than"], 1)
        self.assertEqual(stats["error_envelope"]["ratio_comparison"]["fixed_units_strictly_less_than"], 2)
        ranking = stats["topic_ranking"]
        self.assertEqual([row["topic"] for row in ranking], ["archive", "orders", "payments", "audit"])
        by_topic = {row["topic"]: row for row in ranking}
        self.assertEqual(by_topic["orders"]["fixed_exact_once_ratio"], 250_000)
        self.assertEqual(by_topic["payments"]["fixed_exact_once_ratio"], 250_000)
        self.assertTrue(by_topic["orders"]["error_affected"])
        self.assertTrue(by_topic["payments"]["error_affected"])
        self.assertEqual(stats["comparison_affected_pairs"][0]["topic_a"], "orders")
        self.assertEqual(stats["comparison_affected_pairs"][0]["topic_b"], "payments")
        self.assertEqual(stats["comparison_affected_pairs"][0]["fixed_difference_units"], 0)

    def test_group_order_does_not_change_conclusion_bytes(self) -> None:
        document = load_config("session.json")
        document["consumer_groups"] = list(reversed(document["consumer_groups"]))
        reordered = declarations_from_document(document)
        reordered_analysis = analyze(self.records, reordered)
        self.assertEqual(
            canonical_lines(reordered_analysis["conclusions"]),
            canonical_lines(self.analysis["conclusions"]),
        )
        self.assertEqual(
            canonical_lines(reordered_analysis["group_conclusions"]),
            canonical_lines(self.analysis["group_conclusions"]),
        )


if __name__ == "__main__":
    unittest.main()

