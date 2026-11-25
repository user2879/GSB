from __future__ import annotations

import unittest
from dataclasses import replace

from evidence_diag.engine import DiagnosticEngine
from evidence_diag.stages import (
    compute_boundary,
    compute_dedupe,
    compute_recall,
)

from .helpers import sample_objects


class StageTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config, self.session, _ = sample_objects()
        self.domain = self.config.fixed_point.domain()
        self.evidence_map = self.session.evidence_map()

    def test_recall_authority_covered_channels_and_duplicate_count(self) -> None:
        turn = self.session.turns[0]
        recall = compute_recall(
            turn, self.config, self.evidence_map, self.domain
        )
        by_id = {item["evidence_id"]: item for item in recall["evidence"]}
        e02 = by_id["e02"]
        self.assertEqual(e02["authority_channel"], "keyword")
        self.assertEqual(e02["source_channels"], ["dense", "keyword"])
        self.assertEqual(e02["covered_channels"], ["dense"])
        self.assertEqual(e02["reference_count"], 3)
        breakdown = {
            item["channel"]: item for item in e02["score_breakdown"]
        }
        self.assertEqual(breakdown["keyword"]["occurrence_count"], 2)
        self.assertEqual(breakdown["dense"]["occurrence_count"], 1)

    def test_authority_tie_uses_channel_name_ascending(self) -> None:
        config_raw = self.config.to_dict()
        config_raw["channel_weights"] = {"alpha": "1.0", "beta": "1.0"}
        config = type(self.config).from_dict(config_raw)
        from evidence_diag.models import Session

        session = Session.from_dict(
            {
                "session_id": "tie",
                "turns": [
                    {
                        "turn_id": 1,
                        "query": "tie query",
                        "channels": {"beta": ["e1"], "alpha": ["e1"]},
                    }
                ],
                "evidence": [
                    {"evidence_id": "e1", "text": "tie evidence", "weight": "1"}
                ],
            }
        )
        recall = compute_recall(
            session.turns[0], config, session.evidence_map(), config.fixed_point.domain()
        )
        self.assertEqual(recall["evidence"][0]["authority_channel"], "alpha")
        self.assertEqual(recall["evidence"][0]["covered_channels"], ["beta"])

    def test_dedupe_keeps_breakdown_and_reference_count(self) -> None:
        turn = self.session.turns[0]
        recall = compute_recall(
            turn, self.config, self.evidence_map, self.domain
        )
        dedupe = compute_dedupe(recall)
        e02 = next(
            item for item in dedupe["evidence"] if item["evidence_id"] == "e02"
        )
        self.assertEqual(e02["reference_count"], 3)
        self.assertEqual(dedupe["merge_order"], sorted(dedupe["merge_order"]))
        self.assertEqual(len(e02["score_breakdown"]), 2)

    def test_boundary_exact_equal_continues(self) -> None:
        left, right = self.session.turns[0], self.session.turns[1]
        left_dedupe = compute_dedupe(
            compute_recall(left, self.config, self.evidence_map, self.domain)
        )
        right_dedupe = compute_dedupe(
            compute_recall(right, self.config, self.evidence_map, self.domain)
        )
        probe = compute_boundary(
            left, right, left_dedupe, right_dedupe, self.config
        )
        equal_config = replace(
            self.config,
            topic=replace(
                self.config.topic,
                switch_threshold_bp=probe["combined_score_bp"],
            ),
        )
        boundary = compute_boundary(
            left, right, left_dedupe, right_dedupe, equal_config
        )
        self.assertTrue(boundary["exact_equal"])
        self.assertFalse(boundary["split"])
        self.assertEqual(boundary["decision"], "continue")
        self.assertEqual(boundary["equal_rule"], "continue_on_equal")

    def test_evidence_distribution_contributes_to_topic_score(self) -> None:
        left, right = self.session.turns[0], self.session.turns[1]
        left_dedupe = compute_dedupe(
            compute_recall(left, self.config, self.evidence_map, self.domain)
        )
        right_dedupe = compute_dedupe(
            compute_recall(right, self.config, self.evidence_map, self.domain)
        )
        overlap = compute_boundary(
            left, right, left_dedupe, right_dedupe, self.config
        )
        disjoint_right = {
            "evidence": [
                {"evidence_id": "not-present-1"},
                {"evidence_id": "not-present-2"},
            ]
        }
        disjoint = compute_boundary(
            left, right, left_dedupe, disjoint_right, self.config
        )
        self.assertGreater(
            overlap["evidence_similarity_bp"],
            disjoint["evidence_similarity_bp"],
        )
        self.assertGreater(
            overlap["combined_score_bp"], disjoint["combined_score_bp"]
        )

    def test_cross_topic_evidence_is_not_silently_assigned(self) -> None:
        result = DiagnosticEngine(self.config, self.session).compute().result
        e02 = next(
            item
            for item in result["evidence_attribution"]
            if item["evidence_id"] == "e02"
        )
        self.assertTrue(e02["cross_topic"])
        self.assertGreater(len(e02["topic_ids"]), 1)
        self.assertGreater(len(e02["citing_turns"]), 1)
        boundary_topics = [topic for topic in result["topics"] if topic["boundary_topic"]]
        self.assertTrue(boundary_topics)
        self.assertTrue(all(topic["boundary_start"] for topic in boundary_topics))


if __name__ == "__main__":
    unittest.main()
