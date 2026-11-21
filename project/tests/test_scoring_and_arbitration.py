from __future__ import annotations

import unittest
from decimal import Decimal

from arbitration_workbench.arbitration import apply_relations, decide_ticket, fuse_scores
from arbitration_workbench.config import parse_config
from arbitration_workbench.loader import load_evidence_text
from arbitration_workbench.model import ArbitrationConfig, Evidence
from arbitration_workbench.numeric import read_json
from arbitration_workbench.rules import parse_rules
from arbitration_workbench.scoring import score_ticket

from .helpers import ROOT


class ScoringAndArbitrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rules = parse_rules(read_json(ROOT / "samples" / "rules.json"))
        cls.config = parse_config(read_json(ROOT / "samples" / "config.json"))
        loaded = load_evidence_text((ROOT / "samples" / "evidence.jsonl").read_text(encoding="utf-8"), cls.rules)
        cls.evidence = loaded.evidence_by_ticket

    def test_scoring_is_repeatable_and_fixed_quantized(self):
        rows = self.evidence["T0047"]
        first = score_ticket("T0047", rows, self.rules, self.config)
        second = score_ticket("T0047", rows, self.rules, self.config)
        self.assertEqual(first, second)
        self.assertEqual([1_000_000, 1_000_000, 0], [row["decay_fixed"] for row in first["evidence_scores"]])
        self.assertEqual(1_000_000, first["raw_total_fixed"])
        self.assertEqual([1, 2, 3], [row["evidence_seq"] for row in first["evidence_scores"]])

    def test_threshold_equality_is_boundary_and_upgrades(self):
        rows = self.evidence["T0047"]
        score = score_ticket("T0047", rows, self.rules, self.config)
        relations = apply_relations("T0047", rows, score, self.rules, self.config)
        decision = decide_ticket("T0047", relations, self.rules, self.config)
        self.assertEqual(1_000_000, decision["score_fixed"])
        self.assertTrue(decision["boundary"])
        self.assertEqual(1, decision["upgrade_code"])
        self.assertIn("R_THRESHOLD_BOUNDARY", decision["reason_codes"])

    def test_exclusive_relation_clamps_and_support_adds(self):
        rules = parse_rules(
            {
                "evidence_type_weights": {"a": "0.30", "b": "0.10", "c": "0.20"},
                "role_trust": {"r": "1.0"},
                "relations": [
                    {
                        "id": "ex",
                        "kind": "exclusive",
                        "left": {"evidence_type": "a"},
                        "right": {"evidence_type": "b"},
                    },
                    {
                        "id": "sup",
                        "kind": "support",
                        "left": {"evidence_type": "b"},
                        "right": {"evidence_type": "c"},
                        "bonus": "0.04",
                    },
                ],
                "priority_bands": [{"level": 0, "name": "P0", "min_score": "0"}],
            }
        )
        config = parse_config(
            {
                "fixed_decimal_places": 6,
                "quantization_units": 1,
                "decay_half_life": 3600,
                "exclusion_strength": "0.5",
                "support_bonus": "0.01",
                "fusion": "sum",
                "decision_threshold": "0.0",
                "interaction_budget_ms": 50,
                "worker_threads": 2,
                "snapshot_commit_interval": 10,
                "snapshot_dir": "snapshots",
                "temp_dir": "tmp",
            }
        )
        rows = (
            Evidence("T", 1, "a", "r", 0, ("x",)),
            Evidence("T", 2, "b", "r", 0, ("y",)),
            Evidence("T", 3, "c", "r", 3600, ("z",)),
        )
        scored = score_ticket("T", rows, rules, config)
        relations = apply_relations("T", rows, scored, rules, config)
        # a=300000, b=100000, c=100000. Exclusion splits 250000/250000 and
        # b clamps at zero, so the applied pair delta is -350000. Support then
        # adds 20000/20000 to b and c.
        self.assertEqual(-350000, relations["relation_contributions"][0]["applied_delta_fixed"])
        self.assertEqual(40000, relations["relation_contributions"][1]["applied_delta_fixed"])
        self.assertTrue(all(item["score_fixed"] >= 0 for item in relations["adjusted_evidence_scores"]))
        self.assertEqual(190000, relations["score_fixed"])

    def test_priority_band_selection(self):
        rows = self.evidence["T0001"]
        score = score_ticket("T0001", rows, self.rules, self.config)
        decision = decide_ticket("T0001", apply_relations("T0001", rows, score, self.rules, self.config), self.rules, self.config)
        expected = "P3"
        for band in sorted(self.rules.priority_bands, key=lambda item: item.min_score):
            minimum = int(band.min_score * self.config.scale)
            if decision["score_fixed"] >= minimum:
                expected = band.name
        self.assertEqual(expected, decision["priority_name"])

    def test_fusion_mean_stays_in_fixed_units(self):
        scores = [100_000, 300_000, 500_000]
        self.assertEqual(300_000, fuse_scores(scores, "mean", 1))
        self.assertEqual(800_000, fuse_scores(scores, "top2_sum", 1))
        self.assertEqual(500_000, fuse_scores(scores, "max", 1))
        self.assertEqual(100_000, fuse_scores(scores, "min", 1))
        self.assertEqual(900_000, fuse_scores(scores, "sum", 1))


if __name__ == "__main__":
    unittest.main()
