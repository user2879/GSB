from __future__ import annotations

import unittest
from decimal import Decimal

from ner_editor.constants import LABELS
from ner_editor.decoder import decode_emissions
from ner_editor.emission import score_emissions
from ner_editor.features import build_features
from ner_editor.merging import merge_entities, repair_sequence
from ner_editor.models import CorpusSample, InteractionConfig, VocabEntry
from ner_editor.util import canonical_json
from tests.helpers import load_base


class DeterministicStageTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.samples, cls.model, cls.config, _ = load_base(sample_limit=1)
        cls.sample = next(iter(cls.samples.values()))

    def test_feature_hash_and_fixed_emission_are_bit_stable(self):
        first = build_features(
            self.sample,
            self.model.vocabulary,
            self.model.feature_dim,
            self.model.hash_buckets,
        )
        second = build_features(
            self.sample,
            self.model.vocabulary,
            self.model.feature_dim,
            self.model.hash_buckets,
        )
        self.assertEqual(canonical_json(first), canonical_json(second))
        first_emission = score_emissions(first, self.model, self.config)
        second_emission = score_emissions(second, self.model, self.config)
        self.assertEqual(canonical_json(first_emission), canonical_json(second_emission))
        self.assertEqual(len(first_emission["scores"]), len(self.sample.chars))
        for position in first_emission["positions"]:
            quantization = position["quantization"]
            self.assertIn("residual_units", quantization)
            self.assertIn(quantization["rounding"], {"up", "down", "none"})
            self.assertEqual(set(position["emission_scores"]), set(LABELS))

    def test_viterbi_uses_label_code_as_deterministic_tie_break(self):
        model = self.model.clone()
        model.transition_weights = [[Decimal(0) for _ in LABELS] for _ in LABELS]
        model.start_weights = [Decimal(0) for _ in LABELS]
        emission = {
            "sample_id": "tie",
            "chars": ["甲", "乙"],
            "labels": list(LABELS),
            "scores": [[0, 0, 0, 0, 0], [0, 0, 0, 0, 0]],
        }
        first = decode_emissions(emission, model, self.config)
        second = decode_emissions(emission, model, self.config)
        self.assertEqual(canonical_json(first), canonical_json(second))
        self.assertEqual(first["path_labels"], ["B", "B"])
        self.assertEqual(first["path_total"], 0)
        self.assertTrue(any(record["stage"] == "transition" for record in first["tie_records"]))
        self.assertTrue(any(record["stage"] == "final" for record in first["tie_records"]))
        self.assertEqual(first["tie_records"][-1]["chosen_label"], "B")

    def test_repair_rules_cover_i_b_e_jumps_and_end(self):
        spans, repairs = repair_sequence(list("IEBOESBI"))
        self.assertEqual([(span["start"], span["end"]) for span in spans], [(0, 1), (2, 2), (4, 4), (5, 5), (6, 7)])
        reasons = [item["reason_code"] for item in repairs]
        self.assertIn("I_WITHOUT_B", reasons)
        self.assertIn("B_WITHOUT_CONTINUATION", reasons)
        self.assertIn("E_WITHOUT_B", reasons)
        self.assertIn("ENTITY_NOT_CLOSED_BY_E", reasons)
        for repair in repairs:
            self.assertTrue(repair["basis"])
            self.assertTrue(repair["affected_positions"])

    def test_type_tie_uses_lexical_type_code(self):
        sample = CorpusSample("t", ("甲", "乙"), ("O", "O"), ("_", "_"))
        vocabulary = {
            "v1": VocabEntry("v1", "甲", "PER"),
            "v2": VocabEntry("v2", "甲", "LOC"),
        }
        decoder = {
            "sample_id": "t",
            "chars": ["甲", "乙"],
            "path_labels": ["S", "O"],
            "path_total": 0,
            "position_margins": [{"position": 0, "margin": 10}, {"position": 1, "margin": 10}],
            "fixed_decimals": 4,
            "scale": 10000,
            "quantization_level": 1,
        }
        config = self.config.clone()
        config.boundary_score_tolerance = Decimal(100)
        result = merge_entities(sample, vocabulary, decoder, config, self.model)
        self.assertEqual(len(result["entities"]), 1)
        entity = result["entities"][0]
        self.assertEqual(entity["type"], "LOC")
        self.assertEqual(entity["type_tie"]["tied_types"], ["LOC", "PER"])
        self.assertEqual(entity["type_tie"]["chosen_type"], "LOC")

    def test_boundary_probe_reports_reference_and_flip_stage(self):
        sample = CorpusSample("t", ("甲", "乙"), ("O", "O"), ("_", "_"))
        decoder = {
            "sample_id": "t",
            "chars": ["甲", "乙"],
            "path_labels": ["S", "O"],
            "path_total": 5,
            "position_margins": [{"position": 0, "margin": 5}, {"position": 1, "margin": 9}],
            "fixed_decimals": 2,
            "scale": 100,
            "quantization_level": 1,
        }
        high = {
            "sample_id": "t",
            "chars": ["甲", "乙"],
            "path_labels": ["B", "E"],
            "path_total": 50,
            "position_margins": [{"position": 0, "margin": 50}, {"position": 1, "margin": 50}],
            "fixed_decimals": 3,
            "scale": 1000,
            "quantization_level": 1,
        }
        config = InteractionConfig(
            latency_budget_ms=1000,
            fixed_decimals=2,
            quantization_level=1,
            quantization_levels=(1,),
            thread_pool_workers=1,
            snapshot_commit_interval_commands=10,
            boundary_score_tolerance=Decimal("0.05"),
            temp_dir="artifacts/tmp",
            default_entity_type="ORG",
        )
        result = merge_entities(
            sample,
            {},
            decoder,
            config,
            self.model,
            probe_runner=lambda decimals, level: high,
        )
        self.assertEqual(len(result["boundary_records"]), 1)
        boundary = result["boundary_records"][0]
        self.assertIsNotNone(boundary["reference_score"])
        self.assertEqual(boundary["difference"]["numerator"], 0)
        self.assertEqual(boundary["flip_stage"], "decode")


if __name__ == "__main__":
    unittest.main()
