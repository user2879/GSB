from __future__ import annotations

import unittest

from rope_extrap.encoding import encode_sample, position_vector_summary
from rope_extrap.models import PRUNING_FREQUENCY, SCHEME_LINEAR, SCHEME_NTK, SCHEME_STANDARD
from rope_extrap.scoring import calculate_scores
from tests.helpers import load_config_and_samples


class EncodingAndScoringTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config, self.samples, _ = load_config_and_samples(count=3)
        self.sample = self.samples["s001"]

    def test_three_schemes_are_deterministic_and_actually_different(self) -> None:
        fingerprints = set()
        score_vectors = set()
        for scheme in (SCHEME_STANDARD, SCHEME_LINEAR, SCHEME_NTK):
            config = self.config.changed(scheme=scheme)
            first = encode_sample(config, self.sample)
            second = encode_sample(config, self.sample)
            self.assertEqual(first, second)
            scores = calculate_scores(config, self.sample, first)
            self.assertEqual(first.scheme, scheme)
            fingerprints.add(first.position_fingerprint)
            score_vectors.add(scores.score_fingerprint)
            summary = position_vector_summary(first)
            self.assertEqual(summary["scheme"], scheme)
            self.assertTrue(all(value == value and abs(value) != float("inf") for value in first.query_vector))
        self.assertEqual(len(fingerprints), 3)
        self.assertEqual(len(score_vectors), 3)

    def test_pruning_changes_mask_and_outputs(self) -> None:
        all_dimensions = encode_sample(
            self.config.changed(pruning_policy="none"),
            self.sample,
        )
        pruned = encode_sample(
            self.config.changed(pruning_policy=PRUNING_FREQUENCY, pruning_tolerance=0.5),
            self.sample,
        )
        self.assertEqual(len(all_dimensions.kept_dimensions), self.config.dimension)
        self.assertLess(len(pruned.kept_dimensions), self.config.dimension)
        self.assertNotEqual(all_dimensions.position_fingerprint, pruned.position_fingerprint)

    def test_score_float64_and_intervals(self) -> None:
        encoded = encode_sample(self.config, self.sample)
        scores = calculate_scores(self.config, self.sample, encoded)
        self.assertTrue(scores.has_extrapolation)
        self.assertAlmostEqual(sum(scores.attention_weights), 1.0, places=12)
        self.assertTrue(all(isinstance(entry.score, float) for entry in scores.entries))
        self.assertTrue(any(entry.interval == "extrapolation" for entry in scores.entries))
        self.assertTrue(any(entry.interval == "in_training" for entry in scores.entries))


if __name__ == "__main__":
    unittest.main()

