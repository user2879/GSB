from __future__ import annotations

import unittest

import numpy as np

from posdiag.encoding import encode_sequence, position_digest
from posdiag.scoring import (
    ERROR_STAGES,
    aggregate_error_distributions,
    aggregate_layer_statistics,
    attention_cache_entry,
    attention_scores,
    boundary_reasons,
    error_cache_entry,
    sample_tier_summary,
    trace_errors,
)
from tests.helpers import make_config, make_sample


class EncodingTests(unittest.TestCase):
    def test_three_schemes_use_real_distinct_angles_and_are_deterministic(self) -> None:
        config = make_config(scaling_factor=2.0)
        sample = make_sample("s0001", query_position=48, key_positions=(1, 20, 45, 48))
        outputs = {
            scheme: encode_sequence(config, sample, scheme)
            for scheme in ("rope_standard", "rope_linear", "rope_ntk")
        }
        for scheme, encoded in outputs.items():
            repeated = encode_sequence(config, sample, scheme)
            self.assertTrue(np.array_equal(encoded.query_vector, repeated.query_vector), scheme)
            for actual, expected in zip(encoded.key_vectors, repeated.key_vectors, strict=True):
                self.assertTrue(np.array_equal(actual, expected), scheme)
            self.assertEqual(encoded.query_vector.dtype, np.float64)

        self.assertFalse(np.allclose(outputs["rope_standard"].query_vector, outputs["rope_linear"].query_vector))
        self.assertFalse(np.allclose(outputs["rope_standard"].query_vector, outputs["rope_ntk"].query_vector))
        self.assertFalse(np.allclose(outputs["rope_linear"].query_vector, outputs["rope_ntk"].query_vector))

        digests = {scheme: position_digest(encoded, config) for scheme, encoded in outputs.items()}
        self.assertEqual(len(set(digests.values())), 3)

    def test_attention_scores_change_when_scheme_changes(self) -> None:
        config = make_config(scaling_factor=2.0)
        sample = make_sample("s0001", query_position=64, key_positions=(1, 20, 48, 64))
        scores = {}
        for scheme in ("rope_standard", "rope_linear", "rope_ntk"):
            encoded = encode_sequence(config, sample, scheme)
            result = attention_scores(
                encoded.query_vector,
                encoded.key_vectors,
                sample.query_position,
                sample.key_positions,
                config.training_context_length,
            )
            scores[scheme] = result.scores
        self.assertNotEqual(scores["rope_standard"], scores["rope_linear"])
        self.assertNotEqual(scores["rope_standard"], scores["rope_ntk"])


class AttentionAndErrorTests(unittest.TestCase):
    def test_empty_keys_and_self_attention_follow_fixed_rules(self) -> None:
        config = make_config()
        empty = make_sample("s0001", key_positions=())
        encoded_empty = encode_sequence(config, empty, "rope_standard")
        empty_result = attention_scores(
            encoded_empty.query_vector,
            encoded_empty.key_vectors,
            empty.query_position,
            empty.key_positions,
            config.training_context_length,
        )
        self.assertEqual(empty_result.scores, ())
        self.assertEqual(empty_result.weights, ())
        self.assertTrue(sample_tier_summary(empty_result)["empty_keys"])
        self.assertIn("empty_key_list", boundary_reasons(config, empty))

        self_attention = make_sample("s0002", query_position=20, key_positions=(20,))
        encoded_self = encode_sequence(config, self_attention, "rope_standard")
        self_result = attention_scores(
            encoded_self.query_vector,
            encoded_self.key_vectors,
            self_attention.query_position,
            self_attention.key_positions,
            config.training_context_length,
        )
        self.assertEqual(self_result.distances, (0,))
        self.assertEqual(self_result.weights, (1.0,))
        self.assertEqual(sample_tier_summary(self_result)["self_attention_pairs"], 1)
        self.assertIn("self_attention", boundary_reasons(config, self_attention))

    def test_layer_statistics_group_scores_by_distance_and_bucket(self) -> None:
        config = make_config()
        entries = []
        for index, (query, keys) in enumerate(((40, (39, 1)), (50, (49, 1))), start=1):
            sample = make_sample(f"s000{index}", query_position=query, key_positions=keys)
            encoded = encode_sequence(config, sample, "rope_standard")
            result = attention_scores(
                encoded.query_vector,
                encoded.key_vectors,
                query,
                keys,
                config.training_context_length,
            )
            entries.append(attention_cache_entry(config, sample, "rope_standard", result))
        stats = aggregate_layer_statistics(entries)["rope_standard"]
        self.assertEqual(stats["in_training"]["count"], 2)
        self.assertEqual(stats["extrapolation"]["count"], 2)
        self.assertIn("1", stats["in_training"]["by_distance"])
        self.assertIn("39", stats["extrapolation"]["by_distance"])
        self.assertIn("49", stats["extrapolation"]["by_distance"])

    def test_staged_errors_are_computed_and_first_exceeded_is_stable(self) -> None:
        config = make_config(tolerance=0.35)
        sample = make_sample("s0001", query_position=100, key_positions=(1, 40, 100))
        encoded = encode_sequence(config, sample, "rope_standard")
        attention = attention_scores(
            encoded.query_vector,
            encoded.key_vectors,
            sample.query_position,
            sample.key_positions,
            config.training_context_length,
        )
        trace = trace_errors(config, sample, "rope_standard", encoded, attention)
        self.assertEqual(set(trace.deviations()), set(ERROR_STAGES))
        self.assertTrue(all(value >= 0.0 for value in trace.deviations().values()))
        self.assertIn(trace.first_exceeded, (*ERROR_STAGES, "none"))
        entry = error_cache_entry(config, sample, "rope_standard", trace)
        distributions = aggregate_error_distributions([entry])["rope_standard"]
        self.assertEqual(set(distributions), set(ERROR_STAGES))
        self.assertEqual(distributions["position_encoding"]["count"], 1)

        strict = make_config(tolerance=1e-12)
        strict_encoded = encode_sequence(strict, sample, "rope_standard")
        strict_trace = trace_errors(strict, sample, "rope_standard", strict_encoded, attention)
        self.assertEqual(strict_trace.first_exceeded, "position_encoding")

        loose = make_config(tolerance=1e12)
        loose_trace = trace_errors(loose, sample, "rope_standard", strict_encoded, attention)
        self.assertEqual(loose_trace.first_exceeded, "none")


if __name__ == "__main__":
    unittest.main()
