from __future__ import annotations

import tempfile
import unittest

from ctr_cvr_pipeline.binning import build_edge_tables
from ctr_cvr_pipeline.cache import ArtifactCache
from ctr_cvr_pipeline.features import compute_feature_snapshot
from ctr_cvr_pipeline.fixed import fraction_scale, quantize_real, round_div_half_away

from tests.helpers import build_state, make_raw


class FixedFeatureTests(unittest.TestCase):
    def test_round_half_away_and_residual_direction(self) -> None:
        self.assertEqual(round_div_half_away(3, 2).value, 2)
        self.assertEqual(round_div_half_away(-3, 2).value, -2)
        rounding = round_div_half_away(5, 4)
        self.assertEqual(rounding.value, 1)
        self.assertEqual(rounding.direction, "DOWN")
        clipped = quantize_real(8.7, 4, "Q8")
        self.assertTrue(clipped.clipped)
        self.assertEqual(clipped.direction, "CLIPPED_HIGH")

    def test_binning_and_quantization_are_repeatable(self) -> None:
        state, config, training = build_state()
        samples = state.sorted_samples()
        edges_first = build_edge_tables(config, samples, {})
        edges_second = build_edge_tables(config, samples, {})
        self.assertEqual(
            {name: table.to_dict() for name, table in edges_first.items()},
            {name: table.to_dict() for name, table in edges_second.items()},
        )
        with tempfile.TemporaryDirectory() as temp:
            first = compute_feature_snapshot(
                config, samples, edges_first, "Q16", 8, ArtifactCache(temp), 2
            )
            second = compute_feature_snapshot(
                config, samples, edges_second, "Q16", 8, ArtifactCache(temp), 2
            )
        self.assertEqual(first.entries, second.entries)
        self.assertEqual(first.root_key, second.root_key)
        self.assertGreater(first.recompute_count, 0)
        self.assertEqual(second.recompute_count, 0)
        self.assertGreater(second.reuse_count, 0)
        for sample_entries in first.entries.values():
            for record in sample_entries.values():
                self.assertEqual(record["step_den"], fraction_scale(8))
                self.assertIn(record["rounding_direction"], {"UP", "DOWN", "EXACT", "CLIPPED_HIGH", "CLIPPED_LOW"})
                self.assertIn("residual_units", record)

    def test_degenerate_equal_frequency_range_uses_single_bin(self) -> None:
        raws = []
        for request in range(1, 5):
            raws.append(make_raw(request, 0, dwell=2.0))
        state, config, _ = build_state(raws)
        tables = build_edge_tables(config, state.sorted_samples(), {})
        self.assertTrue(tables["dwell"].degenerate)
        self.assertEqual(tables["dwell"].bin_counts[0], 4)
        self.assertTrue(all(table.fingerprint().startswith("sha256:") for table in tables.values()))


if __name__ == "__main__":
    unittest.main()
