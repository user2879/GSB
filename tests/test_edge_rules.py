import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from quantcalib.calib.quantiles import fixed_quantile
from quantcalib.data.weightpack import load_weight_pack
from quantcalib.errors import E_VERSION_ORDER


class EdgeRuleTests(unittest.TestCase):
    def test_quantile_between_groups_uses_fixed_difference_rule(self):
        ordered = np.array([0.0, 10.0], dtype=np.float64)
        # Position 0.25 => 0 + 0.25 * (10 - 0) = 2.5 deterministically.
        value, rank = fixed_quantile(ordered, 250)
        self.assertAlmostEqual(value, 2.5)
        self.assertEqual(rank, 0)

    def test_monotonic_content_version_validation(self):
        records = [
            {"layer": "a", "shape": [2], "weights": [1.0, 2.0],
             "content_version": 2, "group_granularity": 1},
            {"layer": "b", "shape": [2], "weights": [1.0, 2.0],
             "content_version": 1, "group_granularity": 1},
        ]
        with tempfile.NamedTemporaryFile(
                "w", suffix=".jsonl", delete=False,
                encoding="utf-8") as handle:
            for record in records:
                handle.write(json.dumps(record) + "\n")
            path = handle.name
        layers, issues = load_weight_pack(path)
        self.assertEqual(len(layers), 2)
        self.assertIn(E_VERSION_ORDER, {issue.code for issue in issues})

    def test_quantile_tie_keeps_lower_rank(self):
        ordered = np.array([3.0, 3.0, 3.0, 4.0])
        value, rank = fixed_quantile(ordered, 500)
        self.assertEqual(value, 3.0)
        self.assertEqual(rank, 1)


if __name__ == "__main__":
    unittest.main()
