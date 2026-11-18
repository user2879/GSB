import unittest

from ab_workbench.fixedpoint import FixedPoint
from ab_workbench.hashing import bucket_for_user, assign_users, group_boundaries
from ab_workbench.config import config_from_dict
from ab_workbench.errors import E_HASH_CONFIG_INVALID


def base_config(**overrides):
    raw = {
        "experiment_id": "t",
        "groups": ["c", "t"],
        "control_group": "c",
        "group_weights": {"c": 1, "t": 1},
        "hash_seed": 1,
        "num_buckets": 10,
        "primary_metric": "ctr",
        "guardrail_metrics": ["cvr"],
        "metric_direction": "higher_is_better",
        "min_sample_size": 5,
        "min_lift": "0.01",
        "decimal_width": 4,
        "quantization_step": None,
        "concurrency": 1,
        "batch_size": 8,
        "snapshot_every_batches": 1,
    }
    raw.update(overrides)
    return config_from_dict(raw)


class FixedPointTests(unittest.TestCase):
    def test_ratio_floor_and_text(self):
        fp = FixedPoint(4)
        self.assertEqual(fp.ratio(1, 3), 3333)
        self.assertEqual(fp.to_text(fp.ratio(1, 3)), "0.3333")
        self.assertEqual(fp.to_text(-5), "-0.0005")

    def test_quantization_grid(self):
        fp = FixedPoint(4, "0.01")
        self.assertEqual(fp.grid, 100)
        self.assertEqual(fp.quantize(1234), 1200)

    def test_deterministic_bucket(self):
        first = [bucket_for_user(f"u{i}", 9, 10) for i in range(50)]
        second = [bucket_for_user(f"u{i}", 9, 10) for i in range(50)]
        self.assertEqual(first, second)
        self.assertTrue(all(0 <= b < 10 for b in first))

    def test_group_boundaries_cover_all_buckets(self):
        cfg = base_config()
        boundaries = group_boundaries(cfg)
        self.assertEqual(boundaries[0][1], 0)
        self.assertEqual(boundaries[-1][2], 10)

    def test_invalid_bucket_count(self):
        cfg = base_config(num_buckets=0)
        _assignments, failures = assign_users(["u1"], cfg)
        self.assertTrue(any(f.code == E_HASH_CONFIG_INVALID for f in failures))


if __name__ == "__main__":
    unittest.main()
