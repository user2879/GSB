import tempfile
import unittest

from ab_workbench.config import config_from_dict
from ab_workbench.hashing import assign_users
from ab_workbench.errors import E_HASH_CONFIG_INVALID


def raw_config(**over):
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
    raw.update(over)
    return config_from_dict(raw)


class HashConfigTests(unittest.TestCase):
    def test_weights_exceeding_buckets_rejected(self):
        cfg = raw_config(num_buckets=1, group_weights={"c": 5, "t": 5})
        _assignments, failures = assign_users(["u1"], cfg)
        self.assertTrue(any(f.code == E_HASH_CONFIG_INVALID for f in failures))

    def test_non_positive_weight_rejected(self):
        cfg = raw_config(group_weights={"c": 0, "t": 1})
        _assignments, failures = assign_users(["u1"], cfg)
        self.assertTrue(any(f.code == E_HASH_CONFIG_INVALID for f in failures))

    def test_negative_seed_rejected(self):
        cfg = raw_config(hash_seed=-1)
        _assignments, failures = assign_users(["u1"], cfg)
        self.assertTrue(any(f.code == E_HASH_CONFIG_INVALID for f in failures))


if __name__ == "__main__":
    unittest.main()
