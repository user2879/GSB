import unittest

from regbench import yaml_lite
from regbench.config import load_config, parse_config
from regbench.errors import E_CONFIG_INVALID


class YamlLiteTests(unittest.TestCase):
    def test_nested_mapping_and_flow_list(self):
        parsed = yaml_lite.loads(
            "a:\n  b: 1\n  c: [1.5, 2.5]\nitems:\n  - name: x\n    v: 2\n"
        )
        self.assertEqual(parsed["a"]["b"], 1)
        self.assertEqual(parsed["a"]["c"], [1.5, 2.5])
        self.assertEqual(parsed["items"][0]["name"], "x")

    def test_comments_and_strings(self):
        parsed = yaml_lite.loads('k: "v" # comment\nflag: true\nempty: null\n')
        self.assertEqual(parsed["k"], "v")
        self.assertIs(parsed["flag"], True)
        self.assertIsNone(parsed["empty"])


class ConfigValidationTests(unittest.TestCase):
    def test_sample_config_is_valid(self):
        config, failures = load_config("samples/experiment.yaml")
        self.assertIsNotNone(config)
        self.assertEqual(failures, [])
        self.assertIn("ridge", {s.name for s in config.schemes})

    def test_invalid_config_collects_all_entries(self):
        config, failures = load_config("samples/invalid_config.yaml")
        self.assertIsNotNone(config)
        self.assertTrue(len(failures) >= 10)
        self.assertTrue(all(f.code == E_CONFIG_INVALID for f in failures))
        subjects = {f.subject for f in failures}
        for expected in (
            "optimizer.learning_rate",
            "optimizer.batch_size",
            "scheduler.max_workers",
            "schemes.ridge.penalties",
            "data.target",
        ):
            self.assertIn(expected, subjects)

    def test_unknown_scheme_and_unknown_column(self):
        raw = {
            "data": {
                "n_samples": 10,
                "base_seed": 1,
                "split_ratio": 0.8,
                "target": {"name": "y"},
                "numeric": [{"name": "a", "mean": 0, "std": 1}],
                "redundant": [{"name": "r", "source": "ghost"}],
            },
            "optimizer": {
                "learning_rate": 0.01,
                "max_epochs": 5,
                "tol": 1e-6,
                "batch_size": 4,
            },
            "scheduler": {
                "max_workers": 1,
                "global_epoch_budget": 100,
                "per_scheme_epoch_quota": 5,
                "snapshot_interval": 2,
            },
            "schemes": [{"name": "bayes", "penalties": [0.1]}],
            "seeds": [1],
        }
        _, failures = parse_config(raw)
        messages = " ".join(f.message for f in failures)
        self.assertIn("未知列", messages)
        self.assertIn("未知正则化方案", messages)


if __name__ == "__main__":
    unittest.main()
