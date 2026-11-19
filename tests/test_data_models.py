import unittest

import numpy as np

from regbench.config import load_config
from regbench.data import prepare_data
from regbench.models import (
    L1_ZERO_SUBGRADIENT,
    TaskSpec,
    initial_checkpoint,
    make_cache_key,
    make_task_key,
    train_task,
)


class DataTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config, cls.failures = load_config("samples/experiment.yaml")
        cls.data_a = prepare_data(cls.config)
        cls.data_b = prepare_data(cls.config)

    def test_deterministic_fingerprint(self):
        self.assertEqual(self.data_a.data_fingerprint, self.data_b.data_fingerprint)
        self.assertEqual(self.data_a.n_train + self.data_a.n_val, 240)

    def test_bias_first_and_imputation(self):
        self.assertEqual(self.data_a.feature_names[0], "bias")
        np.testing.assert_equal(self.data_a.x_train[:, 0], 1.0)
        self.assertIn("age", self.data_a.imputation)
        self.assertIn("region", self.data_a.imputation)
        self.assertEqual(self.data_a.imputation["region"]["strategy"], "mode")

    def test_onehot_and_no_nan_after_fill(self):
        self.assertFalse(np.isnan(self.data_a.x_train).any())
        self.assertFalse(np.isnan(self.data_a.x_val).any())
        region_cols = [n for n in self.data_a.feature_names if n.startswith("region=")]
        self.assertTrue(region_cols)


class ModelDeterminismTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config, _ = load_config("samples/experiment.yaml")
        cls.data = prepare_data(cls.config)

    def _spec(self, scheme, penalty, ratio=0.0):
        return TaskSpec(
            scheme=scheme,
            penalty=penalty,
            l1_ratio=ratio,
            learning_rate=self.config.optimizer.learning_rate,
            max_epochs=self.config.optimizer.max_epochs,
            tol=self.config.optimizer.tol,
            batch_size=self.config.optimizer.batch_size,
        )

    def test_repeated_runs_identical_trajectory(self):
        for scheme, penalty, ratio in (
            ("ridge", 0.1, 0.0),
            ("lasso", 0.1, 0.0),
            ("elasticnet", 0.1, 0.5),
        ):
            spec = self._spec(scheme, penalty, ratio)
            key = make_task_key(spec, 11)
            cache = make_cache_key(spec, self.data.data_fingerprint)
            r1, _ = train_task(spec, self.data, key, cache, 11, spec.max_epochs)
            r2, _ = train_task(spec, self.data, key, cache, 11, spec.max_epochs)
            self.assertEqual(r1.loss_history, r2.loss_history, scheme)
            np.testing.assert_array_equal(r1.weights, r2.weights)
            self.assertEqual(r1.validation, r2.validation)

    def test_l1_zero_subgradient_is_fixed_rule(self):
        self.assertEqual(L1_ZERO_SUBGRADIENT, 0.0)
        spec = self._spec("lasso", 0.5)
        checkpoint = initial_checkpoint(self.data.x_train.shape[1], 11, spec)
        # 初始权重全为 0，零系数处次梯度固定为 0，首轮可正常推进。
        result, _ = train_task(
            spec, self.data, "k", "c", 11, 3, checkpoint=checkpoint
        )
        self.assertEqual(len(result.loss_history), 3)

    def test_checkpoint_resume_matches_continuous_run(self):
        spec = self._spec("elasticnet", 0.2, 0.5)
        key = make_task_key(spec, 11)
        cache = make_cache_key(spec, self.data.data_fingerprint)
        full, _ = train_task(spec, self.data, key, cache, 11, spec.max_epochs)

        checkpoint = initial_checkpoint(self.data.x_train.shape[1], 11, spec)
        first, checkpoint = train_task(
            spec, self.data, key, cache, 11, 7, checkpoint=checkpoint
        )
        second, _ = train_task(
            spec,
            self.data,
            key,
            cache,
            11,
            spec.max_epochs - 7,
            checkpoint=checkpoint,
        )
        self.assertEqual(second.loss_history, full.loss_history)
        np.testing.assert_allclose(second.weights, full.weights)
        self.assertEqual(len(first.loss_history), 7)


if __name__ == "__main__":
    unittest.main()
