import unittest

import numpy as np

from gradcomp.config import Config, validate
from gradcomp.data import (all_local_gradients, gradient_shards,
                           initial_parameters, local_gradient)
from gradcomp.errors import E_CONFIG, GradCompError


class TestConfig(unittest.TestCase):
    def test_valid_default(self):
        cfg = validate({})
        self.assertEqual(cfg.workers, 4)

    def test_unknown_field_rejected(self):
        with self.assertRaises(GradCompError) as ctx:
            validate({"bogus": 1})
        self.assertEqual(ctx.exception.code, E_CONFIG)
        self.assertEqual(ctx.exception.field, "bogus")

    def test_bad_dimension_division(self):
        with self.assertRaises(GradCompError) as ctx:
            validate({"workers": 4, "dimension": 10})
        self.assertEqual(ctx.exception.code, E_CONFIG)
        self.assertEqual(ctx.exception.field, "dimension")

    def test_bad_port(self):
        with self.assertRaises(GradCompError) as ctx:
            validate({"base_port": 80})
        self.assertEqual(ctx.exception.code, E_CONFIG)

    def test_bad_scale(self):
        with self.assertRaises(GradCompError) as ctx:
            validate({"scale_exponent": 100})
        self.assertEqual(ctx.exception.code, E_CONFIG)

    def test_bad_truncation(self):
        with self.assertRaises(GradCompError) as ctx:
            validate({"truncation": "ceil"})
        self.assertEqual(ctx.exception.code, E_CONFIG)


class TestDeterminism(unittest.TestCase):
    def test_parameters_bit_identical_across_runs(self):
        cfg = Config(workers=4, dimension=48, seed=42, samples=32)
        a = initial_parameters(cfg)
        b = initial_parameters(cfg)
        np.testing.assert_array_equal(a, b)

    def test_gradient_deterministic_and_rank_distinct(self):
        cfg = Config(workers=4, dimension=48, seed=7, samples=32)
        g0 = local_gradient(cfg, 0, 0)
        g0_again = local_gradient(cfg, 0, 0)
        np.testing.assert_array_equal(g0, g0_again)
        g1 = local_gradient(cfg, 1, 0)
        self.assertFalse(np.array_equal(g0, g1))

    def test_shard_shapes(self):
        cfg = Config(workers=4, dimension=48, seed=1, samples=16)
        grads = all_local_gradients(cfg, 0)
        shards = gradient_shards(grads, cfg)
        self.assertEqual(len(shards), 4)
        self.assertEqual(len(shards[0]), 4)
        self.assertEqual(shards[0][0].shape, (12,))

    def test_different_seed_changes_data(self):
        c1 = Config(seed=1)
        c2 = Config(seed=2)
        self.assertFalse(np.array_equal(initial_parameters(c1),
                                        initial_parameters(c2)))


if __name__ == "__main__":
    unittest.main()
