from __future__ import annotations

import unittest

import numpy as np

from gray_merge_lab.deterministic import (
    SplitMix64,
    build_backend_parameters,
    build_input_tensor,
    forward_fixed,
)
from gray_merge_lab.specs import BackendSpec, SampleSpec


class DeterministicModelTests(unittest.TestCase):
    def setUp(self):
        self.backend = BackendSpec("canary", "127.0.0.1", 20002, 6, 9, 12345, "int8-sym-v1")
        self.sample = SampleSpec("s1", 1, "client", "r", "/r/1", 999)

    def test_splitmix64_known_sequence(self):
        rng = SplitMix64(0)
        values = [rng.next_u64() for _ in range(4)]
        self.assertEqual(
            values,
            [
                0xE220A8397B1DCDAF,
                0x6E789E6AA1B965F4,
                0x06C45D188009454F,
                0xF88BB8A8724C81EC,
            ],
        )

    def test_parameter_generation_is_bitwise_repeatable(self):
        first = build_backend_parameters(self.backend, 1)
        second = build_backend_parameters(self.backend, 1)
        self.assertEqual(first.param_digest, second.param_digest)
        self.assertEqual(first.weights_digest, second.weights_digest)
        self.assertTrue(np.array_equal(first.w1, second.w1))
        self.assertTrue(np.array_equal(first.w2, second.w2))
        self.assertTrue(np.array_equal(first.b1, second.b1))
        self.assertTrue(np.array_equal(first.b2, second.b2))

    def test_param_digest_binds_identity_dimension_profile_and_version(self):
        base = build_backend_parameters(self.backend, 1)
        version = build_backend_parameters(self.backend, 2)
        renamed = build_backend_parameters(
            BackendSpec("renamed", "127.0.0.1", 20002, 6, 9, 12345, "int8-sym-v1"), 1
        )
        requantized = build_backend_parameters(
            BackendSpec("canary", "127.0.0.1", 20002, 6, 9, 12345, "int4-sym-v1"), 1
        )
        self.assertNotEqual(base.param_digest, version.param_digest)
        self.assertNotEqual(base.param_digest, renamed.param_digest)
        self.assertNotEqual(base.param_digest, requantized.param_digest)
        self.assertEqual(version.summary()["version"], 2)
        self.assertEqual(version.summary()["quant_profile_version"], 1)

    def test_input_digest_repeatable_and_dimension_bound(self):
        first = build_input_tensor(self.sample, 6)
        second = build_input_tensor(self.sample, 6)
        other_dim = build_input_tensor(self.sample, 7)
        self.assertEqual(first.digest, second.digest)
        self.assertNotEqual(first.digest, other_dim.digest)
        self.assertTrue(np.array_equal(first.values, second.values))

    def test_forward_output_is_repeatable(self):
        params = build_backend_parameters(self.backend, 1)
        inputs = [build_input_tensor(self.sample, params.model_dim)]
        first = forward_fixed(params, inputs)
        second = forward_fixed(params, inputs)
        self.assertEqual(first[0], second[0])
        self.assertEqual(first[1], second[1])
        self.assertEqual(first[2], second[2])
        self.assertTrue(first[0].startswith("sha256:"))

    def test_supported_quant_profiles_produce_bound_summaries(self):
        digests = set()
        for profile in ("fp32-v1", "int8-sym-v1", "int4-sym-v1"):
            parameters = build_backend_parameters(
                BackendSpec("b", "127.0.0.1", 20001, 4, 5, 42, profile), 1
            )
            digests.add(parameters.param_digest)
            self.assertEqual(parameters.summary()["quant_profile"], profile)
        self.assertEqual(len(digests), 3)


if __name__ == "__main__":
    unittest.main()

