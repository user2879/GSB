from __future__ import annotations

import unittest
from dataclasses import replace

import numpy as np

from matgen.cache import CacheStore
from matgen.engine import MaterialEngine
from matgen.fixed import round_div
from matgen.injection import InjectionRegistry
from matgen.loader import load_input
from matgen.noise import value_noise

from tests.helpers import BASIC


class FixedNoiseTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.loaded = load_input(BASIC)

    def test_rounding_modes(self) -> None:
        values = np.array([-3, -2, -1, 0, 1, 2, 3])
        self.assertEqual(round_div(values, 2, "half_up").tolist(), [-2, -1, -1, 0, 1, 1, 2])
        self.assertEqual(round_div(values, 2, "half_even").tolist(), [-2, -1, 0, 0, 0, 1, 2])
        self.assertEqual(round_div(values, 2, "floor").tolist(), [-2, -1, -1, 0, 0, 1, 1])
        self.assertEqual(round_div(values, 2, "ceil").tolist(), [-1, -1, 0, 0, 1, 1, 2])
        self.assertEqual(round_div(values, 2, "truncate").tolist(), [-1, -1, 0, 0, 0, 1, 1])

    def test_coordinate_noise_is_deterministic(self) -> None:
        seed = np.uint64(123456789)
        first = value_noise(seed, 0.17, 0, 16, 16)
        second = value_noise(seed, 0.17, 0, 16, 16)
        different = value_noise(seed, 0.29, 0, 16, 16)
        self.assertTrue(np.array_equal(first, second))
        self.assertFalse(np.array_equal(first, different))
        self.assertTrue(np.isfinite(first).all())

    def test_quantization_and_width_change_real_integers_and_outputs(self) -> None:
        config = self.loaded.config
        q2 = MaterialEngine(CacheStore(), InjectionRegistry()).build(config, self.loaded.materials, self.loaded.textures)
        q4_config = replace(config, fixed_point=replace(config.fixed_point, quantization_level=4))
        q4 = MaterialEngine(CacheStore(), InjectionRegistry()).build(q4_config, self.loaded.materials, self.loaded.textures)
        self.assertFalse(np.array_equal(q2.layers[0].quantized, q4.layers[0].quantized))
        left = q2.material_artifacts["brushed_steel"].channels["diffuse"].final
        right = q4.material_artifacts["brushed_steel"].channels["diffuse"].final
        self.assertFalse(np.array_equal(left, right))

        width_config = replace(
            config,
            fixed_point=replace(config.fixed_point, bit_width=20, fraction_bits=12),
        )
        width = MaterialEngine(CacheStore(), InjectionRegistry()).build(
            width_config, self.loaded.materials, self.loaded.textures
        )
        self.assertFalse(np.array_equal(q2.layers[0].quantized, width.layers[0].quantized))
        self.assertFalse(
            np.array_equal(
                q2.material_artifacts["brushed_steel"].channels["diffuse"].final,
                width.material_artifacts["brushed_steel"].channels["diffuse"].final,
            )
        )


if __name__ == "__main__":
    unittest.main()
