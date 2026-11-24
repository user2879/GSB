from __future__ import annotations

import unittest
from dataclasses import replace

from matgen.cache import CacheStore
from matgen.engine import MaterialEngine
from matgen.injection import InjectionRegistry
from matgen.loader import load_input
from matgen.trace import STAGE_ORDER, build_error_trace, compare_full_rebuild, high_precision_config

from tests.helpers import BASIC


class TraceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.loaded = load_input(BASIC)
        cls.config = replace(cls.loaded.config, material_mix_mode="multiply")
        cls.cache = CacheStore()
        cls.injections = InjectionRegistry()
        cls.result = MaterialEngine(cls.cache, cls.injections).build(
            cls.config,
            cls.loaded.materials,
            cls.loaded.textures,
        )

    def test_high_precision_config(self) -> None:
        high, width = high_precision_config(self.config)
        self.assertGreater(width, self.config.fixed_point.bit_width)
        self.assertGreater(high.fixed_point.fraction_bits, self.config.fixed_point.fraction_bits)

    def test_stage_differences_and_attribution(self) -> None:
        report, diffs, high = build_error_trace(
            config=self.config,
            materials=self.loaded.materials,
            textures=self.loaded.textures,
            low_result=self.result,
            cache=self.cache,
            injections=self.injections,
        )
        self.assertEqual(set(diffs), set(STAGE_ORDER))
        for array in diffs.values():
            self.assertEqual(array.shape, (32, 32))
        self.assertIn(report["first_exceeded_stage"], set(STAGE_ORDER) | {"none"})
        self.assertIn(
            report["attribution"]["category"],
            {"mix_order", "rounding_boundary", "quantization_truncation", "none"},
        )
        self.assertEqual(report["attribution"]["category"], "mix_order")
        self.assertTrue(report["attribution"]["mix_order"]["supported"])
        self.assertGreater(len(high.layers), 0)
        self.assertIn("pixel_differences", report)

    def test_full_rebuild_is_pixel_exact(self) -> None:
        comparison = compare_full_rebuild(
            incremental=self.result,
            config=self.config,
            materials=self.loaded.materials,
            textures=self.loaded.textures,
            injections=self.injections,
        )
        self.assertTrue(comparison["exact_match"])
        self.assertEqual(comparison["incremental_digest"], comparison["fresh_digest"])


if __name__ == "__main__":
    unittest.main()
