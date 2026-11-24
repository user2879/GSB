from __future__ import annotations

import copy
import unittest

from matgen.errors import (
    E_CONCURRENCY,
    E_LATENCY,
    E_NOISE_PARAMS,
    E_QUANTIZATION,
    E_RESOLUTION_TIERS,
    E_TEXTURE_HEADER,
    E_TEXTURE_MODE,
    E_TEXTURE_SIZE,
)
from matgen.loader import load_input, parse_config

from tests.helpers import BASIC, INJECTED


class LoaderTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.loaded = load_input(BASIC)

    def test_basic_sample_loads_cleanly(self) -> None:
        self.assertIsNotNone(self.loaded.config)
        self.assertEqual(len(self.loaded.textures), 4)
        self.assertEqual(len(self.loaded.materials), 2)
        self.assertEqual(len(self.loaded.commands), 6)
        self.assertEqual(self.loaded.issues, [])

    def test_loader_isolates_bad_pngs(self) -> None:
        injected = load_input(INJECTED)
        codes = {issue.code for issue in injected.issues}
        self.assertEqual(codes, {E_TEXTURE_HEADER, E_TEXTURE_MODE, E_TEXTURE_SIZE})
        self.assertEqual(len(injected.textures), 4)
        self.assertIsNotNone(injected.config)

    def test_config_business_codes(self) -> None:
        raw = copy.deepcopy(self.loaded.config_raw)
        raw["resolution_tiers"] = [32, 16, 48]
        _, issues = parse_config(raw, "test")
        self.assertIn(E_RESOLUTION_TIERS, {issue.code for issue in issues})

        raw = copy.deepcopy(self.loaded.config_raw)
        raw["noise_layers"][0]["amplitude"] = 0
        _, issues = parse_config(raw, "test")
        self.assertIn(E_NOISE_PARAMS, {issue.code for issue in issues})

        raw = copy.deepcopy(self.loaded.config_raw)
        raw["fixed_point"]["quantization_level"] = 3
        _, issues = parse_config(raw, "test")
        self.assertIn(E_QUANTIZATION, {issue.code for issue in issues})

        raw = copy.deepcopy(self.loaded.config_raw)
        raw["concurrency"] = raw["max_concurrency"] + 1
        _, issues = parse_config(raw, "test")
        self.assertIn(E_CONCURRENCY, {issue.code for issue in issues})

        raw = copy.deepcopy(self.loaded.config_raw)
        raw["latency_budget_ms"] = 0
        _, issues = parse_config(raw, "test")
        self.assertIn(E_LATENCY, {issue.code for issue in issues})


if __name__ == "__main__":
    unittest.main()
