import tempfile
import unittest
from pathlib import Path

import numpy as np

from scripts.generate_samples import write_bundle
from texmat.config import load_generation_config, load_materials
from texmat.diagnostics import STAGES, compare_gears, run_gear
from texmat.loader import scan_textures


class DiagnosticsTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)
        self.sample = self.tmp / "sample"
        write_bundle(self.sample, size=20, workers=2, budget=100000)
        self.config = load_generation_config(self.sample / "generation_config.json")
        self.materials = load_materials(self.sample / "materials.json", self.config)
        self.textures, _ = scan_textures(self.sample, self.config.active_tier)

    def tearDown(self):
        self._tmp.cleanup()

    def test_four_stages_reported(self):
        base = run_gear(self.config, self.materials, self.textures, self.config.fractional_bits)
        ref = run_gear(self.config, self.materials, self.textures, self.config.fractional_bits + 1)
        report = compare_gears(base, ref, self.config, self.materials)
        for stage in STAGES:
            self.assertIn(stage, report["stages"])
            self.assertGreaterEqual(report["stages"][stage]["max_abs_diff"], 0)
        self.assertEqual(report["gear_widths"],
                         {"configured": self.config.fractional_bits,
                          "reference": self.config.fractional_bits + 1})

    def test_full_pixel_coverage_via_internal_grids(self):
        # Re-run the comparison math directly to assert per-pixel arrays cover
        # every pixel rather than a sampled subset.
        base = run_gear(self.config, self.materials, self.textures, 8)
        ref = run_gear(self.config, self.materials, self.textures, 9)
        pixels = self.config.active_tier * self.config.active_tier
        for layer_id, base_layer in base.layers.items():
            ref_layer = ref.layers[layer_id]
            diff = np.abs(base_layer.accum.astype(np.int64) - ref_layer.accum.astype(np.int64))
            self.assertEqual(diff.size, pixels)
        for name, material in base.materials.items():
            for channel, channel_result in material.channels.items():
                self.assertEqual(channel_result.values.size, pixels)
                self.assertEqual(ref.materials[name].channels[channel].values.size, pixels)

    def test_boundary_sensitive_structure(self):
        # Force narrow bins so bin-crossing pixels are likely to be detected.
        cfg = load_generation_config(self.sample / "generation_config.json")
        object.__setattr__(cfg, "bin_width", 4)
        base = run_gear(cfg, self.materials, self.textures, 8)
        ref = run_gear(cfg, self.materials, self.textures, 9)
        report = compare_gears(base, ref, cfg, self.materials)
        self.assertGreater(report["boundary_sensitive_count"], 0)
        for item in report["boundary_sensitive_pixels"]:
            self.assertIn(item["dominant_error_stage"], STAGES)
            self.assertIn(item["first_breach_stage"], list(STAGES) + ["within_tolerance"])
            self.assertEqual(len(item["pixel"]), 2)
            self.assertIsInstance(item["base_bin"], int)
            self.assertIsInstance(item["reference_bin"], int)
        self.assertEqual(report["boundary_sensitive_count"],
                         len(report["boundary_sensitive_pixels"]))

    def test_reference_gear_must_be_one_bit_wider(self):
        base = run_gear(self.config, self.materials, self.textures, 8)
        ref = run_gear(self.config, self.materials, self.textures, 10)
        with self.assertRaises(ValueError):
            compare_gears(base, ref, self.config, self.materials)


if __name__ == "__main__":
    unittest.main()
