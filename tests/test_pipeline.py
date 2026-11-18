import tempfile
import unittest
from pathlib import Path

import numpy as np

from scripts.generate_samples import write_bundle
from texmat.config import load_generation_config, load_materials
from texmat.diagnostics import run_gear
from texmat.pipeline import compute_layer
from texmat.loader import scan_textures


class PipelineTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)
        self.sample = self.tmp / "sample"
        write_bundle(self.sample, size=24, workers=1, budget=100000)
        self.config = load_generation_config(self.sample / "generation_config.json")
        self.materials = load_materials(self.sample / "materials.json", self.config)
        self.textures, _ = scan_textures(self.sample, self.config.active_tier)

    def tearDown(self):
        self._tmp.cleanup()

    def test_layer_stack_deterministic_pixel_identical(self):
        first = run_gear(self.config, self.materials, self.textures, self.config.fractional_bits)
        second = run_gear(self.config, self.materials, self.textures, self.config.fractional_bits)
        for node_id in first.layers:
            np.testing.assert_array_equal(first.layers[node_id].accum, second.layers[node_id].accum)
            np.testing.assert_array_equal(first.layers[node_id].quantized,
                                          second.layers[node_id].quantized)
        for name in first.materials:
            np.testing.assert_array_equal(first.materials[name].rgba, second.materials[name].rgba)

    def test_quantize_switch_changes_intermediate_segment(self):
        on = run_gear(self.config, self.materials, self.textures, 8)
        cfg_off = load_generation_config(self.sample / "generation_config.json")
        object.__setattr__(cfg_off, "quantize", False)
        off = run_gear(cfg_off, self.materials, self.textures, 8)
        layer_on = on.layers["layer01"]
        layer_off = off.layers["layer01"]
        self.assertTrue(layer_on.is_quantized)
        self.assertFalse(layer_off.is_quantized)
        # The stored intermediate fixed-point segment must really differ.
        self.assertFalse(np.array_equal(layer_on.accum, layer_off.accum))
        # Quantized layer is explicitly narrowed; disabled run keeps wide Q16.
        self.assertTrue(np.any(np.abs(layer_off.accum) >= (1 << 8)))
        self.assertEqual(layer_off.accum.dtype, np.int32)

    def test_layer_order_is_ascending(self):
        previous = None
        for spec in self.config.noise_layers:
            result = compute_layer(spec, self.textures[spec.source_texture], previous,
                                   self.config, 8, True, "half_even")
            self.assertEqual(result.spec.index, spec.index)
            previous = result.accum
        self.assertEqual([s.index for s in self.config.noise_layers],
                         sorted(s.index for s in self.config.noise_layers))

    def test_four_channel_outputs_and_ranges(self):
        gear = run_gear(self.config, self.materials, self.textures, 8)
        for name, material in gear.materials.items():
            self.assertEqual(material.rgba.shape, (24, 24, 4))
            self.assertEqual(material.rgba.dtype, np.uint8)
            for channel, result in material.channels.items():
                self.assertEqual(result.values.shape, (24, 24))
                self.assertTrue(np.all(result.values >= 0))
                self.assertTrue(np.all(result.values <= (1 << 8) - 1))
                expected_layers = len(next(m for m in self.materials if m.name == name).layers)
                self.assertEqual(result.contributions.shape[0], expected_layers)

    def test_equal_weight_tie_annotation(self):
        # mat_metal uses equal weights [0.5, 0.5]
        gear = run_gear(self.config, self.materials, self.textures, 8)
        metal = gear.materials["mat_metal"]
        for channel_result in metal.channels.values():
            self.assertEqual(len(channel_result.tie_pixels), 24 * 24)
        stone = gear.materials["mat_stone"]
        for channel_result in stone.channels.values():
            self.assertEqual(channel_result.tie_pixels, [])

    def test_blend_modes_differ(self):
        # Use weights that do not sum to 1 so additive and weighted diverge.
        materials = load_materials(self.sample / "materials.json", self.config)
        object.__setattr__(materials[0], "weights", (0.8, 0.6, 0.4))
        results = {}
        for mode in ("weighted", "additive", "multiply"):
            object.__setattr__(materials[0], "blend_mode", mode)
            gear = run_gear(self.config, materials, self.textures, 8)
            results[mode] = gear.materials["mat_stone"].rgba
        self.assertFalse(np.array_equal(results["weighted"], results["additive"]))
        self.assertFalse(np.array_equal(results["weighted"], results["multiply"]))
        self.assertFalse(np.array_equal(results["additive"], results["multiply"]))


if __name__ == "__main__":
    unittest.main()
