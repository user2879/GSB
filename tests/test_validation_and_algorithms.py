import unittest

import numpy as np

from tests._helpers import load_test_config, make_workspace
from workbench import algorithms as algo
from workbench.errors import (
    E_CONFIG_INVALID,
    E_REGION_OUT_OF_BOUND,
    E_SAMPLE_INVALID,
    ConfigError,
    SampleError,
)
from workbench.sample_gen import (
    build_instructions,
    generate_tier,
    render_image,
    sample_paths,
    validate_sample,
)


class ConfigValidationTests(unittest.TestCase):
    def setUp(self):
        self.workspace = make_workspace()
        self.addCleanup(self.workspace.cleanup)

    def test_workers_must_be_positive(self):
        with self.assertRaises(ConfigError) as ctx:
            load_test_config(self.workspace, {"workers": 0})
        self.assertEqual(ctx.exception.code, E_CONFIG_INVALID)

    def test_tiers_must_be_monotonic(self):
        bad_tiers = [
            {"name": "small", "sample_count": 6, "primitives": 3, "regions": 2, "instruction_length": 3},
            {"name": "medium", "sample_count": 5, "primitives": 5, "regions": 3, "instruction_length": 5},
        ]
        with self.assertRaises(ConfigError) as ctx:
            load_test_config(self.workspace, {"tiers": bad_tiers})
        self.assertEqual(ctx.exception.code, E_CONFIG_INVALID)

    def test_invalid_quant_levels(self):
        with self.assertRaises(ConfigError) as ctx:
            load_test_config(self.workspace, {"quant_levels": [1, 32]})
        self.assertEqual(ctx.exception.code, E_CONFIG_INVALID)

    def test_missing_config_file(self):
        # direct call against a nonexistent path
        from workbench.config import load_config

        with self.assertRaises(ConfigError):
            load_config(str(__import__("pathlib").Path(self.workspace.name) / "nope.yaml"))


class SampleValidationTests(unittest.TestCase):
    def setUp(self):
        self.workspace = make_workspace()
        self.addCleanup(self.workspace.cleanup)
        self.config = load_test_config(self.workspace)
        self.tier = self.config.tier()
        generate_tier(self.config, self.tier)

    def test_generated_sample_is_valid(self):
        loaded = validate_sample(self.config, "sample_0000", self.tier)
        self.assertEqual(loaded["image"].shape, (40, 40, 3))

    def test_corrupt_png_header_yields_sample_error(self):
        paths = sample_paths(self.config, "sample_0001", self.tier)
        paths["png"].write_bytes(b"not-a-png-file")
        with self.assertRaises(SampleError) as ctx:
            validate_sample(self.config, "sample_0001", self.tier)
        self.assertEqual(ctx.exception.code, E_SAMPLE_INVALID)

    def test_region_out_of_bound_code(self):
        from workbench.utils import atomic_write_json

        paths = sample_paths(self.config, "sample_0002", self.tier)
        instruction = __import__("json").loads(paths["json"].read_text())
        instruction["regions"][0]["x"] = self.config.width - 1
        atomic_write_json(paths["json"], instruction)
        with self.assertRaises(SampleError) as ctx:
            validate_sample(self.config, "sample_0002", self.tier)
        self.assertEqual(ctx.exception.code, E_REGION_OUT_OF_BOUND)

    def test_non_finite_strength_rejected(self):
        from workbench.utils import atomic_write_json

        paths = sample_paths(self.config, "sample_0003", self.tier)
        instruction = __import__("json").loads(paths["json"].read_text())
        instruction["ops"][0]["style_strength"] = float("inf")
        atomic_write_json(paths["json"], instruction)
        with self.assertRaises(SampleError) as ctx:
            validate_sample(self.config, "sample_0003", self.tier)
        self.assertEqual(ctx.exception.code, E_SAMPLE_INVALID)


class AlgorithmDeterminismTests(unittest.TestCase):
    def setUp(self):
        self.workspace = make_workspace()
        self.addCleanup(self.workspace.cleanup)
        self.config = load_test_config(self.workspace)
        self.tier = self.config.tier()
        self.image = render_image(self.config, self.tier, 0)
        self.instruction = build_instructions(self.config, self.tier, 0)
        self.region = self.instruction["regions"][0]

    def test_mask_paths_are_pixel_identical_on_repeat(self):
        for path, param in (("threshold", 90), ("edge", 60)):
            params = {"mask_threshold": 90, "edge_threshold": 60}
            first = algo.generate_mask(path, self.image, self.region, params)
            second = algo.generate_mask(path, self.image, self.region, params)
            self.assertTrue(np.array_equal(first, second))
            self.assertEqual(set(np.unique(first)).issubset({0, 1}), True)
            self.assertEqual(first.dtype, np.uint8)
            self.assertGreater(int(first.sum()), 0)

    def test_inpaint_paths_integer_output_and_deterministic(self):
        params = {"mask_threshold": 90, "edge_threshold": 60}
        mask = algo.generate_mask("threshold", self.image, self.region, params)
        for path in ("neighbor_mean", "directional"):
            a = algo.inpaint(path, self.image, mask)
            b = algo.inpaint(path, self.image, mask)
            self.assertTrue(np.array_equal(a, b))
            self.assertEqual(a.dtype, np.uint8)

    def test_recolor_lut_is_deterministic_and_summarized(self):
        params = {"mask_threshold": 90, "edge_threshold": 60}
        mask = algo.generate_mask("edge", self.image, self.region, params)
        out_a, summary_a = algo.apply_recolor(self.image, mask, 120, 0.7, "fp-abc")
        out_b, summary_b = algo.apply_recolor(self.image, mask, 120, 0.7, "fp-abc")
        self.assertTrue(np.array_equal(out_a, out_b))
        self.assertIn("changed_pixels", summary_a)
        self.assertIn("lut_summary", summary_a)
        self.assertEqual(summary_a["lut_summary"], summary_b["lut_summary"])

    def test_style_paths_quantization_aware(self):
        for path in ("lut", "linear"):
            out_a, meta_a = algo.apply_style(path, self.image, 32, style_seed=123)
            out_b, meta_b = algo.apply_style(path, self.image, 32, style_seed=123)
            self.assertTrue(np.array_equal(out_a, out_b))
            self.assertEqual(out_a.shape, self.image.shape)
            self.assertEqual(out_a.ndim, 3)
            self.assertEqual(meta_a["quant_levels"], 32)
            self.assertEqual(meta_a["path"], path)

    def test_composite_mask_retention_policy(self):
        params = {"mask_threshold": 90, "edge_threshold": 60}
        mask = algo.generate_mask("threshold", self.image, self.region, params)
        styled, _ = algo.apply_style("linear", self.image, 32, style_seed=7)
        merged = algo.composite(self.image, styled, mask)
        self.assertTrue(np.array_equal(merged[mask == 0], self.image[mask == 0]))
        self.assertTrue(np.array_equal(merged[mask == 1], styled[mask == 1]))


if __name__ == "__main__":
    unittest.main()
