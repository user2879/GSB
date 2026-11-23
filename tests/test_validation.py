"""业务要求 1：样本与配置加载校验。"""

from __future__ import annotations

import json
import os
import tempfile
import unittest

from retbench.config import build_config
from retbench.errors import BenchError, E_CONFIG_INVALID
from retbench.generator import default_config, generate_workspace
from retbench.samples import scan_samples


class ValidationTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.mkdtemp(prefix="retbench-")
        generate_workspace(self.temp, sample_count=6, inject_count=2)
        self.cfg = build_config(default_config())

    def test_scan_collects_injected_failures(self) -> None:
        result = scan_samples(os.path.join(self.temp, "samples"),
                              self.cfg.resolution, self.cfg.vocabulary)
        self.assertEqual(len(result.failures), 2)
        self.assertTrue(all(item.code == "E_SAMPLE_INVALID" for item in result.failures))
        self.assertEqual(len(result.samples), 4)
        self.assertEqual(result.ordered_names(), sorted(result.samples))

    def test_tile_must_divide_resolution(self) -> None:
        cfg = default_config()
        cfg["image"]["tile_size"] = 13
        with self.assertRaises(BenchError) as ctx:
            build_config(cfg)
        self.assertEqual(ctx.exception.code, E_CONFIG_INVALID)

    def test_dimensions_must_be_positive(self) -> None:
        cfg = default_config()
        cfg["image"]["visual_dim"] = 0
        with self.assertRaises(BenchError):
            build_config(cfg)

    def test_weights_must_be_finite(self) -> None:
        cfg = default_config()
        cfg["alignment"]["weights"]["color"] = float("inf")
        with self.assertRaises(BenchError):
            build_config(cfg)

    def test_quantization_enum(self) -> None:
        cfg = default_config()
        cfg["quantization"]["frac_bits"] = 10
        with self.assertRaises(BenchError):
            build_config(cfg)

    def test_scale_tiers_must_increase(self) -> None:
        cfg = default_config()
        cfg["scale"]["tiers"] = [8, 4, 16]
        with self.assertRaises(BenchError):
            build_config(cfg)

    def test_broken_png_header_enters_failure_list(self) -> None:
        sample_dir = os.path.join(self.temp, "samples")
        with open(os.path.join(sample_dir, "s000.png"), "wb") as handle:
            handle.write(b"NOTPNG")
        result = scan_samples(sample_dir, self.cfg.resolution, self.cfg.vocabulary)
        codes = {(item.sample, item.code) for item in result.failures}
        self.assertIn(("s000", "E_SAMPLE_INVALID"), codes)


if __name__ == "__main__":
    unittest.main()
