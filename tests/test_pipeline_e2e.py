import json
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

from texmat_pipeline.error_tracking import (
    ATTRIB_QUANT,
    ATTRIB_ROUND,
    compare_tiers,
    run_tier,
)
from texmat_pipeline.models import load_generation_config, load_materials
from texmat_pipeline.pipeline import TexturePipeline
from texmat_pipeline.stages import StageSpecs, accumulate_layers

ROOT = Path(__file__).resolve().parents[1]
GENERATOR = ROOT / "tools" / "generate_samples.py"


class E2ETestBase(unittest.TestCase):
    seed_args = ["--tiers", "16", "32", "--layers", "4"]

    @classmethod
    def setUpClass(cls):
        cls.tmp = Path(tempfile.mkdtemp(prefix="texmat-e2e-"))
        cls.sample = cls.tmp / "sample"
        subprocess.run(
            [sys.executable, str(GENERATOR), "--out", str(cls.sample), *cls.seed_args],
            check=True, capture_output=True,
        )
        cls.config, _ = load_generation_config(cls.sample / "generation_config.json")
        cls.materials, _ = load_materials(cls.sample / "materials.json")

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp)

    def fresh_pipeline(self, name: str, config=None) -> TexturePipeline:
        work = self.tmp / name
        if work.exists():
            shutil.rmtree(work)
        return TexturePipeline(self.sample, work, config or self.config, self.materials)


class DeterminismTests(E2ETestBase):
    def test_repeated_runs_pixel_identical(self):
        first = self.fresh_pipeline("run_a").run(16)
        second = self.fresh_pipeline("run_b").run(16)
        for la, lb in zip(first.noise_result["layers"], second.noise_result["layers"]):
            np.testing.assert_array_equal(la["image"], lb["image"])
            self.assertEqual(la["fingerprint"], lb["fingerprint"])
        for ma, mb in zip(first.material_result["materials"], second.material_result["materials"]):
            self.assertEqual(ma["material_id"], mb["material_id"])
            for channel in ("diffuse", "roughness", "metallic", "normal"):
                np.testing.assert_array_equal(
                    ma["channels"][channel]["image"], mb["channels"][channel]["image"]
                )
        self.assertEqual(first.composite_keys, second.composite_keys)

    def test_layer_order_is_ascending_and_row_major_shape(self):
        pipe = self.fresh_pipeline("order")
        result = pipe.run(16)
        self.assertEqual(result.noise_result["order"], sorted(result.noise_result["order"]))
        for entry in result.noise_result["layers"]:
            self.assertEqual(entry["image"].shape, (16, 16))
            self.assertEqual(entry["image"].dtype, np.int64)

    def test_metadata_records_quantization_and_width(self):
        result = self.fresh_pipeline("meta").run(16)
        meta = result.specs_meta
        self.assertEqual(meta["frac_bits"], self.config.fixed_point.frac_bits)
        self.assertEqual(meta["unit_levels"], self.config.fixed_point.unit_levels)
        self.assertGreaterEqual(meta["unit_step"], 1)
        report = json.loads((Path(result.snapshots[0]) / "report.json").read_text())
        self.assertEqual(report["fixed_point"], meta)


class RoundingAndTierTests(E2ETestBase):
    def test_round_mode_changes_actual_pixels(self):
        baseline = self.fresh_pipeline("round_base").run(16)
        changed_config = self.config.with_overrides(round_mode="half_up")
        changed = self.fresh_pipeline("round_up", changed_config).run(16)
        diffs = []
        for ma, mb in zip(baseline.material_result["materials"], changed.material_result["materials"]):
            for channel in ("diffuse", "roughness", "metallic", "normal"):
                diffs.append(np.count_nonzero(
                    ma["channels"][channel]["image"] != mb["channels"][channel]["image"]
                ))
        self.assertGreater(sum(diffs), 0)

    def test_quantization_tier_changes_actual_pixels(self):
        baseline = self.fresh_pipeline("level_base").run(16)
        changed_config = self.config.with_overrides(unit_levels=64, signed_levels=128)
        changed = self.fresh_pipeline("level_coarse", changed_config).run(16)
        total = 0
        for ma, mb in zip(baseline.material_result["materials"], changed.material_result["materials"]):
            for channel in ("diffuse", "roughness", "metallic", "normal"):
                total += np.count_nonzero(
                    ma["channels"][channel]["image"] != mb["channels"][channel]["image"]
                )
        self.assertGreater(total, 0)

    def test_error_tracking_attribution_labels(self):
        pipe = self.fresh_pipeline("attr")
        textures, _, _ = pipe.load_textures(16)
        base = run_tier(textures, self.materials, self.config,
                        self.config.fixed_point.frac_bits, self.config.fixed_point.round_mode)
        high = run_tier(textures, self.materials, self.config,
                        self.config.fixed_point.high_tier_frac_bits,
                        self.config.fixed_point.round_mode)
        report = compare_tiers(textures, self.materials, self.config, base, high)
        labels = set(np.unique(report["attribution"]))
        self.assertTrue(labels.issubset({ATTRIB_QUANT, ATTRIB_ROUND, "within_tolerance"}))
        stats = report["attribution_stats"]
        self.assertEqual(
            sum(stats.values()),
            report["attribution"].size,
        )
        self.assertEqual(len(report["first_exceed"]), len(self.materials))
        self.assertGreaterEqual(report["high_frac_bits"], report["base_frac_bits"])

    def test_attribution_fires_with_tight_tier_gap(self):
        # A coarse base width and tight tolerance force inconsistent pixels and
        # exercise both attribution branches across the material set.
        coarse_config = self.config.with_overrides(frac_bits=3, unit_levels=256, signed_levels=512)
        pipe = self.fresh_pipeline("attr_coarse", coarse_config)
        textures, _, _ = pipe.load_textures(16)
        base = run_tier(textures, self.materials, coarse_config, 3, coarse_config.fixed_point.round_mode)
        high = run_tier(textures, self.materials, coarse_config, 4, coarse_config.fixed_point.round_mode)
        report = compare_tiers(textures, self.materials, coarse_config, base, high)
        self.assertGreater(report["inconsistent_pixels"], 0)
        stats = report["attribution_stats"]
        self.assertGreater(stats[ATTRIB_QUANT] + stats[ATTRIB_ROUND], 0)


if __name__ == "__main__":
    unittest.main()


class BudgetEndToEndTests(E2ETestBase):
    seed_args = ["--tiers", "16", "--layers", "4", "--point-budget", "128"]

    def test_small_budget_reports_pending_and_remains_serial_equivalent(self):
        # 16x16 = 256 points per layer but batch budget is only 128.
        self.assertEqual(self.config.point_budget, 128)
        pipe = self.fresh_pipeline("budget")
        result = pipe.run(16)
        codes = [f.code for f in result.failures]
        self.assertIn("E_POINT_BUDGET_PENDING", codes)
        self.assertTrue(result.scheduler["exhausted"])
        self.assertTrue(result.scheduler["serial_equivalent"])
        self.assertGreater(len(result.scheduler["pending"]), 0)
        # Budget deductions are logged in stable layer order.
        seq = result.scheduler["budget_log"]
        self.assertEqual([row[0] for row in seq], sorted(row[0] for row in seq))
