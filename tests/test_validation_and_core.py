from __future__ import annotations

import json
import shutil
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import numpy as np
from panorama_workbench.config import ConfigError, load_config, parse_config
from panorama_workbench.engine import Workbench
from panorama_workbench.inference import (
    DeterministicScorer,
    TileGeometry,
    TileMask,
    infer_tile,
    tile_geometries,
)
from panorama_workbench.merging import merge_masks
from panorama_workbench.samples import generate_demo
from panorama_workbench.validation import load_inputs


class ValidationAndCoreTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.temp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temp.name)
        cls.demo = generate_demo(cls.root / "demo", image_count=2)
        cls.config = load_config(cls.demo.config)

    @classmethod
    def tearDownClass(cls) -> None:
        cls.temp.cleanup()

    def test_validation_failure_codes_and_isolation(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            samples = root / "samples"
            shutil.copytree(self.demo.samples, samples)
            (samples / "bad_format.png").write_bytes(b"not a png")
            (samples / "bad_format.json").write_text("{}", encoding="utf-8")
            truth_path = samples / "scene_01.json"
            truth = json.loads(truth_path.read_text(encoding="utf-8"))
            truth["instances"][0]["category"] = 999
            truth_path.write_text(json.dumps(truth), encoding="utf-8")
            result = load_inputs(samples, self.config)
            codes = {failure.code for failure in result.failures}
            self.assertIn("E_IMAGE_FORMAT", codes)
            self.assertIn("E_TRUTH_INVALID", codes)
            self.assertNotIn("scene_01.png", {sample.name for sample in result.samples})
            self.assertNotIn("bad_format.png", {sample.name for sample in result.samples})
            config_result = load_inputs(samples, replace(self.config, window_size=15))
            self.assertIn(
                "E_CONFIG_INVALID",
                {failure.code for failure in config_result.failures},
            )

    def test_config_rejects_invalid_enums_and_ranges(self) -> None:
        with self.assertRaises(ConfigError):
            parse_config(
                {
                    "categories": [{"code": 0, "name": "background", "weight": 1.0}],
                    "window": {"size": 16, "stride": 8},
                    "merge_iou": 1.5,
                    "min_instance_area": 1,
                    "quantization": {"fixed_bits": 10, "levels": [4], "level": 4},
                    "scheduling": {
                        "max_workers": 1,
                        "pixel_budget": 100,
                        "max_instances_per_image": 3,
                        "cache_budget": 10,
                        "snapshot_interval": 1,
                    },
                }
            )
        with self.assertRaises(ConfigError):
            parse_config(
                {
                    "categories": [{"code": 0, "name": "background", "weight": 1.0}],
                    "window": {"size": 16, "stride": 8},
                    "merge_iou": 0.2,
                    "min_instance_area": 1,
                    "quantization": {"fixed_bits": 10, "levels": [4], "level": 8},
                    "scheduling": {
                        "max_workers": 1,
                        "pixel_budget": 100,
                        "max_instances_per_image": 3,
                        "cache_budget": 10,
                        "snapshot_interval": 1,
                    },
                }
            )

    def test_tile_inference_is_repeatable_and_quantized(self) -> None:
        sample = load_inputs(self.demo.samples, self.config).samples[0]
        geometry = tile_geometries(
            sample.width,
            sample.height,
            self.config.window_size,
            self.config.window_stride,
        )[0]
        scorer = DeterministicScorer(self.config)
        first = infer_tile(sample.rgb, geometry, self.config, scorer)
        second = infer_tile(sample.rgb, geometry, self.config, scorer)
        self.assertEqual(first.feature_key, second.feature_key)
        np.testing.assert_array_equal(first.features, second.features)
        self.assertEqual(first.residual, second.residual)
        self.assertIn("residual_fixed", first.residual)
        self.assertGreater(first.residual["up_count"] + first.residual["down_count"], 0)

        low_quant = replace(self.config, quant_level=1)
        high_quant = replace(self.config, quant_level=64)
        low = infer_tile(sample.rgb, geometry, low_quant, DeterministicScorer(low_quant))
        high = infer_tile(sample.rgb, geometry, high_quant, DeterministicScorer(high_quant))
        self.assertNotEqual(low.feature_key, high.feature_key)
        self.assertNotEqual(low.residual["abs_residual_fixed"], high.residual["abs_residual_fixed"])

    def test_merge_order_min_area_and_instance_limit(self) -> None:
        geometry = TileGeometry(row=0, col=0, x=0, y=0, size=4)
        grid = np.zeros((4, 4), dtype=np.int32)
        grid[0, 0] = 1
        grid[0, 1] = 1
        mask = TileMask(
            geometry=geometry,
            feature_key="a",
            mask_key="a",
            class_grid=grid,
            class_counts={1: 2},
            payload_hash="a",
        )
        merged = merge_masks(
            8,
            8,
            [mask],
            replace(self.config, min_instance_area=1, max_instances_per_image=4),
        )
        self.assertEqual(len(merged.instances), 1)
        self.assertEqual(merged.instances[0].area, 2)
        dropped = merge_masks(
            8,
            8,
            [mask],
            replace(self.config, min_instance_area=3, max_instances_per_image=4),
        )
        self.assertEqual(len(dropped.instances), 0)
        self.assertEqual(dropped.dropped_instances[0]["reason_code"], "E_INSTANCE_BELOW_MIN_AREA")

        second_grid = np.zeros((4, 4), dtype=np.int32)
        second_grid[2, 2] = 2
        second_grid[2, 3] = 2
        second_grid[3, 2] = 2
        second_grid[3, 3] = 2
        second = TileMask(
            geometry=TileGeometry(row=0, col=1, x=4, y=0, size=4),
            feature_key="b",
            mask_key="b",
            class_grid=second_grid,
            class_counts={2: 4},
            payload_hash="b",
        )
        capped = merge_masks(
            8,
            8,
            [mask, second],
            replace(self.config, min_instance_area=1, max_instances_per_image=1),
        )
        self.assertEqual(len(capped.instances), 1)
        self.assertEqual(capped.instances[0].category_code, 2)
        self.assertEqual(capped.truncated_instances[0]["reason_code"], "E_INSTANCE_LIMIT_TRUNCATED")

    def test_quality_failure_and_real_quantization_metric_change(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            one_sample = root / "samples"
            one_sample.mkdir()
            shutil.copy2(self.demo.samples / "scene_01.png", one_sample / "scene_01.png")
            shutil.copy2(self.demo.samples / "scene_01.json", one_sample / "scene_01.json")
            strict = replace(
                self.config,
                min_semantic_accuracy=1.0,
                min_instance_quality=1.0,
            )
            result = Workbench(root / "strict-state").run(one_sample, strict)
            self.assertEqual(result.exit_code, 4)
            self.assertIn("E_QUALITY_BELOW", {failure.code for failure in result.failures})

            low = Workbench(root / "low-state").run(
                one_sample, replace(self.config, quant_level=1)
            )
            high = Workbench(root / "high-state").run(
                one_sample, replace(self.config, quant_level=64)
            )
            low_image = low.report["images"]["scene_01.png"]
            high_image = high.report["images"]["scene_01.png"]
            self.assertNotEqual(low_image["semantic_raster"], high_image["semantic_raster"])
            self.assertNotEqual(
                low_image["evaluation"]["semantic_accuracy"],
                high_image["evaluation"]["semantic_accuracy"],
            )

    def test_concurrent_and_serial_outputs_match(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            serial = Workbench(root / "serial").run(
                self.demo.samples, replace(self.config, max_workers=1)
            )
            parallel = Workbench(root / "parallel").run(
                self.demo.samples, replace(self.config, max_workers=4)
            )
            self.assertLessEqual(parallel.report["run"]["peak_workers"], 4)
            self.assertEqual(_image_core(serial.report), _image_core(parallel.report))

    def test_budget_is_stable_and_pending_is_prioritized(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            samples = root / "samples"
            samples.mkdir()
            for index in (1, 2):
                shutil.copy2(
                    self.demo.samples / f"scene_{index:02d}.png",
                    samples / f"scene_{index:02d}.png",
                )
                shutil.copy2(
                    self.demo.samples / f"scene_{index:02d}.json",
                    samples / f"scene_{index:02d}.json",
                )
            budgeted = replace(
                self.config,
                max_workers=2,
                pixel_budget=30000,
                cache_budget=4096,
            )
            first = Workbench(root / "state").run(samples, budgeted)
            self.assertEqual(first.exit_code, 3)
            self.assertTrue(first.pending)
            self.assertLessEqual(first.report["run"]["peak_workers"], 2)
            codes = {failure.code for failure in first.failures}
            self.assertIn("E_IMAGE_BUDGET_EXHAUSTED", codes)
            self.assertIn("E_PIXEL_BUDGET_PENDING", codes)
            first_completed = set(first.report["state"]["completed_images"])
            second = Workbench(root / "state").run(samples, budgeted)
            second_completed = set(second.report["state"]["completed_images"])
            self.assertTrue(first_completed.issubset(second_completed))
            self.assertGreater(len(second_completed), len(first_completed))


def _image_core(report: dict) -> dict:
    result = {}
    for name, image in report["images"].items():
        result[name] = {
            "dropped_instances": image["dropped_instances"],
            "evaluation": image["evaluation"],
            "instance_raster": image["instance_raster"],
            "instances": image["instances"],
            "semantic_raster": image["semantic_raster"],
            "truncated_instances": image["truncated_instances"],
        }
    return result


if __name__ == "__main__":
    unittest.main()
