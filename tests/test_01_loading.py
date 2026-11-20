"""业务要求 1：加载校验与 E_IMAGE_FORMAT / E_TRUTH_INVALID / E_CONFIG_INVALID。"""

from __future__ import annotations

import json
import os

from vrep.config import Config, ConfigError
from vrep.errors import E_CONFIG_INVALID, E_IMAGE_FORMAT, E_TRUTH_INVALID
from vrep.jsonio import write_json
from vrep.loader import scan_library

from ._base import SampleTestBase


class LoadingTests(SampleTestBase):
    def test_clean_library_scan(self) -> None:
        records, failures, size = scan_library(self.samples)
        self.assertEqual(size, self.IMAGE_SIZE)
        self.assertEqual(len(records), self.COUNT)
        self.assertEqual(failures, [])
        names = [record.name for record in records]
        self.assertEqual(names, sorted(names))

    def test_truncated_png_maps_to_image_format(self) -> None:
        target = os.path.join(self.samples, "img_0001.png")
        with open(target, "rb") as handle:
            raw = handle.read()
        with open(target, "wb") as handle:
            handle.write(raw[:24])
        _, failures, _ = scan_library(self.samples)
        self.assertTrue(any(f.code == E_IMAGE_FORMAT and f.image == "img_0001"
                            for f in failures))
        # 其余图像不受影响
        self.assertEqual(len({f.image for f in failures}), 1)

    def test_out_of_bounds_truth_maps_to_truth_invalid(self) -> None:
        truth_path = os.path.join(self.samples, "img_0002_truth.json")
        payload = {
            "regions": [
                {"category": "bad", "box": [self.IMAGE_SIZE - 2, 2, 10, 10]}
            ]
        }
        write_json(truth_path, payload)
        _, failures, _ = scan_library(self.samples)
        self.assertTrue(any(f.code == E_TRUTH_INVALID and f.image == "img_0002"
                            for f in failures))

    def test_missing_truth_is_truth_invalid(self) -> None:
        os.unlink(os.path.join(self.samples, "img_0003_truth.json"))
        _, failures, _ = scan_library(self.samples)
        self.assertTrue(any(f.code == E_TRUTH_INVALID and f.image == "img_0003"
                            for f in failures))

    def test_non_integer_rectangle_rejected(self) -> None:
        truth_path = os.path.join(self.samples, "img_0004_truth.json")
        payload = {"regions": [{"category": "x", "box": [1.5, 2, 3, 4]}]}
        write_json(truth_path, payload)
        _, failures, _ = scan_library(self.samples)
        self.assertTrue(any(f.code == E_TRUTH_INVALID for f in failures))


class ConfigValidationTests(SampleTestBase):
    def base_config(self) -> dict:
        return {
            "tile_size": self.TILE_SIZE,
            "color_dim": 6,
            "gradient_dim": 6,
            "embed_dim": 8,
            "quant_level": 16,
            "fixed_bits": 12,
            "similarity_threshold": 0.8,
            "max_tiles_per_image": 16,
            "worker_count": 4,
            "point_budget_per_batch": 24,
            "snapshot_commit_interval": 2,
        }

    def test_valid_config(self) -> None:
        config = Config.from_dict(self.base_config(), image_size=self.IMAGE_SIZE)
        self.assertEqual(config.feature_dim, 12)

    def test_tile_must_divide_image_size(self) -> None:
        data = self.base_config()
        data["tile_size"] = 20
        with self.assertRaises(ConfigError):
            Config.from_dict(data, image_size=self.IMAGE_SIZE)

    def test_quant_level_allowed_set(self) -> None:
        data = self.base_config()
        data["quant_level"] = 13
        with self.assertRaises(ConfigError):
            Config.from_dict(data, image_size=self.IMAGE_SIZE)

    def test_fixed_bits_allowed_set(self) -> None:
        data = self.base_config()
        data["fixed_bits"] = 7
        with self.assertRaises(ConfigError):
            Config.from_dict(data, image_size=self.IMAGE_SIZE)

    def test_positive_dimensions_and_budget(self) -> None:
        for field in ("color_dim", "embed_dim", "max_tiles_per_image",
                      "point_budget_per_batch", "worker_count"):
            data = self.base_config()
            data[field] = 0
            with self.assertRaises(ConfigError):
                Config.from_dict(data, image_size=self.IMAGE_SIZE)

    def test_bool_is_not_positive_int(self) -> None:
        data = self.base_config()
        data["worker_count"] = True
        with self.assertRaises(ConfigError):
            Config.from_dict(data, image_size=self.IMAGE_SIZE)

    def test_engine_maps_config_error_to_code(self) -> None:
        from vrep.engine import Engine

        bad_path = os.path.join(self.samples, "bad_config.json")
        data = self.base_config()
        data["quant_level"] = 99
        with open(bad_path, "w", encoding="utf-8") as handle:
            json.dump(data, handle)
        engine = Engine(os.path.join(self.temp_root, "ws-bad-cfg"))
        _, failure = engine.load_config(bad_path)
        self.assertIsNotNone(failure)
        self.assertEqual(failure.code, E_CONFIG_INVALID)
