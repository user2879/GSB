"""业务要求 8：故障注入命中、业务码映射与 JSON 诊断报告。"""

from __future__ import annotations

import os

from vrep.config import Config
from vrep.engine import Engine
from vrep.errors import (
    E_CONFIG_INVALID,
    E_IMAGE_FORMAT,
    E_NONFINITE_FEATURE,
    E_TRUTH_INVALID,
)
from vrep.injection import InjectionManager
from vrep.jsonio import write_json

from ._base import SampleTestBase


class InjectionTests(SampleTestBase):
    def _injection_path(self, entries: list[dict]) -> str:
        path = os.path.join(self.temp_root, "injection.json")
        write_json(path, {"injections": entries})
        return path

    def test_truncated_png_injection(self) -> None:
        path = self._injection_path(
            [{"id": "INJ-001", "type": "truncate_png",
              "image": "img_0001", "keep_bytes": 32}]
        )
        manager = InjectionManager(path)
        engine = Engine(
            os.path.join(self.temp_root, "ws-trunc"), manager
        )
        engine.seed_library(self.samples)
        config = Config.load(self.config_path(), image_size=self.IMAGE_SIZE)
        engine.run_batch(config, batch_no=1)
        codes = {f["code"] for f in engine.ledger["failures"]}
        self.assertIn(E_IMAGE_FORMAT, codes)
        self.assertTrue(manager.diagnostics()["all_hit"])

    def test_truth_out_of_bounds_injection(self) -> None:
        path = self._injection_path(
            [{"id": "INJ-002", "type": "truth_out_of_bounds",
              "image": "img_0002"}]
        )
        manager = InjectionManager(path)
        engine = Engine(os.path.join(self.temp_root, "ws-truth"), manager)
        engine.seed_library(self.samples)
        config = Config.load(self.config_path(), image_size=self.IMAGE_SIZE)
        engine.run_batch(config, batch_no=1)
        codes = {f["code"] for f in engine.ledger["failures"]}
        self.assertIn(E_TRUTH_INVALID, codes)
        self.assertTrue(manager.diagnostics()["all_hit"])

    def test_nonfinite_feature_injection(self) -> None:
        path = self._injection_path(
            [{"id": "INJ-003", "type": "nonfinite_feature",
              "image": "img_0000", "tile_seq": 0}]
        )
        manager = InjectionManager(path)
        engine = Engine(os.path.join(self.temp_root, "ws-nan"), manager)
        engine.seed_library(self.samples)
        config = Config.load(self.config_path(), image_size=self.IMAGE_SIZE)
        engine.run_batch(config, batch_no=1)
        codes = {f["code"] for f in engine.ledger["failures"]}
        self.assertIn(E_NONFINITE_FEATURE, codes)
        diagnostics = manager.diagnostics()
        self.assertTrue(diagnostics["all_hit"])
        self.assertEqual(diagnostics["hits"][0]["id"], "INJ-003")

    def test_quant_level_out_of_range_injection(self) -> None:
        path = self._injection_path(
            [{"id": "INJ-004", "type": "quant_level_out_of_range",
              "bad_level": 13}]
        )
        manager = InjectionManager(path)
        engine = Engine(os.path.join(self.temp_root, "ws-quant"), manager)
        _, failure = engine.load_config(
            self.config_path(), allow_quant_injection=True
        )
        self.assertIsNotNone(failure)
        self.assertEqual(failure.code, E_CONFIG_INVALID)
        self.assertEqual(failure.injection_id, "INJ-004")
        diagnostics = manager.diagnostics()
        self.assertTrue(diagnostics["all_hit"])
        self.assertEqual(diagnostics["items"][0]["expected_code"], "E_QUANT_LEVEL")

    def test_diagnostics_report_structure(self) -> None:
        path = self._injection_path(
            [
                {"id": "INJ-001", "type": "truncate_png",
                 "image": "img_0001", "keep_bytes": 16},
                {"id": "INJ-002", "type": "truth_out_of_bounds",
                 "image": "img_0003"},
            ]
        )
        manager = InjectionManager(path)
        report = manager.diagnostics()
        self.assertEqual(report["injection_count"], 2)
        self.assertEqual(report["hit_count"], 0)
        self.assertFalse(report["all_hit"])
        self.assertEqual([item["id"] for item in report["items"]],
                         ["INJ-001", "INJ-002"])
