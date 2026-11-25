from __future__ import annotations

import copy
import unittest
from pathlib import Path

import numpy as np

from tile_anomaly.config import load_config
from tile_anomaly.detection import (
    compute_robust_statistics,
    compute_window_scores,
    threshold_decisions,
)
from tile_anomaly.features import compute_window_feature
from tile_anomaly.fusion import compute_fusion_band, remove_small_components
from tile_anomaly.manifest import load_manifest
from tile_anomaly.manifest import TileSpec
from tile_anomaly.scheduler import allocate_windows
from tile_anomaly.tiles import TileData, build_mosaic, load_tiles
from tile_anomaly.windows import generate_windows

from .helpers import BASE_SAMPLE, config_path


class QuantizationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.config = load_config(config_path(BASE_SAMPLE))
        manifest, _, invalid = load_manifest(
            BASE_SAMPLE / "input" / "manifest.json", cls.config, {}
        )
        cls.tiles, _ = load_tiles(
            manifest,
            BASE_SAMPLE / "input" / "tiles",
            cls.config,
            {},
            skip_tile_ids=invalid,
        )
        mosaic = build_mosaic(cls.tiles, cls.config, manifest)
        cls.windows, _ = generate_windows(manifest, mosaic, set(cls.tiles), cls.config)
        cls.mosaic = mosaic

    def test_width_and_gear_change_integers_and_dequantized_values(self) -> None:
        window = self.windows[0]
        baseline = compute_window_feature(window, self.mosaic, self.tiles, self.config)
        wider = compute_window_feature(
            window,
            self.mosaic,
            self.tiles,
            self.config,
            fraction_bits=self.config["quantization"]["fraction_bits"] + 1,
        )
        coarser_gear = compute_window_feature(
            window, self.mosaic, self.tiles, self.config, gear=1
        )
        baseline_integers = [dimension["q"] for dimension in baseline["dimensions"]]
        wider_integers = [dimension["q"] for dimension in wider["dimensions"]]
        coarser_integers = [dimension["q"] for dimension in coarser_gear["dimensions"]]
        self.assertNotEqual(baseline_integers, wider_integers)
        self.assertNotEqual(baseline_integers, coarser_integers)
        self.assertTrue(
            any(
                current["dequantized"] != wider_dimension["dequantized"]
                for current, wider_dimension in zip(
                    baseline["dimensions"], wider["dimensions"]
                )
            )
        )
        for dimension in baseline["dimensions"]:
            self.assertIn("quant_step_numerator", dimension)
            self.assertIn("quant_step_denominator", dimension)
            self.assertIn("q", dimension)
            self.assertIn("residual_in_steps", dimension)
        self.assertEqual(len(baseline["observation"]), 4)

    def test_feature_record_keeps_upstream_and_parameter_fingerprints(self) -> None:
        record = compute_window_feature(
            self.windows[0], self.mosaic, self.tiles, self.config
        )
        self.assertEqual(len(record["upstream_tile_keys"]), len(record["source_tile_ids"]))
        self.assertTrue(record["feature_key"])
        self.assertTrue(record["window_parameter_key"])
        self.assertTrue(record["quantization_key"])


class ThresholdTests(unittest.TestCase):
    def test_equality_is_anomalous_and_half_step_is_boundary(self) -> None:
        config = {
            "detector": {
                "threshold_permille": 500,
                "final_score_step_micro": 1000,
            }
        }
        scores = [
            {"window_id": "a", "owner_tile_id": "t", "score_key": "ka", "score_micro": 600000},
            {"window_id": "b", "owner_tile_id": "t", "score_key": "kb", "score_micro": 500000},
            {"window_id": "c", "owner_tile_id": "t", "score_key": "kc", "score_micro": 500500},
            {"window_id": "d", "owner_tile_id": "t", "score_key": "kd", "score_micro": 499500},
        ]
        decisions = {record["window_id"]: record for record in threshold_decisions(scores, config)}
        self.assertTrue(decisions["b"]["is_anomaly"])
        self.assertTrue(decisions["b"]["boundary"])
        self.assertTrue(decisions["c"]["boundary"])
        self.assertTrue(decisions["d"]["boundary"])
        self.assertFalse(decisions["d"]["is_anomaly"])
        self.assertTrue(decisions["a"]["is_anomaly"])
        self.assertFalse(decisions["a"]["boundary"])


def _synthetic_feature(index: int, observation: list[int]) -> dict:
    return {
        "window_id": f"w{index:03d}",
        "owner_tile_id": "tile",
        "core_rect": [index * 8, 0, index * 8 + 8, 8],
        "feature_key": f"feature-{index}",
        "observation": observation,
    }


class StatisticsOrderTests(unittest.TestCase):
    def test_shuffled_inputs_produce_identical_scores(self) -> None:
        config = {
            "detector": {
                "minimum_robust_scale_micro": 1000000,
                "kde_bandwidth_milli": 1400,
                "distance_weight_milli": 600,
                "density_weight_milli": 400,
            }
        }
        features = [
            _synthetic_feature(0, [10, 20, 30, 40]),
            _synthetic_feature(1, [11, 21, 30, 41]),
            _synthetic_feature(2, [9, 19, 31, 39]),
            _synthetic_feature(3, [80, 15, 25, 90]),
        ]
        stats = compute_robust_statistics(features, config)
        first = {record["window_id"]: record for record in compute_window_scores(features, stats, config)}
        shuffled = list(reversed(copy.deepcopy(features)))
        shuffled_stats = compute_robust_statistics(shuffled, config)
        second = {
            record["window_id"]: record
            for record in compute_window_scores(shuffled, shuffled_stats, config)
        }
        self.assertEqual(stats, shuffled_stats)
        self.assertEqual(first, second)
        self.assertIn("accumulation_order", first["w000"])


def _dummy_window(window_id: str, owner: str, row: int, col: int):
    from tile_anomaly.windows import WindowSpec

    return WindowSpec(
        window_id=window_id,
        core_row=row,
        core_col=col,
        core_size=2,
        core_rect=(row, col, row + 2, col + 2),
        context_rect=(row, col, row + 2, col + 2),
        owner_tile_id=owner,
        source_tile_ids=(owner,),
    )


class SchedulerTests(unittest.TestCase):
    def test_quota_aborts_batch_without_consuming(self) -> None:
        first = _dummy_window("w1", "t1", 0, 0)
        second = _dummy_window("w2", "t1", 0, 8)
        allocation = allocate_windows(
            [("t1", [first, second])],
            stage="window_features",
            budget_limit=10,
            consumed_budget=0,
            tile_quota=1,
        )
        self.assertTrue(allocation.aborted)
        self.assertEqual(allocation.selected, [])
        self.assertEqual(allocation.consumed_increment, 0)
        self.assertEqual(allocation.events[0].code, "E_TILE_QUOTA_EXCEEDED")

    def test_budget_keeps_stable_pending_prefix(self) -> None:
        windows = [_dummy_window(f"w{i}", "t1", i * 2, 0) for i in range(4)]
        allocation = allocate_windows(
            [("t1", windows)],
            stage="window_features",
            budget_limit=2,
            consumed_budget=0,
            tile_quota=10,
        )
        self.assertEqual([window.window_id for _, window in allocation.selected], ["w0", "w1"])
        self.assertEqual([entry["window_id"] for entry in allocation.pending], ["w2", "w3"])
        self.assertEqual(allocation.pending[0]["code"], "E_BUDGET_PENDING")


def _dummy_tile(tile_id: str, row: int, col: int, size: int) -> TileData:
    spec = TileSpec(
        tile_id=tile_id,
        row=row,
        col=col,
        size=size,
        band_set_version="v",
        bands={},
        neighbors=[],
    )
    return TileData(spec=spec, arrays={}, band_hashes={}, content_key=f"content-{tile_id}")


class FusionTests(unittest.TestCase):
    def test_tie_uses_lexicographic_tile_decision(self) -> None:
        tile_size = 8
        tiles = {
            "a": _dummy_tile("a", 0, 0, tile_size),
            "b": _dummy_tile("b", 0, 1, tile_size),
        }
        config = {
            "tile": {"size": tile_size},
            "fusion": {"overlap_width": 8},
            "detector": {"threshold_permille": 500, "final_score_step_micro": 1000},
        }
        pixel_maps = {
            "a": {
                "score_micro": np.full((tile_size, tile_size), 600000, dtype=np.int64),
                "decision": np.ones((tile_size, tile_size), dtype=np.int8),
            },
            "b": {
                "score_micro": np.full((tile_size, tile_size), 400000, dtype=np.int64),
                "decision": np.zeros((tile_size, tile_size), dtype=np.int8),
            },
        }
        plan = {
            "band_id": "a__b__vertical",
            "tile_a_id": "a",
            "tile_b_id": "b",
            "tile_a_cache_key": "summary-a",
            "tile_b_cache_key": "summary-b",
            "tile_a_content_key": "content-a",
            "tile_b_content_key": "content-b",
            "side_a_score_keys": ["sa"],
            "side_b_score_keys": ["sb"],
            "fusion_key": "fusion-key",
            "fusion_strategy_version": "v1",
            "geometry": {
                "orientation": "vertical",
                "boundary": 8,
                "top": 0,
                "bottom": 8,
                "left": 0,
                "right": 16,
                "a_side": "left",
                "b_side": "right",
            },
        }
        result = compute_fusion_band(plan, tiles, pixel_maps, config)
        self.assertEqual(result["tie_pixel_count"], 128)
        self.assertTrue(all(value == 1 for value in result["update_values"]))
        self.assertIn("lexicographically", result["tie_reason"])
        self.assertEqual(result["tile_a_cache_key"], "summary-a")
        self.assertEqual(result["overlap_geometry"], plan["geometry"])

    def test_small_anomaly_component_is_backfilled(self) -> None:
        anomaly = np.zeros((12, 12), dtype=np.int8)
        anomaly[1:3, 1:3] = 1
        anomaly[5:11, 5:11] = 1
        info = remove_small_components(anomaly, minimum_size=5)
        self.assertEqual(info["removed_component_count"], 1)
        self.assertEqual(info["removed_pixel_count"], 4)
        self.assertTrue(np.all(anomaly[1:3, 1:3] == 0))
        self.assertTrue(np.all(anomaly[5:11, 5:11] == 1))


if __name__ == "__main__":
    unittest.main()
