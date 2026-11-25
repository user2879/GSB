from __future__ import annotations

import json
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from tile_anomaly.edits import apply_edit_commands, write_edit_commands
from tile_anomaly.pipeline import run_pipeline

from .helpers import (
    config_path,
    load_json,
    make_workspace,
    read_config,
    set_config_value,
    write_config,
)


def _stage(snapshot: Path, name: str):
    return load_json(snapshot / f"{name}.json")


def _raster(root: Path) -> np.ndarray:
    return np.array(Image.open(root / "artifacts" / "output" / "pixel_anomaly.png"))


class BaselineIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.root = make_workspace("tile-baseline")
        cls.result = run_pipeline(config_path(cls.root), reset_state=True)

    def test_complete_snapshot_and_raster(self) -> None:
        self.assertTrue(self.result.complete)
        snapshot = self.result.snapshot_dir
        self.assertTrue((snapshot / "pixel_anomaly.png").exists())
        self.assertTrue((self.root / "artifacts" / "output" / "pixel_anomaly.png").exists())
        features = _stage(snapshot, "features")
        scores = _stage(snapshot, "scores")
        fusion = _stage(snapshot, "fusion")
        pixel_map = _stage(snapshot, "pixel_map")
        self.assertEqual(features["count"], 576)
        self.assertEqual(scores["count"], 576)
        self.assertEqual(len(fusion["by_key"]), 12)
        self.assertEqual(pixel_map["shape"], [192, 192])
        self.assertEqual(pixel_map["unknown_pixel_count"], 0)
        self.assertEqual(self.result.run_manifest["completed_tile_cursor"]["tile_id"], "tile_r2c2")

    def test_feature_and_diagnostic_fields_are_present(self) -> None:
        snapshot = self.result.snapshot_dir
        features = _stage(snapshot, "features")
        record = next(iter(features["by_key"].values()))
        self.assertEqual(len(record["dimensions"]), 4)
        self.assertEqual(len(record["observation"]), 4)
        for dimension in record["dimensions"]:
            self.assertIn("quant_step_numerator", dimension)
            self.assertIn("residual_in_steps", dimension)
        diagnostics = _stage(snapshot, "diagnostics")
        current = diagnostics["by_key"][diagnostics["current_key"]]
        self.assertGreaterEqual(current["boundary_count"], 1)
        self.assertEqual(current["higher_width_fraction_bits"], 13)
        self.assertIn("deviations", current["records"][0])

    def test_fusion_conclusions_carry_cache_keys_and_geometry(self) -> None:
        fusion = _stage(self.result.snapshot_dir, "fusion")
        for conclusion in fusion["by_key"].values():
            self.assertTrue(conclusion["tile_a_cache_key"])
            self.assertTrue(conclusion["tile_b_cache_key"])
            self.assertIn("boundary", conclusion["overlap_geometry"])
            self.assertGreater(conclusion["overlap_pixel_count"], 0)


class FaultIsolationTests(unittest.TestCase):
    def test_four_fault_kinds_are_isolated_per_tile(self) -> None:
        root = make_workspace("tile-fault")
        result = run_pipeline(
            config_path(root),
            fault_config_path=root / "config" / "fault_injection.json",
            reset_state=True,
        )
        faults = {
            (event["tile_id"], event["stage"], event["code"])
            for event in result.events
        }
        self.assertIn(("tile_r0c1", "tile_load", "E_TILE_FORMAT"), faults)
        self.assertIn(("tile_r1c0", "manifest_validation", "E_TILE_FORMAT"), faults)
        self.assertIn(("tile_r2c1", "manifest_validation", "E_TILE_FORMAT"), faults)
        self.assertIn(("tile_r2c2", "quantization_validation", "E_TILE_FORMAT"), faults)
        self.assertGreater(
            sum(1 for event in result.events if event["code"] == "E_WINDOW_CONTEXT_UNAVAILABLE"),
            0,
        )
        self.assertTrue(result.complete)
        pixel_map = _stage(result.snapshot_dir, "pixel_map")
        self.assertGreater(pixel_map["unknown_pixel_count"], 0)
        self.assertGreater(pixel_map["anomaly_pixel_count"], 0)


class IncrementalTests(unittest.TestCase):
    def test_tile_replacement_incremental_equals_full_rebuild(self) -> None:
        incremental = make_workspace("tile-incremental")
        full = make_workspace("tile-full-rebuild")
        base_result = run_pipeline(config_path(incremental), reset_state=True)
        self.assertTrue(base_result.complete)

        edit_path = incremental / "input" / "single_edit.json"
        write_edit_commands(
            edit_path,
            [
                {
                    "op": "replace_band",
                    "tile_id": "tile_r1c2",
                    "band": "red",
                    "method": "patch_add",
                    "rect": [20, 20, 20, 20],
                    "value": 55,
                }
            ],
        )
        apply_edit_commands(config_path(incremental), edit_path)
        incremental_result = run_pipeline(config_path(incremental))
        self.assertTrue(incremental_result.complete)
        self.assertEqual(
            incremental_result.audit_summary["stages"]["window_features"]["reuse"], 486
        )
        self.assertEqual(
            incremental_result.audit_summary["stages"]["window_features"]["invalidate"], 90
        )
        self.assertEqual(
            incremental_result.audit_summary["stages"]["window_anomaly_scores"]["reuse"], 0
        )

        apply_edit_commands(config_path(full), edit_path)
        full_result = run_pipeline(config_path(full), reset_state=True)
        self.assertTrue(full_result.complete)
        self.assertEqual(
            incremental_result.run_manifest["stage_content_keys"],
            full_result.run_manifest["stage_content_keys"],
        )
        for stage_name in (
            "features",
            "statistics",
            "scores",
            "summaries",
            "fusion",
            "pixel_map",
        ):
            self.assertEqual(
                _stage(incremental_result.snapshot_dir, stage_name),
                _stage(full_result.snapshot_dir, stage_name),
                stage_name,
            )
        self.assertTrue(np.array_equal(_raster(incremental), _raster(full)))

    def test_threshold_change_reuses_features_statistics_and_scores(self) -> None:
        root = make_workspace("tile-threshold")
        first = run_pipeline(config_path(root), reset_state=True)
        set_config_value(root, "detector.threshold_permille", 600)
        second = run_pipeline(config_path(root))
        stages = second.audit_summary["stages"]
        self.assertEqual(stages["window_features"]["reuse"], 576)
        self.assertEqual(stages["window_features"]["invalidate"], 0)
        self.assertEqual(stages["robust_statistics"]["reuse"], 1)
        self.assertEqual(stages["window_anomaly_scores"]["reuse"], 576)
        self.assertEqual(stages["window_anomaly_scores"]["invalidate"], 0)
        self.assertEqual(stages["tile_decision_summaries"]["invalidate"], 9)
        self.assertEqual(stages["fusion_conclusions"]["invalidate"], 12)
        self.assertEqual(second.run_manifest["scheduler_budget"]["consumed"], 0)
        self.assertNotEqual(
            first.run_manifest["stage_content_keys"]["summary_set"],
            second.run_manifest["stage_content_keys"]["summary_set"],
        )

    def test_window_and_quantization_edits_rebuild_content(self) -> None:
        root = make_workspace("tile-window-quant")
        baseline = run_pipeline(config_path(root), reset_state=True)
        self.assertTrue(baseline.complete)

        config = read_config(root)
        config["window"] = {"size": 32, "step": 16, "context_overlap": 8}
        write_config(root, config)
        resized = run_pipeline(config_path(root))
        self.assertTrue(resized.complete)
        self.assertEqual(_stage(resized.snapshot_dir, "features")["count"], 144)
        self.assertEqual(resized.audit_summary["stages"]["window_features"]["invalidate"], 144)
        self.assertEqual(resized.audit_summary["stages"]["window_features"]["reuse"], 0)

        config = read_config(root)
        config["quantization"]["gear"] = 3
        write_config(root, config)
        requantized = run_pipeline(config_path(root))
        self.assertTrue(requantized.complete)
        self.assertEqual(
            requantized.audit_summary["stages"]["window_features"]["invalidate"], 144
        )
        self.assertEqual(
            requantized.audit_summary["stages"]["window_anomaly_scores"]["invalidate"], 144
        )
        first_feature = next(
            iter(_stage(requantized.snapshot_dir, "features")["by_key"].values())
        )
        self.assertEqual(first_feature["gear"], 3)

    def test_threshold_boundary_flip_diagnostic(self) -> None:
        root = make_workspace("tile-flip")
        set_config_value(root, "detector.threshold_permille", 184)
        result = run_pipeline(config_path(root), reset_state=True)
        diagnostics = _stage(result.snapshot_dir, "diagnostics")
        current = diagnostics["by_key"][diagnostics["current_key"]]
        self.assertEqual(current["boundary_count"], 1)
        self.assertEqual(current["flip_count"], 1)
        flipped = [record for record in current["records"] if record["decision_flipped"]]
        self.assertEqual(flipped[0]["primary_flip_cause"], "robust_scale_rounding")
        self.assertGreater(abs(flipped[0]["deviations"]["score_delta_micro"]), 1000)
        self.assertIn(
            "index_ratio_quantization_relative", flipped[0]["deviations"]
        )


class ConcurrencyTests(unittest.TestCase):
    def test_serial_and_parallel_runs_are_field_identical(self) -> None:
        serial = make_workspace("tile-serial")
        parallel = make_workspace("tile-parallel")
        serial_config = read_config(serial)
        serial_config["scheduler"]["max_workers"] = 1
        write_config(serial, serial_config)
        parallel_config = read_config(parallel)
        parallel_config["scheduler"]["max_workers"] = 8
        write_config(parallel, parallel_config)
        serial_result = run_pipeline(config_path(serial), reset_state=True)
        parallel_result = run_pipeline(config_path(parallel), reset_state=True)
        self.assertEqual(
            serial_result.run_manifest["stage_content_keys"],
            parallel_result.run_manifest["stage_content_keys"],
        )
        for stage_name in (
            "features",
            "statistics",
            "scores",
            "summaries",
            "fusion",
            "diagnostics",
            "pixel_map",
        ):
            self.assertEqual(
                _stage(serial_result.snapshot_dir, stage_name),
                _stage(parallel_result.snapshot_dir, stage_name),
            )
        self.assertTrue(np.array_equal(_raster(serial), _raster(parallel)))
        self.assertEqual(serial_result.run_manifest["max_workers"], 1)
        self.assertEqual(parallel_result.run_manifest["max_workers"], 8)


class SchedulingIntegrationTests(unittest.TestCase):
    def test_budget_resumes_in_stable_next_batch(self) -> None:
        root = make_workspace("tile-budget")
        set_config_value(root, "scheduler.global_window_budget", 600)
        first = run_pipeline(config_path(root), reset_state=True)
        self.assertFalse(first.complete)
        self.assertGreater(len(first.run_manifest["pending_queue"]), 0)
        self.assertTrue(
            any(event["code"] == "E_BUDGET_PENDING" for event in first.events)
        )
        second = run_pipeline(config_path(root))
        self.assertTrue(second.complete)
        self.assertEqual(second.run_manifest["pending_queue"], [])
        self.assertGreaterEqual(
            second.run_manifest["scheduler_budget"]["cumulative_consumed"], 1152
        )

    def test_tile_quota_aborts_batch(self) -> None:
        root = make_workspace("tile-quota")
        set_config_value(root, "scheduler.tile_window_quota", 60)
        result = run_pipeline(config_path(root), reset_state=True)
        self.assertFalse(result.complete)
        self.assertTrue(
            any(event["code"] == "E_TILE_QUOTA_EXCEEDED" for event in result.events)
        )
        self.assertEqual(result.run_manifest["scheduler_budget"]["consumed"], 0)
        self.assertEqual(len(result.run_manifest["pending_queue"]), 576)

    def test_fusion_limit_advances_until_complete(self) -> None:
        root = make_workspace("tile-fusion-limit")
        set_config_value(root, "scheduler.fusion_recompute_limit", 2)
        completed = False
        pending_sizes = []
        seen_fusion_pending = False
        for _ in range(7):
            result = run_pipeline(config_path(root), reset_state=False)
            pending_sizes.append(len(result.run_manifest["pending_queue"]))
            seen_fusion_pending = seen_fusion_pending or any(
                event["code"] == "E_FUSION_PENDING" for event in result.events
            )
            if result.complete:
                completed = True
                break
        self.assertTrue(completed)
        self.assertIn(0, pending_sizes)
        self.assertTrue(seen_fusion_pending)


if __name__ == "__main__":
    unittest.main()
