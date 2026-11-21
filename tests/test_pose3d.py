from __future__ import annotations

import copy
import json
import shutil
import tempfile
import unittest
from contextlib import contextmanager
from pathlib import Path

import numpy as np
from PIL import Image

from pose3d.cache import StageCache
from pose3d.config import load_config
from pose3d.errors import (
    E_CAMERA_INVALID,
    E_CFG_INVALID,
    E_DISPARITY_INSUFFICIENT,
    E_INLIER_SHORTAGE,
    E_OBS_INVALID,
    E_POINT_QUOTA_EXHAUSTED,
    E_RESIDUAL_BUDGET_EXHAUSTED,
    E_TRIANG_DEGENERATE,
)
from pose3d.geometry import triangulate_dlt, triangulation_stage
from pose3d.injections import load_injections
from pose3d.loader import load_inputs
from pose3d.models import Camera, Observation
from pose3d.pipeline import run_pipeline
from pose3d.quantize import quantize_values
from pose3d.samplegen import HEIGHT, WIDTH, generate_sample
from pose3d.solver import _worst_residual, solve_point


@contextmanager
def sample_environment():
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary) / "sample"
        generate_sample(root)
        yield root, Path(temporary)


def core_report(report: dict) -> dict:
    value = copy.deepcopy(report)
    value.pop("run_mode", None)
    value.pop("snapshot", None)
    value.pop("input", None)
    value.pop("cache", None)
    value.pop("config", None)
    for point in value["points"]:
        point.pop("cache", None)
    return value


def write_config(path: Path, config: dict) -> None:
    path.write_text(
        json.dumps(config, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )


class LoaderTests(unittest.TestCase):
    def test_loader_decodes_views_and_reports_invalid_observation(self) -> None:
        with sample_environment() as (root, _):
            cfg = load_config(root / "solver_config.json")
            bundle = load_inputs(root, cfg, config_path=root / "solver_config.json")
            self.assertEqual(len(bundle.cameras), 6)
            self.assertEqual(bundle.image_resolution, (WIDTH, HEIGHT))
            self.assertEqual(bundle.grayscale_images[1].shape, (HEIGHT, WIDTH))
            self.assertIn("rig/corner_00@instance-A", bundle.points)
            self.assertIn("rig/corner_00@instance-B", bundle.points)
            bounds = bundle.points["bounds/tip@instance-A"]
            self.assertEqual(len(bounds.observations), 6)
            self.assertTrue(any(item.code == E_OBS_INVALID for item in bounds.failures))

    def test_invalid_png_header_and_nonorthogonal_rotation_are_rejected(self) -> None:
        with sample_environment() as (root, _):
            view_path = root / "views" / "view_01.png"
            raw = bytearray(view_path.read_bytes())
            raw[:8] = b"NOTPNG!!"
            view_path.write_bytes(raw)
            camera_path = root / "cameras" / "view_02.json"
            camera = json.loads(camera_path.read_text(encoding="utf-8"))
            camera["R"][0][1] += 0.3
            write_config(camera_path, camera)
            cfg = load_config(root / "solver_config.json")
            bundle = load_inputs(root, cfg, config_path=root / "solver_config.json")
            camera_failures = [
                item for item in bundle.global_failures if item.code == E_CAMERA_INVALID
            ]
            messages = " ".join(item.message for item in camera_failures)
            self.assertIn("PNG", messages)
            self.assertIn("not orthogonal", messages)

    def test_duplicate_and_ambiguous_keypoint_lines_are_not_merged(self) -> None:
        with sample_environment() as (root, _):
            original = (root / "observations.jsonl").read_text(encoding="utf-8")
            first_line = original.splitlines()[0]
            extra = [
                first_line,
                json.dumps(
                    {
                        "keypoint_id": "ambiguous/base",
                        "observations": [{"view_index": 1, "u": 1.0, "v": 1.0}],
                    }
                ),
                json.dumps(
                    {
                        "keypoint_id": "ambiguous/base@instance-A",
                        "observations": [{"view_index": 2, "u": 1.0, "v": 1.0}],
                    }
                ),
            ]
            (root / "observations.jsonl").write_text(
                original + "\n".join(extra) + "\n", encoding="utf-8"
            )
            cfg = load_config(root / "solver_config.json")
            bundle = load_inputs(root, cfg, config_path=root / "solver_config.json")
            self.assertNotIn("ambiguous/base", bundle.points)
            self.assertNotIn("ambiguous/base@instance-A", bundle.points)
            duplicates = [
                failure
                for failure in bundle.global_failures
                if "duplicate keypoint" in failure.message
            ]
            ambiguous = [
                failure
                for failure in bundle.global_failures
                if "ambiguous" in failure.message
            ]
            self.assertTrue(duplicates)
            self.assertEqual(len(ambiguous), 2)

    def test_inconsistent_resolution_is_rejected(self) -> None:
        with sample_environment() as (root, _):
            path = root / "views" / "view_03.png"
            with Image.open(path) as image:
                resized = image.convert("L").resize((WIDTH // 2, HEIGHT // 2))
                resized.save(path)
            camera_path = root / "cameras" / "view_03.json"
            camera = json.loads(camera_path.read_text(encoding="utf-8"))
            camera["width"] = WIDTH // 2
            camera["height"] = HEIGHT // 2
            write_config(camera_path, camera)
            cfg = load_config(root / "solver_config.json")
            bundle = load_inputs(root, cfg, config_path=root / "solver_config.json")
            self.assertNotIn(3, bundle.cameras)
            self.assertTrue(
                any(
                    "resolution" in item.message
                    for item in bundle.global_failures
                    if item.code == E_CAMERA_INVALID
                )
            )


class GeometryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls._temporary = tempfile.TemporaryDirectory()
        cls.root = Path(cls._temporary.name) / "sample"
        generate_sample(cls.root)
        cls.cfg = load_config(cls.root / "solver_config.json")
        cls.bundle = load_inputs(
            cls.root, cls.cfg, config_path=cls.root / "solver_config.json"
        )

    @classmethod
    def tearDownClass(cls) -> None:
        cls._temporary.cleanup()

    def test_triangulation_is_accurate_and_bit_repeatable(self) -> None:
        point = self.bundle.points["aux/marker@instance-A"]
        first = triangulation_stage(
            point.observations,
            self.bundle.cameras,
            self.cfg,
            keypoint_id=point.keypoint_id,
        )
        second = triangulation_stage(
            point.observations,
            self.bundle.cameras,
            self.cfg,
            keypoint_id=point.keypoint_id,
        )
        self.assertEqual(first, second)
        expected = np.array([0.0, 0.68, 0.62])
        self.assertLess(np.linalg.norm(np.asarray(first["point"]) - expected), 0.01)
        self.assertGreaterEqual(len(first["residuals"]), 6)
        self.assertTrue(all(np.isfinite(item["residual_px"]) for item in first["residuals"]))

    def test_degenerate_duplicate_camera_is_reported(self) -> None:
        point = self.bundle.points["aux/marker@instance-A"]
        first = point.observations[0]
        camera = self.bundle.cameras[first.view_index]
        duplicate_camera = Camera(
            view_index=99,
            width=camera.width,
            height=camera.height,
            K=camera.K.copy(),
            R=camera.R.copy(),
            t=camera.t.copy(),
            image_path=camera.image_path,
            camera_path=camera.camera_path,
            image_sha256=camera.image_sha256,
            source_image_sha256=camera.source_image_sha256,
        )
        cameras = dict(self.bundle.cameras)
        cameras[99] = duplicate_camera
        observations = [
            Observation(
                keypoint_id=point.keypoint_id,
                view_index=first.view_index,
                u=first.u,
                v=first.v,
            ),
            Observation(
                keypoint_id=point.keypoint_id,
                view_index=99,
                u=first.u,
                v=first.v,
            ),
        ]
        cfg = self.cfg.replace_for_point(min_views=2)
        with self.assertRaises(Exception) as context:
            triangulate_dlt(
                observations,
                cameras,
                cfg,
                keypoint_id=point.keypoint_id,
            )
        self.assertEqual(context.exception.failure.code, E_TRIANG_DEGENERATE)

    def test_low_disparity_screening_does_not_fall_back_to_all_views(self) -> None:
        point = self.bundle.points["low/parallax@instance-A"]
        with self.assertRaises(Exception) as context:
            triangulation_stage(
                point.observations,
                self.bundle.cameras,
                self.cfg,
                keypoint_id=point.keypoint_id,
            )
        self.assertEqual(context.exception.failure.code, E_DISPARITY_INSUFFICIENT)
        details = context.exception.failure.details
        self.assertEqual(details["accepted_views"], [])
        self.assertEqual(details["excluded_views"], [1, 2, 3])
        self.assertEqual(len(details["excluded_view_pairs"]), 3)

    def test_worst_residual_tie_breaks_by_view_index(self) -> None:
        residuals = [
            {"view_index": 5, "residual_px": 2.0},
            {"view_index": 2, "residual_px": 2.0},
            {"view_index": 3, "residual_px": 1.0},
        ]
        self.assertEqual(_worst_residual(residuals)["view_index"], 2)


class RobustAndQuantizationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls._temporary = tempfile.TemporaryDirectory()
        cls.root = Path(cls._temporary.name) / "sample"
        generate_sample(cls.root)
        cls.cfg = load_config(cls.root / "solver_config.json")
        cls.injections = load_injections(None)
        cls.bundle = load_inputs(
            cls.root, cls.cfg, config_path=cls.root / "solver_config.json"
        )

    @classmethod
    def tearDownClass(cls) -> None:
        cls._temporary.cleanup()

    def test_outlier_is_removed_and_point_is_refinalized(self) -> None:
        point = self.bundle.points["rig/corner_02@instance-A"]
        result = solve_point(
            point, self.bundle, self.cfg, self.injections, StageCache()
        )
        self.assertEqual(result.status, "ok")
        rejection = result.record["outlier_rejection"]
        self.assertEqual(rejection["removed_views"], [4])
        self.assertEqual(rejection["rounds_executed"], 1)
        self.assertLess(rejection["last_round_residual"], 1.5)
        self.assertEqual(result.record["observation_count_final"], 5)
        self.assertTrue(result.record["final"]["observation_fingerprint"])

    def test_inlier_shortage_preserves_trace_without_final_point(self) -> None:
        point = self.bundle.points["fragile/tip@instance-A"]
        result = solve_point(
            point, self.bundle, self.cfg, self.injections, StageCache()
        )
        self.assertEqual(result.status, "error")
        self.assertIn(E_INLIER_SHORTAGE, result.record["error_codes"])
        self.assertIsNone(result.record["final"])
        rejection = result.record["outlier_rejection"]
        self.assertEqual(rejection["stop_reason"], "inlier_shortage")
        self.assertEqual(rejection["trace"][0]["would_remove_view"], 4)

    def test_removal_flip_origin_distinguishes_truncation_and_accumulation(self) -> None:
        truncation_cfg = self.cfg.replace_for_point(
            quantization_levels=(1, 2),
            residual_threshold_px=0.5,
            max_rounds=2,
        )
        truncation_result = solve_point(
            self.bundle.points["rig/corner_00@instance-A"],
            self.bundle,
            truncation_cfg,
            self.injections,
            StageCache(),
        )
        truncation_flip = truncation_result.record["quantization"]["removal_flip"]
        self.assertTrue(truncation_flip["detected"])
        self.assertEqual(truncation_flip["origin"], "quantization_truncation")
        self.assertEqual(truncation_flip["low_removed_views"][:-1], truncation_flip["high_removed_views"][:-1])
        self.assertNotEqual(truncation_flip["low_removed_views"][-1], truncation_flip["high_removed_views"][-1])

        accumulation_cfg = self.cfg.replace_for_point(
            quantization_levels=(0, 1),
            residual_threshold_px=0.05,
            max_rounds=2,
        )
        accumulation_result = solve_point(
            self.bundle.points["aux/marker@instance-A"],
            self.bundle,
            accumulation_cfg,
            self.injections,
            StageCache(),
        )
        accumulation_flip = accumulation_result.record["quantization"]["removal_flip"]
        self.assertTrue(accumulation_flip["detected"])
        self.assertEqual(
            accumulation_flip["origin"], "residual_accumulation_rounding"
        )

    def test_quantization_uses_round_half_even_and_reports_both_widths(self) -> None:
        quantized = quantize_values([0.25, 0.75, 1.25, -0.25, -0.75], 1)
        self.assertEqual(quantized["int"], [0, 2, 2, 0, -2])
        self.assertEqual(quantized["rounding"], "round_half_even")
        point = self.bundle.points["aux/marker@instance-A"]
        result = solve_point(
            point, self.bundle, self.cfg, self.injections, StageCache()
        )
        quantization = result.record["quantization"]
        self.assertEqual(quantization["status"], "ok")
        self.assertEqual(quantization["primary_bits"], 14)
        self.assertEqual(quantization["high_bits"], 15)
        self.assertEqual((quantization["static_primary_point"]["fractional_bits"]), 14)
        propagation = quantization["error_propagation_high"]
        self.assertIn("stages", propagation)
        self.assertIn("first_tolerance_exceeded_stage", propagation)
        self.assertFalse(quantization["removal_flip"]["detected"])


class PipelineTests(unittest.TestCase):
    def test_injection_isolation_reports_all_four_hits(self) -> None:
        with sample_environment() as (root, temporary):
            report = run_pipeline(
                input_dir=root,
                config_path=root / "solver_config.json",
                injections_path=root / "injections.json",
                snapshot_dir=temporary / "state",
                fresh=True,
            )
            hit_records = [item for item in report["injections"] if item["hit"]]
            self.assertEqual(len(hit_records), 4)
            codes = {
                item["id"]: item["observed_business_codes"] for item in hit_records
            }
            self.assertEqual(codes["INJ-001-NONFINITE-PIXEL"], [E_OBS_INVALID])
            self.assertEqual(
                codes["INJ-002-NONORTHOGONAL-ROTATION"], [E_CAMERA_INVALID]
            )
            self.assertEqual(codes["INJ-003-DISPARITY-ABOVE-LIMIT"], [E_CFG_INVALID])
            self.assertEqual(codes["INJ-004-ZERO-THRESHOLD"], [E_CFG_INVALID])
            points = {point["keypoint_id"]: point for point in report["points"]}
            self.assertEqual(points["rig/corner_00@instance-A"]["status"], "ok")
            self.assertIn(E_OBS_INVALID, points["rig/corner_00@instance-A"]["error_codes"])
            self.assertEqual(points["rig/corner_01@instance-B"]["status"], "error")
            self.assertEqual(points["rig/corner_01@instance-A"]["status"], "error")
            self.assertEqual(points["rig/corner_03@instance-B"]["status"], "error")
            self.assertEqual(points["rig/corner_02@instance-A"]["status"], "ok")

    def test_global_budget_exhaustion_keeps_stable_pending_queue(self) -> None:
        with sample_environment() as (root, temporary):
            config_path = root / "solver_config.json"
            config = json.loads(config_path.read_text(encoding="utf-8"))
            config["budget"]["global_residual_budget"] = 0
            write_config(config_path, config)
            report = run_pipeline(
                input_dir=root,
                config_path=config_path,
                snapshot_dir=temporary / "budget-state",
                fresh=True,
            )
            self.assertEqual(report["pending_queue"], sorted(report["pending_queue"]))
            self.assertEqual(len(report["pending_queue"]), 16)
            self.assertEqual(report["budget"]["consumed"], 0)
            self.assertTrue(report["budget"]["exhausted"])
            self.assertTrue(
                all(
                    E_RESIDUAL_BUDGET_EXHAUSTED in point["error_codes"]
                    for point in report["points"]
                )
            )

    def test_per_keypoint_quota_aborts_point_but_not_other_points(self) -> None:
        with sample_environment() as (root, temporary):
            config_path = root / "solver_config.json"
            config = json.loads(config_path.read_text(encoding="utf-8"))
            config["budget"]["per_keypoint_quota"] = 7
            write_config(config_path, config)
            report = run_pipeline(
                input_dir=root,
                config_path=config_path,
                snapshot_dir=temporary / "quota-state",
                fresh=True,
            )
            points = {point["keypoint_id"]: point for point in report["points"]}
            self.assertIn(
                E_POINT_QUOTA_EXHAUSTED,
                points["aux/marker@instance-A"]["error_codes"],
            )
            self.assertGreater(report["budget"]["consumed"], 0)
            self.assertTrue(
                all(
                    point["residual_evaluations"] <= 7
                    for point in report["points"]
                )
            )

    def test_incremental_observation_edit_matches_fresh_rebuild(self) -> None:
        with sample_environment() as (root, temporary):
            config_path = root / "solver_config.json"
            state = temporary / "state"
            base = run_pipeline(
                input_dir=root,
                config_path=config_path,
                snapshot_dir=state,
                fresh=True,
            )
            recovered = run_pipeline(
                input_dir=root,
                config_path=config_path,
                snapshot_dir=state,
            )
            self.assertEqual(
                recovered["snapshot"]["recovery_decision"],
                "exact_snapshot_available",
            )
            self.assertEqual(recovered["snapshot"]["decision"], "unchanged")
            self.assertEqual(recovered["cache"]["recomputed_points"], [])
            self.assertEqual(core_report(base), core_report(recovered))

            observations_path = root / "observations.jsonl"
            edited_lines = []
            for line in observations_path.read_text(encoding="utf-8").splitlines():
                record = json.loads(line)
                if record["keypoint_id"] == "aux/marker@instance-A":
                    for observation in record["observations"]:
                        if observation["view_index"] == 3:
                            observation["u"] += 2.0
                edited_lines.append(
                    json.dumps(record, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
                )
            observations_path.write_text(
                "\n".join(edited_lines) + "\n", encoding="utf-8"
            )
            incremental = run_pipeline(
                input_dir=root,
                config_path=config_path,
                snapshot_dir=state,
            )
            fresh = run_pipeline(
                input_dir=root,
                config_path=config_path,
                snapshot_dir=temporary / "fresh-state",
                fresh=True,
            )
            self.assertEqual(
                incremental["cache"]["recomputed_points"],
                ["aux/marker@instance-A"],
            )
            self.assertEqual(len(incremental["cache"]["reused_points"]), 15)
            self.assertEqual(core_report(incremental), core_report(fresh))
            self.assertNotEqual(
                recovered["snapshot"]["global_version"],
                incremental["snapshot"]["global_version"],
            )

    def test_config_edit_invalidates_rejection_but_reuses_triangulation(self) -> None:
        with sample_environment() as (root, temporary):
            config_path = root / "solver_config.json"
            config = json.loads(config_path.read_text(encoding="utf-8"))
            config["outlier_rejection"]["threshold_px"] = 1.6
            write_config(config_path, config)
            state = temporary / "state"
            run_pipeline(
                input_dir=root,
                config_path=config_path,
                snapshot_dir=state,
            )
            config["outlier_rejection"]["threshold_px"] = 1.7
            write_config(config_path, config)
            incremental = run_pipeline(
                input_dir=root,
                config_path=config_path,
                snapshot_dir=state,
            )
            fresh = run_pipeline(
                input_dir=root,
                config_path=config_path,
                snapshot_dir=temporary / "fresh",
                fresh=True,
            )
            self.assertEqual(core_report(incremental), core_report(fresh))
            for point in incremental["points"]:
                if point["triangulation"] and point["triangulation"].get("status") == "ok":
                    self.assertIn("load", point["cache"]["reused_stages"])
                    self.assertIn("triangulation", point["cache"]["reused_stages"])
                    self.assertIn("rejection", point["cache"]["recomputed_stages"])

    def test_concurrency_does_not_change_core_result(self) -> None:
        with sample_environment() as (root, temporary):
            config_path = root / "solver_config.json"
            config = json.loads(config_path.read_text(encoding="utf-8"))
            config["scheduler"]["max_workers"] = 1
            write_config(config_path, config)
            serial = run_pipeline(
                input_dir=root,
                config_path=config_path,
                snapshot_dir=temporary / "serial",
                fresh=True,
            )
            config["scheduler"]["max_workers"] = 8
            write_config(config_path, config)
            parallel = run_pipeline(
                input_dir=root,
                config_path=config_path,
                snapshot_dir=temporary / "parallel",
                fresh=True,
            )
            self.assertEqual(core_report(serial), core_report(parallel))

    def test_snapshot_manifest_and_atomic_directory(self) -> None:
        with sample_environment() as (root, temporary):
            state = temporary / "state"
            first = run_pipeline(
                input_dir=root,
                config_path=root / "solver_config.json",
                snapshot_dir=state,
                fresh=True,
            )
            second = run_pipeline(
                input_dir=root,
                config_path=root / "solver_config.json",
                snapshot_dir=state,
            )
            self.assertEqual(core_report(first), core_report(second))
            manifest = json.loads(
                (state / "current" / "manifest.json").read_text(encoding="utf-8")
            )
            self.assertEqual(manifest["global_version"], second["snapshot"]["global_version"])
            self.assertEqual(manifest["config_hash"], second["input"]["config_hash"])
            self.assertEqual(manifest["camera_hash"], second["input"]["camera_hash"])
            self.assertEqual(manifest["completed_keypoints"], sorted(manifest["completed_keypoints"]))
            self.assertEqual(manifest["consumed_budget"], second["budget"]["consumed"])
            self.assertFalse(list(state.glob(".snapshot-tmp-*")))
            self.assertTrue((state / "current" / "stages" / "load.json").exists())


if __name__ == "__main__":
    unittest.main()
