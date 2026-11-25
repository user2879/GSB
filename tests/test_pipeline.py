from __future__ import annotations

import json
import random
import shutil
import tempfile
import unittest
from pathlib import Path

from mvrv.editor import apply_command
from mvrv.engine import build_reconstruction, canonical_state_blob
from mvrv.fixedpoint import divide_round
from mvrv.png_codec import PNG_SIGNATURE, write_png_rgb
from mvrv.snapshot import SnapshotStore
from mvrv.scheduler import StableTask, reduce_in_any_order, run_stable
from mvrv.synthetic import generate_project
from mvrv.validation import load_project


ROOT = Path(__file__).resolve().parents[1]


class PipelineTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.temp = tempfile.TemporaryDirectory(prefix="mvrv-test-")
        cls.workspace = Path(cls.temp.name) / "project"
        generate_project(cls.workspace)
        cls.loaded = load_project(cls.workspace)
        assert cls.loaded.ok, [f.to_dict() for f in cls.loaded.failures]
        cls.full = build_reconstruction(cls.loaded, workers=1)
        cls.state = cls.full["state"]

    @classmethod
    def tearDownClass(cls) -> None:
        cls.temp.cleanup()

    def test_01_sample_and_png_container(self) -> None:
        self.assertEqual(self.state["metrics"]["view_count"], 4)
        self.assertEqual(self.state["metrics"]["pair_count"], 6)
        image_path = self.workspace / "data" / "views" / "view_00.png"
        data = image_path.read_bytes()
        self.assertTrue(data.startswith(PNG_SIGNATURE))
        self.assertGreater(self.state["metrics"]["feature_count"], 80)

    def test_02_features_sorted_and_fingerprinted(self) -> None:
        for view in self.state["views"]:
            coords = [tuple(item["xy"]) for item in view["features"]]
            self.assertEqual(coords, sorted(coords))
            self.assertTrue(all(len(item["content_fingerprint"]) == 20 for item in view["features"]))
            self.assertEqual(len(coords), len(set(coords)))

    def test_03_pairs_and_inlier_order(self) -> None:
        pair_keys = [tuple(pair["pair"]) for pair in self.state["pairs"]]
        self.assertEqual(pair_keys, sorted(pair_keys))
        rank = 0
        for pair in self.state["pairs"]:
            left_coords = [tuple(match["left_xy"]) for match in pair["matches"]]
            self.assertEqual(left_coords, sorted(left_coords))
            for match in pair["matches"]:
                rank += 1
                self.assertEqual(match["global_inlier_rank"], rank)
                self.assertEqual(match["global_cumulative_inliers"], rank)

    def test_04_depth_pixels_and_boundary_rule(self) -> None:
        pixels = [pixel for depth_map in self.state["depth_maps"] for pixel in depth_map["pixels"]]
        self.assertEqual(len(pixels), 4 * 64 * 48)
        for pixel in pixels:
            self.assertIn(pixel["rounding"], {"none", "floor", "ceil", "exact"})
            if pixel.get("boundary"):
                self.assertGreater(pixel["spread_q"], 0)
                self.assertLessEqual(pixel["spread_q"], pixel["half_step_q"])

    def test_05_voxel_parent_references(self) -> None:
        by_id = {voxel["voxel_id"]: voxel for voxel in self.state["voxels"]}
        for voxel in self.state["voxels"]:
            if voxel["parent_id"] is not None:
                self.assertIn(voxel["parent_id"], by_id)
                self.assertIn(voxel["octant"], by_id[voxel["parent_id"]]["child_octants"])
            self.assertIn(voxel["status"], {"empty", "occupied", "subdivided"})

    def test_06_unedited_incremental_reuses_all(self) -> None:
        result = build_reconstruction(self.loaded, previous_state=self.state, workers=4)
        self.assertEqual(canonical_state_blob(result["state"]), canonical_state_blob(self.state))
        counts = result["ledger"]["counts"]
        self.assertEqual(counts["features_recomputed"], 0)
        self.assertEqual(counts["pairs_recomputed"], 0)
        self.assertEqual(counts["points_recomputed"], 0)
        self.assertEqual(counts["depth_recomputed"], 0)
        self.assertEqual(counts["voxels_recomputed"], 0)

    def test_07_parallel_degrees_identical(self) -> None:
        expected = canonical_state_blob(self.state)
        for workers in (1, 2, 3, 4, 8):
            with self.subTest(workers=workers):
                result = build_reconstruction(self.loaded, workers=workers)
                self.assertEqual(canonical_state_blob(result["state"]), expected)

    def _copy_workspace(self, name: str) -> Path:
        path = Path(self.temp.name) / name
        shutil.copytree(self.workspace, path)
        return path

    def _fresh(self, path: Path, workers: int = 1):
        loaded = load_project(path)
        self.assertTrue(loaded.ok, [f.to_dict() for f in loaded.failures])
        return build_reconstruction(loaded, workers=workers)

    def test_08_replace_view_precise_invalidation_and_equality(self) -> None:
        path = self._copy_workspace("replace-view")
        apply_command(
            path,
            {"op": "replace_view", "view_index": 1, "source": "replacements/view_01.png"},
        )
        edited = load_project(path)
        incremental = build_reconstruction(edited, previous_state=self.state, workers=3)
        fresh = self._fresh(path)
        self.assertEqual(canonical_state_blob(incremental["state"]), canonical_state_blob(fresh["state"]))
        self.assertEqual(incremental["ledger"]["features"]["recomputed"], [1])
        self.assertEqual(
            sorted(incremental["ledger"]["pairs"]["recomputed"]),
            ["P:00-01", "P:01-02", "P:01-03"],
        )

    def test_09_extrinsic_only_invalidates_referencing_points(self) -> None:
        path = self._copy_workspace("extrinsic")
        apply_command(path, {"op": "set_extrinsic", "view_index": 2, "t": [-3, -1, 0]})
        edited = load_project(path)
        incremental = build_reconstruction(edited, previous_state=self.state, workers=2)
        fresh = self._fresh(path, workers=4)
        self.assertEqual(canonical_state_blob(incremental["state"]), canonical_state_blob(fresh["state"]))
        self.assertEqual(incremental["ledger"]["features"]["recomputed"], [])
        self.assertEqual(incremental["ledger"]["pairs"]["recomputed"], [])
        changed = incremental["ledger"]["points"]["recomputed"]
        self.assertTrue(changed)
        for point_id in changed:
            pair_token = point_id.split(":", 2)[1]
            self.assertIn("02", pair_token.split("-"))

    def test_10_min_parallax_gate_is_deterministic(self) -> None:
        path = self._copy_workspace("parallax")
        apply_command(path, {"op": "set_config", "key": "min_parallax", "value": 5})
        edited = load_project(path)
        incremental = build_reconstruction(edited, previous_state=self.state, workers=2)
        fresh = self._fresh(path)
        self.assertEqual(canonical_state_blob(incremental["state"]), canonical_state_blob(fresh["state"]))
        self.assertEqual(incremental["ledger"]["features"]["recomputed"], [])
        self.assertEqual(incremental["ledger"]["pairs"]["recomputed"], [])
        self.assertTrue(incremental["ledger"]["points"]["gate_changed"])

    def test_11_budget_exhaustion_matches_serial_and_keeps_todo(self) -> None:
        for budget in range(0, 25):
            with self.subTest(budget=budget):
                serial = build_reconstruction(
                    self.loaded, workers=1, global_budget=budget, view_quota=100000
                )
                parallel = build_reconstruction(
                    self.loaded, workers=4, global_budget=budget, view_quota=100000
                )
                self.assertEqual(
                    canonical_state_blob(serial["state"]),
                    canonical_state_blob(parallel["state"]),
                )
                self.assertEqual(len(serial["state"]["todo_queue"]), self.state["metrics"]["match_count"] - budget)

    def test_12_view_quota_is_stable(self) -> None:
        serial = build_reconstruction(self.loaded, workers=1, global_budget=1000, view_quota=3)
        parallel = build_reconstruction(self.loaded, workers=5, global_budget=1000, view_quota=3)
        self.assertEqual(canonical_state_blob(serial["state"]), canonical_state_blob(parallel["state"]))
        self.assertTrue(serial["state"]["todo_queue"])
        self.assertIsNotNone(serial["ledger"]["budget"]["stop_reason"])

    def test_13_random_command_sequences_match_blank_rebuild(self) -> None:
        rng = random.Random(20261126)
        for sequence in range(4):
            path = self._copy_workspace(f"sequence-{sequence}")
            previous = self.state
            for step in range(5):
                choice = rng.randrange(6)
                if choice == 0:
                    command = {"op": "set_config", "key": "min_parallax", "value": rng.choice([2, 3, 5, 8])}
                elif choice == 1:
                    command = {"op": "set_config", "key": "residual_threshold_q", "value": rng.choice([350, 650, 820, 1300])}
                elif choice == 2:
                    command = {"op": "set_config", "key": "voxel_size", "value": rng.choice([3, 4, 5])}
                elif choice == 3:
                    command = {"op": "set_config", "key": "depth_quantization", "value": rng.choice([1, 2, 3])}
                elif choice == 4:
                    view = rng.choice([0, 1, 2, 3])
                    command = {
                        "op": "set_extrinsic",
                        "view_index": view,
                        "t": [rng.choice([-7, -5, -3, -1, 1, 3]), rng.choice([-1, 0, 1]), 0],
                    }
                else:
                    command = {"op": "replace_view", "view_index": 1, "source": "replacements/view_01.png"}
                apply_command(path, command)
                loaded = load_project(path)
                self.assertTrue(loaded.ok)
                incremental = build_reconstruction(loaded, previous_state=previous, workers=rng.choice([1, 2, 4]))
                fresh = build_reconstruction(loaded, workers=1)
                self.assertEqual(
                    canonical_state_blob(incremental["state"]),
                    canonical_state_blob(fresh["state"]),
                    f"sequence {sequence} step {step} command {command}",
                )
                previous = incremental["state"]

    def test_14_snapshot_commit_and_recovery(self) -> None:
        snapshot_root = Path(self.temp.name) / "snapshots"
        store = SnapshotStore(snapshot_root)
        store.commit(
            validation_report=self.loaded.report(),
            state=json.loads(json.dumps(self.state)),
            ledger=self.full["ledger"],
            global_version=1,
        )
        second_state = json.loads(json.dumps(self.state))
        store.commit(
            validation_report=self.loaded.report(),
            state=second_state,
            ledger=self.full["ledger"],
            global_version=2,
        )
        self.assertEqual(store.recover()["status"], "current_valid")
        manifest_pointer = json.loads(store.current_pointer.read_text())
        corrupt = store.snapshot_root / f"version-{manifest_pointer['global_version']:06d}" / "voxels.json"
        corrupt.write_text(corrupt.read_text(encoding="utf-8") + " corruption", encoding="utf-8")
        recovery = store.recover()
        self.assertEqual(recovery["status"], "restored")
        self.assertEqual(recovery["global_version"], 1)

    def test_15_invalid_png_and_camera_business_codes(self) -> None:
        path = self._copy_workspace("invalid")
        image = path / "data" / "views" / "view_00.png"
        image.write_bytes(b"BAD" + image.read_bytes()[3:])
        loaded = load_project(path)
        self.assertFalse(loaded.ok)
        self.assertIn(loaded.failures[0].code, {"E_PNG_SIGNATURE", "E_PNG_CONTAINER", "E_PNG_HEADER"})

        write_png_rgb(path / "data" / "views" / "view_01.png", __import__("numpy").zeros((8, 8, 3), dtype="uint8"))
        loaded = load_project(path)
        codes = {failure.code for failure in loaded.failures}
        self.assertIn("E_PNG_DIMENSIONS", codes)

    def test_16_divide_round_direction(self) -> None:
        self.assertEqual(divide_round(7, 4), (2, "ceil"))
        self.assertEqual(divide_round(5, 4), (1, "floor"))
        self.assertEqual(divide_round(8, 4), (2, "exact"))
        self.assertEqual(divide_round(-5, 4), (-1, "ceil"))

    def test_17_stable_scheduler_ignores_completion_order(self) -> None:
        tasks = [
            StableTask(f"task-{value:03d}", (value,), "fake", value)
            for value in range(80)
        ]

        def handler(task: StableTask):
            return {"id": task.task_id, "sort_key": task.sort_key, "value": task.payload}

        serial, _ = run_stable(tasks, handler, workers=1)
        parallel, report = run_stable(tasks, handler, workers=8)
        self.assertEqual(serial, parallel)
        self.assertEqual([item["value"] for item in parallel], list(range(80)))
        for seed in range(10):
            reduced = reduce_in_any_order(parallel, order_seed=seed)
            self.assertEqual([item["value"] for item in reduced], list(range(80)))
        self.assertEqual(report["scheduled_order"], [task.task_id for task in sorted(tasks, key=lambda t: t.sort_key)])


if __name__ == "__main__":
    unittest.main(verbosity=2)
