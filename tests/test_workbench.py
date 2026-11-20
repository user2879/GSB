from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from terrain_diag.anomalies import run_error_code_injections
from terrain_diag.canonical import canonical_line, read_jsonl
from terrain_diag.diagnostics import find_first_divergence
from terrain_diag.errors import ErrorCode, WorkbenchError
from terrain_diag.heightfield import (
    DEFAULT_SEED,
    HEIGHT_SCALE,
    TerrainConfig,
    generate_heightfield,
    validate_height_range,
)
from terrain_diag.hiz import build_hiz, load_hiz, save_hiz, validate_depth_bias
from terrain_diag.quadtree import build_quadtree
from terrain_diag.runner import RunSettings, prepare_assets, run_workbench
from terrain_diag.streaming import StreamingBudgets, StreamingManager
from terrain_diag.workspace import Workspace


def _file_map(root: Path) -> dict[str, bytes]:
    return {
        path.relative_to(root).as_posix(): path.read_bytes()
        for path in sorted(item for item in root.rglob("*") if item.is_file())
    }


class TerrainAndManifestTests(unittest.TestCase):
    def test_257_generation_is_byte_deterministic_and_uint16(self) -> None:
        first = generate_heightfield(TerrainConfig(257, 12345))
        second = generate_heightfield(TerrainConfig(257, 12345))
        self.assertEqual(first.content_sha256, second.content_sha256)
        self.assertTrue(np.array_equal(first.heights, second.heights))
        self.assertEqual(first.heights.dtype, np.dtype("uint16"))
        self.assertGreaterEqual(int(first.heights.min()), 0)
        self.assertLessEqual(int(first.heights.max()), 65535)
        self.assertEqual(len(first.layer_metadata), 5)

    def test_1025_generation_is_stable(self) -> None:
        first = generate_heightfield(TerrainConfig(1025, 12345))
        second = generate_heightfield(TerrainConfig(1025, 12345))
        self.assertEqual(first.content_sha256, second.content_sha256)
        self.assertEqual(first.heights.shape, (1025, 1025))

    def test_quadtree_manifest_is_deterministic_and_records_ranges(self) -> None:
        terrain = generate_heightfield(TerrainConfig(257, 7))
        first = build_quadtree(terrain.heights, 7, terrain.content_sha256)
        second = build_quadtree(terrain.heights, 7, terrain.content_sha256)
        self.assertEqual(first.content_sha256, second.content_sha256)
        self.assertEqual(len(first.leaves), 64)
        self.assertEqual(first.tree_depth, 3)
        for node in first.leaves:
            tile = terrain.heights[
                node.sample_z0 : node.sample_z1 + 1,
                node.sample_x0 : node.sample_x1 + 1,
            ]
            self.assertEqual(node.height_min_raw, int(tile.min()))
            self.assertEqual(node.height_max_raw, int(tile.max()))
            self.assertEqual(node.height_range_raw, int(tile.max()) - int(tile.min()))
            self.assertAlmostEqual(node.bounds.ymin, int(tile.min()) * HEIGHT_SCALE)
            self.assertAlmostEqual(node.bounds.ymax, int(tile.max()) * HEIGHT_SCALE)


class HiZTests(unittest.TestCase):
    def test_hiz_is_deterministic_conservative_and_persistent(self) -> None:
        rows, columns = np.mgrid[0:32, 0:32]
        depth = ((rows * 17 + columns * 13) % 251).astype(np.float32) / 252.0
        first = build_hiz(depth)
        second = build_hiz(depth)
        self.assertEqual(first.content_sha256, second.content_sha256)
        self.assertEqual(first.max_level, 5)
        for level in range(1, first.max_level + 1):
            child = first.levels[level - 1].depth
            parent = first.levels[level].depth
            expected = child.reshape(parent.shape[0], 2, parent.shape[1], 2).max(axis=(1, 3))
            self.assertTrue(np.array_equal(parent, expected))
            self.assertGreaterEqual(
                first.levels[level].depth_error_upper_bound,
                first.levels[level - 1].depth_error_upper_bound,
            )
        with tempfile.TemporaryDirectory() as temporary:
            prefix = Path(temporary) / "hiz"
            save_hiz(prefix, first)
            loaded = load_hiz(prefix)
            self.assertEqual(loaded.content_sha256, first.content_sha256)
            for left, right in zip(first.levels, loaded.levels):
                self.assertTrue(np.array_equal(left.depth, right.depth))
                self.assertEqual(left.content_sha256, right.content_sha256)
        with self.assertRaises(WorkbenchError) as context:
            first.get_level(99)
        self.assertEqual(context.exception.code, ErrorCode.E_HIZ_LEVEL)

    def test_depth_bias_rejects_invalid_values(self) -> None:
        for value in (-0.01, 0.5, float("nan")):
            with self.assertRaises(WorkbenchError) as context:
                validate_depth_bias(value)
            self.assertEqual(context.exception.code, ErrorCode.E_DEPTH_BIAS)
        self.assertEqual(validate_depth_bias(0.0), 0.0)


class StreamingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.temporary = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temporary.name)
        workspace = Workspace.create(cls.root, minimum_free_bytes=0, enforce_disk=False)
        settings = RunSettings(size=257, seed=11, frame_count=1, minimum_free_bytes=0, enforce_disk=False)
        cls.assets = prepare_assets(workspace, settings)
        cls.blocks = sorted(node.block_id for node in cls.assets.quadtree.leaves if node.block_id)

    @classmethod
    def tearDownClass(cls) -> None:
        cls.temporary.cleanup()

    def test_reload_produces_identical_draw_digest_and_auditable_history(self) -> None:
        first_block = self.blocks[0]
        other_block = self.blocks[-1]
        manager = StreamingManager(
            self.assets.heights,
            self.assets.quadtree,
            StreamingBudgets(memory_bytes=8 * 1024 * 1024, vram_bytes=8 * 1024 * 1024, prefetch_radius=0),
        )
        first = manager.step(0, (first_block,), (), "stamp-0")
        self.assertEqual(first.visible_missing, tuple())
        manager.step(1, (other_block,), (), "stamp-1")
        self.assertNotIn(first_block, manager.resident)
        reloaded = manager.step(2, (first_block,), (), "stamp-2")
        self.assertEqual(first.draw_digest, reloaded.draw_digest)
        history = {row["block_id"]: row for row in manager.history_report()}
        self.assertEqual(history[first_block]["load_count"], 2)
        self.assertGreaterEqual(history[first_block]["unload_count"], 1)
        manager.audit()
        manager.close(3, "stamp-3")
        self.assertEqual(manager.audit()["resident_blocks"], 0)

    def test_visible_working_set_over_budget_is_denied_without_silent_omission(self) -> None:
        manager = StreamingManager(
            self.assets.heights,
            self.assets.quadtree,
            StreamingBudgets(memory_bytes=1, vram_bytes=1, prefetch_radius=0),
        )
        with self.assertRaises(WorkbenchError) as context:
            manager.step(0, (self.blocks[0],), (), "tiny")
        self.assertEqual(context.exception.code, ErrorCode.E_BUDGET_DENIED)
        self.assertIn("visible", context.exception.message)

    def test_lru_pressure_evicts_nonvisible_blocks_only(self) -> None:
        manager = StreamingManager(
            self.assets.heights,
            self.assets.quadtree,
            StreamingBudgets(memory_bytes=2 * 4096, vram_bytes=2 * 4096, prefetch_radius=0),
        )
        result = manager.step(0, (self.blocks[0], self.blocks[1]), (), "lrU-0")
        self.assertEqual(set(result.visible_block_ids), {self.blocks[0], self.blocks[1]})
        manager.step(1, (self.blocks[0],), (self.blocks[2],), "lru-1")
        self.assertIn(self.blocks[0], manager.resident)
        self.assertNotIn(self.blocks[1], manager.resident)
        manager.audit()


class DivergenceTests(unittest.TestCase):
    def test_first_divergence_uses_per_block_status_not_aggregate_counts(self) -> None:
        left = {"frame": 0, "camera": {"eye": [0, 0, 0]}, "parameter_stamp": "a", "decisions": [
            {"block_id": "B000_00000_00000", "status": "VISIBLE", "errors": {"total": 0.1}},
            {"block_id": "B000_00001_00000", "status": "OCCLUDED", "errors": {"total": 0.2}},
        ]}
        right = {"frame": 0, "camera": {"eye": [1, 0, 0]}, "parameter_stamp": "b", "decisions": [
            {"block_id": "B000_00000_00000", "status": "OCCLUDED", "errors": {"total": 0.3}},
            {"block_id": "B000_00001_00000", "status": "VISIBLE", "errors": {"total": 0.4}},
        ]}
        with tempfile.TemporaryDirectory() as temporary:
            left_path = Path(temporary) / "left.jsonl"
            right_path = Path(temporary) / "right.jsonl"
            left_path.write_text(canonical_line(left), encoding="utf-8")
            right_path.write_text(canonical_line(right), encoding="utf-8")
            result = find_first_divergence(left_path, right_path, raise_error=False)
        self.assertTrue(result["diverged"])
        self.assertEqual(result["frame"], 0)
        self.assertEqual(result["block_id"], "B000_00000_00000")
        self.assertEqual(result["left_status"], "VISIBLE")
        self.assertEqual(result["right_status"], "OCCLUDED")


class EndToEndTests(unittest.TestCase):
    def test_error_code_injections_all_pass(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            workspace = Workspace.create(root, minimum_free_bytes=0, enforce_disk=False)
            prepare_assets(
                workspace,
                RunSettings(
                    size=257,
                    seed=DEFAULT_SEED,
                    frame_count=1,
                    minimum_free_bytes=0,
                    enforce_disk=False,
                ),
            )
            report = run_error_code_injections(root)
        self.assertTrue(report["all_passed"], json.dumps(report, indent=2, sort_keys=True))
        self.assertEqual(
            set(report["checks"]),
            {
                "E_HEIGHT_RANGE",
                "E_BLOCK_MISSING",
                "E_HIZ_LEVEL",
                "E_DEPTH_BIAS",
                "E_BUDGET_DENIED",
                "E_RESIDENT_LEAK",
                "E_TRACK_DIVERGE",
            },
        )

    def test_small_two_run_outputs_are_byte_identical_and_uncertain_is_retained(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            settings = RunSettings(
                size=257,
                seed=99,
                frame_count=2,
                minimum_free_bytes=0,
                enforce_disk=False,
            )
            first = run_workbench(root / "first", settings)
            second = run_workbench(root / "second", settings)
            self.assertEqual(_file_map(first.workspace.root), _file_map(second.workspace.root))
            rows = read_jsonl(first.report_path)
            self.assertEqual(len(rows), 2)
            for row in rows:
                decisions = row["decisions"]
                visible = {
                    decision["block_id"]
                    for decision in decisions
                    if decision["status"] in ("VISIBLE", "UNCERTAIN")
                }
                uncertain = {
                    decision["block_id"]
                    for decision in decisions
                    if decision["status"] == "UNCERTAIN"
                }
                self.assertTrue(uncertain.issubset(visible))
                self.assertEqual(len(visible), row["stats"]["visible_blocks"])
                for decision in decisions:
                    errors = decision["errors"]
                    self.assertAlmostEqual(
                        errors["total"],
                        sum(errors[key] for key in ("projection_float", "bbox_expand", "hiz_sampling", "depth_bias")),
                        places=11,
                    )

    def test_height_range_error_rejects_out_of_range_values(self) -> None:
        with self.assertRaises(WorkbenchError) as context:
            validate_height_range(np.array([0, 65536], dtype=np.int64))
        self.assertEqual(context.exception.code, ErrorCode.E_HEIGHT_RANGE)


if __name__ == "__main__":
    unittest.main()
