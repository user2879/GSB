from __future__ import annotations

import json
import shutil
import tempfile
import unittest
from pathlib import Path

import numpy as np

from workbench.commands import RuntimeState, validate_command_stream
from workbench.configuration import load_config
from workbench.diagnostics import load_injection_config, run_injection_diagnostic
from workbench.engine import build_plan
from workbench.errors import WorkbenchError
from workbench.image_io import load_rgb_png, write_png
from workbench.masks import load_mask_document
from workbench.samples import generate_workspace
from workbench.session import WorkbenchSession
from workbench.snapshot import SnapshotStore
from workbench.workflows import build_full_final, compare_sessions


class WorkbenchTestBase(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.temp = tempfile.TemporaryDirectory(prefix="workbench-test-")
        cls.root = Path(cls.temp.name)
        generate_workspace(cls.root)
        cls.config_path = cls.root / "configs" / "workbench.json"
        cls.commands_path = cls.root / "configs" / "edit_stream.json"
        cls.injection_path = cls.root / "configs" / "injection.json"
        cls.config = load_config(cls.config_path)
        cls.rectangles = load_mask_document(cls.config.mask_path, cls.config.width, cls.config.height)
        cls.pixels, cls.image_fingerprint = load_rgb_png(
            cls.config.image_path, cls.config.width, cls.config.height
        )
        cls.commands = validate_command_stream(cls.commands_path, cls.config, cls.rectangles)

    @classmethod
    def tearDownClass(cls) -> None:
        cls.temp.cleanup()

    def fresh_session(self) -> WorkbenchSession:
        return WorkbenchSession.fresh(
            self.config, self.pixels, self.image_fingerprint, self.rectangles
        )

    def completed_session(self, workers: int = 1) -> WorkbenchSession:
        session = self.fresh_session()
        session.run_batch(budget=len(session.planned.order), max_workers=workers)
        return session

    def write_probe_config(self, mutate) -> Path:
        raw = json.loads(self.config_path.read_text(encoding="utf-8"))
        mutate(raw)
        path = self.root / "configs" / "_probe.json"
        path.write_text(json.dumps(raw), encoding="utf-8")
        self.addCleanup(lambda: path.unlink(missing_ok=True))
        return path


class ValidationTests(WorkbenchTestBase):
    def test_valid_bundle_has_expected_geometry(self) -> None:
        self.assertEqual(self.config.levels, 4)
        self.assertEqual(self.config.level_dimensions[0], (16, 12))
        self.assertEqual(len(build_plan(self.config, self.pixels, self.image_fingerprint,
                                        RuntimeState.initial(self.config, self.rectangles)).order), 48)

    def test_config_error_code(self) -> None:
        path = self.write_probe_config(lambda raw: raw["pyramid"]["downsample_factors"].__setitem__(0, 3))
        with self.assertRaises(WorkbenchError) as ctx:
            load_config(path)
        self.assertEqual(ctx.exception.code, "E_CONFIG_INVALID")

    def test_mask_out_of_range_code(self) -> None:
        with self.assertRaises(WorkbenchError) as ctx:
            load_mask_document(self.root / "examples" / "invalid_mask.json",
                               self.config.width, self.config.height)
        self.assertEqual(ctx.exception.code, "E_MASK_OUT_OF_RANGE")

    def test_tile_geometry_code(self) -> None:
        path = self.write_probe_config(lambda raw: raw["pyramid"]["tile_sizes"].__setitem__(1, 9))
        with self.assertRaises(WorkbenchError) as ctx:
            load_config(path)
        self.assertEqual(ctx.exception.code, "E_TILE_GEOMETRY")

    def test_png_resolution_header_is_validated(self) -> None:
        bad_png = self.root / "bad-resolution.png"
        write_png(bad_png, np.zeros((8, 8, 3), dtype=np.uint8))
        with self.assertRaises(WorkbenchError) as ctx:
            load_rgb_png(bad_png, self.config.width, self.config.height)
        self.assertEqual(ctx.exception.code, "E_CONFIG_INVALID")

    def test_invalid_command_is_rejected_with_index(self) -> None:
        bad = {"schema_version": 1, "commands": [{"op": "set_guidance", "level": 99, "strength": 0.5}]}
        with self.assertRaises(WorkbenchError) as ctx:
            validate_command_stream(bad, self.config, self.rectangles)
        self.assertEqual(ctx.exception.code, "E_CONFIG_INVALID")
        self.assertEqual(ctx.exception.details["command_index"], 0)


class DeterminismAndKernelTests(WorkbenchTestBase):
    def test_two_runs_have_identical_pixels_and_keys(self) -> None:
        first = self.completed_session()
        second = self.completed_session()
        for level in range(self.config.levels):
            self.assertTrue(np.array_equal(first.repaired_levels[level], second.repaired_levels[level]))
            self.assertTrue(np.array_equal(first.styled_levels[level], second.styled_levels[level]))
        self.assertEqual(
            {k: v.cache_key for k, v in first.records.items()},
            {k: v.cache_key for k, v in second.records.items()},
        )
        self.assertEqual(
            first.planned.pyramid.layer_fingerprints,
            second.planned.pyramid.layer_fingerprints,
        )

    def test_quantization_changes_lut_keys_and_pixels(self) -> None:
        baseline = self.completed_session()
        changed = self.fresh_session()
        changed.apply({"op": "set_quantization", "level": 3, "levels": 4,
                       "fractional_bits": 8, "id": "q"})
        report = changed.run_batch(budget=48, max_workers=1)
        self.assertIn("L03R000C000", report["recomputed"])
        self.assertNotEqual(
            baseline.planned.plans["L03R000C000"].cache_key,
            changed.planned.plans["L03R000C000"].cache_key,
        )
        self.assertFalse(np.array_equal(baseline.styled_levels[3], changed.styled_levels[3]))

    def test_mask_outside_is_not_style_polluted(self) -> None:
        session = self.completed_session()
        mask = session.planned.pyramid.mask_levels[3]
        np.testing.assert_array_equal(
            session.styled_levels[3][~mask],
            self.pixels[~mask],
        )

    def test_block_mean_rounding_is_deterministic(self) -> None:
        from workbench.pyramid import block_mean_downsample
        pixels = np.array([
            [[0, 0, 0], [2, 4, 6]],
            [[4, 8, 12], [6, 8, 10]],
        ], dtype=np.uint8)
        result = block_mean_downsample(pixels, 2)
        expected = np.array([[[3, 5, 7]]], dtype=np.uint8)
        np.testing.assert_array_equal(result, expected)


class CacheBudgetSnapshotTests(WorkbenchTestBase):
    def test_parameter_invalidation_stops_at_coarser_unaffected_level(self) -> None:
        session = self.completed_session()
        session.apply({"op": "set_guidance", "level": 1, "strength": 0.91, "id": "g"})
        report = session.run_batch(budget=48, max_workers=1)
        recomputed_levels = {tile_id[:3] for tile_id in report["recomputed"]}
        reused_levels = {tile_id[:3] for tile_id in report["reused"]}
        self.assertEqual(recomputed_levels, {"L01", "L02", "L03"})
        self.assertEqual(reused_levels, {"L00"})
        for entry in report["recomputed_entries"] + report["reused_entries"]:
            self.assertEqual(entry["cache_key"], session.records[entry["tile_id"]].cache_key)

    def test_budget_pending_resume_and_snapshot_restore(self) -> None:
        store = SnapshotStore(self.config, self.root / "state-budget")
        session = self.fresh_session()
        first = session.run_batch(budget=5, max_workers=1)
        order = session.planned.order
        self.assertEqual(first["budget_charged"], order[:5])
        self.assertEqual(first["pending"], order[5:])
        self.assertEqual(first["code"], "E_TILE_BUDGET_PENDING")
        committed = session.commit(store)
        self.assertEqual(committed["status"], "E_TILE_BUDGET_PENDING")
        self.assertTrue(store.manifest_path.exists())

        restored, manifest = open_snapshot_session_compat(self, store)
        self.assertEqual(manifest["pending"], order[5:])
        second = restored.run_batch(budget=5, max_workers=1)
        self.assertEqual(second["reused"], order[:5])
        self.assertEqual(second["budget_charged"], order[5:10])
        self.assertEqual(second["pending"], order[10:])
        restored.commit(store)

    def test_snapshot_version_and_hash_are_required(self) -> None:
        store = SnapshotStore(self.config, self.root / "state-stale")
        session = self.completed_session()
        session.commit(store)
        raw = json.loads(self.config_path.read_text(encoding="utf-8"))
        raw["global_version"] += 1
        stale_path = self.root / "configs" / "_stale.json"
        stale_path.write_text(json.dumps(raw), encoding="utf-8")
        stale_config = load_config(stale_path)
        stale_store = SnapshotStore(stale_config, store.root)
        manifest = stale_store.load_manifest()
        with self.assertRaises(WorkbenchError) as ctx:
            stale_store.check_compatible(manifest)
        self.assertEqual(ctx.exception.code, "E_SNAPSHOT_STALE")


class ConsistencyTests(WorkbenchTestBase):
    def test_default_command_stream_matches_full_rebuild(self) -> None:
        incremental = self.fresh_session()
        for command in self.commands:
            incremental.apply(command)
            report = incremental.run_batch(budget=48, max_workers=1)
            self.assertEqual(report["pending"], [])
        full = build_full_final(
            self.config, self.pixels, self.image_fingerprint, self.rectangles,
            self.commands, max_workers=1,
        )
        comparison = compare_sessions(incremental, full)
        self.assertTrue(comparison["equal"], comparison["differences"])

    def test_low_budget_drain_matches_full_rebuild(self) -> None:
        incremental = self.fresh_session()
        for command in self.commands:
            incremental.apply(command)
        for _ in range(20):
            report = incremental.run_batch(budget=7, max_workers=1)
            if not report["pending"]:
                break
        self.assertEqual(report["pending"], [])
        full = build_full_final(
            self.config, self.pixels, self.image_fingerprint, self.rectangles,
            self.commands, max_workers=1,
        )
        self.assertTrue(compare_sessions(incremental, full)["equal"])


    def test_each_command_can_drain_and_matches_full_rebuild(self) -> None:
        incremental = self.fresh_session()
        for command in self.commands:
            incremental.apply(command)
            for _ in range(20):
                report = incremental.run_batch(budget=9, max_workers=1)
                if not report["pending"]:
                    break
            self.assertEqual(report["pending"], [])
        full = build_full_final(
            self.config, self.pixels, self.image_fingerprint, self.rectangles,
            self.commands, max_workers=1,
        )
        self.assertTrue(compare_sessions(incremental, full)["equal"])

    def test_replay_reports_are_reproducible_from_blank(self) -> None:
        first = self.replay_with_budget(7)
        second = self.replay_with_budget(7)
        self.assertEqual(first, second)

    def replay_with_budget(self, budget: int) -> list[dict]:
        session = self.fresh_session()
        summaries = []
        for command in self.commands:
            session.apply(command)
            report = session.run_batch(budget=budget, max_workers=1)
            summaries.append({
                "command": command["id"],
                "reused": report["reused"],
                "recomputed": report["recomputed"],
                "pending": report["pending"],
                "budget_charged": report["budget_charged"],
            })
        return summaries

    def test_deterministic_random_sequences_match_full(self) -> None:
        rng = np.random.default_rng(20261125)
        for sequence_index in range(3):
            commands = self.random_command_sequence(rng, 7 + sequence_index)
            with self.subTest(sequence=sequence_index):
                incremental = self.fresh_session()
                for command in commands:
                    incremental.apply(command)
                    incremental.run_batch(budget=48, max_workers=1)
                full = build_full_final(
                    self.config, self.pixels, self.image_fingerprint, self.rectangles,
                    commands, max_workers=1,
                )
                comparison = compare_sessions(incremental, full)
                self.assertTrue(comparison["equal"], comparison["differences"])

    def random_command_sequence(self, rng: np.random.Generator, length: int) -> list[dict]:
        rect_ids = list(self.rectangles)
        commands = []
        temp_active = False
        for index in range(length):
            op = int(rng.integers(0, 6))
            level = int(rng.integers(0, self.config.levels))
            if op == 0 and rect_ids:
                rect_id = rect_ids[int(rng.integers(0, len(rect_ids)))]
                x = int(rng.integers(0, 70))
                y = int(rng.integers(0, 50))
                commands.append({"op": "upsert_rect", "rectangle": {
                    "id": rect_id, "x": x, "y": y,
                    "width": int(rng.integers(8, min(30, self.config.width - x) + 1)),
                    "height": int(rng.integers(8, min(24, self.config.height - y) + 1)),
                }})
            elif op == 1:
                if not temp_active:
                    commands.append({"op": "upsert_rect", "rectangle": {
                        "id": "random_temp", "x": 40, "y": 40, "width": 18, "height": 16,
                    }})
                    temp_active = True
                else:
                    commands.append({"op": "delete_rect", "rectangle_id": "random_temp"})
                    temp_active = False
            elif op == 2:
                commands.append({"op": "set_iterations", "level": level,
                                 "iterations": int(rng.integers(1, 9))})
            elif op == 3:
                commands.append({"op": "set_guidance", "level": level,
                                 "strength": round(float(rng.random()), 3)})
            elif op == 4:
                commands.append({"op": "set_style_weight", "level": level,
                                 "weights": [round(float(rng.uniform(0.6, 1.4)), 3) for _ in range(3)]})
            else:
                allowed = [4, 8, 16, 32, 64]
                commands.append({"op": "set_quantization", "level": level,
                                 "levels": allowed[int(rng.integers(0, len(allowed)))],
                                 "fractional_bits": 8})
        return commands


class ConcurrencyAndInjectionTests(WorkbenchTestBase):
    def test_serial_and_parallel_normal_runs_match(self) -> None:
        serial = self.completed_session(workers=1)
        parallel = self.completed_session(workers=self.config.max_workers)
        for level in range(self.config.levels):
            self.assertTrue(np.array_equal(serial.styled_levels[level], parallel.styled_levels[level]))
        self.assertEqual(
            {k: v.cache_key for k, v in serial.records.items()},
            {k: v.cache_key for k, v in parallel.records.items()},
        )

    def test_injection_diagnostic_is_isolated_and_reproducible(self) -> None:
        injections = load_injection_config(self.injection_path, self.config)
        report = run_injection_diagnostic(self.config, injections)
        self.assertTrue(report["all_injections_hit"], report)
        self.assertTrue(report["consistent"], report)
        self.assertEqual(len(report["failures"]), 3)
        codes = {item["code"] for item in report["failures"]}
        self.assertEqual(codes, {"E_NONFINITE_PIXELS", "E_MASK_OUT_OF_RANGE", "E_QUANT_OUT_OF_RANGE"})


def open_snapshot_session_compat(testcase: WorkbenchTestBase, store: SnapshotStore):
    from workbench.workflows import open_snapshot_session
    return open_snapshot_session(
        testcase.config, testcase.pixels, testcase.image_fingerprint,
        testcase.rectangles, store,
    )


if __name__ == "__main__":
    unittest.main()
