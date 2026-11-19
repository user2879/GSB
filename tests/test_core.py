from __future__ import annotations

import copy
import json
import os
import shutil
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from npr_edit.engine import NprEngine
from npr_edit.errors import (
    E_CLOSURE_MISMATCH,
    E_FORMAT_ENDIAN,
    E_FORMAT_INDEX,
    E_GL_CONTEXT,
    E_OUTLINE_DEGENERATE,
    E_SCHEMA_VERSION,
    E_SNAPSHOT_STALE,
    E_STROKE_SAMPLE,
    NPRError,
)
from npr_edit.formats import (
    inspect_skeleton_binary,
    migrate_project,
    roundtrip_report,
    scene_to_skeleton_binary,
    skeleton_binary_to_project,
    skeleton_binary_to_scene,
)
from npr_edit.generator import build_element_hash_manifest, generate_scene
from npr_edit.geometry import flatten_element, verify_closure_hashes
from npr_edit.gl_backend import open_offscreen_gl
from npr_edit.model import Point, Segment, sampling_from_dict
from npr_edit.persistence import PersistenceStore
from npr_edit.raster import render_snapshot
from npr_edit.snapshot import RenderState, SnapshotStore
from npr_edit.strategy import StrategySelector, scene_render_stats


class GenerationTests(unittest.TestCase):
    def test_generation_is_byte_identical_and_manifest_is_complete(self) -> None:
        first = generate_scene(200, seed=12345)
        second = generate_scene(200, seed=12345)
        self.assertEqual(first.project_bytes, second.project_bytes)
        self.assertEqual(first.content_hash, second.content_hash)
        manifest = build_element_hash_manifest(first, 200)
        self.assertEqual(manifest["element_count"], 200)
        self.assertEqual(manifest["layer_count"], len(first.layers))
        self.assertEqual(len(manifest["elements"]), 200)
        self.assertEqual(len({entry["content_hash"] for entry in manifest["elements"]}), 200)
        kinds = {segment.kind for element in first.elements for segment in element.segments}
        self.assertEqual(kinds, {"line", "quadratic", "cubic"})


class FlattenTests(unittest.TestCase):
    def setUp(self) -> None:
        self.element = generate_scene(200).elements[3]

    def test_flatten_and_skeleton_are_deterministic_and_derived(self) -> None:
        first, first_skeleton = flatten_element(self.element, target_viewport=(256, 192), canvas_size=(640, 480))
        second, second_skeleton = flatten_element(self.element, target_viewport=(256, 192), canvas_size=(640, 480))
        self.assertEqual(first.points, second.points)
        self.assertEqual(first.geometry_hash, second.geometry_hash)
        self.assertEqual(first_skeleton.vertices, first.points)
        self.assertEqual(first_skeleton.indices, first.indices)
        self.assertEqual(first_skeleton.content_hash, second_skeleton.content_hash)
        self.assertTrue(all(error <= first.tolerance for error in (segment.error_estimate for segment in first.segments)))

    def test_tighter_tolerance_is_monotonic(self) -> None:
        counts: list[int] = []
        for tolerance in (2.0, 1.0, 0.5, 0.25, 0.125):
            flat, _skeleton = flatten_element(self.element, tolerance_override=tolerance)
            counts.append(len(flat.points))
        self.assertEqual(counts, sorted(counts))

    def test_degenerate_segment_returns_business_error(self) -> None:
        degenerate = replace(
            self.element,
            segments=(Segment("line", (Point(10.0, 10.0), Point(10.0, 10.0))),),
        )
        with self.assertRaises(NPRError) as context:
            flatten_element(degenerate)
        self.assertEqual(context.exception.code, E_OUTLINE_DEGENERATE)

    def test_sampling_validation(self) -> None:
        with self.assertRaises(NPRError) as context:
            sampling_from_dict({"tolerance_px": 0.0})
        self.assertEqual(context.exception.code, E_STROKE_SAMPLE)


class MigrationTests(unittest.TestCase):
    def test_v1_migration_fills_fields_and_is_idempotent(self) -> None:
        scene = generate_scene(200)
        legacy = copy.deepcopy(scene.as_project_dict())
        legacy["schema_version"] = 1
        legacy.pop("sampling")
        for layer in legacy["layers"]:
            layer.pop("order", None)
        for element in legacy["elements"]:
            element.pop("style", None)
            element.pop("sampling", None)
            element.pop("layer_order", None)
        migrated = migrate_project(legacy)
        self.assertEqual(migrated["schema_version"], 3)
        self.assertIn("style", migrated["elements"][0])
        self.assertIn("sampling", migrated["elements"][0])
        self.assertIn("layer_order", migrated["elements"][0])
        migrated_twice = migrate_project(migrated)
        self.assertEqual(json.dumps(migrated, sort_keys=True), json.dumps(migrated_twice, sort_keys=True))

    def test_newer_schema_is_rejected(self) -> None:
        with self.assertRaises(NPRError) as context:
            migrate_project({"schema_version": 99})
        self.assertEqual(context.exception.code, E_SCHEMA_VERSION)


class BinaryFormatTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.scene = generate_scene(200)
        cls.binary = scene_to_skeleton_binary(cls.scene)

    def test_header_and_roundtrip(self) -> None:
        header = inspect_skeleton_binary(self.binary)
        self.assertEqual(header["magic"], "NPRSKEL")
        self.assertEqual(header["endian"], "little")
        self.assertEqual(header["index_width"], 4)
        self.assertGreater(header["vertex_count"], 0)
        self.assertGreater(header["index_count"], 0)
        report = roundtrip_report(self.scene, self.binary)
        self.assertTrue(report["within_tolerance"])
        self.assertTrue(report["roundtrip_exact_scene"])
        recovered = skeleton_binary_to_scene(self.binary)
        self.assertEqual(recovered.content_hash, self.scene.content_hash)

    def test_big_endian_u16_width_and_2_byte_indices(self) -> None:
        scene = replace(
            self.scene,
            elements=tuple(
                replace(element, stroke_width=element.stroke_width + 0.001234)
                for element in self.scene.elements
            ),
        )
        binary = scene_to_skeleton_binary(scene, endian="big", index_width=2, width_encoding="u16_fixed")
        header = inspect_skeleton_binary(binary)
        self.assertEqual(header["endian"], "big")
        self.assertEqual(header["index_width"], 2)
        self.assertEqual(header["line_width_encoding"], "u16_fixed")
        report = roundtrip_report(scene, binary)
        self.assertTrue(report["within_tolerance"])
        self.assertLessEqual(report["max_stroke_width_error"], 1.0 / 512.0)

    def test_bad_endian_and_index_width(self) -> None:
        bad_endian = bytearray(self.binary)
        bad_endian[8:10] = b"XX"
        with self.assertRaises(NPRError) as context:
            skeleton_binary_to_project(bytes(bad_endian))
        self.assertEqual(context.exception.code, E_FORMAT_ENDIAN)
        bad_index = bytearray(self.binary)
        bad_index[10] = 3
        with self.assertRaises(NPRError) as context:
            skeleton_binary_to_project(bytes(bad_index))
        self.assertEqual(context.exception.code, E_FORMAT_INDEX)


class StrategyTests(unittest.TestCase):
    def test_tier_choice_is_reproducible_and_cache_is_tier_scoped(self) -> None:
        scene = generate_scene(200)
        selector = StrategySelector()
        flat_map = {}
        for element in scene.elements:
            flat, _ = flatten_element(element, target_viewport=(256, 192), canvas_size=(640, 480))
            flat_map[element.id] = flat
        stats = scene_render_stats(scene, flat_map, (256, 192))
        large_stats = replace(
            stats,
            element_count=20000,
            point_count=stats.point_count * 100,
            avg_bbox_area_px=stats.avg_bbox_area_px * 0.45,
            tile_count=40,
        )
        medium_stats = replace(
            stats,
            element_count=2000,
            point_count=stats.point_count * 10,
            avg_bbox_area_px=stats.avg_bbox_area_px * 0.65,
            tile_count=20,
        )
        first_200 = selector.select(200, stats)
        first_2000 = selector.select(2000, medium_stats)
        first_20000 = selector.select(20000, large_stats)
        self.assertEqual(first_200.chosen, "per_element")
        self.assertEqual(first_2000.chosen, "layer_merge")
        self.assertEqual(first_20000.chosen, "tile_batch")
        calls = selector.selection_calls
        selector.select(200, stats)
        selector.select(20000, large_stats)
        self.assertEqual(selector.selection_calls, calls)
        selector.select(2000, medium_stats, {"frame_ms": 45.0, "memory_mb": 128.0})
        self.assertEqual(selector.selection_calls, calls + 1)


class EngineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.scene = generate_scene(200)
        cls.engine = NprEngine(cls.scene, viewport=(192, 144))
        cls.full = cls.engine.full_render()
        cls.before = cls.engine.current_snapshot

    def test_incremental_point_edit_has_exact_closure(self) -> None:
        engine = NprEngine(self.scene, viewport=(192, 144))
        full = engine.full_render()
        before = engine.current_snapshot
        target = self.scene.elements[7]
        point = target.segments[0].points[0]
        record = engine.edit_control_point(
            target.id,
            0,
            0,
            point.x + 1.0,
            point.y + 0.5,
            expected_version=full.snapshot.version,
            expected_hash=full.snapshot.snapshot_hash,
        )
        after = engine.current_snapshot
        self.assertTrue(record.changed)
        self.assertEqual(set(record.element_ids), {target.id})
        self.assertEqual(set(record.layer_ids), {target.layer_id})
        self.assertEqual(set(record.recomputed_elements), {target.id})
        self.assertEqual(set(record.recomputed_layers), {target.layer_id})
        for element_id in after.element_hashes:
            if element_id != target.id:
                self.assertEqual(before.skeleton_hashes[element_id], after.skeleton_hashes[element_id])
                self.assertEqual(before.element_hashes[element_id], after.element_hashes[element_id])

    def test_style_switch_reuses_all_geometry(self) -> None:
        engine = NprEngine(self.scene, viewport=(192, 144))
        full = engine.full_render()
        build_counts = dict(engine.geometry.build_counts)
        skeleton_hashes = dict(engine.current_snapshot.skeleton_hashes)
        record = engine.set_style_mode(
            "cartoon",
            expected_version=full.snapshot.version,
            expected_hash=full.snapshot.snapshot_hash,
        )
        self.assertTrue(record.changed)
        self.assertEqual(record.geometry_rebuilt, ())
        self.assertEqual(dict(engine.geometry.build_counts), build_counts)
        self.assertEqual(dict(engine.current_snapshot.skeleton_hashes), skeleton_hashes)

    def test_snapshot_stale_is_rejected(self) -> None:
        engine = NprEngine(self.scene, viewport=(192, 144))
        engine.full_render()
        with self.assertRaises(NPRError) as context:
            engine.read_snapshot(0)
        self.assertEqual(context.exception.code, E_SNAPSHOT_STALE)

    def test_render_state_is_immutable_versioned_pair(self) -> None:
        engine = NprEngine(self.scene, viewport=(192, 144))
        engine.full_render()
        state = engine.render_state
        self.assertIsInstance(state, RenderState)
        self.assertEqual(state.version, engine.current_snapshot.version)
        self.assertEqual(state.scene.content_hash, state.snapshot.scene_hash)
        with self.assertRaises(Exception):
            state.version = 99  # type: ignore[misc]

    def test_wrong_dirty_layer_returns_closure_mismatch(self) -> None:
        engine = NprEngine(self.scene, viewport=(192, 144))
        full = engine.full_render()
        current = engine.current_snapshot
        element = self.scene.elements[0]
        wrong_layer = next(layer.id for layer in self.scene.layers if layer.id != element.layer_id)
        with self.assertRaises(NPRError) as context:
            render_snapshot(
                engine.scene,
                engine.geometry,
                engine.mode,
                engine.viewport,
                current.version + 1,
                prior=current,
                dirty_elements={element.id},
                dirty_layers={wrong_layer},
                strategy=engine.strategy,
            )
        self.assertEqual(context.exception.code, E_CLOSURE_MISMATCH)


class PersistenceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.root = Path(tempfile.mkdtemp(prefix="npr-persistence-test-"))
        self.scene = generate_scene(200)
        self.engine = NprEngine(self.scene, viewport=(128, 96))
        self.engine.full_render()

    def tearDown(self) -> None:
        shutil.rmtree(self.root, ignore_errors=True)

    def test_commit_recover_and_uncommitted_orphan(self) -> None:
        store = PersistenceStore(self.root)
        manifest = store.commit(self.engine.scene, self.engine.current_snapshot)
        recovered = store.recover()
        self.assertIsNotNone(recovered)
        assert recovered is not None
        self.assertFalse(recovered.clean_shutdown)
        self.assertEqual(recovered.recovered_scene_hash, self.scene.content_hash)
        self.assertEqual(recovered.recovered_snapshot_hash, self.engine.current_snapshot.snapshot_hash)
        self.assertEqual(recovered.version, self.engine.current_snapshot.version)
        recovered_engine = NprEngine.from_snapshot(recovered.scene, recovered.snapshot)
        target = recovered.scene.elements[0]
        point = target.segments[0].points[0]
        edit = recovered_engine.edit_control_point(
            target.id,
            0,
            0,
            point.x + 0.5,
            point.y + 0.5,
            expected_version=recovered.version,
            expected_hash=recovered.snapshot.snapshot_hash,
        )
        self.assertEqual(edit.old_version, recovered.version)
        store.write_uncommitted_generation(recovered_engine.scene, recovered_engine.current_snapshot)
        recovered_again = store.recover()
        self.assertIsNotNone(recovered_again)
        assert recovered_again is not None
        self.assertTrue(recovered_again.ignored_uncommitted)
        self.assertEqual(recovered_again.recovered_snapshot_hash, self.engine.current_snapshot.snapshot_hash)
        marker = store.close(recovered_engine.scene, recovered_engine.current_snapshot)
        self.assertTrue(marker["clean"])
        final = store.recover()
        self.assertIsNotNone(final)
        assert final is not None
        self.assertTrue(final.clean_shutdown)
        self.assertEqual(final.version, recovered_engine.current_snapshot.version)
        self.assertEqual(manifest["version"], self.engine.current_snapshot.version)

    def test_snapshot_store_rejects_old_publication(self) -> None:
        store = SnapshotStore(initial=self.engine.current_snapshot)
        duplicate = replace(self.engine.current_snapshot, snapshot_hash="")
        with self.assertRaises(NPRError) as context:
            store.publish(duplicate)
        self.assertEqual(context.exception.code, E_SNAPSHOT_STALE)


class GLTests(unittest.TestCase):
    def test_forced_context_failure_is_business_error(self) -> None:
        old = os.environ.get("NPR_DISABLE_GL")
        os.environ["NPR_DISABLE_GL"] = "1"
        try:
            with self.assertRaises(NPRError) as context:
                open_offscreen_gl()
            self.assertEqual(context.exception.code, E_GL_CONTEXT)
        finally:
            if old is None:
                os.environ.pop("NPR_DISABLE_GL", None)
            else:
                os.environ["NPR_DISABLE_GL"] = old


if __name__ == "__main__":
    unittest.main()
