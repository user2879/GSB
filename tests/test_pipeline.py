from __future__ import annotations

import tempfile
import unittest

from diffrender.errors import DiffRenderError, E_SNAPSHOT_VERSION
from diffrender.persistence import StateStore
from diffrender.pipeline import IncrementalPipeline
from diffrender.renderer import (
    grad_aggregate_node_key,
    grad_chunk_node_key,
    raster_node_key,
    render_full,
    shade_node_key,
)
from diffrender.scene import default_scene, target_scene_from


class PipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.scene = default_scene()
        cls.target = render_full(target_scene_from(cls.scene)).image

    def build_pipeline(self) -> IncrementalPipeline:
        return IncrementalPipeline(self.scene, self.target)

    def test_shade_only_edit_reuses_raster_tiles(self) -> None:
        pipeline = self.build_pipeline()
        before = pipeline.graph
        raster_before = {
            key: (node.version, node.content_hash, node.version_stamp)
            for key, node in before.nodes.items()
            if node.kind == "raster_tile"
        }
        entry = pipeline.set_parameter("sphere.1.albedo", 0.55)
        self.assertTrue(entry["meaningful"])
        self.assertEqual(entry["dirty_raster_tiles"], [])
        self.assertEqual(entry["reused_count"], len(pipeline.graph.nodes) - entry["closure_size"])
        raster_after = {
            key: (node.version, node.content_hash, node.version_stamp)
            for key, node in pipeline.graph.nodes.items()
            if node.kind == "raster_tile"
        }
        self.assertEqual(raster_before, raster_after)
        self.assertIn(shade_node_key(2, 3), entry["closure"])
        self.assertFalse(any(key.startswith("raster:") for key in entry["closure"]))
        pipeline.compare_with_full()

    def test_tolerance_noop_never_invalidates(self) -> None:
        pipeline = self.build_pipeline()
        before_hashes = pipeline.node_hash_manifest()
        before_stamps = pipeline.node_stamp_manifest()
        before_image = pipeline.render_result().image_hash
        old = pipeline.scene.value("sphere.0.radius")
        entry = pipeline.set_parameter("sphere.0.radius", old + 5.0e-10)
        self.assertFalse(entry["meaningful"])
        self.assertEqual(entry["closure_size"], 0)
        self.assertEqual(entry["reason"], "within_tolerance")
        self.assertEqual(entry["tolerance"], pipeline.tolerances.sphere_geometry)
        self.assertEqual(before_hashes, pipeline.node_hash_manifest())
        self.assertEqual(before_stamps, pipeline.node_stamp_manifest())
        self.assertEqual(before_image, pipeline.render_result().image_hash)
        self.assertEqual(entry["reused_count"], len(pipeline.graph.nodes))

    def test_geometry_closure_is_exact_and_incremental_matches_full(self) -> None:
        pipeline = self.build_pipeline()
        key = "sphere.0.radius"
        new_value = pipeline.scene.value(key) + 0.045
        old_scene = pipeline.scene
        new_scene = old_scene.with_value(key, new_value)
        dirty_raster, dirty_shade = pipeline.affected_tiles(key, old_scene, new_scene)
        entry = pipeline.set_parameter(key, new_value)
        expected = {f"param:{key}", "image:final", "loss:l2"}
        expected.update(raster_node_key(ty, tx) for ty, tx in dirty_raster)
        expected.update(shade_node_key(ty, tx) for ty, tx in dirty_shade)
        for ty, tx in set(dirty_raster) | set(dirty_shade):
            for param_key in self.scene.param_keys():
                expected.add(grad_chunk_node_key(ty, tx, param_key))
        for param_key in self.scene.param_keys():
            expected.add(grad_aggregate_node_key(param_key))
        self.assertEqual(set(entry["closure"]), expected)
        self.assertEqual(entry["closure_size"], len(expected))
        self.assertTrue(entry["meaningful"])
        pipeline.compare_with_full()

    def test_persistence_restore_and_continue(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = StateStore(tmp)
            pipeline = self.build_pipeline()
            pipeline.commit(store)
            pipeline.set_parameter("camera.yaw", pipeline.scene.value("camera.yaw") + 0.01)
            version = pipeline.commit(store)
            hashes = pipeline.node_hash_manifest()
            stamps = pipeline.node_stamp_manifest()
            state, pointer = store.read_snapshot(expected_snapshot_version=version)
            self.assertFalse(pointer["closed"])
            restored = IncrementalPipeline.from_state_dict(state)
            self.assertEqual(restored.node_hash_manifest(), hashes)
            self.assertEqual(restored.node_stamp_manifest(), stamps)
            restored.set_parameter("sphere.2.albedo", 0.60)
            restored.compare_with_full()
            restored.commit(store)

    def test_snapshot_hash_tamper_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = StateStore(tmp)
            pipeline = self.build_pipeline()
            pipeline.commit(store)
            pointer = store.pointer()
            assert pointer is not None
            snapshot_path = store.committed_dir / pointer["snapshot_file"]
            payload = bytearray(snapshot_path.read_bytes())
            payload[len(payload) // 2] ^= 0x01
            snapshot_path.write_bytes(bytes(payload))
            with self.assertRaises(DiffRenderError) as context:
                store.read_snapshot()
            self.assertEqual(context.exception.code, E_SNAPSHOT_VERSION)

    def test_forced_recovery_uses_last_commit_and_reclaims_orphan(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = StateStore(tmp)
            pipeline = self.build_pipeline()
            pipeline.commit(store)
            pipeline.set_parameter("sphere.0.radius", 0.69)
            version = pipeline.commit(store)
            hashes = pipeline.node_hash_manifest()
            stamps = pipeline.node_stamp_manifest()
            store.write_staging("partial-cache.json", {"uncommitted": True})
            state, recovery = store.recover(expected_snapshot_version=version)
            assert state is not None and recovery is not None
            self.assertTrue(recovery.forced_termination_detected)
            self.assertFalse(recovery.clean_shutdown)
            self.assertFalse(recovery.fabricated_exit_report)
            self.assertEqual(len(recovery.orphans), 1)
            self.assertEqual(list(store.staging_dir.iterdir()), [])
            restored = IncrementalPipeline.from_state_dict(state)
            self.assertEqual(restored.node_hash_manifest(), hashes)
            self.assertEqual(restored.node_stamp_manifest(), stamps)
            self.assertNotIn("exit_report", state)

    def test_controlled_close_writes_marker_and_exit_report(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = StateStore(tmp)
            pipeline = self.build_pipeline()
            version = pipeline.commit(store, closed=True)
            state, pointer = store.read_snapshot(expected_snapshot_version=version)
            self.assertTrue(pointer["closed"])
            self.assertTrue(state["closed"])
            self.assertTrue(state["clean_shutdown"])
            self.assertIn("exit_report", state)
            self.assertTrue((store.committed_dir / "CLOSED.json").exists())
            _, recovery = store.recover()
            assert recovery is not None
            self.assertTrue(recovery.clean_shutdown)


if __name__ == "__main__":
    unittest.main()
