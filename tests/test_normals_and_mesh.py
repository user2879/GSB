from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from pointcloud_studio.consistency import compare_meshes
from pointcloud_studio.engine import StudioEngine
from pointcloud_studio.errors import ErrorCode, StudioError
from pointcloud_studio.ledger import build_ledger, verify_ledger
from pointcloud_studio.mesh import ReconstructionParams
from pointcloud_studio.normals import reestimate_normals_in_region, validate_neighbor_radius
from pointcloud_studio.render import OffscreenViewport
from pointcloud_studio.sample import load_sample_bundle, write_sample_bundle
from pointcloud_studio.snapshots import SnapshotStore, scene_from_engine
from pointcloud_studio.spatial import VoxelGrid, select_sphere
from pointcloud_studio.viewport import CameraPose, project_points, viewport_box_remove_operation


class NormalAndMeshTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.temporary = tempfile.TemporaryDirectory(prefix="pcstudio-mesh-tests-")
        root = Path(cls.temporary.name)
        write_sample_bundle(root / "sample")
        cls.points, cls.normals, cls.confidence, _ = load_sample_bundle(root / "sample")
        cls.engine = StudioEngine(cls.points, cls.normals, cls.confidence, params=ReconstructionParams())

    @classmethod
    def tearDownClass(cls) -> None:
        cls.temporary.cleanup()

    def test_voxel_neighbors_and_normal_region_isolation(self) -> None:
        grid = VoxelGrid.build(self.points, 0.34)
        neighbors = grid.query_radius(self.points, self.points[123], 0.34)
        self.assertGreater(neighbors.size, 10)
        self.assertIn(123, neighbors)
        selected = select_sphere(self.points, [2.7, 1.3, 3.96], 0.72)
        bounds = np.full(self.points.shape[0], 1e-4, dtype=np.float32)
        result = reestimate_normals_in_region(
            self.points,
            self.normals,
            self.confidence,
            bounds,
            selected,
            0.34,
        )
        outside = np.ones(self.points.shape[0], dtype=bool)
        outside[selected] = False
        self.assertTrue(np.array_equal(result.normals[outside], self.normals[outside]))
        self.assertTrue(np.array_equal(result.confidence[outside], self.confidence[outside]))
        self.assertEqual(result.audit()["neighbor_radius"], 0.34)
        self.assertTrue(np.isfinite(result.residuals[selected]).all())
        self.assertTrue((result.neighbor_counts[selected] >= 4).all())
        self.assertTrue((result.error_bounds[selected] <= np.pi / 2.0 + 1e-7).all())
        with self.assertRaises(StudioError) as caught:
            validate_neighbor_radius(0.001)
        self.assertEqual(caught.exception.code, ErrorCode.E_NEIGHBOR_RADIUS)

    def test_incremental_cache_closure_and_full_consistency(self) -> None:
        engine = StudioEngine(self.points, self.normals, self.confidence, params=ReconstructionParams())
        before_vertices = {key: patch.vertices.copy() for key, patch in engine.patches.items()}
        before_objects = dict(engine.patches)
        record = engine.apply_operation({
            "seq": 1,
            "type": "remove_sphere",
            "params": {"center": [2.7, 1.3, 3.985], "radius": 0.36},
        })
        invalidation = record["invalidation"]
        self.assertTrue(invalidation["closure_equals_recomputed"])
        self.assertEqual(invalidation["closure_patch_keys"], invalidation["recomputed_patch_keys"])
        self.assertFalse(invalidation["whole_model_rebuild"])
        self.assertGreater(invalidation["reused_patch_count"], 0)
        self.assertLess(invalidation["recomputed_patch_count"], len(before_objects))
        for key in invalidation["reused_patch_keys"]:
            self.assertIs(engine.patches[key], before_objects[key])
            self.assertTrue(np.array_equal(engine.patches[key].vertices, before_vertices[key]))
        comparison = record["consistency_check"]
        self.assertTrue(comparison["ok"])
        self.assertTrue(all(comparison["criteria"].values()))
        self.assertEqual(comparison["maximum_position_deviation"], 0.0)
        self.assertEqual(record["normal_audit"], None)

    def test_ledger_recompute_and_mismatch_locating_edit(self) -> None:
        engine = StudioEngine(self.points, self.normals, self.confidence, params=ReconstructionParams())
        ledger = build_ledger(
            engine.cloud.positions,
            engine.cloud.normals,
            engine.cloud.confidence,
            engine.patches,
            engine.mesh,
            engine.cloud.params,
            engine.cloud.normal_radius,
        )
        result = verify_ledger(
            engine.cloud.positions,
            engine.cloud.normals,
            engine.cloud.confidence,
            ledger,
        )
        self.assertTrue(result["ok"])
        self.assertEqual(result["maximum_numeric_difference"], 0.0)
        with self.assertRaises(StudioError) as caught:
            compare_meshes(
                engine.mesh,
                engine.mesh,
                engine.patches,
                tolerance=1e-9,
                edit_seq=77,
                injected_vertex_offset=(0.01, 0.0, 0.0),
            )
        self.assertEqual(caught.exception.code, ErrorCode.E_MESH_MISMATCH)
        self.assertEqual(caught.exception.context["first_divergence"]["edit_seq"], 77)
        self.assertEqual(caught.exception.context["first_divergence"]["local_vertex_index"], 0)

    def test_snapshot_store_rejects_stale_and_mutable_publication(self) -> None:
        store = SnapshotStore()
        scene = store.publish(scene_from_engine(self.engine))
        self.assertFalse(scene.positions.flags.writeable)
        self.assertFalse(scene.mesh.vertices.flags.writeable)
        with self.assertRaises(TypeError):
            scene.ledger["entries"] = {}
        with self.assertRaises(StudioError) as caught:
            store.get(scene.version - 1)
        self.assertEqual(caught.exception.code, ErrorCode.E_SNAPSHOT_STALE)
        stale_render = OffscreenViewport(store).render(width=64, height=64, expected_version=scene.version - 1)
        self.assertFalse(stale_render.ok)
        self.assertEqual(stale_render.code, ErrorCode.E_SNAPSHOT_STALE.value)

    def test_viewport_box_selection_builds_exact_edit_operation(self) -> None:
        camera = CameraPose()
        pixels, _, visible = project_points(self.points, camera, 640, 480)
        pixel = pixels[visible][10]
        operation = viewport_box_remove_operation(
            self.points,
            camera=camera,
            width=640,
            height=480,
            rect=[pixel[0] - 3.0, pixel[1] - 3.0, pixel[0] + 3.0, pixel[1] + 3.0],
            seq=88,
        )
        self.assertEqual(operation["type"], "remove_viewport_box")
        self.assertEqual(operation["seq"], 88)


if __name__ == "__main__":
    unittest.main()
