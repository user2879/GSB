"""Business rules 2-4: projection, occlusion, scale invariance, equivalence."""

import json
import unittest

import numpy as np

from pointcloud_engine.engine import render_scene, render_serial_reference
from pointcloud_engine.pointcloud import PointCloud
from pointcloud_engine.render import prepare_points, render_serial
from pointcloud_engine.scene import load_scene
from tests._harness import Workspace, make_cloud


class RenderEquivalenceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.ws = Workspace()

    def tearDown(self) -> None:
        self.ws.cleanup()

    def test_parallel_matches_serial_pixel_for_pixel(self):
        scene_path = self.ws.write()
        result = render_scene(scene_path, install_signals=False)
        self.assertEqual(result["status"], "completed")

        scene, prepared, sg_gray, sg_depth, s_gray, s_depth = render_serial_reference(scene_path)
        p_gray = np.load(self.ws.path("output", "gray.npy"))
        p_depth = np.load(self.ws.path("output", "depth.npy"))
        np.testing.assert_array_equal(p_gray, s_gray)
        np.testing.assert_array_equal(p_depth, s_depth)

    def test_scheduling_order_independence(self):
        # Run the same scene with concurrency 1, 2 and 8; outputs must match.
        digests = []
        for concurrency in (1, 2, 8):
            ws = Workspace()
            path = ws.write()
            with open(path, encoding="utf-8") as handle:
                scene = json.load(handle)
            scene["concurrency"] = concurrency
            scene["memory_budget_bytes"] = scene["tile_width"] * scene["tile_height"] * 8 * concurrency
            with open(path, "w", encoding="utf-8") as handle:
                json.dump(scene, handle)
            render_scene(path, install_signals=False)
            digests.append((np.load(ws.path("output", "gray.npy")),
                            np.load(ws.path("output", "depth.npy"))))
            ws.cleanup()
        for gray, depth in digests[1:]:
            np.testing.assert_array_equal(gray, digests[0][0])
            np.testing.assert_array_equal(depth, digests[0][1])

    def test_occlusion_nearest_point_wins(self):
        # Two points project onto the same samples. The nearer one (smaller
        # view-axis depth) must win even though it is visited second.
        # Point A is far (z=0 -> depth 4.0, gray 0.9); point B is near
        # (z=0.5 -> depth 3.5, gray 0.1) so B must occlude A everywhere.
        positions = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.5]], dtype=np.float64)
        grays = np.array([0.9, 0.1], dtype=np.float64)
        normals = np.array([[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]], dtype=np.float64)
        cloud = PointCloud(positions, grays, normals, "")
        path = self.ws.write(cloud=cloud)
        scene = load_scene(path)
        prepared = prepare_points(scene)
        grid_gray, grid_depth, hits = render_serial(scene, prepared)
        depths = grid_depth[grid_depth > 0]
        # Every written sample must hold the nearer depth 3.5, never 4.0.
        self.assertTrue(np.allclose(depths, 3.5, atol=1e-4))
        # And the near point's attenuated gray (~0.1*factor), not 0.9*factor.
        near_factor = scene.decay.factor(
            np.array([3.5 / scene.camera.reference_distance]))[0]
        written = grid_gray[grid_depth > 0]
        np.testing.assert_allclose(written, 0.1 * near_factor, rtol=1e-5)

    def test_world_scale_does_not_flip_ordering(self):
        # Widely separated, non-overlapping point disks so each projected
        # center is owned by exactly one point at every scale.
        positions = np.array(
            [[-0.18, 0.12, 0.0], [0.15, -0.11, 0.05], [0.0, 0.0, -0.12],
             [-0.13, -0.15, 0.08], [0.17, 0.14, -0.06]], dtype=np.float64)
        grays = np.array([0.9, 0.3, 0.6, 0.2, 0.75], dtype=np.float64)
        normals = np.tile(np.array([0.0, 0.0, 1.0]), (5, 1))
        cloud = PointCloud(positions, grays, normals, "")

        records = {}
        for scale in (1.0, 3.0):
            ws = Workspace()
            path = ws.write(cloud=cloud)
            with open(path, encoding="utf-8") as handle:
                scene_json = json.load(handle)
            scene_json["world_scale"] = scale
            scene_json["point_radius"] = 0.04
            with open(path, "w", encoding="utf-8") as handle:
                json.dump(scene_json, handle)
            scene_obj, prepared, g_gray, g_depth, _, _ = render_serial_reference(path)
            records[scale] = (scene_obj, prepared, g_gray, g_depth)
            ws.cleanup()

        scene1, prep1, g1, d1 = records[1.0]
        scene3, prep3, g3, d3 = records[3.0]
        np.testing.assert_array_equal(prep1.index, prep3.index)

        def attenuation_order(scene_obj, prepared):
            ref = scene_obj.camera.reference_distance
            factor = scene_obj.decay.factor(prepared.depth / ref)
            return np.argsort(prepared.gray * factor)

        # Gray ordering must not flip when world scale changes.
        np.testing.assert_array_equal(attenuation_order(scene1, prep1),
                                      attenuation_order(scene3, prep3))
        # Occlusion/depth ordering is unchanged under a uniform world scale.
        np.testing.assert_array_equal(np.argsort(prep1.depth), np.argsort(prep3.depth))
        # Scaling point coordinates about the origin keeps every depth positive
        # and inside the same unified world-unit metric.
        self.assertTrue(np.all(prep1.depth > 0) and np.all(prep3.depth > 0))

        # Rendered buffers honor the same depth ordering on shared cells.
        occupied = (d1 > 0) & (d3 > 0)
        self.assertGreater(int(occupied.sum()), 0)
        np.testing.assert_array_equal(np.argsort(d1[occupied]), np.argsort(d3[occupied]))

    def test_frustum_and_near_culling_counted(self):
        # One point behind the camera, one far outside the frustum.
        cloud = make_cloud(n=4)
        cloud.positions[0] = [0.0, 0.0, 8.0]   # behind camera (z>4)
        cloud.positions[1] = [50.0, 0.0, 0.0]  # way outside frustum
        path = self.ws.write(cloud=cloud)
        result = render_scene(path, install_signals=False)
        report = result["report"]
        self.assertGreaterEqual(report["points_behind_near"], 1)
        self.assertGreaterEqual(report["points_culled_frustum"], 1)
        self.assertEqual(
            report["points_rendered"] + report["points_behind_near"] + report["points_culled_frustum"],
            report["point_count"],
        )

    def test_depth_uses_world_units(self):
        path = self.ws.write()
        scene, prepared, _, _, _, depth = render_serial_reference(path)
        # Camera at distance 4; every recorded depth must be positive world units.
        self.assertTrue(np.all(depth[depth > 0] > 0))
        self.assertLess(float(depth.max()), 8.0)


if __name__ == "__main__":
    unittest.main()
