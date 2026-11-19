"""Business rule 1: parsing and validation error codes."""

import json
import os
import unittest

import numpy as np

from pointcloud_engine.errors import EngineError
from pointcloud_engine.pointcloud import PointCloud, save_point_cloud
from tests._harness import Workspace, base_scene, make_cloud


class ValidationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.ws = Workspace()

    def tearDown(self) -> None:
        self.ws.cleanup()

    def _scene(self, mutate=None, cloud: PointCloud | None = None):
        path = self.ws.write(cloud=cloud)
        with open(path, encoding="utf-8") as handle:
            scene = json.load(handle)
        if mutate:
            mutate(scene)
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(scene, handle)
        return path

    def test_missing_scene(self):
        with self.assertRaises(EngineError) as ctx:
            from pointcloud_engine.scene import load_scene
            load_scene(self.ws.path("nope.json"))
        self.assertEqual(ctx.exception.code, "E_SCENE")

    def test_bad_world_scale(self):
        for value in (0.0, -1.0, float("nan")):
            path = self._scene(lambda s: s.update(world_scale=value))
            with self.assertRaises(EngineError) as ctx:
                from pointcloud_engine.scene import load_scene
                load_scene(path)
            self.assertEqual(ctx.exception.code, "E_SCALE")

    def test_bad_fov(self):
        for value in (0.0, 180.0, 200.0):
            path = self._scene(lambda s, v=value: s["camera"].update(vfov_deg=v))
            with self.assertRaises(EngineError) as ctx:
                from pointcloud_engine.scene import load_scene
                load_scene(path)
            self.assertEqual(ctx.exception.code, "E_FOV")

    def test_bad_aspect(self):
        path = self._scene(lambda s: s["camera"].update(aspect=0.0))
        with self.assertRaises(EngineError) as ctx:
            from pointcloud_engine.scene import load_scene
            load_scene(path)
        self.assertEqual(ctx.exception.code, "E_SCHEMA")

    def test_parallel_up_vector(self):
        path = self._scene(lambda s: s["camera"].update(up=[0.0, 0.0, -1.0]))
        with self.assertRaises(EngineError) as ctx:
            from pointcloud_engine.scene import load_scene
            load_scene(path)
        self.assertEqual(ctx.exception.code, "E_CAMERA")

    def test_non_integer_resolution(self):
        path = self._scene(lambda s: s.update(width=0))
        with self.assertRaises(EngineError) as ctx:
            from pointcloud_engine.scene import load_scene
            load_scene(path)
        self.assertEqual(ctx.exception.code, "E_DIM")

    def test_tile_does_not_cover_grid(self):
        path = self._scene(lambda s: s.update(tile_width=17))
        with self.assertRaises(EngineError) as ctx:
            from pointcloud_engine.scene import load_scene
            load_scene(path)
        self.assertEqual(ctx.exception.code, "E_TILE")

    def test_gray_out_of_range(self):
        cloud = make_cloud()
        cloud.grays[3] = 1.5
        path = self.ws.write(cloud=cloud)
        with self.assertRaises(EngineError) as ctx:
            from pointcloud_engine.scene import load_scene
            load_scene(path)
        self.assertEqual(ctx.exception.code, "E_GRAY")

    def test_zero_normal(self):
        cloud = make_cloud()
        cloud.normals[5] = 0.0
        path = self.ws.write(cloud=cloud)
        with self.assertRaises(EngineError) as ctx:
            from pointcloud_engine.scene import load_scene
            load_scene(path)
        self.assertEqual(ctx.exception.code, "E_NORMAL")

    def test_non_finite_coordinate(self):
        cloud = make_cloud()
        cloud.positions[2, 1] = float("inf")
        path = self.ws.write(cloud=cloud)
        with self.assertRaises(EngineError) as ctx:
            from pointcloud_engine.scene import load_scene
            load_scene(path)
        self.assertEqual(ctx.exception.code, "E_FINITE")

    def test_memory_budget_too_small(self):
        path = self._scene(lambda s: s.update(memory_budget_bytes=10))
        with self.assertRaises(EngineError) as ctx:
            from pointcloud_engine.scene import load_scene
            load_scene(path)
        self.assertEqual(ctx.exception.code, "E_MEMORY")

    def test_no_output_files_on_validation_error(self):
        path = self._scene(lambda s: s.update(world_scale=-1.0))
        with self.assertRaises(EngineError):
            from pointcloud_engine.engine import render_scene
            render_scene(path, install_signals=False)
        self.assertFalse(os.path.exists(self.ws.path("output")))
        self.assertFalse(os.path.exists(self.ws.path("state")))


if __name__ == "__main__":
    unittest.main()
