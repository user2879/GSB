from __future__ import annotations

import unittest

import numpy as np

from pickworkbench.errors import E_NORMAL_ZERO, E_RAY_DEGENERATE, WorkbenchError
from pickworkbench.generator import _polyhedron_template, generate_sample_with_config
from pickworkbench.geometry import (
    compute_object_bounds,
    intersect_aabb,
    intersect_convex,
    intersect_sphere,
    select_winner,
    selection_key,
    world_planes,
)
from pickworkbench.engine import run_ray_query


def sphere_object(object_id: str, x: float, radius: float = 1.0, priority: int = 10) -> dict:
    obj = {
        "id": object_id,
        "object_index": int(object_id[-1]),
        "kind": "sphere",
        "transform": {"translation": [x, 0.0, 0.0], "rotation_deg": [0.0, 0.0, 0.0]},
        "geometry": {"radius": radius},
        "pick_priority": priority,
    }
    minimum, maximum, center, bound_radius = compute_object_bounds(obj)
    obj["bounding_volume"] = {
        "aabb": {"min": minimum.tolist(), "max": maximum.tolist()},
        "sphere": {"center": center.tolist(), "radius": bound_radius},
    }
    return obj


def box_object(object_id: str = "obj-box", priority: int = 10) -> dict:
    obj = {
        "id": object_id,
        "object_index": 0,
        "kind": "aabb",
        "transform": {"translation": [0.0, 0.0, 0.0], "rotation_deg": [0.0, 0.0, 0.0]},
        "geometry": {"half_extents": [1.0, 1.0, 1.0]},
        "pick_priority": priority,
    }
    minimum, maximum, center, bound_radius = compute_object_bounds(obj)
    obj["bounding_volume"] = {
        "aabb": {"min": minimum.tolist(), "max": maximum.tolist()},
        "sphere": {"center": center.tolist(), "radius": bound_radius},
    }
    return obj


def poly_object(object_id: str = "obj-poly", priority: int = 10) -> dict:
    vertices, planes = _polyhedron_template(1.0)
    obj = {
        "id": object_id,
        "object_index": 0,
        "kind": "convex_polyhedron",
        "transform": {"translation": [0.0, 0.0, 0.0], "rotation_deg": [0.0, 0.0, 0.0]},
        "geometry": {"vertices": vertices, "planes": planes},
        "pick_priority": priority,
    }
    minimum, maximum, center, bound_radius = compute_object_bounds(obj)
    obj["bounding_volume"] = {
        "aabb": {"min": minimum.tolist(), "max": maximum.tolist()},
        "sphere": {"center": center.tolist(), "radius": bound_radius},
    }
    return obj


class GeometryTests(unittest.TestCase):
    def test_sphere_ray_distance_and_method(self) -> None:
        hit = intersect_sphere(
            np.asarray([-3.0, 0.0, 0.0]),
            np.asarray([1.0, 0.0, 0.0]),
            sphere_object("obj-0", 0.0, 1.0),
        )
        self.assertIsNotNone(hit)
        self.assertAlmostEqual(hit["hit_distance"], 2.0, places=12)
        self.assertEqual(hit["hit_point"], [-1.0, 0.0, 0.0])
        self.assertEqual(hit["intersection_method"], "sphere_quadratic")

    def test_aabb_slab_intersection(self) -> None:
        hit = intersect_aabb(
            np.asarray([-3.0, 0.0, 0.0]),
            np.asarray([1.0, 0.0, 0.0]),
            box_object(),
        )
        self.assertIsNotNone(hit)
        self.assertAlmostEqual(hit["hit_distance"], 2.0, places=12)
        self.assertEqual(hit["intersection_method"], "aabb_slab")

    def test_convex_halfspace_intersection(self) -> None:
        obj = poly_object()
        self.assertGreater(len(world_planes(obj)), 6)
        hit = intersect_convex(
            np.asarray([-3.0, 0.0, 0.0]),
            np.asarray([1.0, 0.0, 0.0]),
            obj,
        )
        self.assertIsNotNone(hit)
        self.assertAlmostEqual(hit["hit_distance"], 2.0, places=12)
        self.assertEqual(hit["intersection_method"], "convex_halfspace_clip")

    def test_miss_returns_none(self) -> None:
        hit = intersect_sphere(
            np.asarray([-3.0, 4.0, 0.0]),
            np.asarray([1.0, 0.0, 0.0]),
            sphere_object("obj-0", 0.0, 1.0),
        )
        self.assertIsNone(hit)

    def test_nonunit_ray_is_normalized_by_public_query_path(self) -> None:
        scene, _ = generate_sample_with_config()
        query = {
            "query_index": 0,
            "query_id": "q-test-ray",
            "kind": "ray",
            "origin": [-3.0, 0.0, 0.0],
            "direction": [10.0, 0.0, 0.0],
        }
        scene = {
            **scene,
            "objects": [sphere_object("obj-0", 0.0, 1.0)],
            "object_count": 1,
        }
        record = run_ray_query(scene, query)
        self.assertAlmostEqual(record["winner"]["hit_distance"], 2.0, places=12)

    def test_priority_distance_and_id_tie_breaks(self) -> None:
        near_low_priority = {
            "object_id": "obj-near",
            "pick_priority": 10,
            "hit_distance": 2.0,
            "hit_point": [0.0, 0.0, 0.0],
            "intersection_method": "test",
        }
        far_high_priority = {
            "object_id": "obj-far",
            "pick_priority": 20,
            "hit_distance": 5.0,
            "hit_point": [0.0, 0.0, 0.0],
            "intersection_method": "test",
        }
        self.assertEqual(select_winner([near_low_priority, far_high_priority])["object_id"], "obj-far")
        same_priority_near = {**near_low_priority, "pick_priority": 20}
        self.assertEqual(select_winner([far_high_priority, same_priority_near])["object_id"], "obj-near")
        same_distance = {**same_priority_near, "object_id": "obj-zzz"}
        self.assertEqual(select_winner([same_distance, same_priority_near])["object_id"], "obj-near")
        self.assertLess(selection_key(same_priority_near), selection_key(same_distance))

    def test_degenerate_public_ray_raises_business_error(self) -> None:
        scene, _ = generate_sample_with_config()
        query = {
            "query_index": 0,
            "query_id": "q-zero",
            "kind": "ray",
            "origin": [0.0, 0.0, 0.0],
            "direction": [0.0, 0.0, 0.0],
        }
        with self.assertRaises(WorkbenchError) as context:
            run_ray_query(scene, query)
        self.assertEqual(context.exception.code, E_RAY_DEGENERATE)

    def test_zero_normal_raises_business_error(self) -> None:
        obj = poly_object()
        obj["geometry"]["planes"][0]["normal"] = [0.0, 0.0, 0.0]
        with self.assertRaises(WorkbenchError) as context:
            world_planes(obj)
        self.assertEqual(context.exception.code, E_NORMAL_ZERO)


if __name__ == "__main__":
    unittest.main()
