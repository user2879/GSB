import unittest

import numpy as np

from refrac.errors import IntersectDegenerateError, RayDegenerateError
from refrac.geometry import (
    Ray, intersect_background, intersect_slab, intersect_sphere, make_ray,
    nearest_transparent_hit,
)
from refrac.scene import BackgroundPlane, TransparentBody


def sphere(body_id="s", radius=1.0, center=(0.0, 0.0, 0.0), ior=1.5):
    return TransparentBody(
        body_id=body_id, kind="sphere", ior=ior, absorption=(0.0, 0.0, 0.0),
        center=tuple(center), radius=radius,
    )


class GeometryTests(unittest.TestCase):
    def test_ray_construction_and_degenerate(self):
        ray = make_ray((0, 0, 0), (0, 0, -1))
        np.testing.assert_allclose(ray.direction, [0, 0, -1])
        with self.assertRaises(RayDegenerateError) as ctx:
            make_ray((0, 0, 0), (0, 0, 0))
        self.assertEqual(ctx.exception.code, "E_RAY_DEGENERATE")

    def test_sphere_hit_normal_and_sides(self):
        body = sphere()
        enter_ray = make_ray((0, 0, 3), (0, 0, -1))
        hit = intersect_sphere(enter_ray, body)
        self.assertIsNotNone(hit)
        self.assertEqual(hit.side, "enter")
        np.testing.assert_allclose(hit.normal, [0, 0, 1], atol=1e-12)
        exit_ray = Ray(np.array([0.0, 0.0, 0.0]), np.array([0.0, 0.0, 1.0]))
        hit_exit = intersect_sphere(exit_ray, body)
        self.assertEqual(hit_exit.side, "exit")

    def test_sphere_miss(self):
        ray = make_ray((5, 0, 0), (0, 0, -1))
        self.assertIsNone(intersect_sphere(ray, sphere()))

    def test_negative_radius_degenerate(self):
        ray = make_ray((0, 0, 3), (0, 0, -1))
        with self.assertRaises(IntersectDegenerateError) as ctx:
            intersect_sphere(ray, sphere(radius=-1.0))
        self.assertEqual(ctx.exception.code, "E_INTERSECT_DEGENERATE")

    def test_slab_enter_exit_and_limited_extent(self):
        slab = TransparentBody(
            body_id="slab", kind="slab", ior=1.5, absorption=(0.0, 0.0, 0.0),
            center=(0.0, 0.0, 0.0), axis=2, thickness=0.4, half_extent=1.0,
        )
        ray = make_ray((0, 0, 3), (0, 0, -1))
        hit = intersect_slab(ray, slab)
        self.assertEqual(hit.side, "enter")
        self.assertAlmostEqual(hit.point[2], 0.2, places=10)
        wide_ray = make_ray((5, 0, 3), (0, 0, -1))
        self.assertIsNone(intersect_slab(wide_ray, slab))

    def test_background_plane(self):
        plane = BackgroundPlane("bg", (0, 0, -5), (0, 0, 1),
                                (0.5, 0.5, 0.5), (0.1, 0.1, 0.1), 1.0)
        ray = make_ray((0, 0, 0), (0, 0, -1))
        hit = intersect_background(ray, plane)
        self.assertEqual(hit.kind, "background")
        self.assertAlmostEqual(hit.t, 5.0, places=10)

    def test_nearest_hit_deterministic_order(self):
        bodies = [sphere("z_near", 1.0, (0, 0, -2)),
                  sphere("a_far", 1.0, (0, 0, -4))]
        ray = make_ray((0, 0, 0), (0, 0, -1))
        hit = nearest_transparent_hit(ray, bodies)
        self.assertEqual(hit.body_id, "z_near")


if __name__ == "__main__":
    unittest.main()
