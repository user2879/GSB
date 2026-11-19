from __future__ import annotations

import unittest

import numpy as np

from volumetric_editor.scene import generate_scene, scene_bytes
from volumetric_editor.session import IncrementalSession


class DeterminismTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.scene = generate_scene(seed=424242, width=32, height=24)

    def test_scene_generation_is_byte_identical(self) -> None:
        first = generate_scene(seed=424242, width=32, height=24)
        second = generate_scene(seed=424242, width=32, height=24)
        self.assertEqual(scene_bytes(first), scene_bytes(second))
        for surface in first.surfaces:
            self.assertEqual(len(surface.normal), 3)
            self.assertGreaterEqual(surface.ior, 1.0)
            self.assertLessEqual(surface.ior, 3.0)
            faces = surface.to_dict()["faces"]
            self.assertEqual(len(faces), 2)
            self.assertEqual(faces[0]["ior"], surface.ior)

    def test_image_and_volume_trace_are_invariant_to_threads_and_tiles(self) -> None:
        results = []
        for threads, tile_size in ((1, 8), (2, 8), (3, 8), (1, 12), (2, 12)):
            session = IncrementalSession(self.scene, tile_size=tile_size)
            session.render_full(threads=threads)
            results.append(session)

        baseline_image = results[0].image.tobytes(order="C")
        baseline_trace = results[0].trace.tobytes(order="C")
        baseline_inside = results[0].inside.tobytes(order="C")
        for session in results[1:]:
            self.assertEqual(session.image.dtype, np.float32)
            self.assertEqual(session.image.tobytes(order="C"), baseline_image)
            self.assertEqual(session.trace.tobytes(order="C"), baseline_trace)
            self.assertEqual(session.inside.tobytes(order="C"), baseline_inside)

    def test_repeated_render_is_byte_identical(self) -> None:
        first = IncrementalSession(self.scene, tile_size=8)
        second = IncrementalSession(self.scene, tile_size=8)
        first.render_full(threads=3)
        second.render_full(threads=3)
        self.assertEqual(first.image.tobytes(order="C"), second.image.tobytes(order="C"))
        self.assertEqual(first.trace.tobytes(order="C"), second.trace.tobytes(order="C"))


if __name__ == "__main__":
    unittest.main()
