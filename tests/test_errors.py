from __future__ import annotations

import dataclasses
import unittest

from diffrender.errors import (
    DiffRenderError,
    E_CLOSURE_MISMATCH,
    E_DEPENDENCY_STALE,
    E_GRAD_MISMATCH,
    E_SCENE_INVALID,
    E_TILE_HASH,
)
from diffrender.graph import DependencyGraph, Node
from diffrender.pipeline import IncrementalPipeline
from diffrender.renderer import (
    compare_gradient_results,
    compare_render_results,
    gradient_full,
    image_hash,
    render_full,
)
from diffrender.scene import default_scene, target_scene_from


class ErrorCodeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.scene = default_scene()
        cls.target = render_full(target_scene_from(cls.scene)).image

    def test_scene_invalid(self) -> None:
        with self.assertRaises(DiffRenderError) as context:
            self.scene.with_value("sphere.0.radius", -1.0)
        self.assertEqual(context.exception.code, E_SCENE_INVALID)

    def test_tile_hash_mismatch(self) -> None:
        reference = render_full(self.scene)
        bad_image = list(reference.image)
        bad_image[0] += 1.0
        corrupted = dataclasses.replace(
            reference, image=tuple(bad_image), image_hash=image_hash(bad_image)
        )
        with self.assertRaises(DiffRenderError) as context:
            compare_render_results(corrupted, reference)
        self.assertEqual(context.exception.code, E_TILE_HASH)
        self.assertIsNotNone(context.exception.details.get("first_divergent_tile"))

    def test_gradient_mismatch_locates_tile_and_param(self) -> None:
        pipeline = IncrementalPipeline(self.scene, self.target)
        incremental = pipeline.gradient_result()
        full = gradient_full(self.scene, self.target)
        key = "sphere.0.radius"
        bad_pixels = list(incremental.pixel_gradients[key])
        bad_pixels[0] += 1.0
        bad_chunks = {name: dict(chunks) for name, chunks in incremental.chunks.items()}
        bad_chunks[key][(0, 0)] = tuple(bad_pixels[:64])
        bad = dataclasses.replace(
            incremental,
            pixel_gradients={**incremental.pixel_gradients, key: tuple(bad_pixels)},
            chunks=bad_chunks,
        )
        with self.assertRaises(DiffRenderError) as context:
            compare_gradient_results(bad, full)
        self.assertEqual(context.exception.code, E_GRAD_MISMATCH)
        self.assertEqual(context.exception.details.get("param_key"), key)
        self.assertIsNotNone(context.exception.details.get("first_divergent_tile"))

    def test_closure_mismatch(self) -> None:
        graph = DependencyGraph()
        graph.add_node(Node("a", "test", "h", 0, "s"))
        graph.add_node(Node("b", "test", "h", 0, "s"))
        with self.assertRaises(DiffRenderError) as context:
            graph.assert_closure(["a"], ["b"])
        self.assertEqual(context.exception.code, E_CLOSURE_MISMATCH)

    def test_dependency_stale(self) -> None:
        graph = DependencyGraph()
        graph.add_node(Node("a", "test", "actual", 0, "s"))
        with self.assertRaises(DiffRenderError) as context:
            graph.verify_nodes({"a": "expected"})
        self.assertEqual(context.exception.code, E_DEPENDENCY_STALE)


if __name__ == "__main__":
    unittest.main()

