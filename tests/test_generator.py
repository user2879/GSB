from __future__ import annotations

import unittest
from collections import Counter

import numpy as np

from pickworkbench.canonical import canonical_bytes, content_hash
from pickworkbench.generator import (
    GeneratorConfig,
    generate_sample_with_config,
    summarize_scene,
)


class GeneratorTests(unittest.TestCase):
    def setUp(self) -> None:
        self.scene, self.queries = generate_sample_with_config()

    def test_two_generations_are_byte_identical(self) -> None:
        scene_two, queries_two = generate_sample_with_config()
        self.assertEqual(canonical_bytes(self.scene), canonical_bytes(scene_two))
        self.assertEqual(canonical_bytes(self.queries), canonical_bytes(queries_two))
        self.assertEqual(content_hash(self.scene), content_hash(scene_two))
        self.assertEqual(content_hash(self.queries), content_hash(queries_two))

    def test_expected_object_and_query_counts(self) -> None:
        summary = summarize_scene(self.scene, self.queries)
        self.assertEqual(summary["object_count"], 12)
        self.assertEqual(summary["query_count"], 22)
        self.assertEqual(summary["query_counts"], {"ray": 10, "click": 8, "drag": 4})
        self.assertEqual(
            summary["object_kinds"],
            {"sphere": 4, "aabb": 4, "convex_polyhedron": 4},
        )

    def test_object_ids_are_stable_and_ascending(self) -> None:
        ids = [obj["id"] for obj in self.scene["objects"]]
        self.assertEqual(ids, [f"obj-{index:03d}" for index in range(12)])
        self.assertEqual([q["query_index"] for q in self.queries["queries"]], list(range(22)))

    def test_bounds_cover_every_object_aabb(self) -> None:
        bounds_min = np.asarray(self.scene["bounds"]["min"])
        bounds_max = np.asarray(self.scene["bounds"]["max"])
        for obj in self.scene["objects"]:
            self.assertTrue(np.all(bounds_min <= np.asarray(obj["bounding_volume"]["aabb"]["min"])))
            self.assertTrue(np.all(bounds_max >= np.asarray(obj["bounding_volume"]["aabb"]["max"])))

    def test_all_three_query_kinds_exist(self) -> None:
        counts = Counter(query["kind"] for query in self.queries["queries"])
        self.assertEqual(counts["ray"], 10)
        self.assertEqual(counts["click"], 8)
        self.assertEqual(counts["drag"], 4)
        for query in self.queries["queries"]:
            if query["kind"] == "drag":
                self.assertEqual(query["snap_axes"], ["x", "z"])
                self.assertGreater(float(query["snap_spacing"]), 0.0)


if __name__ == "__main__":
    unittest.main()
