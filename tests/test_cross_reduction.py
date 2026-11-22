from __future__ import annotations

import unittest

from .helpers import TestWorkspace


class CrossReductionTests(unittest.TestCase):
    def test_cross_encoding_and_three_reduction_paths_are_deterministic(self) -> None:
        first_workspace = TestWorkspace()
        second_workspace = TestWorkspace()
        try:
            first_workspace.prepare()
            second_workspace.prepare()
            first = first_workspace.run()
            second = second_workspace.run()
            self.assertEqual(first["status"], "ok")
            self.assertEqual(first["semantic_fingerprint"], second["semantic_fingerprint"])
            self.assertEqual(
                first["cross_encoding"]["features"], second["cross_encoding"]["features"]
            )
            self.assertEqual(
                first["cross_encoding"]["degradations"], second["cross_encoding"]["degradations"]
            )
            self.assertEqual(len(first["cross_encoding"]["features"]), 6)
            self.assertEqual(len(first["cross_encoding"]["degradations"]), 6)
            self.assertEqual(first["reduction"]["path_order"], ["pca", "random_projection", "variance"])
            for path in first["reduction"]["paths"]:
                self.assertIn("retained_dimensions", path)
                self.assertIn("summary", path)
                self.assertIn("included_cross_feature_keys", path)
                self.assertEqual(
                    path["tiers"]["single"]["projected"] == path["tiers"]["double"]["projected"],
                    True,
                )
            pca = next(item for item in first["reduction"]["paths"] if item["path"] == "pca")
            for component in pca["summary"]["components"]:
                pivot = max(range(len(component)), key=lambda index: abs(component[index]))
                self.assertGreaterEqual(component[pivot], 0.0)
            projection = next(
                item for item in first["reduction"]["paths"] if item["path"] == "random_projection"
            )
            self.assertEqual(projection["summary"]["matrix_shape"], [6, 5])
            self.assertIsInstance(projection["summary"]["seed"], int)
        finally:
            first_workspace.close()
            second_workspace.close()


if __name__ == "__main__":
    unittest.main()
