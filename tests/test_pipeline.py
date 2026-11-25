from __future__ import annotations

import copy
import tempfile
import unittest
from pathlib import Path

from mgirecon.config import default_config, save_config
from mgirecon.engine import reconstruct
from mgirecon.synth import initialize_project

from tests.helpers import make_project


class PipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.temporary, cls.root, cls.config = make_project("mgirecon-pipeline-")
        cls.baseline = reconstruct(cls.root, cls.config, previous_state={})

    @classmethod
    def tearDownClass(cls) -> None:
        cls.temporary.cleanup()

    def test_features_matches_and_order(self) -> None:
        result = self.baseline
        self.assertTrue(all(record["feature_count"] > 0 for record in result["features"].values()))
        pair_ids = [pair["pair_id"] for pair in result["match_pairs"]]
        self.assertEqual(pair_ids, sorted(pair_ids))
        for pair in result["match_pairs"]:
            coords = [(match["left_coord"][0], match["left_coord"][1]) for match in pair["matches"]]
            self.assertEqual(coords, sorted(coords))
            self.assertGreaterEqual(pair["inlier_count"], 0)
        self.assertTrue(any(pair["tied"] for pair in result["match_pairs"]))

    def test_min_disparity_and_threshold_change_matching_and_points(self) -> None:
        changed = copy.deepcopy(self.config)
        changed["matching"]["min_inliers"] = 8
        changed["reconstruction"]["min_disparity"] = 30
        result = reconstruct(self.root, changed, previous_state={})
        baseline_selected = [pair["pair_id"] for pair in self.baseline["match_pairs"] if pair["selected"]]
        changed_selected = [pair["pair_id"] for pair in result["match_pairs"] if pair["selected"]]
        self.assertNotEqual(baseline_selected, changed_selected)
        baseline_points = [point["stable_key"] for point in self.baseline["points"] if point["residual_pass"]]
        changed_points = [point["stable_key"] for point in result["points"] if point["residual_pass"]]
        self.assertNotEqual(baseline_points, changed_points)

        threshold_config = copy.deepcopy(self.config)
        threshold_config["reconstruction"]["residual_threshold"] = 1
        threshold_result = reconstruct(self.root, threshold_config, previous_state={})
        baseline_inliers = sum(pair["inlier_count"] for pair in self.baseline["match_pairs"])
        threshold_inliers = sum(pair["inlier_count"] for pair in threshold_result["match_pairs"])
        self.assertLess(threshold_inliers, baseline_inliers)
        baseline_selected = [pair["pair_id"] for pair in self.baseline["match_pairs"] if pair["selected"]]
        threshold_selected = [pair["pair_id"] for pair in threshold_result["match_pairs"] if pair["selected"]]
        self.assertNotEqual(baseline_selected, threshold_selected)
        self.assertLess(len(threshold_result["points"]), len(self.baseline["points"]))

    def test_quantization_changes_coordinates_residuals_and_depth_pixels(self) -> None:
        changed = copy.deepcopy(self.config)
        changed["reconstruction"]["quantization_level"] = 2
        result = reconstruct(self.root, changed, previous_state={})
        first = {point["stable_key"]: point for point in self.baseline["points"]}
        second = {point["stable_key"]: point for point in result["points"]}
        common = sorted(set(first) & set(second))
        self.assertTrue(common)
        self.assertTrue(any(first[key]["xyz"] != second[key]["xyz"] for key in common))
        self.assertTrue(any(first[key]["residual"] != second[key]["residual"] for key in common))
        changed_pixels = 0
        for view_id in self.baseline["depth_maps"]:
            left = self.baseline["depth_maps"][view_id]
            right = result["depth_maps"][view_id]
            self.assertEqual(len(left["depth"]), left["width"] * left["height"])
            self.assertTrue(set(left["rounding"]).issubset({"exact", "up", "down"}))
            self.assertGreater(sum(left["boundary"]), 0)
            changed_pixels += sum(a != b for a, b in zip(left["depth"], right["depth"]))
        self.assertGreater(changed_pixels, 0)
        self.assertGreater(result["voxel_stats"]["subdivided_count"], 0)

    def test_threshold_drives_real_subdivision_and_adjacency(self) -> None:
        low = copy.deepcopy(self.config)
        low["reconstruction"]["residual_threshold"] = 8
        low_result = reconstruct(self.root, low, previous_state={})
        high = copy.deepcopy(self.config)
        high["reconstruction"]["residual_threshold"] = 100000
        high_result = reconstruct(self.root, high, previous_state={})
        self.assertGreater(low_result["voxel_stats"]["subdivided_count"], high_result["voxel_stats"]["subdivided_count"])
        self.assertNotEqual(low_result["adjacency"], high_result["adjacency"])
        self.assertGreater(len(low_result["voxels"]), len(high_result["voxels"]))

    def test_parent_child_occupancy_is_not_borrowed(self) -> None:
        result = self.baseline
        subdivided = [voxel for voxel in result["voxels"] if voxel["decision"] == "SUBDIVIDED"]
        self.assertTrue(subdivided)
        for voxel in subdivided:
            self.assertIsNone(voxel["direct_occupied"])
            self.assertTrue(voxel["children"])
        by_path = {voxel["path"]: voxel for voxel in result["voxels"]}
        for voxel in result["voxels"]:
            for child_path in voxel["children"]:
                self.assertEqual(by_path[child_path]["parent"], voxel["path"])
                self.assertIn(by_path[child_path]["decision"], {"SUBDIVIDED", "SOLID", "EMPTY"})
            if voxel["decision"] != "SUBDIVIDED":
                self.assertIn(voxel["decision"], {"SOLID", "EMPTY"})
                self.assertIsInstance(voxel["direct_occupied"], bool)
                self.assertEqual(voxel["children"], [])

    def test_strategy_changes_with_view_scale_and_depth(self) -> None:
        two_config = copy.deepcopy(self.config)
        two_config["views"] = two_config["views"][:2]
        with tempfile.TemporaryDirectory(prefix="mgirecon-two-views-") as temporary:
            root = Path(temporary)
            initialize_project(root, config=two_config)
            two_result = reconstruct(root, two_config, previous_state={})
        self.assertEqual(two_result["strategy"]["name"], "compact")
        self.assertEqual(two_result["strategy"]["block_size"], 8)
        self.assertEqual(self.baseline["strategy"]["name"], "wide")
        self.assertLessEqual(two_result["strategy"]["effective_max_depth"], self.baseline["strategy"]["effective_max_depth"])
        self.assertLess(two_result["voxel_stats"]["voxel_count"], self.baseline["voxel_stats"]["voxel_count"])


if __name__ == "__main__":
    unittest.main()
