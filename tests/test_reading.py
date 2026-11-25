from __future__ import annotations

import unittest

from layout_diag.models import ColumnTruth, ImageTruth, TruthKeypoint
from layout_diag.reading import index_all_keypoints, run_reading_stage


class ReadingTests(unittest.TestCase):
    def truth(self) -> ImageTruth:
        columns = (
            ColumnTruth("c0", 500, 600, 0, 100, 3, 0, 0),
            ColumnTruth("c1", 400, 500, 0, 100, 2, 3, 1),
        )
        points = (
            TruthKeypoint("k0", "c0", 550, 10, 0),
            TruthKeypoint("k1", "c0", 550, 20, 1),
            TruthKeypoint("k2", "c0", 550, 30, 2),
            TruthKeypoint("k3", "c1", 450, 10, 3),
            TruthKeypoint("k4", "c1", 450, 20, 4),
        )
        return ImageTruth("sample.png", columns, points, (640, 960))

    def detected(self, *items):
        return [
            {
                "keypoint_id": keypoint_id,
                "image": "sample.png",
                "column": column,
                "column_order": order,
                "x_q": x,
                "y_q": y,
            }
            for keypoint_id, column, order, x, y in items
        ]

    def test_missing_and_cross_column_conflicts_are_retained_and_ordered(self) -> None:
        points = self.detected(
            ("kp-a", "c0", 0, 550, 10),
            ("kp-b", "ghost", 9, 300, 5),
            ("kp-c", "c1", 1, 450, 10),
        )
        indexed = index_all_keypoints(points)
        result = run_reading_stage("sample.png", self.truth(), indexed)
        types = {conflict["type"] for conflict in result["conflicts"]}
        self.assertIn("missing_number", types)
        self.assertIn("cross_column", types)
        indices = [conflict["keypoint_index"] for conflict in result["conflicts"]]
        self.assertEqual(indices, sorted(indices))
        self.assertTrue(all(conflict["resolution_priority"] for conflict in result["conflicts"]))

    def test_wrong_order_conflict_for_inconsistent_column_reading_anchors(self) -> None:
        columns = (
            ColumnTruth("c0", 500, 600, 0, 100, 2, 2, 0),
            ColumnTruth("c1", 400, 500, 0, 100, 2, 0, 1),
        )
        points_truth = (
            TruthKeypoint("k0", "c0", 550, 10, 2),
            TruthKeypoint("k1", "c0", 550, 20, 3),
            TruthKeypoint("k2", "c1", 450, 10, 0),
            TruthKeypoint("k3", "c1", 450, 20, 1),
        )
        truth = ImageTruth("sample.png", columns, points_truth, (640, 960))
        detected = self.detected(
            ("kp-a", "c0", 0, 550, 10),
            ("kp-b", "c0", 0, 550, 20),
            ("kp-c", "c1", 1, 450, 10),
            ("kp-d", "c1", 1, 450, 20),
        )
        result = run_reading_stage("sample.png", truth, index_all_keypoints(detected))
        self.assertIn("wrong_order", {conflict["type"] for conflict in result["conflicts"]})


if __name__ == "__main__":
    unittest.main()
