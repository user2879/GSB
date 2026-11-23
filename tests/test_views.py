from __future__ import annotations

import unittest

from cacheproof.views import build_view_snapshot, compute_view_diff, view_is_stale


class ViewTests(unittest.TestCase):
    def test_view_diff_order_is_stable(self) -> None:
        old = build_view_snapshot(
            view_id="v",
            view_version=1,
            segment_set_version=1,
            segments=[
                {
                    "segment_id": "b",
                    "segment_version": 1,
                    "entries": [{"key": "z", "digest": "sha256:" + "1" * 64}],
                },
                {
                    "segment_id": "a",
                    "segment_version": 1,
                    "entries": [{"key": "x", "digest": "sha256:" + "2" * 64}],
                },
            ],
        )
        current = build_view_snapshot(
            view_id="v",
            view_version=2,
            segment_set_version=2,
            segments=[
                {
                    "segment_id": "a",
                    "segment_version": 2,
                    "entries": [
                        {"key": "y", "digest": "sha256:" + "3" * 64},
                        {"key": "x", "digest": "sha256:" + "4" * 64},
                    ],
                },
                {
                    "segment_id": "b",
                    "segment_version": 2,
                    "entries": [{"key": "z", "digest": "sha256:" + "1" * 64}],
                },
            ],
        )
        self.assertTrue(view_is_stale(old, current))
        diff = compute_view_diff(old, current)
        self.assertEqual(
            [segment["segment_id"] for segment in diff["segments"]], ["a", "b"]
        )
        self.assertEqual(
            [item["key"] for item in diff["segments"][0]["added"]], ["y"]
        )
        self.assertEqual(
            [item["key"] for item in diff["segments"][0]["digest_changed"]], ["x"]
        )


if __name__ == "__main__":
    unittest.main()
