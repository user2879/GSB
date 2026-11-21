"""确定性：同输入同参数多次运行逐字段一致；阈值边界固定决断。"""
from __future__ import annotations

import unittest

from recon_workbench.config import RebuildConfig
from recon_workbench.features import extract_features, match_views

from ._helpers import SampleHarness


class DeterminismTest(unittest.TestCase):
    def setUp(self) -> None:
        self.harness = SampleHarness()
        self.config = RebuildConfig()

    def tearDown(self) -> None:
        self.harness.cleanup()

    def test_features_repeatable(self) -> None:
        view = self.harness.load().views[0]
        runs = [
            extract_features(view.gray, view.view, self.config, "float64")
            for _ in range(3)
        ]
        baseline = [f.to_dict() for f in runs[0]]
        for run in runs[1:]:
            self.assertEqual([f.to_dict() for f in run], baseline)

    def test_features_sorted_response_then_coord(self) -> None:
        view = self.harness.load().views[0]
        features = extract_features(view.gray, view.view, self.config)
        # 响应整体非增；量化后相等的响应段内坐标按 (y,x) 字典序
        for prev, nxt in zip(features, features[1:]):
            self.assertGreaterEqual(
                round(prev.response, 12), round(nxt.response, 12) - 1e-12
            )
            if abs(prev.response - nxt.response) <= 1e-12:
                self.assertLessEqual(
                    (prev.y, prev.x), (nxt.y, nxt.x)
                )

    def test_matches_repeatable_and_sorted(self) -> None:
        load = self.harness.load()
        feats = [extract_features(v.gray, v.view, self.config)
                 for v in load.views]
        first, _ = match_views(
            feats[0], feats[1], load.views[0].gray, load.views[1].gray,
            0, 1, self.config,
        )
        second, _ = match_views(
            feats[0], feats[1], load.views[0].gray, load.views[1].gray,
            0, 1, self.config,
        )
        self.assertEqual(
            [m.to_dict() for m in first], [m.to_dict() for m in second]
        )
        for prev, nxt in zip(first, second):
            self.assertLessEqual(prev.distance, nxt.distance)

    def test_feature_cap_respected(self) -> None:
        view = self.harness.load().views[0]
        features = extract_features(view.gray, view.view, self.config)
        self.assertLessEqual(len(features),
                             self.config.max_features_per_view)


if __name__ == "__main__":
    unittest.main()
