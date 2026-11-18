"""缓存键派生与精确失效范围单元测试（业务要求 5）。"""

import unittest

from xmodal.cache import CachePlanner
from xmodal.tests._helpers import DEFAULT_CONFIG, SPEC, Workspace


class CachePlannerTest(unittest.TestCase):
    def setUp(self):
        self.ws = Workspace(count=4)
        self.samples = ["sample_000", "sample_001",
                        "sample_002", "sample_003"]

    def tearDown(self):
        self.ws.cleanup()

    def _planner(self, version=2):
        return CachePlanner(DEFAULT_CONFIG, SPEC, version,
                            self.ws.input_dir, DEFAULT_CONFIG["image_size"])

    def test_blank_run_recomputes_everything(self):
        planner = self._planner()
        plan, keys = planner.plan(self.samples, None)
        self.assertEqual(len(plan["extract"]["recompute"]), 4)
        self.assertEqual(plan["extract"]["reuse"], [])
        self.assertEqual(plan["retrieve"]["recompute"], ["__index__"])
        self.assertIn("extract", keys)

    def test_unchanged_inputs_full_reuse(self):
        planner = self._planner()
        _, keys = planner.plan(self.samples, None)
        previous = {"keys": keys, "valid_samples": self.samples}
        plan, _ = planner.plan(self.samples, previous)
        self.assertEqual(plan["extract"]["reuse"], self.samples)
        self.assertEqual(plan["retrieve"]["reuse"], ["__index__"])

    def test_weight_change_only_invalidates_align(self):
        planner = self._planner()
        _, keys = planner.plan(self.samples, None)
        previous = {"keys": keys, "valid_samples": self.samples}
        cfg2 = dict(DEFAULT_CONFIG)
        cfg2["align_weights"] = dict(DEFAULT_CONFIG["align_weights"])
        cfg2["align_weights"]["text"] = 0.3
        planner2 = CachePlanner(cfg2, SPEC, 2, self.ws.input_dir,
                                DEFAULT_CONFIG["image_size"])
        plan, _ = planner2.plan(self.samples, previous)
        self.assertEqual(plan["extract"]["reuse"], self.samples)
        self.assertEqual(len(plan["align"]["recompute"]), 4)

    def test_format_switch_only_invalidates_convert(self):
        planner = self._planner(version=2)
        _, keys = planner.plan(self.samples, None)
        previous = {"keys": keys, "valid_samples": self.samples}
        planner_v1 = self._planner(version=1)
        plan, _ = planner_v1.plan(self.samples, previous)
        self.assertEqual(plan["extract"]["reuse"], self.samples)
        self.assertEqual(len(plan["convert"]["recompute"]), 4)

    def test_keys_are_stable_across_instances(self):
        p1 = self._planner()
        p2 = self._planner()
        _, k1 = p1.plan(self.samples, None)
        _, k2 = p2.plan(self.samples, None)
        self.assertEqual(k1, k2)


if __name__ == "__main__":
    unittest.main()
