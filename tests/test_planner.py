import tempfile
import unittest
from pathlib import Path

from tier_mover.errors import CapacityError
from tier_mover.heat import heat_value
from tier_mover.manifest import load_manifest
from tier_mover.planner import derive_cycle_plan, derive_initial_plan
from tier_mover.samplegen import build_sample


class PlannerTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.workdir = Path(self.tmp.name)
        build_sample(self.workdir, hot_capacity=16384, cold_capacity=65536,
                     hot_watermark=12288, cold_reclaim=32768, batch_size=4)
        self.manifest = load_manifest(self.workdir / "manifest.json")

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_heat_is_total_order_and_deterministic(self) -> None:
        values = [heat_value(o) for o in self.manifest.objects]
        self.assertEqual(values, sorted(values, reverse=True)[:0] + values)
        # Identical inputs derive identical initial plans.
        first = [a.to_dict() for a in derive_initial_plan(self.manifest)]
        second = [a.to_dict() for a in derive_initial_plan(self.manifest)]
        self.assertEqual(first, second)

    def test_initial_plan_respects_capacity(self) -> None:
        actions = derive_initial_plan(self.manifest)
        used = {"hot": 0, "cold": 0}
        cap = {t.role: t.capacity_bytes for t in self.manifest.tiers}
        for action in actions:
            tier = self.manifest.tier_by_id(action.target_tier).role
            used[tier] += action.size_bytes
            self.assertLessEqual(used[tier], cap[tier])

    def test_cycle_plan_demotes_when_hot_above_watermark(self) -> None:
        ownership = {}
        used = {"hot": 0, "cold": 0}
        for action in derive_initial_plan(self.manifest):
            role = self.manifest.tier_by_id(action.target_tier).role
            ownership[action.key] = action.target_tier
            used[role] += action.size_bytes
        self.assertGreater(used["hot"], self.manifest.rule.hot_watermark_bytes)
        plan = derive_cycle_plan(self.manifest, ownership)
        self.assertTrue(plan["actions"])
        # Projected hot usage must land at or below the watermark.
        hot_id = self.manifest.hot().tier_id
        self.assertLessEqual(plan["projected_usage"][hot_id],
                             self.manifest.rule.hot_watermark_bytes)
        # Every action stays within hard capacity.
        for tier_id, used_bytes in plan["projected_usage"].items():
            self.assertLessEqual(used_bytes,
                                 self.manifest.tier_by_id(tier_id).capacity_bytes)

    def test_capacity_override_raises_capacity_error(self) -> None:
        from dataclasses import replace
        hot = self.manifest.hot()
        tiny = replace(hot, capacity_bytes=1)
        manifest = replace(self.manifest, tiers=[
            tiny if t.tier_id == hot.tier_id else t for t in self.manifest.tiers
        ])
        with self.assertRaises(CapacityError) as ctx:
            derive_initial_plan(manifest)
        self.assertEqual(ctx.exception.code, "E_CAPACITY")


if __name__ == "__main__":
    unittest.main()
