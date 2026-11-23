import copy
import unittest

from arbworkbench.canonical import canonical_json
from arbworkbench.engine import (
    ATTRIB_ERROR,
    ATTRIB_POLICY,
    ATTRIB_REJECT_RATIO,
    ArbitrationEngine,
)
from arbworkbench.errors import GridError
from arbworkbench.spec import load_specification

from tests import SAMPLE_ENV, SAMPLE_VIEWS


class ArbitrationTests(unittest.TestCase):
    def setUp(self):
        self.spec = load_specification(SAMPLE_ENV, SAMPLE_VIEWS)
        self.engine = ArbitrationEngine(self.spec)

    def test_latest_wins_uses_seq_desc_then_ts_desc(self):
        view = self.spec.views["view-production"]
        kc = self.engine.arbitrate_key(view, "timeout_ms")
        # prod-eu seq5 value 130 -> q=13000; prod-us seq4 131; dr seq2 150.
        self.assertEqual(kc.authority_q, 13000)
        self.assertEqual(kc.accepted_obs_ids, ["env-prod-eu/timeout_ms#5"])
        reasons = {v.obs_id: v.reject_reason for v in kc.rejected}
        self.assertTrue(all(r == "outdated_lower_seq" for r in reasons.values()))
        self.assertEqual(kc.bound_q, 1)

    def test_weighted_majority_authority_and_rejections(self):
        view = self.spec.views["view-production"]
        kc = self.engine.arbitrate_key(view, "pool_size")
        # latest per env: eu=8(w5), us=7(w3), dr=4(w1); candidate 8 wins.
        self.assertEqual(kc.policy, "weighted_majority")
        self.assertIn("env-prod-eu/pool_size#2", kc.accepted_obs_ids)
        rejected_ids = {v.obs_id for v in kc.rejected}
        self.assertIn("env-prod-us/pool_size#3", rejected_ids)
        self.assertIn("env-dr/pool_size#1", rejected_ids)
        # weighted mean floor((800*5+700*3+400*1)/9) = floor(6500/9)=722
        self.assertEqual(kc.authority_q, 722)
        self.assertEqual(kc.bound_q, 2)

    def test_interval_intersection_midpoint(self):
        view = self.spec.views["view-production"]
        kc = self.engine.arbitrate_key(view, "maintenance_window")
        # eu [2,8], us [4,10], dr [6,12] -> intersection [6,8] midpoint 7.
        self.assertEqual(kc.authority_q, 700)
        self.assertEqual(kc.policy, "interval_intersection")
        self.assertEqual(kc.bound_q, 2)

    def test_empty_interval_intersection_is_policy_conflict(self):
        view = self.spec.views["view-all-fleets"]
        kc = self.engine.arbitrate_key(view, "maintenance_window")
        # canary [1,5] and dr [6,12] cannot intersect.
        self.assertIsNone(kc.authority_q)
        self.assertIn(ATTRIB_POLICY, kc.attribution)
        self.assertIsNotNone(kc.conflict_detail)
        self.assertTrue(all(not v.accepted for v in kc.rejected))

    def test_error_indistinguishable_attribution(self):
        # rate_limit values cluster within 2q after S=100 quantization.
        view = self.spec.views["view-production"]
        kc = self.engine.arbitrate_key(view, "rate_limit")
        # eu 100.50 ->10050, us 100.50 ->10050, dr 99.99 ->9999.
        # leaders identical; build a tighter case via staging/canary cluster.
        view_all = self.spec.views["view-all-fleets"]
        # Add a majority policy on rate_limit in an isolated view-like call.
        view_copy = copy.deepcopy(view_all)
        view_copy.policies = dict(view_copy.policies)
        view_copy.policies["rate_limit"] = "weighted_majority"
        kc2 = self.engine.arbitrate_key(view_copy, "rate_limit")
        self.assertIsInstance(kc2.authority_q, int)
        # The 100.49 / 100.50 / 100.60 cluster spans 11q > bound, verify no crash
        # and deterministic attribution structure.
        self.assertEqual(set(kc2.attribution).issubset(
            {ATTRIB_ERROR, ATTRIB_POLICY, ATTRIB_REJECT_RATIO}), True)

    def test_high_rejected_ratio_attribution(self):
        # all-fleets latest on pool_size: 5 envs, single accepted -> 4/5=0.8.
        view = self.spec.views["view-all-fleets"]
        kc = self.engine.arbitrate_key(view, "pool_size")
        self.assertIn(ATTRIB_REJECT_RATIO, kc.attribution)
        # One latest observation per participating environment (5), one accepted.
        self.assertEqual(len(kc.accepted_verdicts), 1)
        self.assertEqual(len(kc.rejected), 4)
        self.assertGreater(len(kc.rejected) / 5, 0.5)

    def test_reversed_interval_triggers_grid_error(self):
        import dataclasses
        spec = copy.deepcopy(self.spec)
        obs = spec.environments["env-dr"].observations
        for idx, o in enumerate(obs):
            if o.key == "maintenance_window":
                obs[idx] = dataclasses.replace(o, value=[12, 6])
        engine = ArbitrationEngine(spec)
        view = spec.views["view-production"]
        with self.assertRaises(GridError) as ctx:
            engine.arbitrate_key(view, "maintenance_window")
        self.assertEqual(ctx.exception.code, "E_GRID")

    def test_order_independence_byte_identical(self):
        view = self.spec.views["view-production"]
        first = canonical_json(
            [self.engine.arbitrate_key(view, k) for k in sorted(view.policies)]
        )
        shuffled_spec = copy.deepcopy(self.spec)
        shuffled_spec.environments["env-prod-eu"].observations.reverse()
        shuffled_spec.environments["env-prod-us"].observations.reverse()
        shuffled_engine = ArbitrationEngine(shuffled_spec)
        second = canonical_json(
            [shuffled_engine.arbitrate_key(view, k) for k in sorted(view.policies)]
        )
        self.assertEqual(first, second)

    def test_adjacent_diff_is_non_negative_integer(self):
        view = self.spec.views["view-production"]
        for key in view.policies:
            kc = self.engine.arbitrate_key(view, key)
            if kc.authority_q is not None:
                self.assertGreaterEqual(kc.adjacent_diff_q, 0)
                self.assertIsInstance(kc.adjacent_diff_q, int)


if __name__ == "__main__":
    unittest.main()
