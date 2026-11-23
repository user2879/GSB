from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from config_arbiter.arbitration import arbitrate_key
from config_arbiter.chain import build_chain
from config_arbiter.util import canonical_bytes
from tests.support import build_engine


class ArbitrationTests(unittest.TestCase):
    def test_binding_can_skip_higher_priority_runtime_source(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            engine, ledger, specs, _, _ = build_engine(Path(temporary))
            try:
                chain = build_chain(specs, "prod-cn", "http.timeout_ms")
                self.assertEqual([step["source_type"] for step in chain["steps"]], ["runtime", "override", "baseline"])
                self.assertEqual([step["binding_eligible"] for step in chain["steps"]], [False, True, False])
                self.assertEqual(chain["effective"]["source_type"], "override")
                self.assertEqual(chain["chain_length"], 3)

                staging = build_chain(specs, "staging-eu", "http.timeout_ms")
                self.assertEqual([step["source_type"] for step in staging["steps"]], ["override", "baseline"])
                self.assertEqual(staging["effective"]["source_type"], "baseline")
                self.assertNotEqual(chain["chain_signature"], staging["chain_signature"])
            finally:
                ledger.close()

    def test_three_strategies_and_error_influence(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            engine, ledger, specs, _, _ = build_engine(Path(temporary))
            try:
                ops = specs.views["ops-authority"]
                latest = arbitrate_key(specs, ops, "edge.latency_ms")
                self.assertTrue(latest["error_influenced"])
                self.assertEqual(latest["authoritative"]["environment_id"], "prod-cn")
                self.assertEqual(latest["authoritative"]["value_fixed"], 100000)
                self.assertTrue(latest["error_attribution"]["stable_tie_break_applied"])
                self.assertEqual(latest["error_model"]["scale_policy_version"], "scale-policy/1000.1")
                self.assertEqual(latest["arbitration_policy_version"], "arb-policy/2026-11-23.1")

                weighted = arbitrate_key(specs, ops, "retry.backoff_ms")
                accepted = sorted(item["environment_id"] for item in weighted["accepted"])
                self.assertEqual(accepted, ["prod-cn", "staging-eu"])
                self.assertTrue(weighted["error_influenced"])
                self.assertEqual(weighted["rejected"][0]["environment_id"], "dr-ap")
                self.assertEqual(
                    weighted["rejected"][0]["reason"],
                    "error_band_lost_by_stable_weighted_tie_break",
                )

                interval = arbitrate_key(specs, ops, "cache.ttl_s")
                self.assertEqual(interval["authoritative"]["value_fixed"], 62250)
                self.assertEqual(len(interval["accepted"]), 2)
                self.assertEqual(interval["rejected"][0]["environment_id"], "prod-cn")
                self.assertEqual(
                    interval["rejected"][0]["reason"],
                    "interval_disjoint_from_authoritative_cluster",
                )
                self.assertGreater(interval["key_aggregation_error_upper_fixed_units"], 0)
            finally:
                ledger.close()

    def test_environment_source_and_observation_order_does_not_change_bytes(self) -> None:
        with tempfile.TemporaryDirectory() as left_temp, tempfile.TemporaryDirectory() as right_temp:
            left_engine, left_ledger, _, _, _ = build_engine(Path(left_temp))
            right_engine, right_ledger, _, _, _ = build_engine(Path(right_temp), reverse_order=True)
            try:
                left_engine.run_all()
                right_engine.run_all()
                for view_id in left_engine.specs.views:
                    left = left_ledger.current_view(view_id)
                    right = right_ledger.current_view(view_id)
                    self.assertIsNotNone(left)
                    self.assertIsNotNone(right)
                    self.assertEqual(canonical_bytes(left["result"]), canonical_bytes(right["result"]))
            finally:
                left_ledger.close()
                right_ledger.close()


if __name__ == "__main__":
    unittest.main()
