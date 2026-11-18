import json
import unittest

from coldstart_diag.engine import Engine, ceil_div, stable_allocate, synthesize_error_budget
from coldstart_diag.samples import base_manifest


class EngineTest(unittest.TestCase):
    def setUp(self):
        self.manifest = base_manifest(1, 1, 10000, 2500)
        self.fid = "fn-order-processor"

    def test_ceil_div(self):
        self.assertEqual(ceil_div(10, 3), 4)
        self.assertEqual(ceil_div(9, 3), 3)

    def test_stable_allocation_deterministic(self):
        a = stable_allocate(10, 3)
        b = stable_allocate(10, 3)
        self.assertEqual(a, [4, 3, 3])
        self.assertEqual(a, b)
        self.assertEqual(sum(a), 10)

    def test_deterministic_re_adjudication(self):
        m = json.loads(json.dumps(self.manifest))
        r1 = Engine(m, self.fid, 1).run_all(m["call_sequence"])
        r2 = Engine(json.loads(json.dumps(m)), self.fid, 1).run_all(
            json.loads(json.dumps(m["call_sequence"]))
        )
        for a, b in zip(r1, r2):
            self.assertEqual(a.instance_set, b.instance_set)
            self.assertEqual(a.cold_starts, b.cold_starts)
            self.assertEqual(a.to_dict(), b.to_dict())

    def test_hot_cold_split_and_cold_starts(self):
        results = Engine(self.manifest, self.fid, 1).run_all(self.manifest["call_sequence"])
        for r in results:
            self.assertEqual(r.hot_calls + r.cold_calls, r.calls)
            self.assertGreaterEqual(len(r.warm_set) + len(r.cold_set), 0)
            self.assertEqual(len(r.instance_set), r.available_instances)
            self.assertGreaterEqual(r.cold_starts, 0)

    def test_clamp_to_min_max_instances(self):
        m = json.loads(json.dumps(self.manifest))
        m["functions"][0]["min_instances"] = 2
        m["functions"][0]["max_instances"] = 5
        results = Engine(m, self.fid, 1).run_all(m["call_sequence"])
        for r in results:
            self.assertLessEqual(r.available_instances, 5)
            self.assertLessEqual(r.required_instances, 5)
            self.assertGreaterEqual(r.required_instances, 2)

    def test_global_cap_conflict_records_reason(self):
        m = json.loads(json.dumps(self.manifest))
        m["policy"]["global_concurrency_cap"] = 20  # 极低全局并发
        m["policy"]["reserved_instances"] = 8       # 预留超过容量
        results = Engine(m, self.fid, 1).run_all(m["call_sequence"])
        reasons = set()
        for r in results:
            reasons.update(r.arbitration_reasons)
        self.assertIn("GLOBAL_CONCURRENCY_CAP_BINDING", reasons)
        self.assertIn("RESERVED_VIOLATES_GLOBAL_CAP", reasons)

    def test_error_budget_contributions_sum(self):
        engine = Engine(self.manifest, self.fid, 1)
        r = engine.run_minute(self.manifest["call_sequence"][3])
        block = r.error_block
        contrib = block["contributions_ulp"]
        self.assertEqual(
            block["end_to_end_bound_ulp"],
            contrib["memory_mib"] + contrib["concurrency"]
            + contrib["warmup_permyriad"] + contrib["intrinsic_quantization"],
        )
        shares = block["contribution_shares_q"]
        self.assertLessEqual(sum(shares.values()), self.manifest["scale"])

    def test_warm_pool_persists_across_minutes(self):
        results = Engine(self.manifest, self.fid, 1).run_all(self.manifest["call_sequence"])
        # 第二分钟起 warm 不应为 0（预热池跨分钟保留）
        self.assertGreaterEqual(results[1].warm_instances, 1)

    def test_queue_and_wait(self):
        m = json.loads(json.dumps(self.manifest))
        m["policy"]["global_concurrency_cap"] = 10
        m["functions"][0]["max_instances"] = 2
        results = Engine(m, self.fid, 1).run_all(m["call_sequence"])
        has_queue = any(r.queued_calls > 0 for r in results)
        self.assertTrue(has_queue)
        for r in results:
            if r.queued_calls:
                self.assertEqual(r.total_wait_ms, r.queued_calls * m["functions"][0]["init_ms"])


if __name__ == "__main__":
    unittest.main()
