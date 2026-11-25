import copy
import json
import os
import shutil
import tempfile
import unittest

from coljoin.canonical import canonical_bytes
from coljoin.cost import CostEstimator
from coljoin.invalidation import (
    PlanVersionState,
    compute_change_set,
    cost_input_digest,
    data_digests,
    recompute_invalidated_only,
    trace_invalidation,
)
from coljoin.loader import load_query, load_session, load_weights
from coljoin.planner import enumerate_plans
from coljoin.storage import load_generated_specs
from scripts.generate_sample import main as generate_main
import sys


class InvalidationTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        old_argv = sys.argv
        sys.argv = ["generate_sample.py", "--workspace", self.tmp]
        generate_main()
        sys.argv = old_argv
        self.data_dir = os.path.join(self.tmp, "data")
        self.cfg = os.path.join(self.tmp, "configs")

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _context(self):
        groups = load_generated_specs(self.data_dir)
        query = load_query(os.path.join(self.cfg, "query.json"), groups)
        weights = load_weights(os.path.join(self.cfg, "weights.json"))
        session = load_session(os.path.join(self.cfg, "session.json"))
        return groups, query, weights, session

    def test_weight_change_monotonic_version_and_full_reasons(self):
        groups, query, weights, session = self._context()
        first_digest = cost_input_digest(groups, weights, session)
        state = PlanVersionState(
            version=1,
            cost_input_digest=first_digest,
            plan_digests={p["plan_id"]: "x" for p in enumerate_plans(query)},
            shard_digests={f"{p['plan_id']}:0000": "y" for p in enumerate_plans(query)},
            receipt_digests={f"{p['plan_id']}:0000": "z" for p in enumerate_plans(query)},
        )
        changed_weights = copy.deepcopy(weights)
        object.__setattr__(changed_weights, "scan", weights.scan + 7)
        new_digest = cost_input_digest(groups, changed_weights, session)
        trace = trace_invalidation(
            query,
            groups,
            changed_weights,
            session,
            changed_block_groups=[],
            old_state=state,
            new_cost_digest=new_digest,
        )
        self.assertEqual(trace["new_version"], 2)
        self.assertTrue(trace["cost_input_changed"])
        self.assertTrue(trace["invalidated_plans"])
        self.assertTrue(trace["invalidated_shards"])
        self.assertTrue(trace["invalidated_receipts"])
        self.assertTrue(all(item["basis"] for item in trace["reasons"]))

    def test_data_change_targets_only_related_plans(self):
        groups, query, weights, session = self._context()
        digest_before = data_digests(self.data_dir, groups)
        # 篡改 payments 列文件后重新计算数据指纹。
        col_path = os.path.join(self.data_dir, "payments", "pay_amount.col.json")
        with open(col_path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
        if "data" in payload:
            payload["data"] = [value + 1 for value in payload["data"]]
        elif "runs" in payload:
            payload["runs"] = [[value + 1, length] for value, length in payload["runs"]]
        else:
            payload["indices"] = [value + 1 for value in payload["indices"]]
        with open(col_path, "w", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, sort_keys=True))
        digest_after = data_digests(self.data_dir, groups)
        changed = compute_change_set(digest_before, digest_after)
        self.assertEqual(changed, ["payments"])
        trace = trace_invalidation(
            query,
            groups,
            weights,
            session,
            changed_block_groups=changed,
            old_state=PlanVersionState(
                version=1,
                cost_input_digest=cost_input_digest(groups, weights, session),
            ),
            new_cost_digest=cost_input_digest(groups, weights, session),
        )
        plans = enumerate_plans(query)
        for plan_id in trace["invalidated_plans"]:
            plan = next(item for item in plans if item["plan_id"] == plan_id)
            self.assertIn("payments", plan["body"]["order"])

    def test_unaffected_estimate_byte_stable(self):
        groups, query, weights, session = self._context()
        plans = enumerate_plans(query)
        estimator = CostEstimator(groups, weights, session)
        before = [estimator.evaluate(plan) for plan in plans]
        # 重新加载同一磁盘状态，未发生任何变更 => 估算逐字节一致。
        groups2 = load_generated_specs(self.data_dir)
        estimator2 = CostEstimator(groups2, weights, session)
        after = [estimator2.evaluate(plan) for plan in plans]
        self.assertEqual(canonical_bytes(before), canonical_bytes(after))

    def test_recompute_only_invalidated_subset(self):
        groups, query, weights, session = self._context()
        plans = enumerate_plans(query)
        estimator = CostEstimator(groups, weights, session)
        previous = {plan["plan_id"]: estimator.evaluate(plan) for plan in plans}
        invalid = sorted(previous)[:3]
        merged = recompute_invalidated_only(
            query, groups, weights, session, previous, invalid
        )
        stable = [plan_id for plan_id in previous if plan_id not in invalid]
        self.assertTrue(all(merged[plan_id] == previous[plan_id] for plan_id in stable))
        self.assertEqual(set(merged), set(previous))


if __name__ == "__main__":
    unittest.main()
