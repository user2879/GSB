import copy
import unittest

from batch_receipt_ledger.plan import build_plan, plan_fingerprint
from batch_receipt_ledger.spec import parse_spec

from tests._support import make_spec


class PlanTest(unittest.TestCase):
    def test_deterministic_under_dict_and_dispatch_order(self):
        raw = make_spec(deliveries=10)
        spec = parse_spec(raw)
        plans_a = build_plan(spec)

        shuffled = copy.deepcopy(raw)
        shuffled["workers"] = list(reversed(shuffled["workers"]))
        shuffled["deliveries"] = list(reversed(shuffled["deliveries"]))
        shuffled["quota_events"] = list(reversed(shuffled["quota_events"]))
        plans_b = build_plan(parse_spec(shuffled))

        self.assertEqual(plan_fingerprint(plans_a), plan_fingerprint(plans_b))
        self.assertEqual([p.batch_id for p in plans_a],
                         [p.batch_id for p in plans_b])

    def test_batch_id_derived_from_explicit_identity(self):
        spec = parse_spec(make_spec(deliveries=7))
        plan = build_plan(spec)[0]
        payload = plan.identity_payload()
        self.assertEqual(payload["worker_id"], plan.worker_id)
        self.assertEqual(payload["arrival_seqs"], list(plan.arrival_seqs))
        self.assertTrue(plan.batch_id.startswith("b_"))

    def test_superseded_requests_excluded(self):
        raw = make_spec(deliveries=6)
        raw["deliveries"].append({
            "request_id": "req-new", "arrival_seq": 7, "value": 1,
            "replaces": "req-001"})
        plans = build_plan(parse_spec(raw))
        all_ids = [rid for plan in plans for rid in plan.request_ids]
        self.assertNotIn("req-001", all_ids)
        self.assertIn("req-new", all_ids)

    def test_batch_size_change_repartitions(self):
        raw = make_spec(deliveries=12, batch_size=3)
        plans_3 = build_plan(parse_spec(raw))
        raw["model"]["batch_size"] = 4
        plans_4 = build_plan(parse_spec(raw))
        self.assertEqual(len(plans_3), 4)
        self.assertEqual(len(plans_4), 3)
        self.assertNotEqual(plan_fingerprint(plans_3), plan_fingerprint(plans_4))


if __name__ == "__main__":
    unittest.main()
