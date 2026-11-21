"""Deterministic arbitration tests (pure in-memory, no sockets)."""

import unittest

from helpers import free_ports, make_spec_dict

from quota_workbench.arbiter import run_arbitration
from quota_workbench.probe import Transition
from quota_workbench.spec import Request, load_spec


def make_transition(instance_id, to_state, rnd, interval_ms=60, consec=1):
    return Transition(
        instance_id=instance_id, from_state="unknown", to_state=to_state,
        probe_round=rnd, consec_failures=consec, mono_ms=0,
        virt_ms=rnd * interval_ms, reason="test",
    )


def make_requests(rows):
    return [Request(r[0], r[1], r[2], r[3]) for r in rows]


class ArbiterTest(unittest.TestCase):
    def setUp(self):
        self.spec = load_spec("<test>", _raw_override=make_spec_dict(free_ports(2)))

    def test_determinism_byte_identical(self):
        transitions = [make_transition("svc-a", "healthy", 0)]
        requests = make_requests([
            ("r0", "alpha", 3, 0), ("r1", "alpha", 2, 10),
            ("r2", "alpha", 4, 20), ("r3", "alpha", 1, 30),
        ])
        first = run_arbitration(self.spec, requests, transitions)
        second = run_arbitration(self.spec, requests, transitions)
        for rid in first.outcomes:
            a, b = first.outcomes[rid], second.outcomes[rid]
            self.assertEqual(a.outcome, b.outcome)
            self.assertEqual(a.basis_blob(), b.basis_blob())
            self.assertEqual(
                (a.wait_start_ms, a.wait_end_ms, a.grant_ms, a.done_ms),
                (b.wait_start_ms, b.wait_end_ms, b.grant_ms, b.done_ms),
            )

    def test_quota_exhaustion_rejects(self):
        transitions = [make_transition("svc-a", "healthy", 0)]
        requests = make_requests([("r%d" % i, "alpha", 6, i * 200) for i in range(8)])
        result = run_arbitration(self.spec, requests, transitions)
        rejected = [o for o in result.outcomes.values() if o.outcome == "rejected"]
        self.assertTrue(rejected)
        self.assertIn("R_QUOTA_EXHAUSTED", rejected[-1].basis_blob())
        self.assertEqual(result.tenant_consumed["alpha"], 30)

    def test_tenant_slots_full_queues_then_grants(self):
        transitions = [make_transition("svc-a", "healthy", 0)]
        requests = make_requests([("r%d" % i, "alpha", 1, 0) for i in range(4)])
        result = run_arbitration(self.spec, requests, transitions)
        queued = [o for o in result.outcomes.values() if o.wait_start_ms >= 0]
        self.assertTrue(queued)
        for oc in queued:
            self.assertEqual(oc.outcome, "granted")
            self.assertGreaterEqual(oc.wait_end_ms, oc.wait_start_ms)
            self.assertIn("R_TENANT_SLOTS_FULL", oc.basis_blob())

    def test_burst_flag_recorded(self):
        transitions = [make_transition("svc-a", "healthy", 0)]
        requests = make_requests([("r%d" % i, "alpha", 1, 0) for i in range(2)])
        result = run_arbitration(self.spec, requests, transitions)
        bursts = [o for o in result.outcomes.values() if "burst_used=1" in o.basis_blob()]
        self.assertEqual(len(bursts), 1)

    def test_ejected_instance_rejects(self):
        transitions = [make_transition("svc-b", "ejected", 0, consec=3)]
        requests = make_requests([("r0", "beta", 1, 100)])
        result = run_arbitration(self.spec, requests, transitions)
        oc = result.outcomes["r0"]
        self.assertEqual(oc.outcome, "rejected")
        self.assertIn("R_INSTANCE_EJECTED", oc.basis_blob())

    def test_unhealthy_instance_queues_then_end_of_stream(self):
        transitions = [make_transition("svc-b", "unhealthy", 0, consec=2)]
        requests = make_requests([("r0", "beta", 1, 100)])
        result = run_arbitration(self.spec, requests, transitions)
        oc = result.outcomes["r0"]
        self.assertEqual(oc.outcome, "rejected")
        self.assertIn("R_INSTANCE_NOT_HEALTHY", oc.basis_blob())
        self.assertIn("R_END_OF_STREAM", oc.basis_blob())

    def test_unknown_instance_waits_until_healthy(self):
        transitions = [make_transition("svc-a", "healthy", 2)]
        requests = make_requests([("r0", "alpha", 1, 0)])
        result = run_arbitration(self.spec, requests, transitions)
        oc = result.outcomes["r0"]
        self.assertEqual(oc.outcome, "granted")
        self.assertEqual(oc.wait_start_ms, 0)
        self.assertEqual(oc.grant_ms, 120)

    def test_inflight_never_exceeds_caps(self):
        transitions = [make_transition("svc-a", "healthy", 0)]
        requests = make_requests([("r%d" % i, "alpha", 1, i * 5) for i in range(20)])
        result = run_arbitration(self.spec, requests, transitions)
        granted = [o for o in result.outcomes.values() if o.outcome == "granted"]
        for oc in granted:
            overlapping = sum(
                1 for other in granted
                if other.grant_ms <= oc.grant_ms < other.done_ms
            )
            self.assertLessEqual(overlapping, 2)  # max_concurrency 1 + burst 1


if __name__ == "__main__":
    unittest.main()
