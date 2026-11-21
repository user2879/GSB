"""Dependency-graph propagation tests."""

import unittest

from helpers import free_ports, make_spec_dict

from quota_workbench.arbiter import run_arbitration
from quota_workbench.errors import SpecError
from quota_workbench.probe import Transition
from quota_workbench.propagation import (
    DependencyGraph,
    affected_for_transition,
    affected_set,
    build_graph,
    verify_unaffected_unchanged,
)
from quota_workbench.spec import Request, load_spec


def transition(instance_id, to_state, rnd, consec=1):
    return Transition(instance_id, "unknown", to_state, rnd, consec, 0, rnd * 60, "test")


class PropagationTest(unittest.TestCase):
    def setUp(self):
        self.spec = load_spec("<test>", _raw_override=make_spec_dict(free_ports(2)))
        self.requests = [
            Request("r0", "alpha", 1, 0),
            Request("r1", "beta", 1, 10),
            Request("r2", "alpha", 1, 20),
            Request("r3", "beta", 1, 30),
        ]

    def test_graph_integrity_check(self):
        graph = DependencyGraph()
        graph.add("a", ["missing"])
        graph.deps["a"].add("ghost")
        with self.assertRaises(SpecError):
            graph.validate()

    def test_affected_set_for_health_transition(self):
        affected, steps = affected_for_transition(self.spec, self.requests, "svc-b")
        self.assertIn("tenant:beta", affected)
        self.assertIn("request:r1", affected)
        self.assertIn("request:r3", affected)
        self.assertIn("stats", affected)
        self.assertNotIn("tenant:alpha", affected)
        self.assertNotIn("request:r0", affected)
        self.assertEqual(affected, sorted(affected))
        self.assertTrue(all(isinstance(s, tuple) and len(s) == 2 for s in steps))

    def test_rule_version_change_affects_all_tenants(self):
        graph = build_graph(self.spec, self.requests)
        affected, _ = affected_set(graph, ["rule"])
        self.assertIn("tenant:alpha", affected)
        self.assertIn("tenant:beta", affected)
        for req in self.requests:
            self.assertIn(f"request:{req.request_id}", affected)

    def test_unaffected_requests_byte_identical(self):
        transitions_before = [
            transition("svc-a", "healthy", 0),
            transition("svc-b", "unhealthy", 1, consec=2),
        ]
        transitions_after = [
            transition("svc-a", "healthy", 0),
            transition("svc-b", "healthy", 1),
        ]
        before = run_arbitration(self.spec, self.requests, transitions_before)
        after = run_arbitration(self.spec, self.requests, transitions_after)
        affected, _ = affected_for_transition(self.spec, self.requests, "svc-b")
        violations = verify_unaffected_unchanged(before.outcomes, after.outcomes, affected)
        self.assertEqual(violations, [])
        self.assertNotEqual(
            before.outcomes["r1"].basis_blob(), after.outcomes["r1"].basis_blob())


if __name__ == "__main__":
    unittest.main()
