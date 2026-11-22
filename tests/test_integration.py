import tempfile
import unittest
from pathlib import Path

from degprop.collector import CallScheduler
from degprop.engine import DiagnosticsEngine
from degprop.loader import load_events, load_topology
from degprop.mock_service import MockCluster
from degprop.models import Outcome
from degprop.propagation import PropagationPlanner
from degprop.availability import AvailabilityEvaluator

ROOT = Path(__file__).resolve().parents[1]
TOPO = ROOT / "samples" / "topology.json"
EVENTS = ROOT / "samples" / "events.ndjson"


class LoopbackIntegrationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.topology = load_topology(TOPO)
        cls.events = load_events(EVENTS, cls.topology)
        cls.cluster = MockCluster(cls.topology, slow_ms=400)
        bindings = cls.cluster.start()
        cls.port_map = {sid: {"port": b.port, "refused_port": b.refused_port}
                        for sid, b in bindings.items()}

    @classmethod
    def tearDownClass(cls):
        cls.cluster.stop()

    def test_real_loopback_outcomes_cover_all_classes(self):
        scheduler = CallScheduler(self.topology, self.port_map)
        records = scheduler.run(self.events)
        outcomes = {r.outcome for r in records}
        self.assertIn(Outcome.SUCCESS.value, outcomes)
        self.assertIn(Outcome.TIMEOUT.value, outcomes)
        self.assertIn(Outcome.REFUSED.value, outcomes)
        # 真实连接拒绝必须来自 refused 端口且带真实 socket 错误。
        refused = [r for r in records if r.outcome == Outcome.REFUSED.value]
        self.assertTrue(refused)
        for record in refused:
            self.assertEqual(record.port_kind, "refused")
            self.assertIn("ConnectionRefusedError", record.error)
        timeouts = [r for r in records if r.outcome == Outcome.TIMEOUT.value]
        for record in timeouts:
            self.assertIn("socket.timeout", record.error)

    def test_dual_concurrency_limits_respected(self):
        scheduler = CallScheduler(self.topology, self.port_map)
        scheduler.run(self.events)
        self.assertLessEqual(scheduler.stats.max_global_in_flight,
                             self.topology.params.global_concurrency)
        for did, quota in self.topology.domains.items():
            self.assertLessEqual(scheduler.stats.max_domain_in_flight[did],
                                 quota.in_flight_quota)
        self.assertGreaterEqual(scheduler.stats.queued_calls, 1)

    def test_propagation_chain_and_domain_convergence(self):
        scheduler = CallScheduler(self.topology, self.port_map)
        records = scheduler.run(self.events)
        evaluator = AvailabilityEvaluator(self.topology)
        evaluator.ingest(records)
        planner = PropagationPlanner(self.topology, evaluator.states,
                                     evaluator.triggers)
        result = planner.run()
        self.assertIn("svc-gateway", result.degraded_services)
        self.assertIn("svc-db", result.degraded_services)
        # 传播输出顺序与稳定拓扑序一致。
        self.assertEqual(result.degraded_services,
                         sorted(result.degraded_services,
                                key=self.topology.order_index.index))
        domains = result.domain_plans
        self.assertEqual(set(domains), {"dom-edge", "dom-core", "dom-data"})
        for plan in domains.values():
            self.assertEqual(plan.degraded_services,
                             sorted(plan.degraded_services))

    def test_success_and_degrade_are_disjoint(self):
        scheduler = CallScheduler(self.topology, self.port_map)
        records = scheduler.run(self.events)
        evaluator = AvailabilityEvaluator(self.topology)
        evaluator.ingest(records)
        degraded_seqs = {t.seq for t in evaluator.triggers
                         if t.source != "forced_propagation"}
        for record in records:
            if record.seq in degraded_seqs and record.downstream in {
                    t.service_id for t in evaluator.triggers}:
                self.assertFalse(record.is_success)


class TieBreakTest(unittest.TestCase):
    def test_indistinguishable_tie_break_is_stable(self):
        from degprop.fixedpoint import Scaling
        from degprop.propagation import DomainPlan
        from degprop.fixedpoint import judgment_bound, quantized_interval
        from fractions import Fraction

        scaling = Scaling(2)
        a = quantized_interval(Fraction(50, 100), scaling, "a")
        b = quantized_interval(Fraction(50, 100), scaling, "b")
        verdict = judgment_bound(a, b, scaling.factor, "tie")
        self.assertTrue(verdict["indistinguishable"])
        plans = [
            DomainPlan("dom-z", ["svc-z"], 10, 1000, 50, "cross", False, {}),
            DomainPlan("dom-a", ["svc-a"], 10, 1000, 50, "local", False, {}),
        ]
        ordered = sorted(plans, key=lambda p: (-p.total_capacity_fixed,
                                               p.domain_id,
                                               ",".join(p.degraded_services)))
        self.assertEqual(ordered[0].domain_id, "dom-a")
        # 交换枚举顺序不改变决胜结果。
        self.assertEqual(ordered[0], sorted(reversed(plans),
                         key=lambda p: (-p.total_capacity_fixed, p.domain_id,
                                        ",".join(p.degraded_services)))[0])


if __name__ == "__main__":
    unittest.main()
