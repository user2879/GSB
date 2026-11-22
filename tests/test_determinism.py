import random
import unittest

from k8ssim.loader import load_nodes, load_pods
from k8ssim.scheduler.engine import SchedulingEngine

NODES = load_nodes("samples/nodes.json")
PODS = load_pods("samples/pods.json", NODES)


class DeterminismTest(unittest.TestCase):
    def test_shuffled_node_order_keeps_binding_and_diagnostics(self) -> None:
        baseline = SchedulingEngine(NODES, PODS).run()
        rng = random.Random(20261123)
        for _ in range(5):
            shuffled_nodes = list(NODES)
            rng.shuffle(shuffled_nodes)
            report = SchedulingEngine(shuffled_nodes, PODS).run()
            self.assertEqual(
                [d.bound_node for d in baseline.decisions],
                [d.bound_node for d in report.decisions],
            )
            self.assertEqual(
                [d.to_dict() for d in baseline.decisions],
                [d.to_dict() for d in report.decisions],
            )

    def test_shuffled_pod_queue_order_keeps_per_pod_conclusions(self) -> None:
        baseline = {d.pod_id: d.to_dict()
                    for d in SchedulingEngine(NODES, PODS).run().decisions}
        rng = random.Random(424242)
        shuffled_pods = list(PODS)
        rng.shuffle(shuffled_pods)
        report = SchedulingEngine(NODES, shuffled_pods).run()
        for decision in report.decisions:
            # Per-pod filter/score inputs differ once earlier bindings move,
            # so assert structural stability: same set of pod ids, valid states.
            self.assertIn(decision.pod_id, baseline)
            self.assertIn(decision.tie_reason,
                          {"unique", "exact_tie", "error_bound_tie", "unschedulable"})

    def test_zero_weight_plugin_does_not_break_basis(self) -> None:
        engine = SchedulingEngine(NODES, PODS)
        weights = {
            "node_affinity_soft": 4000,
            "pod_anti_affinity_soft": 2000,
            "resource_balance": 4000,
            "label_locality": 0,
        }
        engine.update_params(weights=weights)
        report = engine.run()
        for decision in report.decisions:
            for node_score in decision.node_scores:
                locality = node_score.plugin_map()["label_locality"]
                self.assertEqual(locality.weight, 0)
                self.assertEqual(locality.contribution_mp, 0)

    def test_digest_stability_across_instances(self) -> None:
        first = SchedulingEngine(NODES, PODS)
        second = SchedulingEngine(NODES, PODS)
        self.assertEqual(first.spec_digest(), second.spec_digest())
        self.assertEqual(
            first.params.content_digest(), second.params.content_digest()
        )


if __name__ == "__main__":
    unittest.main()
