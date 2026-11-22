import unittest

from k8ssim.loader import load_nodes, load_pods
from k8ssim.scheduler.engine import SchedulingEngine
from k8ssim.scheduler.filter import (
    GATE_NODE_AFFINITY,
    GATE_POD_ANTI_AFFINITY,
    GATE_RESOURCE,
    GATE_TAINT,
    NodeCapacity,
    Placement,
    evaluate_filter,
)

NODES = load_nodes("samples/nodes.json")
PODS = load_pods("samples/pods.json", NODES)
NODE_LOOKUP = {node.name: node for node in NODES}


class FilterTest(unittest.TestCase):
    def setUp(self) -> None:
        self.capacities = {
            node.name: NodeCapacity(node.cpu, node.memory) for node in NODES
        }

    def test_taint_gate_first_for_gpu_pod(self) -> None:
        pod = next(p for p in PODS if p.pod_id == "gpu-job-1")
        # Without tolerations node-c must fail at the taint gate.
        bare = pod.__class__(**{**pod.__dict__, "tolerations": ()})
        _, rejections = evaluate_filter(
            bare, NODES, self.capacities, [], NODE_LOOKUP
        )
        node_c = next(r for r in rejections if r.node == "node-c")
        self.assertEqual(node_c.gate, GATE_TAINT)

    def test_gate_order_records_first_failure_only(self) -> None:
        pod = next(p for p in PODS if p.pod_id == "web-3")
        placements = [Placement("web-x", {"app": "web"}, "node-a")]
        candidates, rejections = evaluate_filter(
            pod, NODES, self.capacities, placements, NODE_LOOKUP
        )
        gates = {r.node: r.gate for r in rejections}
        self.assertEqual(gates["node-b"], GATE_NODE_AFFINITY)
        self.assertEqual(gates["node-c"], GATE_TAINT)
        self.assertEqual(gates["node-a"], GATE_POD_ANTI_AFFINITY)
        self.assertNotIn("node-a", candidates)

    def test_anti_affinity_topology_conflict(self) -> None:
        web2 = next(p for p in PODS if p.pod_id == "web-2")
        placements = [Placement("web-1", {"app": "web"}, "node-d")]
        candidates, rejections = evaluate_filter(
            web2, NODES, self.capacities, placements, NODE_LOOKUP
        )
        self.assertNotIn("node-d", candidates)
        rejection = next(r for r in rejections if r.node == "node-d")
        self.assertEqual(rejection.gate, GATE_POD_ANTI_AFFINITY)

    def test_resource_gate_after_others(self) -> None:
        big = PODS[0].__class__(
            pod_id="huge",
            cpu_request=100,
            memory_request=100,
            labels={},
        )
        candidates, rejections = evaluate_filter(
            big, NODES, self.capacities, [], NODE_LOOKUP
        )
        self.assertEqual(candidates, [])
        # Every feasible-by-label/taint node reports the resource gate.
        resource_nodes = {r.node for r in rejections if r.gate == GATE_RESOURCE}
        self.assertIn("node-a", resource_nodes)

    def test_determinism_independent_of_input_order(self) -> None:
        pod = PODS[0]
        first_candidates, first_rejections = evaluate_filter(
            pod, NODES, self.capacities, [], NODE_LOOKUP
        )
        shuffled = list(reversed(NODES))
        second_candidates, second_rejections = evaluate_filter(
            pod, shuffled, self.capacities, [], NODE_LOOKUP
        )
        self.assertEqual(first_candidates, second_candidates)
        self.assertEqual(
            [(r.node, r.gate, r.reason) for r in first_rejections],
            [(r.node, r.gate, r.reason) for r in second_rejections],
        )

    def test_engine_rejects_unschedulable_pod(self) -> None:
        engine = SchedulingEngine(nodes=NODES, pods=PODS)
        report = engine.run()
        web3 = next(d for d in report.decisions if d.pod_id == "web-3")
        self.assertFalse(web3.feasible)
        self.assertIsNone(web3.bound_node)
        self.assertEqual(web3.tie_reason, "unschedulable")


if __name__ == "__main__":
    unittest.main()
