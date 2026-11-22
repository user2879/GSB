import json
import unittest
from pathlib import Path
import tempfile

from k8ssim.errors import SpecError
from k8ssim.loader import (
    load_nodes,
    load_pods,
    validate_scale_factor,
    validate_weights,
)

ROOT = Path(__file__).resolve().parent.parent
NODES = ROOT / "samples" / "nodes.json"
PODS = ROOT / "samples" / "pods.json"


class LoaderValidationTest(unittest.TestCase):
    def test_valid_samples_load(self) -> None:
        nodes = load_nodes(NODES)
        pods = load_pods(PODS, nodes)
        self.assertEqual([n.name for n in nodes],
                         ["node-a", "node-b", "node-c", "node-d"])
        self.assertEqual(len(pods), 6)

    def _write(self, payload) -> str:
        handle = tempfile.NamedTemporaryFile("w", suffix=".json", delete=False,
                                             encoding="utf-8")
        json.dump(payload, handle)
        handle.close()
        return handle.name

    def test_duplicate_node_name_rejected(self) -> None:
        nodes = {"nodes": [
            {"name": "n1", "labels": {}, "cpu": 1, "memory": 1, "taints": []},
            {"name": "n1", "labels": {}, "cpu": 1, "memory": 1, "taints": []},
        ]}
        with self.assertRaises(SpecError) as ctx:
            load_nodes(self._write(nodes))
        self.assertEqual(ctx.exception.code, "E_SPEC")
        self.assertIn("$.nodes[1].name", ctx.exception.path)

    def test_boolean_rejected_as_integer(self) -> None:
        nodes = {"nodes": [
            {"name": "n1", "labels": {}, "cpu": True, "memory": 1, "taints": []},
        ]}
        with self.assertRaises(SpecError) as ctx:
            load_nodes(self._write(nodes))
        self.assertEqual(ctx.exception.path, "$.nodes[0].cpu")

    def test_non_positive_memory_rejected(self) -> None:
        nodes = {"nodes": [
            {"name": "n1", "labels": {}, "cpu": 4, "memory": 0, "taints": []},
        ]}
        with self.assertRaises(SpecError) as ctx:
            load_nodes(self._write(nodes))
        self.assertEqual(ctx.exception.path, "$.nodes[0].memory")

    def test_bad_taint_effect_rejected(self) -> None:
        nodes = {"nodes": [
            {"name": "n1", "labels": {}, "cpu": 4, "memory": 4,
             "taints": [{"key": "k", "value": "v", "effect": "Explode"}]},
        ]}
        with self.assertRaises(SpecError) as ctx:
            load_nodes(self._write(nodes))
        self.assertEqual(ctx.exception.path, "$.nodes[0].taints[0].effect")

    def test_duplicate_pod_id_rejected(self) -> None:
        nodes = load_nodes(NODES)
        pods = {"pods": [
            {"pod_id": "p", "cpu_request": 1, "memory_request": 1},
            {"pod_id": "p", "cpu_request": 1, "memory_request": 1},
        ]}
        with self.assertRaises(SpecError) as ctx:
            load_pods(self._write(pods), nodes)
        self.assertEqual(ctx.exception.path, "$.pods[1].pod_id")

    def test_missing_node_label_reference_rejected(self) -> None:
        nodes = load_nodes(NODES)
        pods = {"pods": [
            {"pod_id": "p", "cpu_request": 1, "memory_request": 1,
             "node_affinity": {"required": [
                 {"key": "nonexistent", "operator": "Exists"}]}},
        ]}
        with self.assertRaises(SpecError) as ctx:
            load_pods(self._write(pods), nodes)
        self.assertIn("node_affinity.required[0].key", ctx.exception.path)

    def test_missing_pod_label_reference_rejected(self) -> None:
        nodes = load_nodes(NODES)
        pods = {"pods": [
            {"pod_id": "p", "cpu_request": 1, "memory_request": 1,
             "pod_anti_affinity": {
                 "required": [{"key": "app", "operator": "In", "values": ["web"]}],
                 "required_topology_key": "kubernetes.io/hostname"}},
        ]}
        with self.assertRaises(SpecError) as ctx:
            load_pods(self._write(pods), nodes)
        self.assertIn("pod_anti_affinity.required[0].key", ctx.exception.path)

    def test_negative_operator_allows_unknown_key(self) -> None:
        nodes = load_nodes(NODES)
        pods = {"pods": [
            {"pod_id": "p", "cpu_request": 1, "memory_request": 1,
             "node_affinity": {"required": [
                 {"key": "future-key", "operator": "DoesNotExist"}]}},
        ]}
        loaded = load_pods(self._write(pods), nodes)
        self.assertEqual(loaded[0].pod_id, "p")

    def test_weights_must_sum_to_basis(self) -> None:
        from k8ssim.scheduler.engine import DEFAULT_WEIGHTS
        bad = dict(DEFAULT_WEIGHTS)
        bad["resource_balance"] += 1
        with self.assertRaises(SpecError):
            validate_weights(bad)
        self.assertEqual(sum(validate_weights(dict(DEFAULT_WEIGHTS)).values()), 10000)

    def test_scale_factor_range(self) -> None:
        self.assertEqual(validate_scale_factor(100), 100)
        with self.assertRaises(SpecError):
            validate_scale_factor(0)


if __name__ == "__main__":
    unittest.main()
