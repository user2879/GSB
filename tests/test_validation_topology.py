import unittest

from alert_pipeline.errors import SpecError
from alert_pipeline.rules import load_rule_spec
from alert_pipeline.topology import DependencyGraph

SERVICES = ["a", "b", "c", "d"]


def graph_doc(edges=None, order=None):
    edges = edges or {"a": ["b"], "b": ["c"], "c": ["d"], "d": []}
    order = order or SERVICES
    return {"services": [
        {"service_id": sid, "depends_on": sorted(edges.get(sid, []))}
        for sid in order]}


class ValidationTest(unittest.TestCase):
    def test_duplicate_service_id_rejected(self):
        doc = {"services": [
            {"service_id": "a"}, {"service_id": "a"}]}
        with self.assertRaises(SpecError) as ctx:
            DependencyGraph(doc)
        self.assertEqual(ctx.exception.code, "E_SPEC")
        self.assertEqual(ctx.exception.field, "service_id")

    def test_dangling_dependency_rejected(self):
        doc = {"services": [{"service_id": "a",
                             "depends_on": ["ghost"]}]}
        with self.assertRaises(SpecError) as ctx:
            DependencyGraph(doc)
        self.assertEqual(ctx.exception.field, "depends_on")

    def test_cycle_rejected_even_when_hidden_by_insertion_order(self):
        edges = {"a": ["b"], "b": ["a"]}
        with self.assertRaises(SpecError) as ctx:
            DependencyGraph(graph_doc(edges, ["b", "a"]))
        self.assertEqual(ctx.exception.code, "E_SPEC")

    def test_topo_order_stable_under_insertion_permutation(self):
        edges = {"a": ["b"], "b": ["c"], "c": ["d"], "d": []}
        g1 = DependencyGraph(graph_doc(edges, ["a", "b", "c", "d"]))
        g2 = DependencyGraph(graph_doc(edges, ["d", "c", "b", "a"]))
        self.assertEqual(g1.topo_order, g2.topo_order)
        # Dependency d (no outgoing edges) sorts first.
        self.assertEqual(g1.topo_order, ("d", "c", "b", "a"))
        up = g1.shortest_paths_from("a")
        self.assertEqual(up["d"], ("a", "b", "c", "d"))
        down = g1.shortest_paths_to("d")
        self.assertEqual(down["a"], ("d", "c", "b", "a"))

    def test_rule_spec_rejects_unknown_service_and_negative_window(self):
        with self.assertRaises(SpecError):
            load_rule_spec(
                {"declared_version": 1, "schema_version": 3,
                 "default_silence_window_ms": 1, "window_version": 1,
                 "rule_version": 1,
                 "silence_overrides": {"by_fingerprint": {},
                                       "by_service": {"ghost": 2}}},
                set(SERVICES))
        with self.assertRaises(SpecError):
            load_rule_spec(
                {"declared_version": 1, "schema_version": 3,
                 "default_silence_window_ms": -1, "window_version": 1,
                 "rule_version": 1}, set(SERVICES))


if __name__ == "__main__":
    unittest.main()
