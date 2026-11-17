import unittest

from graphidx import GraphError, PropertyGraph

from tests.helpers import load_sample_graph, sample_changesets


class TestGraphLoading(unittest.TestCase):
    def test_valid_graph_loads(self):
        graph = load_sample_graph()
        self.assertEqual(len(graph.nodes), 7)
        self.assertEqual(len(graph.edges), 8)
        self.assertEqual(graph.version, 0)

    def test_digest_independent_of_input_order(self):
        a = PropertyGraph.from_dict({
            "nodes": [{"id": 1, "label": "A"}, {"id": 2, "label": "B"}],
            "edges": [{"id": "e1", "src": 1, "dst": 2, "weight": 1, "label": "x"}],
        })
        b = PropertyGraph.from_dict({
            "edges": [{"id": "e1", "src": 1, "dst": 2, "weight": 1, "label": "x"}],
            "nodes": [{"id": 2, "label": "B"}, {"id": 1, "label": "A"}],
        })
        self.assertEqual(a.content_digest(), b.content_digest())

    def test_duplicate_node_id_rejected(self):
        with self.assertRaises(GraphError) as ctx:
            PropertyGraph.from_dict({"nodes": [{"id": 1, "label": "A"}, {"id": 1, "label": "B"}], "edges": []})
        self.assertEqual(ctx.exception.code, "E_GRAPH")
        self.assertEqual(ctx.exception.details[0]["element"], "nodes[1]")
        self.assertEqual(ctx.exception.details[0]["field"], "id")

    def test_dangling_edge_endpoint_rejected(self):
        with self.assertRaises(GraphError) as ctx:
            PropertyGraph.from_dict({
                "nodes": [{"id": 1, "label": "A"}],
                "edges": [{"id": "e1", "src": 1, "dst": 99, "weight": 1, "label": "x"}],
            })
        self.assertEqual(ctx.exception.details[0]["field"], "dst")

    def test_non_integer_weight_rejected(self):
        for bad in (1.5, "3", True):
            with self.assertRaises(GraphError) as ctx:
                PropertyGraph.from_dict({
                    "nodes": [{"id": 1, "label": "A"}, {"id": 2, "label": "B"}],
                    "edges": [{"id": "e1", "src": 1, "dst": 2, "weight": bad, "label": "x"}],
                })
            self.assertEqual(ctx.exception.details[0]["field"], "weight")


class TestChangeset(unittest.TestCase):
    def test_version_monotonic_and_deterministic_digest(self):
        cs = sample_changesets()[0]
        g1, g2 = load_sample_graph(), load_sample_graph()
        r1 = g1.apply_changeset(cs)
        self.assertEqual(g1.version, 1)
        r2 = g2.apply_changeset(cs)
        self.assertEqual(r1.digest_after, r2.digest_after)
        g1.apply_changeset(sample_changesets()[1])
        self.assertEqual(g1.version, 2)

    def test_remove_node_with_incident_edges_rejected(self):
        graph = load_sample_graph()
        with self.assertRaises(GraphError) as ctx:
            graph.apply_changeset({"id": "bad", "ops": [{"op": "remove_node", "node": 1}]})
        err = ctx.exception
        self.assertEqual(err.code, "E_GRAPH")
        self.assertEqual(err.details[0]["field"], "edges")
        self.assertIn("e1", err.details[0]["message"])
        self.assertEqual(graph.version, 0)  # 整体拒绝，图未变

    def test_missing_references_rejected(self):
        graph = load_sample_graph()
        with self.assertRaises(GraphError):
            graph.apply_changeset({"id": "bad", "ops": [{"op": "remove_edge", "edge": "nope"}]})
        with self.assertRaises(GraphError):
            graph.apply_changeset({"id": "bad", "ops": [{"op": "set_node_label", "node": 99, "label": "X"}]})
        with self.assertRaises(GraphError):
            graph.apply_changeset({"id": "bad", "ops": [{"op": "add_edge", "edge": "ex", "src": 1, "dst": 99, "weight": 1, "label": "x"}]})

    def test_atomic_rejection_leaves_graph_untouched(self):
        graph = load_sample_graph()
        before = graph.content_digest()
        with self.assertRaises(GraphError):
            graph.apply_changeset({"id": "bad", "ops": [
                {"op": "set_edge_weight", "edge": "e1", "weight": 100},
                {"op": "remove_edge", "edge": "missing"},
            ]})
        self.assertEqual(graph.content_digest(), before)
        self.assertEqual(graph.version, 0)

    def test_remove_node_after_removing_edges_succeeds(self):
        graph = load_sample_graph()
        graph.apply_changeset({"id": "cs", "ops": [
            {"op": "remove_edge", "edge": "e5"},
            {"op": "remove_edge", "edge": "e6"},
            {"op": "remove_node", "node": 4},
        ]})
        self.assertNotIn(4, graph.nodes)


if __name__ == "__main__":
    unittest.main()
