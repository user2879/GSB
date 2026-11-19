import unittest

from pcl.errors import CycleError, ReferenceError
from pcl.graph import RefGraph


def build_graph():
    graph = RefGraph()
    for key in ("a", "b", "c", "d", "e"):
        graph.add_node(key)
    graph.add_edge("a", "b")
    graph.add_edge("a", "c")
    graph.add_edge("b", "d")
    graph.add_edge("c", "d")
    graph.add_edge("d", "e")
    return graph


class GraphTests(unittest.TestCase):
    def test_transitive_downstream(self):
        graph = build_graph()
        self.assertEqual(
            graph.transitive_downstream(["a"]), {"b", "c", "d", "e"}
        )
        self.assertEqual(graph.transitive_downstream(["b"]), {"d", "e"})

    def test_invalidation_order_is_stable(self):
        graph = build_graph()
        order = graph.invalidation_order(["a"])
        self.assertEqual(order[0], "a")
        self.assertEqual(order[-1], "e")
        self.assertLess(order.index("b"), order.index("d"))
        self.assertLess(order.index("c"), order.index("d"))
        # Deterministic regardless of how edges were inserted.
        shuffled = RefGraph()
        for key in ("e", "d", "c", "b", "a"):
            shuffled.add_node(key)
        for edge in (("d", "e"), ("c", "d"), ("b", "d"),
                     ("a", "c"), ("a", "b")):
            shuffled.add_edge(*edge)
        self.assertEqual(
            shuffled.invalidation_order(["a"]),
            graph.invalidation_order(["a"]),
        )

    def test_cycle_detection(self):
        graph = RefGraph()
        for key in ("a", "b", "c"):
            graph.add_node(key)
        graph.add_edge("a", "b")
        graph.add_edge("b", "c")
        graph.add_edge("c", "a")
        with self.assertRaises(CycleError) as ctx:
            graph.invalidation_order(["a"])
        self.assertEqual(ctx.exception.code, "E_CYCLE")
        self.assertEqual(set(ctx.exception.details["cycle"][:-1]),
                         {"a", "b", "c"})

    def test_referenced_delete_rejected(self):
        graph = build_graph()
        with self.assertRaises(ReferenceError) as ctx:
            graph.remove_node("d")
        self.assertEqual(ctx.exception.code, "E_REF")
        self.assertEqual(ctx.exception.details["referrers"], ["b", "c"])
        with self.assertRaises(ReferenceError):
            graph.remove_node("d")
        graph.remove_node("a")
        self.assertNotIn("a", graph.nodes())
        self.assertEqual(graph.referrers_of("b"), [])


if __name__ == "__main__":
    unittest.main()
