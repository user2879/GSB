import unittest

from opdiag.errors import (
    E_ACTIVATION_PARAM, E_EDGE_ENDPOINT, E_GRAPH_CYCLE, E_INPLACE_CONFLICT,
    E_MULTIPLE_PRODUCER, E_NODE_DUPLICATE, E_PERM_INVALID, E_UNKNOWN_OP,
    E_INPUT_UNBOUND,
)
from opdiag.graph import build_graph


def chain(nodes, edges):
    return {"nodes": nodes, "edges": edges}


class TestGraphValidation(unittest.TestCase):
    def _shapes(self):
        return {"in0": (2, 3), "in1": (3, 4)}

    def test_duplicate_node_id(self):
        g = build_graph(
            chain([{"id": "A", "op": "relu", "input_ref": "in0"},
                   {"id": "A", "op": "tanh", "input_ref": "in0"}], []),
            self._shapes())
        self.assertIn(E_NODE_DUPLICATE, {f.code for f in g.failures})

    def test_edge_endpoint_missing(self):
        g = build_graph(
            chain([{"id": "A", "op": "relu", "input_ref": "in0"}],
                  [["A", "GHOST"]]), self._shapes())
        self.assertIn(E_EDGE_ENDPOINT, {f.code for f in g.failures})

    def test_cycle_detected(self):
        g = build_graph(
            chain([{"id": "A", "op": "relu"}, {"id": "B", "op": "tanh"}],
                  [["A", "B"], ["B", "A"]]), self._shapes())
        self.assertIn(E_GRAPH_CYCLE, {f.code for f in g.failures})

    def test_multiple_producers(self):
        g = build_graph(
            chain([{"id": "A", "op": "relu", "input_ref": "in0"},
                   {"id": "B", "op": "tanh", "input_ref": "in1"},
                   {"id": "C", "op": "gelu"}],
                  [["A", "C"], ["B", "C"]]), self._shapes())
        self.assertIn(E_MULTIPLE_PRODUCER, {f.code for f in g.failures})

    def test_unknown_op(self):
        g = build_graph(
            chain([{"id": "A", "op": "warp", "input_ref": "in0"}], []),
            self._shapes())
        self.assertIn(E_UNKNOWN_OP, {f.code for f in g.failures})

    def test_unbound_input(self):
        g = build_graph(
            chain([{"id": "A", "op": "relu"}], []), self._shapes())
        self.assertIn(E_INPUT_UNBOUND, {f.code for f in g.failures})

    def test_activation_param_out_of_range(self):
        g = build_graph(
            chain([{"id": "A", "op": "relu", "input_ref": "in0",
                    "params": {"slope": 1.5}}], []), self._shapes())
        self.assertIn(E_ACTIVATION_PARAM, {f.code for f in g.failures})

    def test_perm_must_be_complete_permutation(self):
        g = build_graph(
            chain([{"id": "A", "op": "relu", "input_ref": "in0"},
                   {"id": "T", "op": "transpose", "perm": [0, 0]}],
                  [["A", "T"]]), self._shapes())
        self.assertIn(E_PERM_INVALID, {f.code for f in g.failures})

    def test_perm_rank_mismatch(self):
        g = build_graph(
            chain([{"id": "A", "op": "relu", "input_ref": "in0"},
                   {"id": "T", "op": "transpose", "perm": [0, 1, 2]}],
                  [["A", "T"]]), self._shapes())
        self.assertIn(E_PERM_INVALID, {f.code for f in g.failures})

    def test_inplace_fanout_conflict(self):
        g = build_graph(
            chain([{"id": "A", "op": "relu", "input_ref": "in0"},
                   {"id": "B", "op": "tanh", "inplace": True},
                   {"id": "C", "op": "gelu"}],
                  [["A", "B"], ["A", "C"]]), self._shapes())
        self.assertIn(E_INPLACE_CONFLICT, {f.code for f in g.failures})

    def test_inplace_transpose_conflict(self):
        g = build_graph(
            chain([{"id": "A", "op": "relu", "input_ref": "in0"},
                   {"id": "T", "op": "transpose", "perm": [1, 0],
                    "inplace": True}], [["A", "T"]]), self._shapes())
        self.assertIn(E_INPLACE_CONFLICT, {f.code for f in g.failures})

    def test_valid_chain_passes_and_toposort(self):
        g = build_graph(
            chain([{"id": "A", "op": "relu", "input_ref": "in0"},
                   {"id": "T", "op": "transpose", "perm": [1, 0]}],
                  [["A", "T"]]), self._shapes())
        self.assertEqual(g.failures, [])
        self.assertEqual(g.order, ["A", "T"])
        self.assertEqual(g.nodes["T"].shape, (3, 2))
        self.assertEqual(len(g.fusion_candidates), 1)


if __name__ == "__main__":
    unittest.main()
