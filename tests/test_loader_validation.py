import json
import os
import tempfile
import unittest

from graphdiag.core.errors import GraphValidationError
from graphdiag.core.loader import load_declaration

VALID = {
    "config": {"scale_exponent": 2},
    "nodes": [
        {"node_id": 1, "label": 10},
        {"node_id": 2, "label": 20},
    ],
    "edges": [{"source": 1, "target": 2, "weight_micro": 1500}],
    "queries": [
        {"query_id": "q1", "source": 1, "targets": [2], "cap_micro": 5000}
    ],
}


class LoaderValidationTest(unittest.TestCase):
    def _write(self, payload) -> str:
        handle = tempfile.NamedTemporaryFile(
            "w", suffix=".json", delete=False, encoding="utf-8"
        )
        json.dump(payload, handle)
        handle.close()
        self.addCleanup(os.unlink, handle.name)
        return handle.name

    def test_valid_load(self):
        decl = load_declaration(self._write(VALID))
        self.assertEqual(decl.scale_exponent, 2)
        self.assertEqual(len(decl.graph.nodes), 2)
        self.assertEqual(decl.queries[0].targets, (2,))

    def test_duplicate_node_rejected_with_first_conflict(self):
        payload = json.loads(json.dumps(VALID))
        payload["nodes"].append({"node_id": 1, "label": 99})
        with self.assertRaises(GraphValidationError) as ctx:
            load_declaration(self._write(payload))
        self.assertEqual(ctx.exception.code, "E_GRAPH")
        self.assertEqual(ctx.exception.detail["node_id"], 1)

    def test_edge_missing_endpoint_rejected(self):
        payload = json.loads(json.dumps(VALID))
        payload["edges"].append({"source": 1, "target": 99, "weight_micro": 1})
        with self.assertRaises(GraphValidationError) as ctx:
            load_declaration(self._write(payload))
        self.assertEqual(ctx.exception.code, "E_GRAPH")
        self.assertIn("edge", ctx.exception.detail)

    def test_negative_weight_rejected(self):
        payload = json.loads(json.dumps(VALID))
        payload["edges"][0]["weight_micro"] = -1
        with self.assertRaises(GraphValidationError) as ctx:
            load_declaration(self._write(payload))
        self.assertEqual(ctx.exception.code, "E_GRAPH")

    def test_bool_weight_rejected(self):
        payload = json.loads(json.dumps(VALID))
        payload["edges"][0]["weight_micro"] = True
        with self.assertRaises(GraphValidationError):
            load_declaration(self._write(payload))

    def test_query_missing_target_rejected(self):
        payload = json.loads(json.dumps(VALID))
        payload["queries"][0]["targets"] = [7]
        with self.assertRaises(GraphValidationError) as ctx:
            load_declaration(self._write(payload))
        self.assertEqual(ctx.exception.detail["query_id"], "q1")

    def test_non_positive_cap_rejected(self):
        payload = json.loads(json.dumps(VALID))
        payload["queries"][0]["cap_micro"] = 0
        with self.assertRaises(GraphValidationError):
            load_declaration(self._write(payload))

    def test_scale_exponent_range(self):
        for bad in (-1, 7, "3", 2.0):
            payload = json.loads(json.dumps(VALID))
            payload["config"]["scale_exponent"] = bad
            with self.assertRaises(GraphValidationError):
                load_declaration(self._write(payload))


if __name__ == "__main__":
    unittest.main()
