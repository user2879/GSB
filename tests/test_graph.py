from __future__ import annotations

import copy
import unittest

from bucketpack.errors import E_CYCLE, E_REF, BusinessError
from bucketpack.graph import build_graph, stable_edges
from bucketpack.spec import validate_spec

from .common import tiny_raw


class GraphTests(unittest.TestCase):
    def test_dependency_edge_and_topological_order(self) -> None:
        raw = tiny_raw()
        raw["rules"].append(
            {
                "id": "vb2",
                "source": "s",
                "input": {"kind": "bucket", "name": "vb"},
                "boundaries": [1],
                "scale": 0,
            }
        )
        graph = build_graph(validate_spec(raw))
        self.assertEqual(stable_edges(graph), [{"from": "vb", "to": "vb2"}])
        self.assertLess(graph.topo.index("vb"), graph.topo.index("vb2"))
        self.assertEqual(graph.children["vb"], ("vb2",))

    def test_cycle_is_reported_with_stable_ids(self) -> None:
        raw = tiny_raw()
        raw["rules"].extend(
            [
                {
                    "id": "a",
                    "source": "s",
                    "input": {"kind": "bucket", "name": "b"},
                    "boundaries": [0],
                    "scale": 0,
                },
                {
                    "id": "b",
                    "source": "s",
                    "input": {"kind": "bucket", "name": "a"},
                    "boundaries": [0],
                    "scale": 0,
                },
            ]
        )
        outcomes = []
        for rules in (raw["rules"], list(reversed(raw["rules"]))):
            candidate = copy.deepcopy(raw)
            candidate["rules"] = rules
            with self.assertRaises(BusinessError) as caught:
                build_graph(validate_spec(candidate))
            outcomes.append(caught.exception)
        self.assertEqual(outcomes[0].code, E_CYCLE)
        self.assertEqual(outcomes[0].details["cycle"], ["a", "b"])
        self.assertEqual(outcomes[1].details["cycle"], ["a", "b"])

    def test_delete_referenced_bucket_reports_referrers(self) -> None:
        raw = tiny_raw()
        raw["rules"].append(
            {
                "id": "vb2",
                "source": "s",
                "input": {"kind": "bucket", "name": "vb"},
                "boundaries": [1],
                "scale": 0,
            }
        )
        spec = validate_spec(raw)
        graph = build_graph(spec)
        with self.assertRaises(BusinessError) as caught:
            graph.assert_deletable(spec, ["vb"])
        error = caught.exception
        self.assertEqual(error.code, E_REF)
        self.assertEqual(error.details["referrers"][0]["referrer_id"], "vb2")
        self.assertEqual(error.details["referrers"][0]["field"], "input.name")
        self.assertTrue(any(item["referrer_type"] == "statistic" for item in error.details["referrers"]))

    def test_deleting_whole_dependent_chain_is_allowed(self) -> None:
        raw = tiny_raw()
        raw["statistics"] = []
        raw["rules"].append(
            {
                "id": "vb2",
                "source": "s",
                "input": {"kind": "bucket", "name": "vb"},
                "boundaries": [1],
                "scale": 0,
            }
        )
        spec = validate_spec(raw)
        build_graph(spec).assert_deletable(spec, ["vb", "vb2"])

    def test_missing_bucket_reference_is_ref_error(self) -> None:
        raw = tiny_raw()
        raw["rules"][0]["input"] = {"kind": "bucket", "name": "missing"}
        spec = validate_spec(raw)
        with self.assertRaises(BusinessError) as caught:
            build_graph(spec)
        self.assertEqual(caught.exception.code, E_REF)


if __name__ == "__main__":
    unittest.main()
