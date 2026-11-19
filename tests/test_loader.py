import copy
import unittest

from cache_invalidation.errors import EGraph, EInherit
from tests.support import SAMPLE_ENTRIES, SAMPLE_NAMESPACES, sample_graph
from cache_invalidation.loader import build_graph


class LoaderValidationTest(unittest.TestCase):
    def test_valid_graph_loads_with_sorted_identifiers(self):
        graph = sample_graph()
        self.assertEqual(
            [entry_id.to_str() for entry_id in graph.entry_ids_sorted()],
            ["catalog/100", "catalog/101", "inventory/300", "pricing/200",
             "search/500", "storefront/400"],
        )
        self.assertEqual(
            graph.namespace_parents["storefront"],
            ("pricing", "inventory"),
        )

    def test_duplicate_namespace_is_rejected(self):
        namespaces = copy.deepcopy(SAMPLE_NAMESPACES)
        namespaces["namespaces"].append(
            {"name": "catalog", "parents": []})
        with self.assertRaises(EGraph) as caught:
            build_graph(namespaces, copy.deepcopy(SAMPLE_ENTRIES))
        self.assertEqual(caught.exception.code, "E_GRAPH")
        self.assertEqual(caught.exception.key, "catalog")

    def test_duplicate_entry_key_in_namespace_is_rejected(self):
        entries = copy.deepcopy(SAMPLE_ENTRIES)
        entries["entries"].append(
            {"key": 100, "namespace": "catalog", "expire_ms": 1,
             "digest": "", "digest_algorithm": "none",
             "annotations": []})
        with self.assertRaises(EGraph) as caught:
            build_graph(copy.deepcopy(SAMPLE_NAMESPACES), entries)
        self.assertEqual(caught.exception.key, "catalog/100")

    def test_negative_expiry_is_rejected(self):
        entries = copy.deepcopy(SAMPLE_ENTRIES)
        entries["entries"][0]["expire_ms"] = -1
        with self.assertRaises(EGraph) as caught:
            build_graph(copy.deepcopy(SAMPLE_NAMESPACES), entries)
        self.assertIn("expire_ms", caught.exception.field_path)

    def test_boolean_is_not_accepted_as_integer(self):
        entries = copy.deepcopy(SAMPLE_ENTRIES)
        entries["entries"][0]["expire_ms"] = True
        with self.assertRaises(EGraph):
            build_graph(copy.deepcopy(SAMPLE_NAMESPACES), entries)

    def test_missing_annotation_target_is_rejected(self):
        entries = copy.deepcopy(SAMPLE_ENTRIES)
        entries["entries"][0]["annotations"].append(
            {"namespace": "search", "key": 999})
        with self.assertRaises(EGraph) as caught:
            build_graph(copy.deepcopy(SAMPLE_NAMESPACES), entries)
        self.assertEqual(caught.exception.key, "catalog/100")
        self.assertIn("search/999",
                      caught.exception.details["missing_target"])

    def test_annotation_cycle_reports_cycle_sequence(self):
        entries = copy.deepcopy(SAMPLE_ENTRIES)
        entries["entries"][-1]["annotations"].append(
            {"namespace": "catalog", "key": 100})
        with self.assertRaises(EGraph) as caught:
            build_graph(copy.deepcopy(SAMPLE_NAMESPACES), entries)
        cycle = caught.exception.details["cycle"]
        self.assertEqual(cycle[0], cycle[-1])
        self.assertIn("catalog/100", cycle)

    def test_inheritance_cycle_reports_namespace_sequence(self):
        namespaces = copy.deepcopy(SAMPLE_NAMESPACES)
        for item in namespaces["namespaces"]:
            if item["name"] == "catalog":
                item["parents"] = ["search"]
        with self.assertRaises(EGraph) as caught:
            build_graph(namespaces, copy.deepcopy(SAMPLE_ENTRIES))
        cycle = caught.exception.details["cycle"]
        self.assertEqual(cycle[0], cycle[-1])

    def test_duplicate_parent_in_chain_is_rejected(self):
        namespaces = copy.deepcopy(SAMPLE_NAMESPACES)
        for item in namespaces["namespaces"]:
            if item["name"] == "storefront":
                item["parents"] = ["pricing", "pricing"]
        with self.assertRaises(EGraph) as caught:
            build_graph(namespaces, copy.deepcopy(SAMPLE_ENTRIES))
        self.assertEqual(caught.exception.key, "storefront")

    def test_missing_inheritance_parent_is_e_inherit(self):
        namespaces = copy.deepcopy(SAMPLE_NAMESPACES)
        namespaces["namespaces"][-1]["parents"].append("nowhere")
        with self.assertRaises(EInherit) as caught:
            build_graph(namespaces, copy.deepcopy(SAMPLE_ENTRIES))
        self.assertEqual(caught.exception.code, "E_INHERIT")
        self.assertEqual(
            caught.exception.details["missing_parent"], "nowhere")

    def test_ambiguous_bare_integer_reference_is_rejected(self):
        entries = copy.deepcopy(SAMPLE_ENTRIES)
        entries["entries"].append({
            "key": 100, "namespace": "pricing", "expire_ms": 1,
            "digest": "", "digest_algorithm": "none",
            "annotations": []})
        entries["entries"][1]["annotations"].append(100)
        with self.assertRaises(EGraph) as caught:
            build_graph(copy.deepcopy(SAMPLE_NAMESPACES), entries)
        self.assertEqual(caught.exception.key, "catalog/101")


if __name__ == "__main__":
    unittest.main()
