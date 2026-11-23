from __future__ import annotations

import unittest
from dataclasses import replace

from mvcc_workbench.errors import DivergenceError
from mvcc_workbench.loader import load_spec, validate_spec_object
from mvcc_workbench.model import build_index, conclusion_digest
from mvcc_workbench.visibility import (
    compare_results,
    result_digest_bytes,
    run_materialized,
    run_per_query,
)


class VisibilityTests(unittest.TestCase):
    def test_two_paths_are_identical_on_large_sample(self) -> None:
        spec = load_spec("data/demo_large.json")
        index = build_index(spec)
        per_query = run_per_query(index)
        materialized = run_materialized(index)
        report = compare_results(per_query, materialized.results)
        self.assertTrue(report.ok)
        self.assertEqual(report.compared_queries, 60)
        self.assertEqual(conclusion_digest(per_query), conclusion_digest(materialized.results))

    def test_input_permutations_are_byte_identical(self) -> None:
        paths = [
            "data/demo_large.json",
            "data/demo_large_shuffled.json",
            "data/demo_large.ndjson",
        ]
        digests = []
        bytes_values = []
        for path in paths:
            index = build_index(load_spec(path))
            results = run_per_query(index)
            digests.append(conclusion_digest(results))
            bytes_values.append(result_digest_bytes(results))
        self.assertEqual(len(set(digests)), 1)
        self.assertEqual(len(set(bytes_values)), 1)

    def test_visibility_rules_and_self_uncommitted(self) -> None:
        data = {
            "format": "mvcc-diagnostic-spec",
            "format_version": 1,
            "name": "self-visibility",
            "transactions": [
                {"id": "T1", "start_seq": 1, "outcome": "committed", "commit_seq": 4},
                {"id": "T2", "start_seq": 2, "outcome": "rolledback", "commit_seq": None},
                {"id": "T3", "start_seq": 3, "outcome": "pending", "commit_seq": None},
            ],
            "records": [
                {
                    "row_key": "r",
                    "tx_id": "T1",
                    "version": 1,
                    "value_digest": "sha256:a",
                    "prev_version": None,
                },
                {
                    "row_key": "r",
                    "tx_id": "T2",
                    "version": 2,
                    "value_digest": "sha256:b",
                    "prev_version": 1,
                },
                {
                    "row_key": "r",
                    "tx_id": "T3",
                    "version": 3,
                    "value_digest": "sha256:c",
                    "prev_version": 2,
                },
            ],
            "read_points": [
                {"id": "RPSELF", "snapshot_seq": 4, "tx_id": "T3"},
                {"id": "RPOTHER", "snapshot_seq": 4, "tx_id": "T1"},
            ],
            "queries": [
                {"id": "QSELF", "read_point_id": "RPSELF", "row_key": "r", "predicate": {"op": "exists"}},
                {"id": "QOTHER", "read_point_id": "RPOTHER", "row_key": "r", "predicate": {"op": "exists"}},
            ],
            "budget": {"global_concurrency": 2, "per_read_point_concurrency": 1},
        }
        spec = validate_spec_object(data, allow_pending=True)
        results = {item.query_id: item for item in run_per_query(build_index(spec))}
        self.assertEqual(results["QSELF"].visible_version, 3)
        self.assertEqual(results["QSELF"].visibility_basis, "SELF_UNCOMMITTED")
        self.assertEqual(results["QOTHER"].visible_version, 1)
        codes = [reason.code for reason in results["QOTHER"].exclusion_reasons]
        self.assertEqual(codes, ["WRITER_PENDING_NOT_SELF", "WRITER_ROLLED_BACK"])

    def test_divergence_is_located_precisely(self) -> None:
        index = build_index(load_spec("data/demo_small.json"))
        left = list(run_per_query(index))
        right = list(left)
        right[2] = replace(right[2], visible_version=999999)
        with self.assertRaises(DivergenceError) as raised:
            compare_results(left, right)
        self.assertEqual(raised.exception.code, "E_DIVERGE")
        self.assertEqual(raised.exception.details["query_id"], "Q0003")
        self.assertEqual(raised.exception.details["row_key"], right[2].row_key)
        self.assertEqual(raised.exception.details["field"], "visible_version")


if __name__ == "__main__":
    unittest.main()
