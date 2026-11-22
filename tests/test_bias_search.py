from __future__ import annotations

import sqlite3
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from qplan.bias import _classify_bias
from qplan.data import materialize_database
from qplan.pipeline import build_estimates
from qplan.search import search_indexes
from qplan.stats import collect_statistics

from tests.helpers import make_spec


class BiasAndSearchTests(unittest.TestCase):
    def test_bias_categories_are_explicit(self) -> None:
        category, _ = _classify_bias(
            coverage_mismatch=True,
            selectivity_ratio=1.0,
            row_ratio=1.0,
            join_order_mismatch=False,
            threshold=1.75,
        )
        self.assertEqual(category, "index_coverage_decision_error")
        category, _ = _classify_bias(
            coverage_mismatch=False,
            selectivity_ratio=4.0,
            row_ratio=1.0,
            join_order_mismatch=False,
            threshold=1.75,
        )
        self.assertEqual(category, "selectivity_error")
        category, _ = _classify_bias(
            coverage_mismatch=False,
            selectivity_ratio=1.0,
            row_ratio=3.0,
            join_order_mismatch=True,
            threshold=1.75,
        )
        self.assertEqual(category, "join_order_deviation")

    def test_search_reports_e_search_when_greedy_oracle_differs(self) -> None:
        spec = make_spec()
        with tempfile.TemporaryDirectory() as temp:
            db_path = Path(temp) / "sample.sqlite"
            materialize_database(spec, db_path)
            connection = sqlite3.connect(db_path)
            try:
                stats = collect_statistics(spec, connection, spec.stats)
            finally:
                connection.close()
        from qplan.paths import build_candidate_paths
        from qplan.cost import estimate_query_paths

        paths_by_query = {query.id: build_candidate_paths(spec, query) for query in spec.queries}
        estimates_by_query = {
            query.id: estimate_query_paths(spec, stats, query, paths_by_query[query.id])["estimates"]
            for query in spec.queries
        }
        baseline = search_indexes(spec, paths_by_query, estimates_by_query)
        all_indexes = list(baseline["candidate_enumeration"])
        different = next(index for index in all_indexes if index not in baseline["selected_indexes"])
        fake_greedy = {
            "key": (1, 1, (different,)),
            "total_cost": 1,
            "selected_indexes": [different],
            "queries": {},
            "evaluations": 1,
            "trace": [],
        }
        with patch("qplan.search._greedy", return_value=fake_greedy):
            result = search_indexes(spec, paths_by_query, estimates_by_query)
        self.assertEqual(result["selected_indexes"], baseline["selected_indexes"])
        self.assertEqual(result["truth_source"], "exhaustive")
        self.assertTrue(any(item["code"] == "E_SEARCH" for item in result["diagnostics"]))
        diagnostic = next(item for item in result["diagnostics"] if item["code"] == "E_SEARCH")
        self.assertIn(different, diagnostic["difference_indexes"])
        self.assertEqual(diagnostic["truth_source"], "exhaustive")


if __name__ == "__main__":
    unittest.main()
