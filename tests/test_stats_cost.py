from __future__ import annotations

import sqlite3
import tempfile
import unittest
from pathlib import Path

from qplan.canonical import content_digest
from qplan.cost import estimate_path
from qplan.data import materialize_database
from qplan.paths import build_candidate_paths
from qplan.pipeline import build_estimates
from qplan.stats import collect_statistics, estimate_filter_selectivity, statistics_digest

from tests.helpers import make_spec, reorder_candidate_indexes, small_spec_dict


class StatsAndCostTests(unittest.TestCase):
    def test_exact_and_sampled_statistics_are_both_archived(self) -> None:
        spec = make_spec()
        with tempfile.TemporaryDirectory() as temp:
            db_path = Path(temp) / "sample.sqlite"
            materialize_database(spec, db_path)
            connection = sqlite3.connect(db_path)
            try:
                stats_first = collect_statistics(spec, connection, spec.stats)
                stats_second = collect_statistics(spec, connection, spec.stats)
            finally:
                connection.close()
        self.assertEqual(statistics_digest(stats_first), statistics_digest(stats_second))
        self.assertEqual(stats_first["tables"]["users"]["adaptive_method"], "exact")
        self.assertEqual(stats_first["tables"]["events"]["adaptive_method"], "sampled")
        event_status = stats_first["tables"]["events"]["columns"]["status"]
        self.assertGreater(len(event_status["exact"]["histogram"]), 0)
        self.assertGreater(len(event_status["sampled"]["histogram"]), 0)
        self.assertEqual(stats_first["tables"]["events"]["sample_size"], 32)
        self.assertNotEqual(
            event_status["exact"]["histogram"], event_status["sampled"]["histogram"]
        )
        filter_spec = spec.queries[0].filters[0]
        estimate = estimate_filter_selectivity(spec, stats_first, filter_spec)
        self.assertEqual(estimate["adaptive_method"], "sampled")
        self.assertEqual(estimate["final_stats_method"], "sampled")
        self.assertGreater(estimate["error_upper_bound"], 0.0)
        self.assertIn("histogram_bucket_fractions", estimate["exact"])

    def test_path_estimates_are_byte_stable_and_cover_four_classes(self) -> None:
        spec = make_spec()
        with tempfile.TemporaryDirectory() as temp:
            db_path = Path(temp) / "sample.sqlite"
            materialize_database(spec, db_path)
            connection = sqlite3.connect(db_path)
            try:
                stats = collect_statistics(spec, connection, spec.stats)
            finally:
                connection.close()
        estimates_first = build_estimates(spec, stats)
        estimates_second = build_estimates(spec, stats)
        self.assertEqual(estimates_first["estimates_digest"], estimates_second["estimates_digest"])
        query = spec.queries[0]
        paths = build_candidate_paths(spec, query)
        classes = {path.kind for path in paths}
        self.assertEqual(classes, {"full_scan", "single_index", "composite_index", "covering_index"})
        for path in paths:
            first = estimate_path(spec, stats, query, path)
            second = estimate_path(spec, stats, query, path)
            self.assertEqual(content_digest(first), content_digest(second))
            self.assertIn("access_rows", first)
            self.assertIn("comparisons", first)
            self.assertIn("lookups", first)
            self.assertIn("io_pages", first)
            self.assertIn("cost", first)

    def test_candidate_declaration_order_does_not_change_selection(self) -> None:
        first_spec = make_spec()
        reversed_spec = make_spec(reorder_candidate_indexes(small_spec_dict()))
        with tempfile.TemporaryDirectory() as temp:
            first_db = Path(temp) / "first.sqlite"
            second_db = Path(temp) / "second.sqlite"
            materialize_database(first_spec, first_db)
            materialize_database(reversed_spec, second_db)
            first_connection = sqlite3.connect(first_db)
            second_connection = sqlite3.connect(second_db)
            try:
                first_stats = collect_statistics(first_spec, first_connection, first_spec.stats)
                second_stats = collect_statistics(reversed_spec, second_connection, reversed_spec.stats)
            finally:
                first_connection.close()
                second_connection.close()
        first = build_estimates(first_spec, first_stats)
        second = build_estimates(reversed_spec, second_stats)
        self.assertEqual(first["index_search"]["selected_indexes"], second["index_search"]["selected_indexes"])
        self.assertEqual(first["estimates_digest"], second["estimates_digest"])
        self.assertEqual(
            first["index_search"]["candidate_enumeration"],
            second["index_search"]["candidate_enumeration"],
        )


if __name__ == "__main__":
    unittest.main()

