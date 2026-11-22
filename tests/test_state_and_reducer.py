from __future__ import annotations

import copy
import tempfile
import unittest
from pathlib import Path

from slowdiag.codecs import Evidence
from slowdiag.config import default_config_dict, load_config_data
from slowdiag.invalidation import commit_plan, decide_plan
from slowdiag.reducer import reduce_integers, reduce_rows, select_replays
from slowdiag.storage import Ledger


def make_evidence(row_no: int, elapsed: int, scans: int) -> Evidence:
    return Evidence(
        source_id="unit",
        source_type="db_exec_stats",
        query_id="q001",
        row_no=row_no,
        record_version=2,
        source_format_version=2,
        original_fields=("query_id",),
        padded_fields=(),
        elapsed_raw=elapsed,
        scanned_raw=scans,
        elapsed_value=elapsed,
        scanned_value=scans,
        elapsed_semantics="p95",
        scan_semantics="exact",
        sample_period_ms=100,
    )


class ReducerTests(unittest.TestCase):
    def test_replay_selection_is_stable_and_bounded(self) -> None:
        rows = [make_evidence(index, index, index) for index in range(1, 8)]
        selected = select_replays(rows, 3)
        self.assertEqual([item.row_no for item in selected], [1, 3, 5])
        self.assertEqual(
            [item.row_no for item in select_replays(list(reversed(rows)), 3)],
            [1, 3, 5],
        )

    def test_median_and_trimmed_mean_are_integer_deterministic(self) -> None:
        self.assertEqual(reduce_integers([1, 2, 9, 10], "median"), 6)
        self.assertEqual(reduce_integers([1, 2, 9, 10], "trimmed_mean", 25), 6)
        rows = [
            make_evidence(1, 100, 1000),
            make_evidence(2, 200, 2000),
            make_evidence(3, 900, 9000),
        ]
        reduction = reduce_rows(rows, 3, "trimmed_mean", 0, "elapsed_ms")
        self.assertEqual(reduction.value, 400)
        self.assertEqual(reduction.raw_value, 400)


class PlanVersionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.workspace = Path(self.temp.name)
        self.ledger = Ledger(self.workspace)
        self.raw = default_config_dict()
        self.config = load_config_data(self.raw, self.workspace)

    def tearDown(self) -> None:
        self.temp.cleanup()

    def test_plan_version_bumps_for_each_declared_mutable_parameter(self) -> None:
        decision = decide_plan(self.ledger, self.config)
        self.assertEqual(decision.plan_version, 1)
        commit_plan(self.ledger, decision, self.config)
        self.assertEqual(decide_plan(self.ledger, self.config).plan_version, 1)

        mutations = (
            ("sample_rows", 25),
            ("sample_period_ms", 101),
            ("format_version", 1),
            ("elapsed_semantics", "mean"),
            ("scan_semantics", "sampled"),
        )
        expected = 1
        for field, value in mutations:
            raw = copy.deepcopy(self.raw)
            raw["sources"][0][field] = value
            config = load_config_data(raw, self.workspace)
            decision = decide_plan(self.ledger, config)
            expected += 1
            self.assertEqual(decision.plan_version, expected)
            self.assertIn("app-prod", decision.changed_sources)
            commit_plan(self.ledger, decision, config)


if __name__ == "__main__":
    unittest.main()
