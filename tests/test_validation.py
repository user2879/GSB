from __future__ import annotations

import copy
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from colpage_recovery.errors import (
    CommitMarkerConflictError,
    RangeOutOfBoundsError,
    WalValidationError,
)
from colpage_recovery.loader import build_bundle, load_json_file


ROOT = Path(__file__).resolve().parents[1]
SAMPLES = ROOT / "samples"


class ValidationTest(unittest.TestCase):
    def setUp(self) -> None:
        self.pages = load_json_file(SAMPLES / "pages.json")
        self.wal = load_json_file(SAMPLES / "wal.json")
        self.budget = load_json_file(SAMPLES / "budget.json")

    def test_sample_is_valid(self) -> None:
        bundle = build_bundle(self.pages, self.wal, self.budget)
        self.assertEqual(8, len(bundle.entries))
        self.assertEqual(2, len(bundle.pages))

    def test_duplicate_lsn_reports_first_lsn_and_field(self) -> None:
        wal = copy.deepcopy(self.wal)
        wal["entries"][2]["lsn"] = wal["entries"][1]["lsn"]
        with self.assertRaises(WalValidationError) as ctx:
            build_bundle(self.pages, wal, self.budget)
        self.assertEqual("E_WAL", ctx.exception.code)
        self.assertEqual("lsn", ctx.exception.field)
        self.assertEqual(2, ctx.exception.lsn)

    def test_duplicate_page_id_rejected(self) -> None:
        pages = copy.deepcopy(self.pages)
        pages["pages"][1]["page_id"] = pages["pages"][0]["page_id"]
        with self.assertRaises(WalValidationError) as ctx:
            build_bundle(pages, self.wal, self.budget)
        self.assertEqual("page_id", ctx.exception.field)

    def test_positive_row_count_is_enforced(self) -> None:
        pages = copy.deepcopy(self.pages)
        pages["pages"][0]["row_count"] = 0
        with self.assertRaises(WalValidationError) as ctx:
            build_bundle(pages, self.wal, self.budget)
        self.assertIn("row_count", ctx.exception.field)

    def test_range_error_code_and_location(self) -> None:
        wal = copy.deepcopy(self.wal)
        wal["entries"][0]["row_range"] = [0, 99]
        with self.assertRaises(RangeOutOfBoundsError) as ctx:
            build_bundle(self.pages, wal, self.budget)
        self.assertEqual("E_RANGE", ctx.exception.code)
        self.assertEqual(1, ctx.exception.lsn)
        self.assertEqual("row_range", ctx.exception.field)

    def test_commit_conflict_code_and_location(self) -> None:
        wal = copy.deepcopy(self.wal)
        wal["entries"][1]["commit_flag"] = False
        with self.assertRaises(CommitMarkerConflictError) as ctx:
            build_bundle(self.pages, wal, self.budget)
        self.assertEqual("E_COMMIT", ctx.exception.code)
        self.assertEqual(2, ctx.exception.lsn)
        self.assertEqual("commit_flag", ctx.exception.field)

    def test_transaction_grouping_required(self) -> None:
        wal = copy.deepcopy(self.wal)
        entries = wal["entries"]
        entries[1]["tx_id"] = "T2"
        entries[2]["tx_id"] = "T1"
        with self.assertRaises(WalValidationError) as ctx:
            build_bundle(self.pages, wal, self.budget)
        self.assertEqual("tx_id", ctx.exception.field)

    def test_wal_strict_rejects_uncommitted_persisted_pair(self) -> None:
        budget = copy.deepcopy(self.budget)
        budget["marker_protocol"] = "wal-strict"
        with self.assertRaises(WalValidationError) as ctx:
            build_bundle(self.pages, self.wal, budget)
        self.assertEqual("persist_flag", ctx.exception.field)
        self.assertEqual(6, ctx.exception.lsn)


if __name__ == "__main__":
    unittest.main()
