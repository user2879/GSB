from __future__ import annotations

import copy
import unittest

from mvcc_workbench.errors import CycleError, SpecError
from mvcc_workbench.loader import validate_spec_object
from mvcc_workbench.samples import large_sample, shuffled_copy, small_sample


class LoaderTests(unittest.TestCase):
    def test_duplicate_transaction_is_rejected_with_identifier(self) -> None:
        data = small_sample()
        data["transactions"].append(copy.deepcopy(data["transactions"][0]))
        with self.assertRaises(SpecError) as raised:
            validate_spec_object(data)
        self.assertEqual(raised.exception.code, "E_SPEC")
        self.assertEqual(raised.exception.details["entity_id"], "T0001")

    def test_commit_sequence_rules(self) -> None:
        data = small_sample()
        data["transactions"][0]["commit_seq"] = data["transactions"][0]["start_seq"] - 1
        with self.assertRaises(SpecError) as raised:
            validate_spec_object(data)
        self.assertEqual(raised.exception.details["entity_id"], "T0001")
        self.assertEqual(raised.exception.details["field"], "commit_seq")

    def test_rolledback_must_not_have_commit_sequence(self) -> None:
        data = small_sample()
        target = next(tx for tx in data["transactions"] if tx["outcome"] == "rolledback")
        target["commit_seq"] = target["start_seq"] + 2
        with self.assertRaises(SpecError) as raised:
            validate_spec_object(data)
        self.assertEqual(raised.exception.details["entity_id"], target["id"])

    def test_missing_previous_version_reports_row_key(self) -> None:
        data = small_sample()
        data["records"][1]["prev_version"] = 999
        with self.assertRaises(SpecError) as raised:
            validate_spec_object(data)
        self.assertEqual(raised.exception.code, "E_SPEC")
        self.assertEqual(raised.exception.details["entity_id"], data["records"][1]["row_key"])

    def test_cycle_reports_e_cycle(self) -> None:
        data = small_sample()
        row_key = data["records"][0]["row_key"]
        row_records = [record for record in data["records"] if record["row_key"] == row_key]
        row_records[0]["prev_version"] = row_records[1]["version"]
        row_records[1]["prev_version"] = row_records[0]["version"]
        with self.assertRaises(CycleError) as raised:
            validate_spec_object(data)
        self.assertEqual(raised.exception.code, "E_CYCLE")
        self.assertEqual(raised.exception.details["row_key"], row_key)

    def test_query_references_are_validated(self) -> None:
        data = small_sample()
        data["queries"][0]["row_key"] = "missing-row"
        with self.assertRaises(SpecError) as raised:
            validate_spec_object(data)
        self.assertEqual(raised.exception.details["entity_id"], "Q0001")
        self.assertEqual(raised.exception.details["row_key"], "missing-row")

    def test_array_permutation_produces_identical_normalized_spec(self) -> None:
        original = large_sample()
        shuffled = shuffled_copy(original)
        left = validate_spec_object(original)
        right = validate_spec_object(shuffled)
        self.assertEqual(left.to_dict(), right.to_dict())


if __name__ == "__main__":
    unittest.main()
