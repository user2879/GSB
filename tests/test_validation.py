from __future__ import annotations

import unittest

from mvcc_workbench.errors import DiagnosticError
from mvcc_workbench.loader import dataset_from_dict, load_dataset
from mvcc_workbench.visibility import compute_visibility

from tests.helpers import base_dataset, deepcopy, sample


class ValidationTests(unittest.TestCase):
    def test_valid_samples_load(self) -> None:
        dataset = load_dataset("samples/base.json")
        self.assertEqual(len(dataset.transactions), 10)
        self.assertEqual(len(dataset.chains), 3)
        self.assertEqual(len(dataset.queries), 8)
        self.assertEqual(dataset.manifest.txn_set_version, 1)

    def test_chain_cycle_is_rejected_with_location(self) -> None:
        with self.assertRaises(DiagnosticError) as caught:
            dataset_from_dict(sample("invalid_chain_cycle.json"))
        error = caught.exception
        self.assertEqual(error.code, "E_CHAIN")
        self.assertEqual(error.location["row_key"], "acct:3")
        self.assertEqual(error.location["field"], "prev_version")
        self.assertEqual(error.details["cycle"][0], 1)

    def test_commit_regression_is_rejected_with_first_txn_id(self) -> None:
        with self.assertRaises(DiagnosticError) as caught:
            dataset_from_dict(sample("invalid_txn_commit_regression.json"))
        error = caught.exception
        self.assertEqual(error.code, "E_TXN")
        self.assertEqual(error.location["txn_id"], "T4")
        self.assertEqual(error.location["field"], "commit_seq")

    def test_invalid_query_is_rejected_as_e_query(self) -> None:
        payload = deepcopy(sample("base.json"))
        payload["queries"][0]["row_key"] = "missing:row"
        with self.assertRaises(DiagnosticError) as caught:
            dataset_from_dict(payload)
        self.assertEqual(caught.exception.code, "E_QUERY")
        self.assertEqual(caught.exception.location["query_id"], "Q01")

    def test_duplicate_transaction_and_rolledback_commit_are_rejected(self) -> None:
        duplicate = deepcopy(sample("base.json"))
        duplicate["transactions"].append(deepcopy(duplicate["transactions"][0]))
        with self.assertRaises(DiagnosticError) as caught:
            dataset_from_dict(duplicate)
        self.assertEqual(caught.exception.code, "E_TXN")
        self.assertEqual(caught.exception.location["txn_id"], "T1")

        rolledback = deepcopy(sample("base.json"))
        for txn in rolledback["transactions"]:
            if txn["txn_id"] == "T5":
                txn["commit_seq"] = 9
        with self.assertRaises(DiagnosticError) as caught:
            dataset_from_dict(rolledback)
        self.assertEqual(caught.exception.code, "E_TXN")
        self.assertEqual(caught.exception.location["txn_id"], "T5")

    def test_record_order_does_not_change_reason_chain(self) -> None:
        first = base_dataset()
        payload = sample("base.json")
        payload["records"].reverse()
        second = dataset_from_dict(payload)
        for query_id in sorted(first.queries_by_id):
            left = compute_visibility(first, first.queries_by_id[query_id])
            right = compute_visibility(second, second.queries_by_id[query_id])
            self.assertEqual(left.visible_version_no, right.visible_version_no)
            self.assertEqual(left.reason_digest, right.reason_digest)


if __name__ == "__main__":
    unittest.main()
