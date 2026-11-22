from __future__ import annotations

import copy
import unittest

from qplan.errors import QPlanError
from qplan.spec import parse_spec

from tests.helpers import small_spec_dict


class SpecValidationTests(unittest.TestCase):
    def test_first_violation_location_is_reported(self) -> None:
        data = small_spec_dict()
        data["tables"][0]["columns"][1]["type"] = "text"
        with self.assertRaises(QPlanError) as context:
            parse_spec(data)
        error = context.exception
        self.assertEqual(error.code, "E_SPEC")
        self.assertEqual(error.location, "tables[0].columns[1].type")

    def test_duplicate_index_column_rejected(self) -> None:
        data = small_spec_dict()
        data["tables"][0]["candidate_indexes"][0]["columns"] = ["region", "region"]
        with self.assertRaises(QPlanError) as context:
            parse_spec(data)
        self.assertEqual(context.exception.code, "E_SPEC")
        self.assertEqual(context.exception.location, "tables[0].candidate_indexes[0].columns[1]")

    def test_unknown_query_table_rejected(self) -> None:
        data = small_spec_dict()
        data["queries"][0]["tables"][1] = "missing_table"
        with self.assertRaises(QPlanError) as context:
            parse_spec(data)
        self.assertEqual(context.exception.code, "E_SPEC")
        self.assertEqual(context.exception.location, "queries[0].tables[1]")

    def test_unsupported_filter_operator_rejected(self) -> None:
        data = small_spec_dict()
        data["queries"][0]["filters"][0]["op"] = "like"
        with self.assertRaises(QPlanError) as context:
            parse_spec(data)
        self.assertEqual(context.exception.code, "E_SPEC")
        self.assertEqual(context.exception.location, "queries[0].filters[0].op")

    def test_bad_between_value_rejected(self) -> None:
        data = small_spec_dict()
        data["queries"][1]["filters"][1]["value"] = [30, 20]
        with self.assertRaises(QPlanError) as context:
            parse_spec(data)
        self.assertEqual(context.exception.code, "E_SPEC")
        self.assertEqual(context.exception.location, "queries[1].filters[1].value")

    def test_valid_spec_digest_is_stable(self) -> None:
        first = parse_spec(copy.deepcopy(small_spec_dict()))
        second = parse_spec(copy.deepcopy(small_spec_dict()))
        self.assertEqual(first.digest, second.digest)


if __name__ == "__main__":
    unittest.main()

