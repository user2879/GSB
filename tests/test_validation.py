from __future__ import annotations

import unittest

from mlpipe_workbench.errors import E_SPEC, WorkbenchError
from mlpipe_workbench.loader import prepare_from_objects

from tests.helpers import base_objects


class ValidationTests(unittest.TestCase):
    def assert_spec_error(self, data: dict, pipeline: dict, expected_path: str) -> WorkbenchError:
        with self.assertRaises(WorkbenchError) as captured:
            prepare_from_objects(data, pipeline)
        error = captured.exception
        self.assertEqual(error.code, E_SPEC)
        self.assertIn(expected_path, error.location or "")
        return error

    def test_duplicate_stage_id_is_rejected(self) -> None:
        data, pipeline = base_objects()
        pipeline["stages"][2]["id"] = pipeline["stages"][1]["id"]
        self.assert_spec_error(data, pipeline, "pipeline.stages[2].id")

    def test_missing_upstream_is_rejected(self) -> None:
        data, pipeline = base_objects()
        pipeline["stages"][2]["upstream"] = "missing-stage"
        self.assert_spec_error(data, pipeline, "pipeline.stages[id=normalize_amount].upstream")

    def test_cycle_is_rejected(self) -> None:
        data, pipeline = base_objects()
        pipeline["stages"][0]["upstream"] = "pack"
        error = self.assert_spec_error(data, pipeline, "upstream")
        self.assertEqual(error.code, E_SPEC)

    def test_null_bitmap_length_is_rejected(self) -> None:
        data, pipeline = base_objects()
        data["sources"][0]["columns"][1]["nulls"].append(0)
        self.assert_spec_error(data, pipeline, "data.sources[0].columns[1].nulls")

    def test_unsupported_column_type_is_rejected(self) -> None:
        data, pipeline = base_objects()
        data["sources"][0]["columns"][0]["type"] = "string"
        self.assert_spec_error(data, pipeline, "data.sources[0].columns[0].type")

    def test_parameter_bounds_are_rejected(self) -> None:
        data, pipeline = base_objects()
        pipeline["stages"][3]["params"]["max_outlier_permille"] = 1001
        error = self.assert_spec_error(
            data,
            pipeline,
            "pipeline.stages[id=clip_amount].params.max_outlier_permille",
        )
        self.assertEqual(error.param, "max_outlier_permille")
        self.assertEqual(error.param_value, 1001)

    def test_non_integer_parameter_is_rejected(self) -> None:
        data, pipeline = base_objects()
        pipeline["stages"][3]["params"]["column"] = True
        self.assert_spec_error(data, pipeline, "pipeline.stages[3].params.column")

    def test_branching_chain_is_rejected(self) -> None:
        data, pipeline = base_objects()
        pipeline["stages"][3]["upstream"] = "fill_missing"
        self.assert_spec_error(data, pipeline, "pipeline.stages[id=fill_missing]")

    def test_stage_types_must_remain_ordered(self) -> None:
        data, pipeline = base_objects()
        pipeline["stages"][2], pipeline["stages"][3] = pipeline["stages"][3], pipeline["stages"][2]
        pipeline["stages"][2]["upstream"] = "fill_missing"
        pipeline["stages"][3]["upstream"] = "clip_amount"
        pipeline["stages"][4]["upstream"] = "normalize_amount"
        self.assert_spec_error(data, pipeline, "pipeline.stages[id=normalize_amount].type")

    def test_no_partial_object_is_returned(self) -> None:
        data, pipeline = base_objects()
        pipeline["stages"][0]["version"] = 0
        with self.assertRaises(WorkbenchError):
            prepare_from_objects(data, pipeline)


if __name__ == "__main__":
    unittest.main()
