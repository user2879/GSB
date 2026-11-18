from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from mixedprec_diag.deterministic import (
    generate_artifacts,
    load_expected_manifest,
    verify_artifacts,
    write_manifest,
)
from mixedprec_diag.errors import E_DATA, DiagnosticError


class DeterministicTests(unittest.TestCase):
    def test_same_arguments_are_bitwise_identical(self) -> None:
        left = generate_artifacts(1234, 12, 8, 16)
        right = generate_artifacts(1234, 12, 8, 16)
        self.assertEqual(left.data_sha256, right.data_sha256)
        self.assertEqual(left.params_sha256, right.params_sha256)
        np.testing.assert_array_equal(left.dataset.inputs, right.dataset.inputs)
        for left_weight, right_weight in zip(left.model.weights, right.model.weights):
            np.testing.assert_array_equal(left_weight, right_weight)
        for left_bias, right_bias in zip(left.model.biases, right.model.biases):
            np.testing.assert_array_equal(left_bias, right_bias)

    def test_different_seed_changes_artifacts(self) -> None:
        left = generate_artifacts(1, 8, 6, 12)
        right = generate_artifacts(2, 8, 6, 12)
        self.assertNotEqual(left.data_sha256, right.data_sha256)
        self.assertNotEqual(left.params_sha256, right.params_sha256)

    def test_manifest_match_and_mismatch_location(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            artifacts = generate_artifacts(77, 6, 5, 9)
            manifest_path = directory / "manifest.json"
            write_manifest(artifacts, manifest_path)
            manifest = load_expected_manifest(manifest_path)
            status = verify_artifacts(artifacts, manifest)
            self.assertEqual(status["status"], "MATCH")

            mismatch = dict(manifest)
            mismatch["data_sha256"] = "0" * 64
            with self.assertRaises(DiagnosticError) as raised:
                verify_artifacts(artifacts, mismatch)
            self.assertEqual(raised.exception.code, E_DATA)
            self.assertEqual(raised.exception.field_path, "$.data_sha256")
            self.assertEqual(raised.exception.details["mismatches"][0]["artifact"], "dataset")
            self.assertIn("dataset.inputs", raised.exception.details["mismatches"][0]["field_path"])

    def test_report_digest_scope_is_serializable(self) -> None:
        artifacts = generate_artifacts(9, 4, 5, 7)
        payload = artifacts.summary()
        json.dumps(payload, allow_nan=False, sort_keys=True)
        self.assertEqual(len(artifacts.data_sha256), 64)
        self.assertEqual(len(artifacts.params_sha256), 64)


if __name__ == "__main__":
    unittest.main()
