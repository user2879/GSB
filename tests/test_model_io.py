from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from tabular_lab.errors import (
    E_CONFIG_INVALID,
    E_FIELD_TRUNCATED,
    E_MIX_RANGE,
    E_SCALE_RANGE,
    E_SCHEME_UNKNOWN,
    LabError,
)
from tabular_lab.model_io import (
    V1_LINE_WIDTH,
    compare_records,
    parse_v1_text,
    parse_v2_payload,
    read_model_file,
    write_downgrade_sidecar,
    write_v1,
    write_v2,
)


ROOT = Path(__file__).resolve().parents[1]


def v1_line(name: str, index: int, coefficient: int, code: int, intercept: int) -> str:
    return f"{name:<20}{index:>4}{coefficient:>12}{code:>2}{intercept:>12}"


class ModelIOTests(unittest.TestCase):
    def test_sample_v1_and_v2_parse_to_common_representation(self) -> None:
        v2 = read_model_file(ROOT / "samples/models.v2.json")
        self.assertEqual([record.name for record in v2], ["sample_ridge", "sample_lasso", "sample_elastic"])
        self.assertEqual(v2[0].precision_label, "float64-v2-sample")
        self.assertEqual(v2[2].scheme, "elastic_net")
        self.assertEqual(v2[2].mix, 0.55)

        v1 = read_model_file(ROOT / "samples/models.v1.txt", sidecar_dir=ROOT / "samples/sidecars")
        self.assertEqual([record.name for record in v1], ["sample_ridge", "sample_lasso", "sample_elastic"])
        self.assertEqual(v1[0].coefficients, [0.422, -0.183, 0.0])
        self.assertEqual(v1[1].convergence_status, "max_epochs")
        self.assertEqual(v1[2].mix, 0.55)
        self.assertIn("downgrade_sidecar", v1[0].metadata)

    def test_v1_auto_sidecar_discovery(self) -> None:
        records = read_model_file(ROOT / "samples/models.v1.txt")
        self.assertEqual(records[2].penalty, 0.04)
        self.assertEqual(records[2].mix, 0.55)

    def test_v2_to_v1_roundtrip_preserves_precision_within_tolerance(self) -> None:
        source = read_model_file(ROOT / "samples/models.v2.json")
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            sidecars = root / "sidecars"
            v1_path = root / "models.v1.txt"
            write_v1(source, v1_path)
            for record in source:
                write_downgrade_sidecar(record, sidecars)

            for line in v1_path.read_text(encoding="utf-8").splitlines():
                self.assertEqual(len(line), V1_LINE_WIDTH)
            self.assertEqual(len(list(sidecars.glob("*.downgrade.json"))), 3)

            parsed = read_model_file(v1_path)
            comparison = compare_records(source, parsed)
            self.assertTrue(comparison["equal_names"])
            self.assertLessEqual(comparison["max_abs_coefficient_delta"], 0.001)
            self.assertLessEqual(comparison["max_abs_intercept_delta"], 0.001)
            self.assertEqual(parsed[2].mix, 0.55)

            sidecar = json.loads(next(sidecars.glob("sample_elastic*.json")).read_text(encoding="utf-8"))
            self.assertEqual(sidecar["precision_label"], "float64-v2-sample")
            self.assertEqual(sidecar["discarded_names"]["scheme"], "elastic_net")
            self.assertEqual(len(sidecar["coefficients_residual"]), 3)
            self.assertTrue(any(abs(value) > 0 for value in sidecar["coefficients_residual"]))

            roundtrip_v2 = root / "roundtrip.v2.json"
            write_v2(parsed, roundtrip_v2)
            reloaded = read_model_file(roundtrip_v2)
            second_comparison = compare_records(parsed, reloaded)
            self.assertEqual(second_comparison["max_abs_coefficient_delta"], 0.0)
            self.assertEqual(reloaded[2].precision_label, "float64-v2-sample")

    def test_v1_business_error_codes(self) -> None:
        with self.assertRaises(LabError) as context:
            parse_v1_text(v1_line("short", 0, 100, 11, 0)[: V1_LINE_WIDTH - 1])
        self.assertEqual(context.exception.code, E_FIELD_TRUNCATED)

        with self.assertRaises(LabError) as context:
            parse_v1_text(v1_line("scale", 0, 1_000_000_000, 11, 0))
        self.assertEqual(context.exception.code, E_SCALE_RANGE)

        with self.assertRaises(LabError) as context:
            parse_v1_text(v1_line("scheme", 0, 100, 99, 0))
        self.assertEqual(context.exception.code, E_SCHEME_UNKNOWN)

        with self.assertRaises(LabError) as context:
            parse_v1_text(v1_line("bad", 0, 100, 11, 0) + v1_line("bad", 0, 200, 11, 0))
        self.assertEqual(context.exception.code, E_CONFIG_INVALID)

        sparse = parse_v1_text(v1_line("sparse", 0, 100, 11, 0) + "\n" + v1_line("sparse", 2, 250, 11, 0))
        self.assertEqual(sparse[0].feature_indices, [0, 2])
        self.assertEqual(sparse[0].coefficients, [0.1, 0.25])

    def test_v2_business_error_codes(self) -> None:
        base = {
            "name": "bad",
            "scheme": "unknown_scheme",
            "coefficients": [0.1],
            "feature_indices": [0],
            "intercept": 0.0,
            "convergence_status": "converged",
            "precision_label": "float64",
        }
        with self.assertRaises(LabError) as context:
            parse_v2_payload({"models": [base]})
        self.assertEqual(context.exception.code, E_SCHEME_UNKNOWN)

        mix_bad = dict(base, scheme="elastic_net", mix=1.2)
        with self.assertRaises(LabError) as context:
            parse_v2_payload({"models": [mix_bad]})
        self.assertEqual(context.exception.code, E_MIX_RANGE)

        mix_nan = dict(base, scheme="elastic_net", mix=float("nan"))
        with self.assertRaises(LabError) as context:
            parse_v2_payload({"models": [mix_nan]})
        self.assertEqual(context.exception.code, E_MIX_RANGE)

        non_finite = dict(base, scheme="l2", coefficients=[float("nan")])
        with self.assertRaises(LabError) as context:
            parse_v2_payload({"models": [non_finite]})
        self.assertEqual(context.exception.code, E_CONFIG_INVALID)

        with self.assertRaises(LabError) as context:
            read_model_file(
                ROOT / "samples/models.v2.json",
                known_feature_names=["income", "tenure"],
            )
        self.assertEqual(context.exception.code, E_CONFIG_INVALID)


if __name__ == "__main__":
    unittest.main()
