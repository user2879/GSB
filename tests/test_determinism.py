from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from tests.helpers import run_audit


def _core_payload(payload: dict, include_evaluation: bool = True) -> dict:
    result = {
        "step": payload["step"],
        "operator_type": payload["operator_type"],
        "execution_count": payload["execution_count"],
        "fixed": payload["mesh"]["fixed"],
        "reference": payload["mesh"]["reference"],
        "faces": payload["mesh"]["faces"],
        "sources": payload["error"]["direct_source_vertices"],
        "exceeding": payload["error"]["exceeding_vertices"],
        "error_max": payload["error"]["max_abs_error"],
        "operator_effect": payload["operator_effect"],
        "propagation": [],
    }
    for trace in payload["propagation_sources"]:
        result["propagation"].append(
            {
                "source": trace["source_vertex"],
                "layers": [
                    {
                        "layer": layer["layer"],
                        "vertices": layer["vertices"],
                        "max": layer["max_abs_error"],
                    }
                    for layer in trace["layers"]
                ],
            }
        )
    if include_evaluation:
        result["propagation_evaluation"] = payload["propagation"]["evaluation"]
    return result


class DeterminismTests(unittest.TestCase):
    def test_repeated_runs_are_vertex_identical(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            first, _ = run_audit(root, state_name="state-a", output_name="output-a")
            second, _ = run_audit(root, state_name="state-b", output_name="output-b")
            self.assertEqual(first.version_id, second.version_id)
            self.assertEqual(len(first.report["steps"]), len(second.report["steps"]))
            for left, right in zip(first.report["steps"], second.report["steps"]):
                left_payload = self._record_payload(first, left["step"])
                right_payload = self._record_payload(second, right["step"])
                self.assertEqual(
                    _core_payload(left_payload),
                    _core_payload(right_payload),
                    f"step {left['step']} is not deterministic",
                )

    def _record_payload(self, result, step: int) -> dict:
        record = next(
            record
            for record in result.state_inspection.snapshots
            if record.step == step
        )
        self.assertIsNotNone(record.payload)
        return record.payload

    def test_first_tolerance_exceedance_is_fixed_point_authoritative(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            result, _ = run_audit(Path(temporary))
            summary = result.report["summary"]
            exceedance = summary["first_tolerance_exceedance"]
            self.assertIsNotNone(exceedance)
            self.assertEqual(exceedance["operator_index"], 4)
            self.assertEqual(exceedance["vertex"], 0)
            self.assertGreater(exceedance["max_abs_error"], exceedance["tolerance"])
            verdict = summary["fixed_point_verdict"]
            self.assertEqual(verdict["status"], "FAIL_FIXED_POINT_AUDIT")
            self.assertEqual(verdict["decision_basis"], "fixed_point_only")
            self.assertFalse(verdict["float64_reference_used_for_verdict"])
            smoothing = result.report["steps"][4]
            self.assertIn("W_QUANTIZATION_TOLERANCE_EXCEEDED", smoothing["business_codes"])
            self.assertTrue(smoothing["propagation"]["source_vertices"])
            self.assertGreaterEqual(smoothing["propagation"]["max_reached_layer"], 1)


if __name__ == "__main__":
    unittest.main()
