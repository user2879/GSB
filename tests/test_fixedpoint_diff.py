import json
import tempfile
import unittest
from pathlib import Path

from configreg.canon import canonical_bytes
from configreg.diff import (
    ADDED,
    COEFF_CHANGED,
    REMOVED,
    TARGET_CHANGED,
    VALUE_CHANGED,
    build_diff_report,
)
from configreg.fixedpoint import SAT_MAX, make_spec, quantize, quantize_manifest, sat_mul_q
from configreg.manifest import load_manifest_pair

REGISTRY = {"nodes": [{"node_id": "node_a", "port": 23001}, {"node_id": "node_b", "port": 23002}]}


def write(tmp: Path, name: str, payload: dict) -> Path:
    path = tmp / name
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


class FixedPointTests(unittest.TestCase):
    def test_truncation_toward_zero_and_saturation(self) -> None:
        spec = make_spec(3)
        self.assertEqual(quantize(1.2349, spec), 1234)
        self.assertEqual(quantize(-1.2349, spec), -1234)
        self.assertEqual(quantize(10**30, spec), SAT_MAX)
        self.assertEqual(sat_mul_q(quantize(2.5, spec), quantize(3.5, spec), spec), quantize(8.75, spec))

    def test_derived_integer_pipeline(self) -> None:
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            write(root, "registry.json", REGISTRY)
            manifest = {
                "version": 1,
                "strategy": "broadcast",
                "items": {
                    "x": {"derived": False, "value": 1.25, "targets": ["node_a"]},
                    "y": {"derived": True, "coeffs": {"x": 2.5}, "const": 0.25, "targets": ["node_a"]},
                },
            }
            path = write(root, "m.json", manifest)
            from configreg.manifest import load_manifest, load_registry

            loaded = load_manifest(path, load_registry(root / "registry.json"))
            q = quantize_manifest(loaded, make_spec(4))
            self.assertEqual(q.item_q["x"], 12500)
            self.assertEqual(q.item_q["y"], 33750)


class DiffDeterminismTests(unittest.TestCase):
    def _pair(self, root: Path):
        write(root, "registry.json", REGISTRY)
        baseline = {
            "version": 1,
            "strategy": "rolling",
            "items": {
                "a": {"derived": False, "value": 1.0, "targets": ["node_a"]},
                "gone": {"derived": False, "value": 2.0, "targets": ["node_a"]},
                "d": {"derived": True, "coeffs": {"a": 1.0}, "targets": ["node_a"]},
            },
        }
        candidate = {
            "version": 2,
            "strategy": "rolling",
            "items": {
                "a": {"derived": False, "value": 1.00001, "targets": ["node_a", "node_b"]},
                "fresh": {"derived": False, "value": 1.0, "targets": ["node_b"]},
                "d": {"derived": True, "coeffs": {"a": 2.0}, "targets": ["node_a"]},
            },
        }
        base = write(root, "baseline.json", baseline)
        cand = write(root, "candidate.json", candidate)
        return load_manifest_pair(base, cand, root / "registry.json")

    def test_change_types_impact_chain_and_byte_stability(self) -> None:
        with tempfile.TemporaryDirectory() as name:
            baseline, candidate, _ = self._pair(Path(name))
            first = build_diff_report(baseline, candidate, 4)
            second = build_diff_report(baseline, candidate, 4)
            self.assertEqual(canonical_bytes(first.to_dict()), canonical_bytes(second.to_dict()))
            self.assertEqual(first.report_digest, second.report_digest)
            self.assertIn(VALUE_CHANGED, first.item_diffs["a"].change_types)
            self.assertIn(TARGET_CHANGED, first.item_diffs["a"].change_types)
            self.assertEqual(first.item_diffs["fresh"].change_types, (ADDED,))
            self.assertEqual(first.item_diffs["gone"].change_types, (REMOVED,))
            self.assertIn(COEFF_CHANGED, first.item_diffs["d"].change_types)
            downstream = [entry["item"] for entry in first.impact_chains["a"].downstream]
            self.assertEqual(downstream, ["d"])

    def test_indistinguishable_under_error_bound(self) -> None:
        with tempfile.TemporaryDirectory() as name:
            baseline, candidate, _ = self._pair(Path(name))
            report = build_diff_report(baseline, candidate, 4)
            diff = report.item_diffs["a"]
            self.assertTrue(diff.indistinguishable)
            self.assertIsNotNone(diff.indistinguishable_reason)
            self.assertGreaterEqual(diff.error_bound, diff.abs_diff)


if __name__ == "__main__":
    unittest.main()
