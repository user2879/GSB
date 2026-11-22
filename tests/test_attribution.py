import json
import tempfile
import unittest
from pathlib import Path

from configreg.collect import Observation
from configreg.diff import build_diff_report
from configreg.manifest import load_manifest_pair
from configreg.regression import (
    CAUSE_AMPLIFIED,
    CAUSE_DIRECT,
    CAUSE_DISCRETE,
    Budget,
    NodeVerdict,
    attribute,
    evaluate,
)

REGISTRY = {"nodes": [{"node_id": "node_a", "port": 25001}]}


def write(root: Path, name: str, payload: dict) -> Path:
    path = root / name
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def make_pair(root: Path):
    write(root, "registry.json", REGISTRY)
    baseline = {
        "version": 1,
        "strategy": "broadcast",
        "items": {
            "a": {"derived": False, "value": 1.0, "targets": ["node_a"]},
            "d": {"derived": True, "coeffs": {"a": 1.0}, "targets": ["node_a"]},
        },
    }
    candidate = {
        "version": 2,
        "strategy": "broadcast",
        "items": {
            "a": {"derived": False, "value": 3.0, "targets": ["node_a"]},
            "d": {"derived": True, "coeffs": {"a": 2.0}, "targets": ["node_a"]},
        },
    }
    return load_manifest_pair(
        write(root, "b.json", baseline), write(root, "c.json", candidate), root / "registry.json"
    )


class AttributionTests(unittest.TestCase):
    def test_direct_and_amplified_causes(self) -> None:
        with tempfile.TemporaryDirectory() as name:
            baseline, candidate, _ = make_pair(Path(name))
            diff = build_diff_report(baseline, candidate, 4)
            observations = {
                (1, "a", "node_a"): Observation(1, "a", "node_a", {"checksum": 1.0}, 5, 1, 0, 0, 0),
                (2, "a", "node_a"): Observation(2, "a", "node_a", {"checksum": 9.0}, 6, 1, 0, 0, 0),
                (1, "d", "node_a"): Observation(1, "d", "node_a", {"checksum": 1.0}, 5, 1, 0, 0, 0),
                (2, "d", "node_a"): Observation(2, "d", "node_a", {"checksum": 9.0}, 6, 1, 0, 0, 0),
            }
            report = evaluate(baseline, candidate, diff, observations, Budget(50, 0), None)  # type: ignore[arg-type]
            self.assertEqual(report.item_verdicts["a"].attribution.cause, CAUSE_DIRECT)
            self.assertEqual(report.item_verdicts["d"].attribution.cause, CAUSE_AMPLIFIED)
            self.assertEqual(report.item_verdicts["d"].attribution.trigger, "d")
            self.assertEqual(report.item_verdicts["d"].attribution.path, ("d",))

    def test_dispersion_when_only_timing_breaches(self) -> None:
        with tempfile.TemporaryDirectory() as name:
            baseline, candidate, _ = make_pair(Path(name))
            diff = build_diff_report(baseline, candidate, 4)
            node_verdict = NodeVerdict(
                node_id="node_a",
                baseline_present=True,
                candidate_present=True,
                summary_equal=True,
                baseline_elapsed_ms=5,
                candidate_elapsed_ms=80,
                elapsed_delta_ms=75,
                elapsed_breach=True,
                mismatch_fields=("elapsed_ms",),
            )
            attribution = attribute(
                "a",
                baseline,
                candidate,
                diff,
                [node_verdict],
                {"a": 1.0, "d": 1.0},
                {"a": 3.0, "d": 6.0},
            )
            self.assertEqual(attribution.cause, CAUSE_DISCRETE)

    def test_attribution_sorting_is_input_order_independent(self) -> None:
        with tempfile.TemporaryDirectory() as name:
            baseline, candidate, _ = make_pair(Path(name))
            diff = build_diff_report(baseline, candidate, 4)
            first = attribute(
                "d", baseline, candidate, diff, [],
                {"a": 1.0, "d": 1.0}, {"a": 3.0, "d": 6.0},
            )
            second = attribute(
                "d", baseline, candidate, diff, [],
                {"a": 1.0, "d": 1.0}, {"a": 3.0, "d": 6.0},
            )
            self.assertEqual(first.layer_contributions, second.layer_contributions)


if __name__ == "__main__":
    unittest.main()
