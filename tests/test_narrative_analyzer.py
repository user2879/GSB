from __future__ import annotations

import copy
import json
import tempfile
import unittest
from pathlib import Path

from narrative_analyzer.canonical import canonical_json, pretty_json_bytes
from narrative_analyzer.engine import ReplayOptions, replay
from narrative_analyzer.errors import (
    E_GRAPH_CYCLE,
    E_INJECT_CONFIG,
    E_PATH_DIVERGE,
    E_PRECISION_BOUND,
    E_SELECTION_UNKNOWN,
    E_SNAPSHOT_STALE,
    E_SPEC_INVALID,
    ERROR_CODES,
    BusinessError,
)
from narrative_analyzer.inject import parse_injection, run_injection
from narrative_analyzer.pathcheck import analyze_paths
from narrative_analyzer.samplegen import (
    _yaml_bytes,
    build_demo_artifacts,
    build_demo_bundle,
    sample_spec_value,
)
from narrative_analyzer.sensitivity import ScanConfig, scan_sensitivity
from narrative_analyzer.snapshot import verify_snapshot_file
from narrative_analyzer.spec import parse_spec


PRIMARY = ("path_a", "join", "continue")
BRANCH_B = ("path_b", "join", "continue")


def sample_spec():
    return parse_spec(_yaml_bytes(sample_spec_value()), "test_spec.yaml")


def mutate_spec(mutator):
    value = copy.deepcopy(sample_spec_value())
    mutator(value)
    return _yaml_bytes(value)


class SpecificationValidationTests(unittest.TestCase):
    def assert_code(self, code, func):
        with self.assertRaises(BusinessError) as ctx:
            func()
        self.assertEqual(ctx.exception.code, code)
        return ctx.exception

    def test_valid_spec_has_expected_shape(self):
        spec = sample_spec()
        self.assertEqual(spec.state.components, ("trust", "reputation", "fear"))
        self.assertEqual(len(spec.edges), 6)
        self.assertEqual(len(spec.conditions), 1)
        self.assertEqual(len(spec.coefficients), 8)

    def test_unknown_variable_is_spec_invalid(self):
        data = mutate_spec(lambda value: value["edges"][0]["effects"].__setitem__(0, "trust += ghost"))
        self.assert_code(E_SPEC_INVALID, lambda: parse_spec(data))

    def test_unknown_condition_component_is_spec_invalid(self):
        data = mutate_spec(lambda value: value["conditions"][0].__setitem__("component", "luck"))
        self.assert_code(E_SPEC_INVALID, lambda: parse_spec(data))

    def test_missing_threshold_is_spec_invalid(self):
        data = mutate_spec(lambda value: value["conditions"][0].pop("threshold"))
        error = self.assert_code(E_SPEC_INVALID, lambda: parse_spec(data))
        self.assertIn("threshold", error.message)

    def test_subtraction_expression_is_rejected_by_whitelist(self):
        data = mutate_spec(lambda value: value["edges"][0]["effects"].__setitem__(0, "trust = trust - 0.1"))
        self.assert_code(E_SPEC_INVALID, lambda: parse_spec(data))

    def test_cycle_is_graph_cycle(self):
        def add_cycle(value):
            value["edges"].append(
                {"id": "e_cycle", "source": "merge", "target": "start", "choice": "loop", "effects": []}
            )

        error = self.assert_code(E_GRAPH_CYCLE, lambda: parse_spec(mutate_spec(add_cycle)))
        self.assertIn("cycle_nodes", error.details)

    def test_duplicate_yaml_key_is_spec_invalid(self):
        text = _yaml_bytes(sample_spec_value()).decode("utf-8")
        text = text.replace("spec_version: 1\n", "spec_version: 1\nspec_version: 1\n", 1)
        self.assert_code(E_SPEC_INVALID, lambda: parse_spec(text))


class ReplayTests(unittest.TestCase):
    def test_replay_is_bitwise_and_hash_deterministic(self):
        spec = sample_spec()
        first = replay(spec, PRIMARY)
        second = replay(spec, PRIMARY)
        self.assertEqual(first["final_state"]["bits"], second["final_state"]["bits"])
        self.assertEqual(first["edge_sequence_hash"], second["edge_sequence_hash"])
        self.assertEqual(first["content_hash"], second["content_hash"])
        self.assertEqual(first["steps"], second["steps"])
        self.assertEqual(first["final_state"]["values"], [0.30000000000000004, 0.15000000000000002, 0.03])

    def test_effect_order_is_left_associative_and_declared(self):
        spec = sample_spec()
        first = replay(spec, PRIMARY)
        branch_b = replay(spec, BRANCH_B)
        self.assertEqual(first["final_state"]["bits"][0], "3fd3333333333334")
        self.assertEqual(branch_b["final_state"]["bits"][0], "3fd3333333333333")
        self.assertNotEqual(first["edge_sequence_hash"], branch_b["edge_sequence_hash"])

    def test_boundary_judgment_is_uncertain_and_reported(self):
        spec = sample_spec()
        result = replay(spec, PRIMARY)
        decision = result["decisions"][0]
        self.assertEqual(decision["condition_id"], "c_trust_high")
        self.assertEqual(decision["direction"], ">")
        self.assertEqual(decision["signed_margin"], 5.551115123125783e-17)
        self.assertTrue(decision["boundary"])
        self.assertTrue(decision["uncertain"])
        self.assertTrue(decision["reachable"])
        self.assertGreater(decision["margin_error_upper"], abs(decision["signed_margin"]))
        self.assertLessEqual(decision["state_interval"][0], decision["threshold"])
        self.assertGreaterEqual(decision["state_interval"][1], decision["threshold"])
        self.assertLessEqual(decision["margin_lower"], 0.0)
        self.assertGreaterEqual(decision["margin_upper"], 0.0)

    def test_each_effect_reports_coefficient_and_roundoff_bounds(self):
        result = replay(sample_spec(), PRIMARY)
        effect_count = 0
        for step in result["steps"]:
            for effect in step["effects"]:
                effect_count += 1
                self.assertGreaterEqual(effect["coefficient_error_upper"], 0.0)
                self.assertGreaterEqual(effect["roundoff_error_upper"], 0.0)
                self.assertGreaterEqual(effect["after_error_upper"], 0.0)
        self.assertEqual(effect_count, 5)

    def test_unknown_selection(self):
        with self.assertRaises(BusinessError) as ctx:
            replay(sample_spec(), ("nope",))
        self.assertEqual(ctx.exception.code, E_SELECTION_UNKNOWN)

    def test_precision_bound(self):
        value = sample_spec_value()
        value["numeric"]["max_abs_error"] = 1.0e-16
        spec = parse_spec(_yaml_bytes(value))
        with self.assertRaises(BusinessError) as ctx:
            replay(spec, PRIMARY)
        self.assertEqual(ctx.exception.code, E_PRECISION_BOUND)


class SensitivityTests(unittest.TestCase):
    def test_scan_is_reproducible_and_spec_is_not_mutated(self):
        spec = sample_spec()
        before_values = [coefficient.value for coefficient in spec.coefficients]
        config = ScanConfig((1.0e-15, 1.0e-12, 1.0e-9), direction=1)
        first = scan_sensitivity(spec, BRANCH_B, config)
        second = scan_sensitivity(spec, BRANCH_B, config)
        self.assertEqual(canonical_json(first), canonical_json(second))
        self.assertEqual(before_values, [coefficient.value for coefficient in spec.coefficients])
        self.assertGreaterEqual(first["critical_count"], 1)
        self.assertEqual(first["ranking"][0]["condition_id"], "c_trust_high")
        self.assertEqual(first["ranking"][0]["relative_delta"], 1.0e-15)
        deltas = [entry["relative_delta"] for entry in first["ranking"]]
        self.assertEqual(deltas, sorted(deltas))


class PathAndInjectionTests(unittest.TestCase):
    def test_path_divergence_reports_first_component_and_edge_sequences(self):
        report = analyze_paths(sample_spec())
        self.assertEqual(report["business_code"], E_PATH_DIVERGE)
        first = report["first_divergence"]
        self.assertEqual(first["component"], "trust")
        self.assertEqual(first["left_edge_ids"], ["e_start_a", "e_a_join"])
        self.assertEqual(first["right_edge_ids"], ["e_start_b", "e_b_join"])
        self.assertGreater(first["absolute_difference"], report["path_tolerance"])
        with self.assertRaises(BusinessError) as ctx:
            analyze_paths(sample_spec(), raise_on_diverge=True)
        self.assertEqual(ctx.exception.code, E_PATH_DIVERGE)

    def test_threshold_injection_flips_first_judgment_and_outcome(self):
        spec = sample_spec()
        plan = parse_injection(
            _yaml_bytes(
                {
                    "version": 1,
                    "name": "flip",
                    "coefficient_perturbations": [],
                    "threshold_offsets": [{"condition": "c_trust_high", "delta": -1.0e-9}],
                    "initial_offsets": [],
                }
            )
        )
        first = run_injection(spec, BRANCH_B, plan)
        second = run_injection(spec, BRANCH_B, plan)
        self.assertEqual(canonical_json(first), canonical_json(second))
        self.assertTrue(first["outcome_changed"])
        self.assertEqual(first["baseline"]["outcome"], "bad")
        self.assertEqual(first["injected"]["outcome"], "good")
        self.assertEqual(first["first_flip"]["condition_id"], "c_trust_high")
        self.assertFalse(first["first_flip"]["before"]["reachable"])
        self.assertTrue(first["first_flip"]["after"]["reachable"])
        self.assertEqual(first["first_flip"]["before"]["signed_margin"], 0.0)
        self.assertGreater(first["first_flip"]["after"]["signed_margin"], 0.0)

    def test_coefficient_and_initial_offsets_are_applied_to_copies(self):
        spec = sample_spec()
        base_coefficient = spec.coefficients[3].value
        plan = parse_injection(
            _yaml_bytes(
                {
                    "version": 1,
                    "coefficient_perturbations": [
                        {"coefficient": "e_start_b/effect[0]/coef[0]", "relative": 0.1}
                    ],
                    "threshold_offsets": [],
                    "initial_offsets": [{"component": "fear", "delta": 0.25}],
                }
            )
        )
        report = run_injection(spec, BRANCH_B, plan)
        self.assertEqual(spec.coefficients[3].value, base_coefficient)
        self.assertEqual(spec.state.initial[spec.state.components.index("fear")], 0.0)
        operations = {operation["kind"]: operation for operation in report["effective_operations"]}
        self.assertAlmostEqual(operations["coefficient"]["after"], base_coefficient * 1.1)
        self.assertAlmostEqual(operations["initial"]["after"], 0.25)

    def test_invalid_injection_target(self):
        spec = sample_spec()
        plan = parse_injection(
            _yaml_bytes(
                {
                    "version": 1,
                    "coefficient_perturbations": [{"coefficient": "missing", "relative": 0.1}],
                    "threshold_offsets": [],
                    "initial_offsets": [],
                }
            )
        )
        with self.assertRaises(BusinessError) as ctx:
            run_injection(spec, BRANCH_B, plan)
        self.assertEqual(ctx.exception.code, E_INJECT_CONFIG)


class SnapshotAndDemoTests(unittest.TestCase):
    def test_demo_artifacts_are_byte_reproducible_and_snapshot_verifies(self):
        first = build_demo_bundle()
        second = build_demo_bundle()
        self.assertEqual(first, second)
        summary = json.loads(first["demo_summary.json"])
        self.assertEqual(summary["primary"]["outcome"], "good")
        self.assertEqual(summary["branch_paths"]["business_code"], E_PATH_DIVERGE)
        self.assertTrue(summary["injection_flip"]["outcome_changed"])
        self.assertEqual(summary["error_code_coverage"], list(ERROR_CODES))
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for name, data in first.items():
                (root / name).write_bytes(data)
            verified = verify_snapshot_file(root / "snapshot.json")
            self.assertEqual(verified["artifact_count"], len(first) - 1)
            (root / "demo_summary.json").write_bytes(b"stale\n")
            with self.assertRaises(BusinessError) as ctx:
                verify_snapshot_file(root / "snapshot.json")
            self.assertEqual(ctx.exception.code, E_SNAPSHOT_STALE)

    def test_demo_bundle_has_deterministic_pretty_json(self):
        artifacts = build_demo_bundle()
        self.assertTrue(artifacts["spec.yaml"].endswith(b"\n"))
        self.assertTrue(artifacts["demo_summary.json"].endswith(b"\n"))
        self.assertEqual(pretty_json_bytes(json.loads(artifacts["demo_summary.json"])), artifacts["demo_summary.json"])


if __name__ == "__main__":
    unittest.main()
