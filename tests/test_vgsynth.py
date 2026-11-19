from __future__ import annotations

import copy
import tempfile
import unittest
from pathlib import Path

import numpy as np

from vgsynth.canonical import canonical_json_bytes, read_json
from vgsynth.composite import layer_content_hash, render_path_layer
from vgsynth.errors import ErrorCode
from vgsynth.flatten import flatten_scene
from vgsynth.inject import apply_injections, default_render_params
from vgsynth.pipeline import compare_runs, demo_cases, run_case, run_demo
from vgsynth.rng import replay_float, replay_int
from vgsynth.scene import generate_scene, normalized_scene_for_render, validate_scene, validate_tolerance
from vgsynth.verifier import verify_case_dir


class SceneAndFlattenTests(unittest.TestCase):
    def test_scene_bytes_and_seed_audit_are_replayable(self) -> None:
        first = generate_scene("simple_cross", 0x1234567890, "audit-scene")
        second = generate_scene("simple_cross", 0x1234567890, "audit-scene")
        self.assertEqual(canonical_json_bytes(first), canonical_json_bytes(second))
        audit = first["seed_audit"]
        self.assertGreater(audit["entry_count"], 0)
        for entry in audit["entries"]:
            if entry["algorithm"] == "splitmix64-mod":
                replayed = replay_int(first["seed"], entry["decision_id"], entry["upper"])
                self.assertEqual(replayed, entry["value"])
            else:
                replayed = replay_float(
                    first["seed"],
                    entry["decision_id"],
                    float(entry["low"]),
                    float(entry["high"]),
                    int(entry["digits"]),
                )
                self.assertEqual(replayed, entry["value"])

    def test_flatten_is_deterministic_and_monotonic(self) -> None:
        for kind in ("self_intersect_loop", "overlapping_large"):
            scene = normalized_scene_for_render(generate_scene(kind, 991, kind))
            loose = flatten_scene(scene, 0.5)
            loose_again = flatten_scene(scene, 0.5)
            self.assertEqual(canonical_json_bytes(loose), canonical_json_bytes(loose_again))
            tight = flatten_scene(scene, 0.1)
            for loose_path, tight_path in zip(loose["paths"], tight["paths"]):
                for loose_subpath, tight_subpath in zip(loose_path["subpaths"], tight_path["subpaths"]):
                    self.assertGreaterEqual(tight_subpath["vertex_count"], loose_subpath["vertex_count"])
                for subpath in loose_path["subpaths"]:
                    for segment in subpath["segments"]:
                        if not segment["action"].startswith(("skip:", "degrade:")):
                            self.assertLessEqual(segment["error_estimate"], 0.5 + 1e-12)
        strict = flatten_scene(scene, 0.05)
        for tight_path, strict_path in zip(tight["paths"], strict["paths"]):
            for tight_subpath, strict_subpath in zip(tight_path["subpaths"], strict_path["subpaths"]):
                self.assertGreaterEqual(strict_subpath["vertex_count"], tight_subpath["vertex_count"])

    def test_degenerate_segments_have_declared_actions(self) -> None:
        scene = normalized_scene_for_render(generate_scene("degenerate_segments", 17, "degenerate"))
        flat = flatten_scene(scene, 0.25)
        actions = [
            segment["action"]
            for path in flat["paths"]
            for subpath in path["subpaths"]
            for segment in subpath["segments"]
        ]
        self.assertIn("skip:zero-length-line", actions)
        self.assertIn("degrade:collinear-cubic-to-line", actions)
        self.assertGreaterEqual(flat["skipped_segment_count"], 1)
        self.assertGreaterEqual(flat["degraded_segment_count"], 1)


class RasterAndCompositeTests(unittest.TestCase):
    def test_two_fill_rules_are_deterministic_and_locatable(self) -> None:
        scene = normalized_scene_for_render(generate_scene("self_intersect_loop", 42, "star"))
        flat = flatten_scene(scene, 0.25)
        first = render_path_layer("L0000", scene["paths"][0], flat["paths"][0], 192, 144)
        second = render_path_layer("L0000", scene["paths"][0], flat["paths"][0], 192, 144)
        self.assertEqual(first["layer_hash"], second["layer_hash"])
        self.assertTrue(np.array_equal(first["fill"]["nonzero"], second["fill"]["nonzero"]))
        self.assertTrue(np.array_equal(first["fill"]["evenodd"], second["fill"]["evenodd"]))
        difference = first["fill"]["first_difference"]
        self.assertIsNotNone(difference)
        self.assertGreater(difference["count"], 0)
        self.assertEqual(first["fill"]["selected_rule"], "evenodd")
        self.assertGreater(int(first["stroke"]["coverage"].max()), 0)
        self.assertEqual(first["fill"]["nonzero"].dtype, np.uint8)

    def test_layer_hash_changes_when_content_is_replaced(self) -> None:
        scene = normalized_scene_for_render(generate_scene("simple_cross", 9, "hash"))
        flat = flatten_scene(scene, 0.25)
        layer = render_path_layer("L0000", scene["paths"][0], flat["paths"][0], 192, 144)
        changed = layer["pixels"].copy()
        changed[:, :, 3] = (changed[:, :, 3].astype(np.uint16) + 1) % 256
        changed_hash = layer_content_hash(layer["header"], changed.astype(np.uint8))
        self.assertNotEqual(layer["layer_hash"], changed_hash)


class InjectionAndVerifierTests(unittest.TestCase):
    def _issue_codes(self, injection_type: str, target: dict, value) -> set[str]:
        scene = generate_scene("simple_cross", 5, "injection")
        config = {
            "schema": "vgsynth.injections/v1",
            "operations": [{"id": "inj", "type": injection_type, "target": target, "value": value}],
        }
        result = apply_injections(scene, config, default_render_params())
        issues = result.issues + validate_scene(result.scene, check_content_hash=False) + validate_tolerance(
            result.render_params["tolerance"]
        )
        return {str(issue.code) for issue in issues}

    def test_all_required_injections_are_classified_without_crash(self) -> None:
        cases = [
            ("coordinate_out_of_bounds", {"path_id": "p0", "subpath_index": 0, "segment_index": 0, "point": "p0"}, {"x": 999, "y": -1}, ErrorCode.SCENE_INVALID),
            ("degenerate_segment", {"path_id": "p0", "subpath_index": 0, "segment_index": 1}, "zero_length", ErrorCode.PATH_DEGENERATE),
            ("fill_rule_invalid", {"path_id": "p0"}, "bogus", ErrorCode.FILL_RULE),
            ("fill_alpha_out_of_range", {"path_id": "p0"}, 400, ErrorCode.COLOR_RANGE),
            ("tolerance_negative", {}, -2.0, ErrorCode.TOLERANCE_RANGE),
            ("stroke_width_zero", {"path_id": "p0"}, 0, ErrorCode.PATH_DEGENERATE),
        ]
        for injection_type, target, value, code in cases:
            with self.subTest(injection_type=injection_type):
                self.assertIn(str(code), self._issue_codes(injection_type, target, value))

    def test_malformed_injection_values_are_classified(self) -> None:
        scene = generate_scene("simple_cross", 6, "malformed")
        config = {
            "schema": "vgsynth.injections/v1",
            "operations": [
                "not-an-object",
                {
                    "id": "bad-coordinate",
                    "type": "coordinate_out_of_bounds",
                    "target": {"path_id": "p0", "subpath_index": 0, "segment_index": 0, "point": "p0"},
                    "value": {"x": "not-a-number", "y": []},
                },
                {"id": "bad-tolerance", "type": "tolerance_negative", "target": {}, "value": "bad"},
            ],
        }
        result = apply_injections(scene, config, default_render_params())
        issues = result.issues + validate_scene(result.scene, check_content_hash=False) + validate_tolerance(
            result.render_params["tolerance"]
        )
        codes = {str(issue.code) for issue in issues}
        self.assertIn(str(ErrorCode.INJECTION_CONFLICT), codes)
        self.assertIn(str(ErrorCode.TOLERANCE_RANGE), codes)

    def test_verifier_passes_clean_and_locates_first_mutated_pixel(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            clean_case = dict(demo_cases()[0])
            clean = run_case(clean_case, root / "clean", minimize=False, verify=True)
            self.assertEqual(clean.report["verification"]["status"], "ok")
            mutation_case = next(case for case in demo_cases() if case["case_id"] == "verifier_pixel_mismatch_injected")
            mutated = run_case(mutation_case, root / "mutated", minimize=False, verify=True)
            first = mutated.report["verification"]["first_mismatch"]
            self.assertEqual(mutated.report["verification"]["status"], "mismatch")
            self.assertEqual(first["layer_id"], "L0000")
            self.assertEqual((first["x"], first["y"]), (60, 54))
            self.assertEqual(first["channel"], 0)
            self.assertEqual(mutated.report["verification"]["code"], "E_VERIFIER_MISMATCH")

    def test_minimal_counterexample_is_irreducible(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            for case_id in ("alpha_overflow_injected", "verifier_pixel_mismatch_injected"):
                case = next(item for item in demo_cases() if item["case_id"] == case_id)
                execution = run_case(case, root / case_id, minimize=True, verify=True)
                minimal = execution.report["minimal_counterexample"]
                self.assertEqual(minimal["status"], "reproduced")
                self.assertTrue(minimal["irreducible"])
                self.assertTrue(minimal["elements"])
                for check in minimal["removal_checks"]:
                    self.assertFalse(check["still_triggers"], check)
                    self.assertTrue(check["necessary"], check)


class EndToEndTests(unittest.TestCase):
    def test_two_complete_runs_are_byte_identical(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            first_report = run_demo(root / "run_a", minimize=False)
            second_report = run_demo(root / "run_b", minimize=False)
            self.assertTrue(first_report["success"])
            self.assertTrue(second_report["success"])
            comparison = compare_runs(root / "run_a", root / "run_b")
            self.assertTrue(comparison["equivalent"], comparison)
            self.assertEqual(comparison["differing"], [])
            stored = read_json(root / "run_a" / "cases" / "normal_cross" / "verification.json")
            self.assertEqual(stored["status"], "ok")


if __name__ == "__main__":
    unittest.main()
