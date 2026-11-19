from __future__ import annotations

import copy
import unittest

from level_workbench.canonical import grid_hash
from level_workbench.counterexample import build_minimal_counterexample, replay_minimal_counterexample
from level_workbench.generator import generate_level
from level_workbench.runner import run_case
from level_workbench.samples import normal_spec
from level_workbench.spec import parse_spec
from level_workbench.verifier import verify_artifact


class VerifierCounterexampleTests(unittest.TestCase):
    def test_verifier_ignores_stale_generator_evidence(self) -> None:
        spec = parse_spec(normal_spec())
        outcome = generate_level(spec)
        document = copy.deepcopy(outcome.level_document())
        document["grid"]["cells"][6][8] = "rock"
        document["grid"]["cells"][6][9] = "water"
        document["hashes"]["grid_sha256"] = grid_hash(document["grid"]["cells"])
        verification = verify_artifact(spec, document)
        self.assertEqual(verification["status"], "fail")
        self.assertEqual(verification["first_violation"]["code"], "E_FORBIDDEN_ADJACENT")
        self.assertEqual(document["generation"]["status"], "pass")

    def test_first_violation_uses_row_major_order(self) -> None:
        raw = normal_spec()
        raw["resources"]["count"] = [0, 0]
        spec = parse_spec(raw)
        outcome = generate_level(spec)
        document = copy.deepcopy(outcome.level_document())
        for left, right in (((1, 1), (2, 1)), ((10, 2), (11, 2))):
            document["grid"]["cells"][left[1]][left[0]] = "rock"
            document["grid"]["cells"][right[1]][right[0]] = "water"
        document["hashes"]["grid_sha256"] = grid_hash(document["grid"]["cells"])
        verification = verify_artifact(spec, document)
        self.assertEqual(verification["first_violation"]["coord"], [1, 1])
        self.assertEqual(len([issue for issue in verification["issues"] if issue["code"] == "E_FORBIDDEN_ADJACENT"]), 2)

    def test_forbidden_counterexample_is_minimal_and_replayable(self) -> None:
        case = next(case for case in __import__("level_workbench.samples", fromlist=["demo_cases"]).demo_cases() if case["id"] == "illegal_adjacent")
        result = run_case(case)
        counterexample = result["minimal_counterexample"]
        self.assertIsNotNone(counterexample)
        self.assertTrue(counterexample["checks"]["reproduces"])
        self.assertEqual(len(counterexample["elements"]), 2)
        self.assertEqual(result["first_violation"]["coord"], [8, 6])
        self.assertTrue(counterexample["checks"]["all_single_element_removals_disappear"])
        self.assertTrue(all(not check["violation_present_after_removal"] for check in counterexample["checks"]["removal_checks"]))
        spec = parse_spec(normal_spec())
        replay = replay_minimal_counterexample(spec, counterexample)
        self.assertTrue(replay["reproduced"])
        self.assertTrue(replay["minimal"])

    def test_resource_overflow_counterexample_has_max_plus_one_elements(self) -> None:
        case = next(case for case in __import__("level_workbench.samples", fromlist=["demo_cases"]).demo_cases() if case["id"] == "resource_pipeline_overflow")
        result = run_case(case)
        counterexample = result["minimal_counterexample"]
        self.assertIsNotNone(counterexample)
        self.assertEqual(counterexample["violation_code"], "E_RESOURCE_RANGE")
        self.assertEqual(len(counterexample["elements"]), 15)
        self.assertTrue(counterexample["checks"]["reproduces"])
        self.assertTrue(counterexample["checks"]["all_single_element_removals_disappear"])
        replay = replay_minimal_counterexample(parse_spec(normal_spec()), counterexample)
        self.assertTrue(replay["reproduced"])
        self.assertTrue(replay["minimal"])

    def test_connectivity_counterexample_is_a_minimal_blocked_cut(self) -> None:
        raw = normal_spec()
        raw["resources"]["count"] = [0, 0]
        spec = parse_spec(raw)
        outcome = generate_level(spec)
        document = copy.deepcopy(outcome.level_document())
        for y in range(spec.height):
            document["grid"]["cells"][y][10] = "water"
        document["hashes"]["grid_sha256"] = grid_hash(document["grid"]["cells"])
        verification = verify_artifact(spec, document)
        self.assertEqual(verification["status"], "fail")
        self.assertEqual(verification["first_violation"]["code"], "E_CONNECT_UNSAT")
        counterexample = build_minimal_counterexample(spec, document, verification)
        self.assertIsNotNone(counterexample)
        self.assertEqual(counterexample["reproduction"]["kind"], "blocked_cut")
        self.assertTrue(counterexample["checks"]["reproduces"])
        self.assertTrue(counterexample["checks"]["all_single_element_removals_disappear"])
        replay = replay_minimal_counterexample(spec, counterexample)
        self.assertTrue(replay["reproduced"])
        self.assertTrue(replay["minimal"])


if __name__ == "__main__":
    unittest.main()
