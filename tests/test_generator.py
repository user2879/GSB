from __future__ import annotations

import copy
import unittest

from level_workbench.canonical import canonical_json_bytes, grid_hash
from level_workbench.generator import generate_level
from level_workbench.injection import normalize_injection
from level_workbench.samples import connectivity_unsat_spec, normal_spec
from level_workbench.spec import parse_spec
from level_workbench.verifier import verify_artifact


class GeneratorTests(unittest.TestCase):
    def test_same_spec_and_seed_are_byte_identical(self) -> None:
        spec = parse_spec(normal_spec())
        first = generate_level(spec)
        second = generate_level(spec)
        self.assertTrue(first.ok)
        self.assertTrue(second.ok)
        doc_first = first.level_document()
        doc_second = second.level_document()
        self.assertIsNotNone(doc_first)
        self.assertIsNotNone(doc_second)
        self.assertEqual(canonical_json_bytes(doc_first), canonical_json_bytes(doc_second))
        self.assertEqual(doc_first["hashes"]["grid_sha256"], grid_hash(first.level.cells))

    def test_terrain_subseed_offset_changes_grid_reproducibly(self) -> None:
        spec = parse_spec(normal_spec())
        baseline = generate_level(spec)
        injection = normalize_injection({"pipeline": {"subseed_offsets": {"terrain": 17}}})
        offset = generate_level(spec, injection)
        repeat = generate_level(spec, injection)
        self.assertTrue(baseline.ok)
        self.assertTrue(offset.ok)
        self.assertNotEqual(baseline.level_document()["hashes"]["grid_sha256"], offset.level_document()["hashes"]["grid_sha256"])
        self.assertEqual(canonical_json_bytes(offset.level_document()), canonical_json_bytes(repeat.level_document()))
        terrain_seeds = [item for item in offset.subseeds if item["purpose"] == "terrain"]
        self.assertEqual(terrain_seeds[0]["offset"], 17)

    def test_connectivity_evidence_contains_paths_for_all_goals(self) -> None:
        spec = parse_spec(normal_spec())
        outcome = generate_level(spec)
        self.assertTrue(outcome.ok)
        evidence = {item["constraint"]: item for item in outcome.evidence["constraints"]}
        connectivity = evidence["connectivity"]
        self.assertEqual(connectivity["status"], "pass")
        destinations = {path["to"] for path in connectivity["details"]["paths"]}
        self.assertEqual(destinations, {"target", "region:grove", "region:marsh", "region:ridge"})
        for path in connectivity["details"]["paths"]:
            self.assertEqual(path["tiles"][0], [1, 7])

    def test_forbidden_repair_records_rounds(self) -> None:
        spec = parse_spec(normal_spec())
        outcome = generate_level(spec)
        trace = outcome.repair_trace["forbidden_adjacent"]
        self.assertGreater(trace["initial_count"], 0)
        self.assertEqual(trace["status"], "pass")
        self.assertGreaterEqual(len(trace["rounds"]), 1)
        self.assertTrue(any(change for round_item in trace["rounds"] for change in round_item["changes"]))

    def test_unsatisfiable_connectivity_reports_conflict_without_level(self) -> None:
        spec = parse_spec(connectivity_unsat_spec())
        outcome = generate_level(spec)
        self.assertFalse(outcome.ok)
        self.assertIsNone(outcome.level)
        self.assertEqual(outcome.error["code"], "E_CONNECT_UNSAT")
        conflict = outcome.error["details"]["conflict"]
        self.assertEqual(conflict["conflict_pair"], ["spawn", "target"])
        self.assertGreater(len(conflict["blocking_cells"]), 0)

    def test_skip_repair_returns_exhaustion_and_preserves_candidate(self) -> None:
        spec = parse_spec(normal_spec())
        injection = normalize_injection(
            {
                "pipeline": {
                    "skip_local_repair": True,
                    "pre_repair_force_terrain": [
                        {"coord": [8, 6], "terrain": "rock"},
                        {"coord": [9, 6], "terrain": "water"},
                    ],
                }
            }
        )
        outcome = generate_level(spec, injection)
        self.assertFalse(outcome.ok)
        self.assertIsNotNone(outcome.level)
        self.assertEqual(outcome.error["code"], "E_REPAIR_EXHAUST")
        self.assertEqual(outcome.repair_trace["forbidden_adjacent"]["status"], "skipped")
        verification = verify_artifact(spec, outcome.level_document())
        self.assertEqual(verification["status"], "fail")
        self.assertEqual(verification["first_violation"]["code"], "E_FORBIDDEN_ADJACENT")

    def test_resource_override_uses_seeded_local_repair(self) -> None:
        spec = parse_spec(normal_spec())
        injection = normalize_injection({"pipeline": {"resource_count_override": 20}})
        outcome = generate_level(spec, injection)
        self.assertTrue(outcome.ok)
        self.assertEqual(len(outcome.level.resources), 14)
        resource_trace = outcome.repair_trace["resources"]
        self.assertEqual(resource_trace["status"], "pass")
        self.assertEqual(resource_trace["override"]["after"], 20)
        self.assertTrue(any(round_item["action"] == "remove_to_maximum" for round_item in resource_trace["rounds"]))


if __name__ == "__main__":
    unittest.main()
