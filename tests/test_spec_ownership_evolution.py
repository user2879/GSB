from __future__ import annotations

import json
import random
import tempfile
import unittest
from pathlib import Path

from leasemesh.canonical import canonical_bytes
from leasemesh.errors import SpecError
from leasemesh.evolution import propagate_invalidation
from leasemesh.ownership import adjudicate_events
from leasemesh.spec import load_declaration, load_events, validate_events
from leasemesh.jsonio import load_json_document


ROOT = Path(__file__).resolve().parents[1]


class SpecValidationTests(unittest.TestCase):
    def test_shipped_samples_are_valid(self) -> None:
        spec = load_declaration(ROOT / "samples/declarations.json")
        events = load_events(ROOT / "samples/events.json")
        self.assertEqual(3, len(spec["nodes"]))
        self.assertEqual(11, len(events))

    def test_bad_interval_reports_field_and_line(self) -> None:
        data = json.loads((ROOT / "samples/declarations.json").read_text(encoding="utf-8"))
        data["renew_interval_ms"] = data["lease_duration_ms"] + 1
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.json"
            path.write_text(json.dumps(data, indent=2), encoding="utf-8")
            with self.assertRaises(SpecError) as caught:
                load_declaration(path)
        self.assertEqual("E_SPEC", caught.exception.code)
        self.assertEqual("renew_interval_ms", caught.exception.field)
        self.assertEqual(5, caught.exception.line)

    def test_duplicate_port_and_bad_event_seq(self) -> None:
        data = json.loads((ROOT / "samples/declarations.json").read_text(encoding="utf-8"))
        data["nodes"][1]["port"] = data["nodes"][0]["port"]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "ports.json"
            path.write_text(json.dumps(data, indent=2), encoding="utf-8")
            with self.assertRaises(SpecError) as caught:
                load_declaration(path)
            self.assertEqual("nodes[1].port", caught.exception.field)
            self.assertGreaterEqual(caught.exception.line or 0, 1)

            events = json.loads((ROOT / "samples/events.json").read_text(encoding="utf-8"))
            events["events"][-1]["seq"] = 1
            event_path = Path(directory) / "events.json"
            event_path.write_text(json.dumps(events, indent=2), encoding="utf-8")
            with self.assertRaises(SpecError) as event_error:
                validate_events(load_json_document(event_path))
            self.assertIn("seq", event_error.exception.field or "")
            self.assertGreaterEqual(event_error.exception.line or 0, 1)


class OwnershipEvolutionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.spec = load_declaration(ROOT / "samples/declarations.json")
        self.events = load_events(ROOT / "samples/events.json")

    def test_unique_adjudication_and_reclaim_exclusion(self) -> None:
        result = adjudicate_events(self.events, self.spec)
        self.assertTrue(result["invariants"]["no_double_hold"])
        self.assertTrue(result["invariants"]["no_reclaim_double_count"])
        self.assertEqual(
            {"expired": 1, "quota_fenced": 1, "unique_epochs": 2, "overlap": []},
            result["reclamation_totals"],
        )
        decisions = {item["lease_id"]: item for item in result["decisions"]}
        self.assertEqual("expired", decisions["lease-a1"]["reason_category"])
        self.assertIn("claim-fence-a1", decisions["lease-a1"]["suppressed_claim_ids"])
        self.assertEqual("quota_fenced", decisions["lease-c1"]["reason_category"])
        self.assertIn("claim-exp-c1", decisions["lease-c1"]["suppressed_claim_ids"])

    def test_arbitrary_node_processing_order_is_canonical(self) -> None:
        expected = canonical_bytes(adjudicate_events(self.events, self.spec))
        for seed in range(20):
            ordered = list(self.events)
            random.Random(seed).shuffle(ordered)
            actual = canonical_bytes(adjudicate_events(ordered, self.spec))
            self.assertEqual(expected, actual, f"order seed {seed} changed adjudication")

    def test_invalidation_subset_and_byte_proofs(self) -> None:
        new_spec = load_declaration(ROOT / "samples/declarations.v2.json")
        report = propagate_invalidation(self.spec, new_spec, self.events)
        invalid_ids = [item["lease_id"] for item in report["invalid_set"]]
        self.assertEqual(["lease-a2", "lease-c2"], invalid_ids)
        self.assertTrue(report["all_unaffected_bytes_identical"])
        self.assertGreater(len(report["trace"]), 5)
        for proof in report["unaffected_proofs"]:
            self.assertTrue(proof["identical"], proof)
            self.assertEqual(proof["sha256_before"], proof["sha256_after"])


if __name__ == "__main__":
    unittest.main()
