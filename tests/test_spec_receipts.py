from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from edge_shard_diag.demo import write_demo_spec
from edge_shard_diag.errors import Codes, DiagnosticError
from edge_shard_diag.fixedpoint import attribute_decision, fixed_ratio
from edge_shard_diag.invalidation import compare_semantic_views, propagate_invalidation
from edge_shard_diag.receipts import adopt_receipts, normalize_receipt_dict
from edge_shard_diag.spec import load_spec_bundle


class SpecValidationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.spec = write_demo_spec(self.root / "spec", allocate_ports=True)

    def tearDown(self) -> None:
        self.temp.cleanup()

    def test_valid_demo_loads(self) -> None:
        bundle = load_spec_bundle(self.spec)
        self.assertEqual(len(bundle.nodes), 3)
        self.assertEqual(len(bundle.shards), 4)
        self.assertIn(1, bundle.capacity_by_version)

    def test_bad_host_reports_field_and_real_line(self) -> None:
        path = self.spec / "nodes.json"
        lines = path.read_text().splitlines()
        expected_line = next(i for i, line in enumerate(lines, 1) if '"host"' in line and "127.0.0.1" in line)
        text = path.read_text().replace('"host": "127.0.0.1"', '"host": "localhost"', 1)
        path.write_text(text)
        with self.assertRaises(DiagnosticError) as ctx:
            load_spec_bundle(self.spec)
        self.assertEqual(ctx.exception.code, Codes.SPEC)
        self.assertEqual(ctx.exception.field, "host")
        self.assertEqual(ctx.exception.line, expected_line)

    def test_duplicate_port_and_cycle_and_bad_scale(self) -> None:
        nodes = self.spec / "nodes.json"
        payload = json.loads(nodes.read_text())
        payload["nodes"][1]["port"] = payload["nodes"][0]["port"]
        nodes.write_text(json.dumps(payload, indent=2))
        with self.assertRaises(DiagnosticError) as ctx:
            load_spec_bundle(self.spec)
        self.assertEqual(ctx.exception.code, Codes.SPEC)
        self.assertEqual(ctx.exception.field, "port")

        spec2 = write_demo_spec(self.root / "cycle", allocate_ports=True)
        shards_path = spec2 / "shards.json"
        payload = json.loads(shards_path.read_text())
        payload["shards"][0]["dependencies"] = ["shard-0004"]
        shards_path.write_text(json.dumps(payload, indent=2))
        with self.assertRaises(DiagnosticError) as ctx:
            load_spec_bundle(spec2)
        self.assertEqual(ctx.exception.code, Codes.SPEC)
        self.assertIn("cycle", ctx.exception.message)

        spec3 = write_demo_spec(self.root / "scale", allocate_ports=True)
        revisions = spec3 / "fixed_revisions.json"
        payload = json.loads(revisions.read_text())
        payload["revisions"][0]["scale_factor"] = 7
        revisions.write_text(json.dumps(payload, indent=2))
        with self.assertRaises(DiagnosticError) as ctx:
            load_spec_bundle(spec3)
        self.assertEqual(ctx.exception.code, Codes.SPEC)
        self.assertEqual(ctx.exception.field, "scale_factor")


class ReceiptNormalizationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.spec = write_demo_spec(Path(self.temp.name) / "spec", allocate_ports=True)
        self.bundle = load_spec_bundle(self.spec)

    def tearDown(self) -> None:
        self.temp.cleanup()

    def test_d1_defaults_and_supplement_basis(self) -> None:
        receipt = normalize_receipt_dict(
            {"format": "d1", "node_seq": 1, "seq_start": 1001, "seq_end": 1002, "summary_prefix": "a" * 16},
            3,
            "x.d1",
            bundle=self.bundle,
        ).to_dict()
        self.assertEqual(receipt["capacity_version"], 0)
        self.assertEqual(receipt["scale_factor"], 1)
        self.assertEqual(receipt["stable_rank"], -1)
        self.assertEqual(receipt["duplicate_refs"], [])
        self.assertIn("capacity_version", receipt["supplement_basis"])

    def test_format_field_dup_errors_have_lines(self) -> None:
        with self.assertRaises(DiagnosticError) as ctx:
            normalize_receipt_dict({"format": "d9"}, 7, "bad.jsonl", bundle=self.bundle)
        self.assertEqual(ctx.exception.code, Codes.FORMAT)
        self.assertEqual(ctx.exception.line, 7)

        with self.assertRaises(DiagnosticError) as ctx:
            normalize_receipt_dict({"format": "d2", "receipt_id": "x"}, 8, "bad.jsonl", bundle=self.bundle)
        self.assertEqual(ctx.exception.code, Codes.FIELD)
        self.assertIn("missing required", ctx.exception.message)

        d3 = {
            "format": "d3",
            "receipt_id": "r",
            "shard_id": "shard-0001",
            "summary_prefix": "x",
            "capacity_version": 1,
            "scale_factor": 1,
            "duration_ms": 1,
            "stable_rank": 99,
            "duplicate_refs": [],
            "sample_snapshot": "s",
        }
        with self.assertRaises(DiagnosticError) as ctx:
            normalize_receipt_dict(d3, 9, "bad.jsonl", bundle=self.bundle)
        self.assertEqual(ctx.exception.code, Codes.FIELD)
        self.assertEqual(ctx.exception.details["reason"], "stable_rank_out_of_bounds")

        d3["stable_rank"] = 1
        d3["duplicate_refs"] = ["delivery-missing"]
        with self.assertRaises(DiagnosticError) as ctx:
            normalize_receipt_dict(d3, 10, "bad.jsonl", bundle=self.bundle, known_delivery_refs={"other"})
        self.assertEqual(ctx.exception.code, Codes.DUP)

    def test_adoption_prefers_capacity_then_format(self) -> None:
        records = [
            normalize_receipt_dict({"format": "d1", "node_seq": 1, "seq_start": 1001, "seq_end": 1002, "summary_prefix": "a" * 16}, 1, bundle=self.bundle).to_dict(),
            normalize_receipt_dict(
                {"format": "d2", "receipt_id": "r2", "shard_id": "shard-0001", "summary_prefix": "b", "capacity_version": 1, "scale_factor": 10, "duration_ms": 1},
                2,
                bundle=self.bundle,
            ).to_dict(),
            normalize_receipt_dict(
                {
                    "format": "d3", "receipt_id": "r3", "shard_id": "shard-0001", "summary_prefix": "c",
                    "capacity_version": 2, "scale_factor": 100, "duration_ms": 1, "stable_rank": 1,
                    "duplicate_refs": [], "sample_snapshot": "s",
                },
                3,
                bundle=self.bundle,
            ).to_dict(),
        ]
        ordered = adopt_receipts(records)
        self.assertEqual(ordered[0]["receipt_id"], "r3")


class FixedPointAndCompareTests(unittest.TestCase):
    def test_scale_changes_close_water_comparison(self) -> None:
        low_a, low_b = fixed_ratio(3, 6, 1)["q"], fixed_ratio(4, 6, 1)["q"]
        high_a, high_b = fixed_ratio(3, 6, 100)["q"], fixed_ratio(4, 6, 100)["q"]
        self.assertEqual((low_a, low_b), (0, 0))
        self.assertEqual((high_a, high_b), (50, 66))
        low = attribute_decision(candidate_ids=["a", "b"], candidate_levels_q=[low_a, low_b], slack_q=0, scale=1)
        high = attribute_decision(candidate_ids=["a", "b"], candidate_levels_q=[high_a, high_b], slack_q=100, scale=100)
        self.assertTrue(low["error_affected"])
        self.assertFalse(high["error_affected"])
        self.assertTrue(low["level_close"])
        self.assertFalse(high["level_close"])
        self.assertEqual(low["selected_id"], "a")

    def test_compare_locates_divergent_shard_field(self) -> None:
        left = {"s1": {"summary": "a", "water": {"q": 1}}}
        right = {"s1": {"summary": "b", "water": {"q": 1}}}
        with self.assertRaises(DiagnosticError) as ctx:
            compare_semantic_views(left, right, code=Codes.DIVERGE, order=["s1"])
        self.assertEqual(ctx.exception.code, Codes.DIVERGE)
        self.assertEqual(ctx.exception.details["first_mismatch_shard"], "s1")
        self.assertEqual(ctx.exception.details["first_mismatch_field"], "summary")

    def test_propagation_uses_dependency_edges(self) -> None:
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        spec = write_demo_spec(Path(temp.name) / "spec", allocate_ports=True)
        bundle = load_spec_bundle(spec)
        report = propagate_invalidation(bundle, changed_samples=[bundle.shards_by_id["shard-0002"].sample_ids[0]])
        self.assertEqual(report["invalid_shards"], ["shard-0002", "shard-0004"])
        self.assertEqual(report["unchanged_shards"], ["shard-0001", "shard-0003"])


if __name__ == "__main__":
    unittest.main()
