from __future__ import annotations

import json
import unittest
from copy import deepcopy

from drift_diag.canonical import canonical_json
from drift_diag.config import EvaluationConfig
from drift_diag.declarations import normalize_declaration
from drift_diag.errors import DiagnosticError
from drift_diag.propagation import ChangeSet, apply_change_set, parse_change_set, TargetUpdate

from .helpers import complete_actual, config, decisions_for, sample_path, target_v3


class PropagationTests(unittest.TestCase):
    def setUp(self):
        self.target = target_v3()
        self.actual = complete_actual(self.target)
        self.decisions = decisions_for(self.target, self.actual)

    def decision_map(self, decisions):
        return {(decision["workload"], decision["field"]): decision for decision in decisions}

    def test_single_field_update_preserves_unaffected_bytes_and_versions(self):
        before = self.decision_map(self.decisions)
        result = apply_change_set(
            self.target,
            self.actual,
            self.decisions,
            config(),
            ChangeSet(
                target_updates=(TargetUpdate("api-gateway", "replicas", 4),)
            ),
        )
        after = self.decision_map(result.decisions)
        self.assertEqual(result.report["affected_decisions"], ["api-gateway/replicas"])
        self.assertTrue(result.report["unaffected_byte_stable"])
        self.assertTrue(after[("api-gateway", "replicas")]["drift"])
        self.assertNotEqual(
            before[("api-gateway", "replicas")]["field_version"],
            after[("api-gateway", "replicas")]["field_version"],
        )
        for key in before:
            if key != ("api-gateway", "replicas"):
                self.assertEqual(canonical_json(before[key]), canonical_json(after[key]))

    def test_format_upgrade_marks_every_field_affected(self):
        raw_v2 = json.loads(sample_path("declaration-v2.json").read_text(encoding="utf-8"))
        target_v2 = normalize_declaration(raw_v2, defaults_version="defaults-v1")
        actual = complete_actual(target_v2)
        decisions = decisions_for(target_v2, actual)
        result = apply_change_set(
            target_v2,
            actual,
            decisions,
            config(),
            ChangeSet(format_from=2, format_to=3),
        )
        self.assertEqual(len(result.report["affected_fields"]), 18)
        self.assertEqual(result.target.declaration_format_versions, (3,))
        for workload in result.target.ordered_workloads():
            self.assertEqual(workload.source_format_version, 3)
            for field in workload.provenance:
                self.assertEqual(workload.provenance[field].format_version, 3)

    def test_default_rule_change_only_recomputes_default_originated_fields(self):
        raw_v2 = json.loads(sample_path("declaration-v2.json").read_text(encoding="utf-8"))
        target_v2 = normalize_declaration(raw_v2, defaults_version="defaults-v1")
        actual = complete_actual(target_v2)
        decisions = decisions_for(target_v2, actual)
        result = apply_change_set(
            target_v2,
            actual,
            decisions,
            config(),
            ChangeSet(new_defaults_version="defaults-v2"),
        )
        affected = {item["field"] for item in result.report["affected_fields"]}
        self.assertEqual(affected, {"probe_ms", "labels", "parent_ref"})
        self.assertTrue(all(item["reasons"][0].startswith("default_rule_change:") for item in result.report["affected_fields"]))
        self.assertEqual(result.target.workloads["worker"].fields["probe_ms"], 750)
        self.assertEqual(result.target.workloads["worker"].fields["image"], "registry.local/worker:3.2.1")
        before = self.decision_map(decisions)
        after = self.decision_map(result.decisions)
        for field in ("replicas", "image", "resource_limit"):
            key = ("worker", field)
            self.assertEqual(canonical_json(before[key]), canonical_json(after[key]))

    def test_unreferenced_workload_can_be_deleted(self):
        result = apply_change_set(
            self.target,
            self.actual,
            self.decisions,
            config(),
            ChangeSet(deleted_workloads=("cache",)),
        )
        self.assertNotIn("cache", result.target.workloads)
        self.assertEqual(len(result.report["removed_decisions"]), 6)
        self.assertTrue(all(not path.startswith("cache/") for path in result.report["unaffected_decisions"]))

    def test_referenced_workload_deletion_returns_e_ref(self):
        with self.assertRaises(DiagnosticError) as context:
            apply_change_set(
                self.target,
                self.actual,
                self.decisions,
                config(),
                ChangeSet(deleted_workloads=("api-gateway",)),
            )
        self.assertEqual(context.exception.code, "E_REF")
        self.assertEqual(context.exception.field, "parent_ref")
        referrers = {item["workload"] for item in context.exception.details["referrers"]}
        self.assertEqual(referrers, {"worker", "cache"})

    def test_parent_update_to_missing_workload_is_e_ref(self):
        with self.assertRaises(DiagnosticError) as context:
            apply_change_set(
                self.target,
                self.actual,
                self.decisions,
                config(),
                ChangeSet(
                    target_updates=(TargetUpdate("worker", "parent_ref", "missing"),)
                ),
            )
        self.assertEqual(context.exception.code, "E_REF")

    def test_change_set_parser_and_sample(self):
        raw = json.loads(sample_path("change-set.json").read_text(encoding="utf-8"))
        parsed = parse_change_set(raw)
        self.assertEqual(len(parsed.target_updates), 1)
        self.assertEqual(parsed.new_defaults_version, "defaults-v2")
        self.assertEqual(parsed.deleted_workloads, ("cache",))
        with self.assertRaises(DiagnosticError):
            parse_change_set({"unknown": True})


if __name__ == "__main__":
    unittest.main()
