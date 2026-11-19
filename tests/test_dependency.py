from __future__ import annotations

import unittest
from dataclasses import replace

from meshdiag.errors import MeshError
from meshdiag.models import (
    normalize_revision,
    parse_service_rows,
    validate_deleted_rules,
)


SERVICES = parse_service_rows(
    [
        {
            "id": "svc-a",
            "max_concurrency": 1,
            "instances": [{"id": "a-1", "host": "127.0.0.1", "port": 19001}],
        },
        {
            "id": "svc-b",
            "max_concurrency": 1,
            "instances": [{"id": "b-1", "host": "127.0.0.1", "port": 19002}],
        },
    ]
)


def rule(rule_id, service, depends_on, suffix=None):
    target = suffix or rule_id
    return {
        "id": rule_id,
        "format_version": 3,
        "prefix": f"/{rule_id}",
        "target_service": service,
        "instance_weights": {
            f"{'a' if service == 'svc-a' else 'b'}-1": 1
        },
        "timeout_ms": 50,
        "retries": 0,
        "circuit_threshold": 1,
        "cooldown_ms": 100,
        "depends_on": depends_on,
    }


def normalize(rules, revision=1):
    return normalize_revision(
        {"revision": revision, "rules": rules},
        SERVICES,
        workspace_id="dependency-test",
        defaults={"timeout_ms": 100, "retries": 0, "cooldown_ms": 100},
    )


class DependencyTests(unittest.TestCase):
    def test_edges_are_sorted_and_include_rule_and_service_references(self):
        revision = normalize(
            [
                rule("z", "svc-b", ["service:svc-a", "rule:a"]),
                rule("a", "svc-a", []),
            ]
        )
        expected = sorted(revision.dependency_graph.edges)
        self.assertEqual(list(revision.dependency_graph.edges), expected)
        self.assertIn(
            ("rule:z", "rule:a", "depends_on_rule"), revision.dependency_graph.edges
        )
        self.assertIn(
            ("rule:z", "service:svc-a", "depends_on_service"),
            revision.dependency_graph.edges,
        )

    def test_unknown_rule_and_service_reference_are_e_ref(self):
        with self.assertRaises(MeshError) as context:
            normalize([rule("a", "svc-a", ["rule:missing", "service:missing"])])
        self.assertEqual(context.exception.code, "E_REF")
        targets = {entry.value for entry in context.exception.diagnostics}
        self.assertEqual(targets, {"missing"})

    def test_cycle_is_deterministic_and_starts_at_lexicographic_minimum(self):
        rules = [
            rule("c", "svc-a", ["rule:a"]),
            rule("b", "svc-a", ["rule:c"]),
            rule("a", "svc-a", ["rule:b"]),
        ]
        sequences = []
        for _ in range(5):
            with self.assertRaises(MeshError) as context:
                normalize(rules)
            self.assertEqual(context.exception.code, "E_CYCLE")
            sequences.append(tuple(context.exception.extra["cycle"]))
        self.assertEqual(len(set(sequences)), 1)
        self.assertEqual(sequences[0], ("a", "b", "c", "a"))

    def test_self_cycle_is_reported(self):
        with self.assertRaises(MeshError) as context:
            normalize([rule("a", "svc-a", ["rule:a"])])
        self.assertEqual(context.exception.code, "E_CYCLE")
        self.assertEqual(context.exception.extra["cycle"], ["a", "a"])

    def test_deleting_referenced_rule_reports_referrers(self):
        previous = normalize(
            [rule("a", "svc-a", []), rule("b", "svc-b", ["rule:a"])]
        )
        new_rules = [replace(previous.rule_map()["b"], revision_version=2)]
        with self.assertRaises(MeshError) as context:
            validate_deleted_rules(previous.rules, new_rules)
        self.assertEqual(context.exception.code, "E_REF")
        diagnostic = context.exception.diagnostics[0]
        self.assertEqual(diagnostic.extra["referrers"], ["b"])

if __name__ == "__main__":
    unittest.main()
