from __future__ import annotations

import json
import unittest

from meshdiag.errors import MeshError
from meshdiag.models import normalize_revision, parse_service_rows


SERVICES = parse_service_rows(
    [
        {
            "id": "svc",
            "max_concurrency": 2,
            "instances": [
                {"id": "svc-a", "host": "127.0.0.1", "port": 19001},
                {"id": "svc-b", "host": "127.0.0.1", "port": 19002},
            ],
        }
    ]
)


def normalize(document, defaults=None):
    return normalize_revision(
        document,
        SERVICES,
        workspace_id="normalize-test",
        defaults=defaults or {"timeout_ms": 500, "retries": 0, "cooldown_ms": 1000},
    )


class NormalizeTests(unittest.TestCase):
    def test_three_formats_normalize_to_same_internal_shape(self):
        revision = normalize(
            {
                "revision": 7,
                "rules": [
                    {
                        "id": "v1",
                        "format_version": 1,
                        "prefix": "/v1",
                        "target_service": "svc",
                    },
                    {
                        "id": "v2",
                        "format_version": 2,
                        "prefix": "/v2",
                        "target_service": "svc",
                        "instance_weights": {"svc-a": 2, "svc-b": 1},
                        "timeout_ms": 50,
                    },
                    {
                        "id": "v3",
                        "format_version": 3,
                        "prefix": "/v3",
                        "target_service": "svc",
                        "instance_weights": {"svc-a": 0, "svc-b": 1},
                        "timeout_ms": 25,
                        "retries": 2,
                        "circuit_threshold": 3,
                        "cooldown_ms": 400,
                        "depends_on": ["service:svc"],
                    },
                ],
            }
        )
        rules = revision.rule_map()
        self.assertEqual([rule.rule_id for rule in revision.rules], ["v1", "v2", "v3"])
        self.assertEqual(rules["v1"].weights(), {"svc-a": 1, "svc-b": 1})
        self.assertEqual(rules["v2"].timeout_ms, 50)
        self.assertEqual(rules["v3"].retries, 2)
        self.assertEqual(rules["v3"].circuit_threshold, 3)
        self.assertEqual(rules["v3"].cooldown_ms, 400)
        self.assertIn("service:svc", revision.dependency_graph.nodes)

    def test_key_order_and_whitespace_do_not_change_summary(self):
        first = json.loads(
            '{"revision":1,"rules":[{"id":"r","format_version":1,'
            '"prefix":"/x","target_service":"svc"}]}'
        )
        second = json.loads(
            '{ "rules" : [ { "target_service" : "svc", "prefix" : "/x", '
            '"format_version" : 1, "id" : "r" } ], "revision" : 1 }'
        )
        a = normalize(first)
        b = normalize(second)
        self.assertEqual(a.canonical_text, b.canonical_text)
        self.assertEqual(a.digest, b.digest)

    def test_rule_array_order_is_canonicalized(self):
        a = normalize(
            {
                "revision": 1,
                "rules": [
                    {"id": "z", "format_version": 1, "prefix": "/z", "target_service": "svc"},
                    {"id": "a", "format_version": 1, "prefix": "/a", "target_service": "svc"},
                ],
            }
        )
        b = normalize(
            {
                "revision": 1,
                "rules": [
                    {"id": "a", "format_version": 1, "prefix": "/a", "target_service": "svc"},
                    {"id": "z", "format_version": 1, "prefix": "/z", "target_service": "svc"},
                ],
            }
        )
        self.assertEqual(a.canonical_text, b.canonical_text)

    def test_cross_format_equivalent_defaults_have_same_digest(self):
        v1 = normalize(
            {
                "revision": 1,
                "rules": [
                    {"id": "r", "format_version": 1, "prefix": "/r", "target_service": "svc"}
                ],
            }
        )
        v2 = normalize(
            {
                "revision": 1,
                "rules": [
                    {
                        "id": "r",
                        "format_version": 2,
                        "prefix": "/r",
                        "target_service": "svc",
                        "instance_weights": {"svc-a": 1, "svc-b": 1},
                        "timeout_ms": 500,
                    }
                ],
            }
        )
        self.assertEqual(v1.digest, v2.digest)

    def test_unknown_format_has_error_path(self):
        with self.assertRaises(MeshError) as context:
            normalize(
                {
                    "revision": 1,
                    "rules": [
                        {"id": "r", "format_version": 9, "prefix": "/r", "target_service": "svc"}
                    ],
                }
            )
        self.assertEqual(context.exception.code, "E_RULE")
        self.assertEqual(
            context.exception.diagnostics[0].path, "rules[0].format_version"
        )

    def test_negative_weight_has_error_path(self):
        with self.assertRaises(MeshError) as context:
            normalize(
                {
                    "revision": 1,
                    "rules": [
                        {
                            "id": "r",
                            "format_version": 2,
                            "prefix": "/r",
                            "target_service": "svc",
                            "instance_weights": {"svc-a": -1, "svc-b": 1},
                            "timeout_ms": 20,
                        }
                    ],
                }
            )
        self.assertEqual(context.exception.code, "E_RULE")
        self.assertEqual(
            context.exception.diagnostics[0].path,
            "rules[0].instance_weights.svc-a",
        )

    def test_missing_target_service_uses_e_ref(self):
        with self.assertRaises(MeshError) as context:
            normalize(
                {
                    "revision": 1,
                    "rules": [
                        {
                            "id": "r",
                            "format_version": 1,
                            "prefix": "/r",
                            "target_service": "missing",
                        }
                    ],
                }
            )
        self.assertEqual(context.exception.code, "E_REF")
        self.assertEqual(
            context.exception.diagnostics[0].path, "rules[0].target_service"
        )

    def test_zero_weight_sum_rejects_whole_revision(self):
        with self.assertRaises(MeshError) as context:
            normalize(
                {
                    "revision": 1,
                    "rules": [
                        {
                            "id": "r",
                            "format_version": 2,
                            "prefix": "/r",
                            "target_service": "svc",
                            "instance_weights": {"svc-a": 0, "svc-b": 0},
                            "timeout_ms": 20,
                        }
                    ],
                }
            )
        self.assertEqual(context.exception.code, "E_RULE")
        self.assertEqual(
            context.exception.diagnostics[0].path, "rules[0].instance_weights"
        )

    def test_v3_missing_required_fields_are_reported_together(self):
        with self.assertRaises(MeshError) as context:
            normalize(
                {
                    "revision": 1,
                    "rules": [
                        {
                            "id": "r",
                            "format_version": 3,
                            "prefix": "/r",
                            "target_service": "svc",
                            "instance_weights": {"svc-a": 1},
                        }
                    ],
                }
            )
        self.assertEqual(context.exception.code, "E_RULE")
        paths = {entry.path for entry in context.exception.diagnostics}
        self.assertIn("rules[0].timeout_ms", paths)
        self.assertIn("rules[0].retries", paths)
        self.assertIn("rules[0].circuit_threshold", paths)
        self.assertIn("rules[0].cooldown_ms", paths)
        self.assertIn("rules[0].depends_on", paths)

    def test_v1_rejects_later_format_fields(self):
        with self.assertRaises(MeshError) as context:
            normalize(
                {
                    "revision": 1,
                    "rules": [
                        {
                            "id": "r",
                            "format_version": 1,
                            "prefix": "/r",
                            "target_service": "svc",
                            "timeout_ms": 20,
                        }
                    ],
                }
            )
        self.assertEqual(context.exception.code, "E_RULE")
        self.assertEqual(context.exception.diagnostics[0].path, "rules[0].timeout_ms")


if __name__ == "__main__":
    unittest.main()
