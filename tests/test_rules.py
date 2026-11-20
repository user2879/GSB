from __future__ import annotations

import json
import unittest

from meshgov.errors import RuleError
from meshgov.rules import parse_rules
from tests.helpers import instance, make_rules


class RuleValidationTests(unittest.TestCase):
    def test_valid_rules_are_canonical_and_sorted(self) -> None:
        rules = make_rules()
        self.assertEqual(rules.rules_version, "sha256:" + rules.source_sha256)
        self.assertEqual(rules.global_concurrency, 3)
        self.assertEqual([item.port for item in rules.service("alpha").instances], [28001, 28002])
        self.assertEqual(rules.match_route("/api/alpha/1").service, "alpha")

    def test_field_path_and_overall_rejection_for_invalid_cases(self) -> None:
        base_service = {
            "name": "alpha",
            "concurrency_quota": 1,
            "instances": [instance("a", 28001, 1), instance("b", 28002, 1)],
        }
        base_route = {
            "prefix": "/api/alpha",
            "service": "alpha",
            "timeout_ms": 100,
            "max_retries": 1,
        }
        cases = []
        duplicate_service = json.loads(json.dumps([base_service, base_service]))
        cases.append(("duplicate service", duplicate_service, [base_route], "services[1].name"))
        non_loopback = json.loads(json.dumps([base_service]))
        non_loopback[0]["instances"][0]["host"] = "localhost"
        cases.append(("loopback", non_loopback, [base_route], "services[0].instances[0].host"))
        duplicate_port = json.loads(json.dumps([base_service]))
        duplicate_port[0]["instances"][1]["port"] = 28001
        cases.append(("port unique", duplicate_port, [base_route], "services[0].instances[1].port"))
        zero_sum = json.loads(json.dumps([base_service]))
        zero_sum[0]["instances"][0]["weight"] = 0
        zero_sum[0]["instances"][1]["weight"] = 0
        cases.append(("weight sum", zero_sum, [base_route], "services[0].instances"))
        bad_prefix = json.loads(json.dumps([base_service]))
        bad_route = dict(base_route, prefix="api/alpha")
        cases.append(("prefix", bad_prefix, [bad_route], "routes[0].prefix"))
        bad_timeout = json.loads(json.dumps([base_service]))
        cases.append(
            (
                "timeout",
                bad_timeout,
                [dict(base_route, timeout_ms=0)],
                "routes[0].timeout_ms",
            )
        )
        bad_retry = json.loads(json.dumps([base_service]))
        cases.append(
            (
                "retry",
                bad_retry,
                [dict(base_route, max_retries=11)],
                "routes[0].max_retries",
            )
        )
        for name, services, routes, expected_path in cases:
            with self.subTest(name=name):
                document = {
                    "schema_version": 1,
                    "limits": {"global_concurrency": 2},
                    "services": services,
                    "routes": routes,
                }
                with self.assertRaises(RuleError) as raised:
                    parse_rules(json.dumps(document).encode(), "<test>")
                self.assertEqual(raised.exception.code, "E_RULE")
                self.assertIn(expected_path, raised.exception.field_path)

    def test_malformed_json_and_duplicate_keys_are_rule_errors(self) -> None:
        with self.assertRaises(RuleError) as malformed:
            parse_rules(b'{"schema_version":', "<bad>")
        self.assertEqual(malformed.exception.code, "E_RULE")
        duplicate = b'{"schema_version":1,"schema_version":1}'
        with self.assertRaises(RuleError) as duplicate_error:
            parse_rules(duplicate, "<duplicate>")
        self.assertIn("duplicate", duplicate_error.exception.message)

    def test_bool_is_not_accepted_as_integer(self) -> None:
        service = {
            "name": "alpha",
            "instances": [dict(instance("a", 28001, 1), port=True)],
        }
        document = {
            "schema_version": 1,
            "limits": {"global_concurrency": 2},
            "services": [service],
            "routes": [
                {
                    "prefix": "/api/alpha",
                    "service": "alpha",
                    "timeout_ms": 100,
                    "max_retries": 1,
                }
            ],
        }
        with self.assertRaises(RuleError) as raised:
            parse_rules(json.dumps(document).encode(), "<test>")
        self.assertIn("port", raised.exception.field_path)


if __name__ == "__main__":
    unittest.main()

