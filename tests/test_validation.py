from __future__ import annotations

import unittest

from config_diag.errors import DiagnosticError
from config_diag.validation import (
    load_bundle,
    load_change_sequence,
    load_config_spec,
    load_subscription_sample,
)

from .support import SAMPLES, local_temp, read_json, write_json


class ValidationTests(unittest.TestCase):
    def test_bundle_validates_and_has_stable_shape(self) -> None:
        spec, clients, changes = load_bundle(SAMPLES)
        self.assertEqual(len(spec.keys), 11)
        self.assertEqual(len(clients.clients), 3)
        self.assertEqual([item.sequence for item in changes.operations], [1, 2, 3, 4])
        self.assertEqual(spec.by_key["pricing.net_fee"].refs, ("pricing.base_discount", "pricing.base_fee"))

    def test_first_invalid_key_is_reported(self) -> None:
        raw = read_json(SAMPLES / "config_keys.json")
        raw["keys"][0]["value"] = True
        raw["keys"][1]["ttl_ms"] = 0
        with local_temp("spec-first-") as tmp:
            path = tmp / "config.json"
            write_json(path, raw)
            with self.assertRaises(DiagnosticError) as raised:
                load_config_spec(path)
        self.assertEqual(raised.exception.code, "E_SPEC")
        self.assertEqual(raised.exception.key, "pricing.base_fee")
        self.assertEqual(raised.exception.field, "value")

    def test_duplicate_key_and_missing_reference_are_spec_errors(self) -> None:
        raw = read_json(SAMPLES / "config_keys.json")
        raw["keys"][1]["key"] = raw["keys"][0]["key"]
        with local_temp("spec-dup-") as tmp:
            path = tmp / "config.json"
            write_json(path, raw)
            with self.assertRaises(DiagnosticError) as raised:
                load_config_spec(path)
        self.assertEqual((raised.exception.code, raised.exception.key, raised.exception.field), ("E_SPEC", "pricing.base_fee", "key"))

        raw = read_json(SAMPLES / "config_keys.json")
        raw["keys"][0]["refs"] = ["pricing.missing"]
        raw["keys"][0]["formula"] = {
            "op": "linear",
            "terms": [{"ref": "pricing.missing", "weight": 1}],
            "divisor": 1,
            "bias": 0,
        }
        with local_temp("spec-missing-") as tmp:
            path = tmp / "config.json"
            write_json(path, raw)
            with self.assertRaises(DiagnosticError) as raised:
                load_config_spec(path)
        self.assertEqual(raised.exception.code, "E_SPEC")
        self.assertEqual(raised.exception.key, "pricing.base_fee")
        self.assertEqual(raised.exception.field, "refs")

    def test_cycle_is_reported_with_stable_key_and_path(self) -> None:
        raw = read_json(SAMPLES / "config_keys.json")
        by_key = {item["key"]: item for item in raw["keys"]}
        by_key["pricing.net_fee"]["refs"] = ["tax.amount"]
        by_key["pricing.net_fee"]["formula"] = {
            "op": "linear",
            "terms": [{"ref": "tax.amount", "weight": 1000}],
            "divisor": 1000,
            "bias": 0,
        }
        by_key["tax.amount"]["refs"] = ["pricing.net_fee"]
        by_key["tax.amount"]["formula"] = {
            "op": "linear",
            "terms": [{"ref": "pricing.net_fee", "weight": 1000}],
            "divisor": 1000,
            "bias": 0,
        }
        with local_temp("spec-cycle-") as tmp:
            path = tmp / "config.json"
            write_json(path, raw)
            with self.assertRaises(DiagnosticError) as raised:
                load_config_spec(path)
        exc = raised.exception
        self.assertEqual(exc.code, "E_CYCLE")
        self.assertEqual(exc.key, "pricing.net_fee")
        self.assertEqual(exc.field, "refs")
        self.assertEqual(exc.context["spec_category"], "E_SPEC")
        self.assertEqual(exc.context["cycle"][0], exc.context["cycle"][-1])

    def test_subscription_client_reference_is_strict(self) -> None:
        spec = load_config_spec(SAMPLES / "config_keys.json")
        raw = read_json(SAMPLES / "subscription_clients.json")
        raw["clients"][0]["keys"].append("unknown.key")
        with local_temp("clients-bad-") as tmp:
            path = tmp / "clients.json"
            write_json(path, raw)
            with self.assertRaises(DiagnosticError) as raised:
                load_subscription_sample(path, set(spec.by_key))
        self.assertEqual(raised.exception.code, "E_SPEC")
        self.assertEqual(raised.exception.key, "sub-alpha")
        self.assertEqual(raised.exception.field, "keys")

    def test_change_versions_must_increase(self) -> None:
        spec = load_config_spec(SAMPLES / "config_keys.json")
        raw = read_json(SAMPLES / "change_sequence.json")
        raw["operations"][1]["changes"][0]["version"] = 1
        with local_temp("changes-bad-") as tmp:
            path = tmp / "changes.json"
            write_json(path, raw)
            with self.assertRaises(DiagnosticError) as raised:
                load_change_sequence(path, spec)
        self.assertEqual(raised.exception.code, "E_SPEC")
        self.assertEqual(raised.exception.key, "tax.rate_permille")
        self.assertEqual(raised.exception.field, "version")


if __name__ == "__main__":
    unittest.main()
