from __future__ import annotations

import unittest

from quota_app.codec import parse_document
from quota_app.derive import changed_fields, resolve_effective
from quota_app.errors import E_DERIVE, DeriveError

from .helpers import document, document_raw


class DerivationTests(unittest.TestCase):
    def test_effective_capacity_and_arbitration(self) -> None:
        effective = resolve_effective(document())
        self.assertEqual(effective["child"].derived_capacity_bytes, 1100)
        self.assertEqual(effective["child"].capacity_bytes, 700)
        self.assertEqual(effective["child"].arbitration, "explicit")
        self.assertEqual(effective["child"].derivation_chain, ("root", "child"))
        self.assertEqual(effective["mirror"].derived_capacity_bytes, 1000)
        self.assertEqual(effective["mirror"].capacity_bytes, 1000)
        self.assertEqual(effective["mirror"].arbitration, "derived")

    def test_chain_backtracking(self) -> None:
        parsed = parse_document(
            {
                "format_version": 3,
                "volumes": [
                    {"id": "a", "capacity_bytes": 100, "ops_per_sec": 0, "burst_capacity": 0, "limit_dimension": "bytes"},
                    {"id": "b", "capacity_bytes": 999, "ops_per_sec": 0, "burst_capacity": 0, "limit_dimension": "bytes", "derivation": {"upstream_id": "a", "factor": 2, "offset": 1}},
                    {"id": "c", "capacity_bytes": 9999, "ops_per_sec": 0, "burst_capacity": 0, "limit_dimension": "bytes", "derivation": {"upstream_id": "b", "factor": 2, "offset": 1}},
                ],
            }
        )
        effective = resolve_effective(parsed)
        self.assertEqual(effective["c"].derived_capacity_bytes, 403)
        self.assertEqual(effective["c"].derivation_chain, ("a", "b", "c"))

    def test_cycle_reports_volume_sequence(self) -> None:
        parsed = parse_document(
            {
                "format_version": 3,
                "volumes": [
                    {"id": "a", "capacity_bytes": 10, "ops_per_sec": 0, "burst_capacity": 0, "limit_dimension": "bytes", "derivation": {"upstream_id": "b", "factor": 1, "offset": 0}},
                    {"id": "b", "capacity_bytes": 10, "ops_per_sec": 0, "burst_capacity": 0, "limit_dimension": "bytes", "derivation": {"upstream_id": "a", "factor": 1, "offset": 0}},
                ],
            }
        )
        with self.assertRaises(DeriveError) as context:
            resolve_effective(parsed)
        self.assertEqual(context.exception.code, E_DERIVE)
        self.assertEqual(context.exception.details["cycle"], ["a", "b", "a"])

    def test_missing_reference_is_e_derive(self) -> None:
        parsed = parse_document(
            {
                "format_version": 3,
                "volumes": [
                    {"id": "a", "capacity_bytes": 10, "ops_per_sec": 0, "burst_capacity": 0, "limit_dimension": "bytes", "derivation": {"upstream_id": "missing", "factor": 1, "offset": 0}}
                ],
            }
        )
        with self.assertRaises(DeriveError) as context:
            resolve_effective(parsed)
        self.assertEqual(context.exception.volume_id, "a")
        self.assertEqual(context.exception.details["reference_chain"], ["a", "missing"])

    def test_changed_fields_sorted(self) -> None:
        old = document(3)
        raw = document_raw(4)
        raw["volumes"][1]["capacity_bytes"] = 800
        raw["volumes"][3]["ops_per_sec"] = 9
        new = parse_document(raw)
        changes = changed_fields(old, new)
        self.assertEqual(changes, [
            {"volume_id": "child", "fields": ["capacity_bytes", "effective_capacity_bytes"]},
            {"volume_id": "mirror", "fields": ["ops_per_sec"]},
        ])

    def test_upstream_change_marks_downstream_effective_volume(self) -> None:
        old = document(3)
        raw = document_raw(4)
        raw["volumes"][0]["capacity_bytes"] = 500
        new = parse_document(raw)
        changes = changed_fields(old, new)
        by_volume = {item["volume_id"]: item["fields"] for item in changes}
        self.assertEqual(by_volume["root"], ["capacity_bytes", "effective_capacity_bytes"])
        self.assertEqual(by_volume["child"], ["effective_capacity_bytes", "arbitration"])
        self.assertEqual(by_volume["mirror"], ["effective_capacity_bytes"])


if __name__ == "__main__":
    unittest.main()
