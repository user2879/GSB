from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from quota_app.codec import load_document, parse_document, write_document
from quota_app.errors import E_FORMAT, ConfigFormatError


class CodecTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)

    def tearDown(self) -> None:
        self.temp.cleanup()

    def test_three_versions_normalize_to_v3(self) -> None:
        v1 = parse_document(
            {
                "format_version": 1,
                "revision": 1,
                "volumes": [{"id": "a", "capacity_bytes": 10}],
            }
        )
        self.assertEqual(v1.schema_version, 3)
        self.assertEqual(v1.original_version, 1)
        self.assertEqual(v1.volumes[0].ops_per_sec, 0)
        self.assertEqual(v1.volumes[0].burst_capacity, 0)
        self.assertEqual(v1.volumes[0].limit_dimension, "bytes")
        self.assertIsNone(v1.volumes[0].derivation)

        v2 = parse_document(
            {
                "version": 2,
                "revision": 2,
                "volumes": [{"id": "a", "capacity_bytes": 10, "ops_per_sec": 3, "burst_capacity": 4}],
            }
        )
        self.assertEqual(v2.volumes[0].ops_per_sec, 3)
        self.assertEqual(v2.volumes[0].limit_dimension, "bytes")

        v3 = parse_document(
            {
                "schema_version": 3,
                "revision": 3,
                "volumes": [
                    {
                        "volume_id": "a",
                        "capacity_bytes": 10,
                        "ops_per_sec": 3,
                        "burst_capacity": 4,
                        "limit_dimension": "ops",
                        "derivation": {"upstream_id": "b", "factor": 2, "offset": 1},
                    }
                ],
            }
        )
        self.assertEqual(v3.volumes[0].limit_dimension, "operations")
        self.assertEqual(v3.volumes[0].derivation.factor, 2)

    def test_unknown_version_is_e_format(self) -> None:
        with self.assertRaises(ConfigFormatError) as context:
            parse_document({"format_version": 9, "volumes": []})
        self.assertEqual(context.exception.code, E_FORMAT)
        self.assertEqual(context.exception.field_path, "$.format_version")

    def test_first_missing_field_has_volume_and_path(self) -> None:
        raw = {
            "format_version": 2,
            "volumes": [
                {"id": "ok", "capacity_bytes": 1, "ops_per_sec": 1, "burst_capacity": 1},
                {"id": "bad", "capacity_bytes": 2, "burst_capacity": 1},
                {"id": "later", "capacity_bytes": 3},
            ],
        }
        with self.assertRaises(ConfigFormatError) as context:
            parse_document(raw)
        self.assertEqual(context.exception.code, E_FORMAT)
        self.assertEqual(context.exception.volume_id, "bad")
        self.assertEqual(context.exception.field_path, "$.volumes[1].ops_per_sec")

    def test_negative_value_rejected_before_partial_effect(self) -> None:
        raw = {
            "format_version": 3,
            "volumes": [
                {
                    "id": "a",
                    "capacity_bytes": -1,
                    "ops_per_sec": 1,
                    "burst_capacity": 1,
                    "limit_dimension": "bytes",
                }
            ],
        }
        with self.assertRaises(ConfigFormatError) as context:
            parse_document(raw)
        self.assertEqual(context.exception.volume_id, "a")
        self.assertEqual(context.exception.field_path, "$.volumes[0].capacity_bytes")

    def test_boolean_is_not_an_integer(self) -> None:
        with self.assertRaises(ConfigFormatError):
            parse_document(
                {
                    "format_version": 1,
                    "volumes": [{"id": "a", "capacity_bytes": True}],
                }
            )

    def test_duplicate_ids_rejected(self) -> None:
        with self.assertRaises(ConfigFormatError) as context:
            parse_document(
                {
                    "format_version": 1,
                    "volumes": [
                        {"id": "a", "capacity_bytes": 1},
                        {"id": "a", "capacity_bytes": 2},
                    ],
                }
            )
        self.assertEqual(context.exception.volume_id, "a")

    def test_round_trip_and_upgraded_from(self) -> None:
        original = parse_document(
            {"format_version": 1, "revision": 7, "volumes": [{"id": "a", "capacity_bytes": 8}]}
        )
        path = self.root / "config.json"
        write_document(path, original)
        loaded = load_document(path)
        self.assertEqual(loaded.revision, 7)
        self.assertEqual(loaded.original_version, 1)
        self.assertEqual(loaded.volumes[0].capacity_bytes, 8)
        payload = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(payload["format_version"], 3)
        self.assertEqual(payload["upgraded_from"], 1)

    def test_invalid_json_e_format(self) -> None:
        path = self.root / "broken.json"
        path.write_text('{"broken":', encoding="utf-8")
        with self.assertRaises(ConfigFormatError) as context:
            load_document(path)
        self.assertEqual(context.exception.code, E_FORMAT)

    def test_flat_derivation_accepted(self) -> None:
        parsed = parse_document(
            {
                "format_version": 3,
                "volumes": [
                    {
                        "id": "a",
                        "capacity_bytes": 10,
                        "ops_per_sec": 1,
                        "burst_capacity": 1,
                        "limit_dimension": "bytes",
                        "derived_from": "b",
                        "factor": 2,
                        "offset": 3,
                    }
                ],
            }
        )
        self.assertEqual(parsed.volumes[0].derivation.upstream_id, "b")
        self.assertEqual(parsed.volumes[0].derivation.factor, 2)


if __name__ == "__main__":
    unittest.main()
