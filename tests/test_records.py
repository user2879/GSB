from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from affinity_scheduler.errors import E_FORMAT, E_WIDTH, FormatError, WidthError
from affinity_scheduler.model import PLUGIN_NAMES
from affinity_scheduler.records import (
    NormalizationParams,
    NormalizedRecord,
    V1_WIDTH,
    decode_line,
    encode_v1,
    encode_v2,
    encode_v3,
    load_records,
)
from affinity_scheduler.samples import write_sample
from affinity_scheduler.spec import load_spec


class RecordFormatTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.sample = write_sample(self.root, port_base=40200)

    def tearDown(self) -> None:
        self.temp.cleanup()

    def test_mixed_versions_load_and_normalize(self) -> None:
        batch = load_records(self.sample["records"])
        self.assertEqual(batch.format_versions, (1, 2, 3))
        v1, v2, v3 = batch.records
        self.assertIn("plugin_scores.node_affinity", v1.filled_fields)
        self.assertEqual(v1.plugin_scores["node_affinity"], 0)
        self.assertIn("plugin_contributions.resource_fit", v2.filled_fields)
        self.assertEqual(v2.source_format_version, 2)
        self.assertEqual(v3.filled_fields, ())
        for record in batch.records:
            self.assertEqual(set(record.plugin_scores), set(PLUGIN_NAMES))
            self.assertEqual(record.to_v3_dict()["format_version"], 3)

    def test_encoders_round_trip(self) -> None:
        params = NormalizationParams.from_spec(load_spec(self.sample["spec"]))
        batch = load_records(self.sample["records"], params=params)
        for record in batch.records:
            for text, version in (
                (encode_v1(record), 1),
                (encode_v2(record), 2),
                (encode_v3(record), 3),
            ):
                decoded = decode_line(
                    text.strip() if text.endswith("\n") else text,
                    source_line=1,
                    source_path="<roundtrip>",
                    params=params,
                )
                self.assertEqual(decoded.pod_id, record.pod_id)
                self.assertEqual(decoded.node_name, record.node_name)
                self.assertEqual(decoded.veto_code, record.veto_code)
                if version == 3:
                    self.assertEqual(decoded.total_score, record.total_score)

    def test_v1_short_width_is_format_error(self) -> None:
        line = encode_v1(load_records(self.sample["records"]).records[0])
        with self.assertRaises(FormatError) as caught:
            decode_line(line[:-1], source_line=1, source_path="x")
        self.assertEqual(caught.exception.code, E_FORMAT)

    def test_v1_long_width_is_width_error(self) -> None:
        line = encode_v1(load_records(self.sample["records"]).records[0])
        with self.assertRaises(WidthError) as caught:
            decode_line(line + "X", source_line=1, source_path="x")
        self.assertEqual(caught.exception.code, E_WIDTH)
        self.assertEqual(caught.exception.details["actual_width"], V1_WIDTH + 1)

    def test_unknown_version(self) -> None:
        with self.assertRaises(FormatError) as caught:
            decode_line(
                json.dumps({"format_version": 9}), source_line=4, source_path="x"
            )
        self.assertEqual(caught.exception.code, E_FORMAT)

    def test_json_declared_as_v1_is_rejected_as_format_error(self) -> None:
        with self.assertRaises(FormatError) as caught:
            decode_line(
                json.dumps({"format_version": 1, "pod_id": "pod-0001"}),
                source_line=1,
                source_path="x",
            )
        self.assertEqual(caught.exception.code, E_FORMAT)

    def test_missing_v2_field(self) -> None:
        value = {
            "format_version": 2,
            "pod_id": "p",
            "node_name": "n",
            "plugin_scores": {name: 1.0 for name in PLUGIN_NAMES},
            "total_score": 1.0,
            "tie_break": "total_score,node_name",
        }
        with self.assertRaises(FormatError) as caught:
            decode_line(json.dumps(value), source_line=1, source_path="x")
        self.assertEqual(caught.exception.field_path, "veto_code")

    def test_score_and_veto_validation(self) -> None:
        batch = load_records(self.sample["records"])
        value = batch.records[1].to_v2_dict()
        value["total_score"] = 101.0
        with self.assertRaises(FormatError) as caught:
            decode_line(json.dumps(value), source_line=1, source_path="x")
        self.assertEqual(caught.exception.field_path, "total_score")
        value = batch.records[1].to_v2_dict()
        value["veto_code"] = 99
        with self.assertRaises(FormatError) as caught:
            decode_line(json.dumps(value), source_line=1, source_path="x")
        self.assertEqual(caught.exception.field_path, "veto_code")


if __name__ == "__main__":
    unittest.main()
