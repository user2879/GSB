import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from quantcalib.data.weightpack import load_weight_pack
from quantcalib.errors import (
    E_GROUP_GRANULARITY,
    E_LAYER_DUPLICATE,
    E_WEIGHT_NONFINITE,
)
from quantcalib.models import WeightLayer
from quantcalib.parsers.receipts import (
    parse_q1,
    parse_q2,
    parse_q3,
    validate_references,
    write_q1,
    write_q2,
    write_q3,
)
from quantcalib.errors import E_LAYER_UNKNOWN, E_RECEIPT_REFERENCE
from quantcalib.models import UnifiedReceipt
from quantcalib.parsers.sidecar import downgrade_q3_to_q1, upgrade_q1_to_q3


class ValidationTests(unittest.TestCase):
    def _write(self, records):
        handle = tempfile.NamedTemporaryFile(
            "w", suffix=".jsonl", delete=False, encoding="utf-8")
        for record in records:
            handle.write(json.dumps(record) + "\n")
        handle.close()
        return handle.name

    def test_duplicate_nonfinite_and_granularity_codes(self):
        good = {
            "layer": "a", "shape": [2, 2], "weights": [1.0] * 4,
            "content_version": 1, "group_granularity": 2,
        }
        dup = dict(good)
        bad_finite = {
            "layer": "b", "shape": [2], "weights": [1.0, float("nan")],
            "content_version": 2, "group_granularity": 2,
        }
        bad_group = {
            "layer": "c", "shape": [3], "weights": [1.0, 2.0, 3.0],
            "content_version": 3, "group_granularity": 2,
        }
        path = self._write([good, dup, bad_finite, bad_group])
        layers, issues = load_weight_pack(path)
        codes = {issue.code for issue in issues}
        self.assertIn(E_LAYER_DUPLICATE, codes)
        self.assertIn(E_WEIGHT_NONFINITE, codes)
        self.assertIn(E_GROUP_GRANULARITY, codes)
        self.assertEqual([layer.name for layer in layers], ["a"])

    def test_receipt_unknown_layer_and_future_param_version(self):
        receipts = [
            UnifiedReceipt(layer="ghost", coverage_permille=500,
                           source_version="q2", param_version=1, time_ms=3),
        ]
        issues = validate_references(receipts, {"real": 1})
        self.assertEqual(issues[0].code, E_LAYER_UNKNOWN)

        receipts = [
            UnifiedReceipt(layer="real", coverage_permille=500,
                           source_version="q2", param_version=9, time_ms=3),
        ]
        issues = validate_references(receipts, {"real": 1})
        self.assertEqual(issues[0].code, E_RECEIPT_REFERENCE)

    def test_q1_q2_q3_roundtrip(self):
        receipts = [
            UnifiedReceipt(
                layer="layer.fc1", coverage_permille=925, source_version="q3",
                param_version=2, time_ms=11, group_scale_digest="digest-1",
                duplicate_refs=["layer.fc2"], weight_snapshot="snap-1",
                precision_labels={},
            )
        ]
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            q1, q2, q3 = tmp / "a.q1", tmp / "a.json", tmp / "a.bin"
            write_q1(q1, [UnifiedReceipt(
                layer="layer.fc1", coverage_permille=925, source_version="q1",
                precision_labels={})])
            write_q2(q2, [UnifiedReceipt(
                layer="layer.fc1", coverage_permille=925, source_version="q2",
                param_version=2, time_ms=11, precision_labels={})])
            write_q3(q3, receipts)
            self.assertEqual(parse_q1(q1)[0].coverage_permille, 925)
            self.assertEqual(parse_q2(q2)[0].time_ms, 11)
            parsed = parse_q3(q3)[0]
            self.assertEqual(parsed.group_scale_digest, "digest-1")
            self.assertEqual(parsed.duplicate_refs, ["layer.fc2"])

    def test_sidecar_roundtrip_restores_without_defaults(self):
        receipts = [
            UnifiedReceipt(
                layer="layer.fc1", coverage_permille=925, source_version="q3",
                param_version=2, time_ms=11, group_scale_digest="digest-x",
                duplicate_refs=["layer.fc2"], weight_snapshot="snap-x",
                precision_labels={},
            )
        ]
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            q3, q1, sidecar, q3_back = (
                tmp / "a.bin", tmp / "a.q1", tmp / "side.json", tmp / "b.bin")
            write_q3(q3, receipts)
            downgraded = downgrade_q3_to_q1(q3, q1, sidecar)
            self.assertIsNone(downgraded[0].group_scale_digest)
            restored = upgrade_q1_to_q3(q1, sidecar, q3_back)
            self.assertEqual(restored[0].group_scale_digest, "digest-x")
            self.assertEqual(restored[0].duplicate_refs, ["layer.fc2"])
            self.assertEqual(restored[0].weight_snapshot, "snap-x")


if __name__ == "__main__":
    unittest.main()
