from __future__ import annotations

import json
import unittest

from receipt_diag.canonical import canonical_json
from receipt_diag.errors import DiagnosticError
from receipt_diag.receipts import (
    NormalizationContext,
    decode_receipt,
    decode_receipt_batch,
    encode_receipt,
    normalize_batch,
)


def values_for(request_id: str, rank: int) -> dict:
    return {
        "batch_id": "B_test",
        "request_id": request_id,
        "output_digest": "out-digest",
        "param_version": 7,
        "duration_ms": 4,
        "worker_id": "w-one",
        "rank": rank,
        "duplicate_refs": ["req-000"],
        "param_digest": "param-digest",
    }


def contexts() -> dict[str, NormalizationContext]:
    return {
        request_id: NormalizationContext(
            worker_id="w-one",
            param_version=7,
            duration_ms=4,
            rank=rank,
            duplicate_refs=("req-000",),
            param_digest="param-digest",
        )
        for rank, request_id in enumerate(("req-001", "req-002"))
    }


class ReceiptTests(unittest.TestCase):
    def test_three_versions_normalize_to_same_attribution(self) -> None:
        payloads = [
            encode_receipt(fmt, values_for("req-001", 0))
            for fmt in ("r1", "r2", "r3")
        ]
        normalized = []
        for payload in payloads:
            decoded = decode_receipt(payload, known_request_ids=["req-000", "req-001"])
            norm = normalize_batch(
                decode_receipt_batch(
                    [payload], known_request_ids=["req-000", "req-001"], batch_size=1
                ),
                {"req-001": contexts()["req-001"]},
            )[0]
            normalized.append(norm)
            self.assertEqual(decoded.source_format, norm.source_format)
            self.assertIsInstance(norm.filled_fields, tuple)
        self.assertEqual(normalized[0].attribution(), normalized[1].attribution())
        self.assertEqual(normalized[1].attribution(), normalized[2].attribution())
        self.assertEqual(len({item.attribution_digest() for item in normalized}), 1)
        self.assertIn("param_version", normalized[0].filled_fields)
        self.assertIn("worker_id", normalized[0].fill_basis)
        self.assertEqual(normalized[2].filled_fields, ())

    def test_batch_r3_rank_validation(self) -> None:
        payloads = [
            encode_receipt("r3", values_for("req-001", 0)),
            encode_receipt("r3", values_for("req-002", 0)),
        ]
        with self.assertRaises(DiagnosticError) as caught:
            decode_receipt_batch(
                payloads,
                known_request_ids=["req-000", "req-001", "req-002"],
                batch_size=2,
            )
        self.assertEqual(caught.exception.code, "E_FIELD")
        self.assertIn("rank", caught.exception.path)

    def test_rank_out_of_range(self) -> None:
        payload = encode_receipt("r3", values_for("req-001", 2))
        with self.assertRaises(DiagnosticError) as caught:
            decode_receipt_batch(
                [payload],
                known_request_ids=["req-000", "req-001"],
                batch_size=2,
            )
        self.assertEqual(caught.exception.code, "E_FIELD")
        self.assertIn("rank", caught.exception.path)

    def test_dangling_duplicate_reference(self) -> None:
        payload = encode_receipt("r3", values_for("req-001", 0))
        with self.assertRaises(DiagnosticError) as caught:
            decode_receipt(payload, known_request_ids=["req-001"])
        self.assertEqual(caught.exception.code, "E_FIELD")
        self.assertIn("duplicate_refs", caught.exception.path)

    def test_field_count_mismatch(self) -> None:
        payload = json.loads(encode_receipt("r1", values_for("req-001", 0)))
        payload["field_count"] = 4
        with self.assertRaises(DiagnosticError) as caught:
            decode_receipt(canonical_json(payload))
        self.assertEqual(caught.exception.code, "E_FIELD")
        self.assertIn("field_count", caught.exception.path)

    def test_declared_length_mismatch(self) -> None:
        payload = json.loads(encode_receipt("r1", values_for("req-001", 0)))
        payload["fields"][0]["length"] += 1
        with self.assertRaises(DiagnosticError) as caught:
            decode_receipt(canonical_json(payload))
        self.assertEqual(caught.exception.code, "E_FIELD")
        self.assertIn("length", caught.exception.path)

    def test_negative_duration(self) -> None:
        payload = encode_receipt("r2", {**values_for("req-001", 0), "duration_ms": -1})
        with self.assertRaises(DiagnosticError) as caught:
            decode_receipt(payload)
        self.assertEqual(caught.exception.code, "E_FIELD")
        self.assertIn("duration_ms", caught.exception.path)

    def test_rank_requires_batch_size(self) -> None:
        payload = encode_receipt("r3", values_for("req-001", 0))
        with self.assertRaises(DiagnosticError) as caught:
            decode_receipt_batch(
                [payload], known_request_ids=["req-000", "req-001"]
            )
        self.assertEqual(caught.exception.code, "E_FIELD")
        self.assertIn("batch_size", caught.exception.path)


if __name__ == "__main__":
    unittest.main()
