import unittest

from receipt_ledger.convert import (
    build_sidecar,
    completeness_label,
    downgrade,
    receipt_ref,
    restore,
    round_trip,
    to_payload,
    upgrade,
)
from receipt_ledger.errors import (
    E_NON_FINITE_DURATION,
    E_RECEIPT_MALFORMED,
    E_RECEIPT_TRUNCATED,
)
from receipt_ledger.models import MISSING
from receipt_ledger.receipts import parse_receipt_line

R3 = (
    '{"version":"r3","batch_id":"b1","request_id":"q1","output_digest":"d1",'
    '"param_version":2,"duration_ms":30,"worker_id":"w1","rank_in_batch":0,'
    '"supersedes":["ref-x"],"param_summary":"ps-2"}'
)


class ReceiptParseTests(unittest.TestCase):
    def test_r1_keeps_missing_markers(self):
        receipt, failure = parse_receipt_line(
            '{"version":"r1","batch_id":"b","request_id":"q","output_digest":"d"}',
            "receipt_r1.jsonl",
            1,
        )
        self.assertIsNone(failure)
        self.assertEqual(receipt.source_version, "r1")
        self.assertEqual(
            receipt.missing,
            (
                "param_version",
                "duration_ms",
                "worker_id",
                "rank_in_batch",
                "supersedes",
                "param_summary",
            ),
        )

    def test_r2_and_r3_extra_field_is_malformed(self):
        _, failure = parse_receipt_line(
            '{"version":"r2","batch_id":"b","request_id":"q","output_digest":"d",'
            '"param_version":1,"duration_ms":1,"worker_id":"w","rank_in_batch":1}',
            "receipt_r2.jsonl",
            2,
        )
        self.assertEqual(failure.code, E_RECEIPT_MALFORMED)
        self.assertEqual(failure.line, 2)
        self.assertIn("r2", failure.message)

    def test_truncated_line_code_and_non_finite_duration(self):
        _, truncated = parse_receipt_line(
            '{"version":"r3","batch_id":"b"', "receipt_r3.jsonl", 7
        )
        self.assertEqual(truncated.code, E_RECEIPT_TRUNCATED)
        self.assertEqual(truncated.line, 7)

        _, non_finite = parse_receipt_line(
            '{"version":"r2","batch_id":"b","request_id":"q","output_digest":"d",'
            '"param_version":1,"duration_ms":NaN,"worker_id":"w"}',
            "receipt_r2.jsonl",
            1,
        )
        self.assertEqual(non_finite.code, E_NON_FINITE_DURATION)

    def test_round_trip_preserves_ref_and_attribution_identity(self):
        original, _ = parse_receipt_line(R3, "receipt_r3.jsonl", 1)
        restored, r1_payload, sidecar = round_trip(original)
        self.assertEqual(r1_payload["version"], "r1")
        self.assertNotIn("worker_id", r1_payload)
        self.assertEqual(sidecar["fields"]["worker_id"], "w1")
        self.assertEqual(restored.rank_in_batch, 0)
        self.assertEqual(restored.supersedes, ("ref-x",))
        self.assertEqual(restored.param_summary, "ps-2")
        self.assertEqual(receipt_ref(restored), receipt_ref(original))
        self.assertEqual(restored.integrity_tag, original.integrity_tag)
        self.assertEqual(build_sidecar(original)["original_ref"], receipt_ref(original))

    def test_upgrade_r1_to_r3_explicit_missing_and_label(self):
        r1, _ = parse_receipt_line(
            '{"version":"r1","batch_id":"b","request_id":"q","output_digest":"d"}',
            "receipt_r1.jsonl",
            1,
        )
        r3 = upgrade(r1, "r3")
        payload = to_payload(r3, "r3")
        self.assertEqual(payload["param_version"], MISSING)
        self.assertEqual(payload["supersedes"], MISSING)
        self.assertTrue(completeness_label(r3).startswith("incomplete"))

    def test_restore_from_sidecar_matches_original(self):
        original, _ = parse_receipt_line(R3, "receipt_r3.jsonl", 1)
        downgraded, _ = downgrade(original, "r1")
        r1_payload = to_payload(downgraded, "r1")
        sidecar = build_sidecar(original)
        restored = restore(r1_payload, sidecar)
        self.assertEqual(restored.as_ledger_dict(), original.as_ledger_dict())


if __name__ == "__main__":
    unittest.main()
