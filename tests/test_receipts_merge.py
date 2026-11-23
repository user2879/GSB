from __future__ import annotations

import copy
import random
import unittest

from gray_merge_lab.errors import LabError
from gray_merge_lab.merge import merge_receipts
from gray_merge_lab.receipts import (
    decode_receipt,
    decode_receipts,
    encode_receipt,
    make_receipt,
)


DIGEST_A = "sha256:" + "a" * 64
DIGEST_B = "sha256:" + "b" * 64


def h1(batch="batch-1", destination="canary"):
    return make_receipt("h1", batch, destination, DIGEST_A)


def h2(batch="batch-1", destination="canary"):
    return make_receipt(
        "h2",
        batch,
        destination,
        DIGEST_A,
        client_bucket=123,
        weight_bp=2500,
        duration_ms=5,
    )


def h3(batch="batch-1", *, seq=2, version=1, samples=("s1", "s2"), digest=DIGEST_A):
    return make_receipt(
        "h3",
        batch,
        "canary",
        digest,
        client_bucket=123,
        weight_bp=2500,
        duration_ms=5,
        sample_ids=samples,
        backend_param_digest=DIGEST_B,
        backend_param_version=version,
        merge_seq=seq,
    )


class ReceiptTests(unittest.TestCase):
    def test_roundtrip_h1_h2_h3(self):
        for receipt in (h1(), h2(), h3()):
            encoded = encode_receipt(receipt)
            decoded = decode_receipt(encoded)
            self.assertEqual(decoded, receipt)

    def test_h1_contains_only_base_fields(self):
        encoded = encode_receipt(h3(), "h1")
        self.assertEqual(set(encoded), {"h", "batch_id", "destination", "output_digest"})

    def test_h2_requires_h2_fields(self):
        with self.assertRaises(LabError) as caught:
            encode_receipt(h1(), "h2")
        self.assertEqual(caught.exception.code, "E_FORMAT")

    def test_h3_requires_h3_fields(self):
        with self.assertRaises(LabError) as caught:
            encode_receipt(h2(), "h3")
        self.assertEqual(caught.exception.code, "E_FORMAT")

    def test_unsupported_version(self):
        raw = h3().to_dict()
        raw["h"] = "h9"
        with self.assertRaises(LabError) as caught:
            decode_receipt(raw)
        self.assertEqual(caught.exception.code, "E_FORMAT")
        self.assertEqual(caught.exception.field, "h")

    def test_missing_required_field_preserves_unknown_candidates(self):
        raw = h3().to_dict()
        del raw["backend_param_digest"]
        raw["legacy_param_summary"] = "candidate"
        with self.assertRaises(LabError) as caught:
            decode_receipt(raw)
        self.assertEqual(caught.exception.code, "E_FORMAT")
        self.assertEqual(caught.exception.field, "backend_param_digest")
        self.assertIn("legacy_param_summary", caught.exception.details["pending_candidates"])

    def test_invalid_digest_and_negative_duration(self):
        raw = h3().to_dict()
        raw["output_digest"] = "sha256:oops"
        with self.assertRaises(LabError) as caught:
            decode_receipt(raw)
        self.assertEqual(caught.exception.code, "E_FORMAT")
        raw = h2().to_dict()
        raw["duration_ms"] = -1
        with self.assertRaises(LabError) as caught:
            decode_receipt(raw)
        self.assertEqual(caught.exception.code, "E_FORMAT")
        self.assertEqual(caught.exception.field, "duration_ms")

    def test_h3_sample_ids_must_be_unique(self):
        raw = h3().to_dict()
        raw["sample_ids"] = ["s1", "s1"]
        with self.assertRaises(LabError) as caught:
            decode_receipt(raw)
        self.assertEqual(caught.exception.code, "E_FORMAT")
        self.assertEqual(caught.exception.field, "sample_ids[1]")

    def test_duplicate_batch_sequence_in_collection(self):
        with self.assertRaises(LabError) as caught:
            decode_receipts([h3(seq=2), h3(seq=2, samples=("s2",))])
        self.assertEqual(caught.exception.code, "E_FORMAT")
        self.assertEqual(caught.exception.details["merge_seq"], 2)

    def test_unknown_fields_are_preserved(self):
        raw = h3().to_dict()
        raw["controller_hint"] = {"repair": True}
        decoded = decode_receipt(raw)
        self.assertEqual(decoded.extras["controller_hint"], {"repair": True})
        self.assertEqual(encode_receipt(decoded)["controller_hint"], {"repair": True})


class MergeTests(unittest.TestCase):
    def test_three_formats_adopt_h3(self):
        conclusion = merge_receipts([h2(), h1(), h3()], expected_sample_ids=["s1", "s2"])
        self.assertEqual(conclusion["destination"], "canary")
        self.assertEqual(conclusion["merge_seq"], 2)
        self.assertEqual(conclusion["backend_param_version"], 1)
        self.assertEqual([item["format"] for item in conclusion["comparison_order"]], ["h1", "h2", "h3"])
        self.assertEqual(len(conclusion["rejected"]), 2)
        self.assertTrue(all(item["rejection_reason"] == "superseded_by_higher_merge_seq" for item in conclusion["rejected"]))

    def test_arrival_order_does_not_change_conclusion(self):
        receipts = [h1(), h2(), h3(seq=2, version=1), h3(seq=3, version=2, digest="sha256:" + "c" * 64)]
        expected = merge_receipts(receipts, expected_sample_ids=["s1", "s2"])
        for seed in range(12):
            shuffled = copy.deepcopy(receipts)
            random.Random(seed).shuffle(shuffled)
            self.assertEqual(merge_receipts(shuffled, expected_sample_ids=["s1", "s2"]), expected)

    def test_hot_update_prefers_higher_sequence_then_version(self):
        old = h3(seq=2, version=1, samples=("s1", "s2"))
        new = h3(seq=3, version=2, samples=("s1", "s3"), digest="sha256:" + "d" * 64)
        conclusion = merge_receipts([old, h1(), h2(), new], expected_sample_ids=["s1", "s2", "s3"])
        self.assertEqual(conclusion["merge_seq"], 3)
        self.assertEqual(conclusion["backend_param_version"], 2)
        self.assertEqual(conclusion["sample_ids"], ["s1", "s3"])
        discrepancies = [
            item
            for rejected in conclusion["rejected"]
            for item in rejected["sample_discrepancies"]
        ]
        self.assertIn({"sample_id": "s2", "in_rejected": True, "in_adopted": False, "kind": "rejected_only"}, discrepancies)
        self.assertIn({"sample_id": "s3", "in_rejected": False, "in_adopted": True, "kind": "adopted_only"}, discrepancies)

    def test_same_sequence_is_rejected_even_with_different_param_version(self):
        first = h3(seq=5, version=1, digest="sha256:" + "1" * 64)
        second = h3(seq=5, version=2, digest="sha256:" + "2" * 64)
        with self.assertRaises(LabError) as caught:
            merge_receipts([first, second], expected_sample_ids=["s1", "s2"])
        self.assertEqual(caught.exception.code, "E_FORMAT")
        self.assertEqual(caught.exception.details["merge_seq"], 5)

    def test_param_version_can_be_inferred_from_digest_map(self):
        raw = h3(seq=2).to_dict()
        raw.pop("backend_param_version", None)
        conclusion = merge_receipts(
            [raw],
            expected_sample_ids=["s1", "s2"],
            param_versions={DIGEST_B: 7},
        )
        self.assertEqual(conclusion["backend_param_version"], 7)

    def test_h1_h2_missing_sample_set_uses_expected_batch_members(self):
        conclusion = merge_receipts([h1(), h2()], expected_sample_ids=["s2", "s1"])
        self.assertEqual(conclusion["sample_ids"], ["s1", "s2"])
        self.assertEqual(conclusion["adopted_receipt"]["format"], "h2")

    def test_mixed_batch_ids_rejected(self):
        with self.assertRaises(LabError) as caught:
            merge_receipts([h1(batch="a"), h1(batch="b")])
        self.assertEqual(caught.exception.code, "E_FORMAT")


if __name__ == "__main__":
    unittest.main()
