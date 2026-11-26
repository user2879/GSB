import unittest

from leasebatch.errors import FieldError, RankError, RefError
from leasebatch.receipt import decode, encode, normalize


KNOWN = ["r01", "r02", "r03"]

R1 = {
    "batch_id": "w1-b0000-abc",
    "request_ids": ["r01", "r02"],
    "output_digest": "digest-x",
}
R2 = dict(R1, param_version=2, elapsed_ms=7, worker_id="w1")
R3 = dict(R2, ranks=[1, 0], superseded_refs=["r03"],
          param_digest="pd-1")


class ReceiptCodecTest(unittest.TestCase):
    def test_detect_and_round_trip_formats(self):
        for fmt, record in (("r1", R1), ("r2", R2), ("r3", R3)):
            decoded = decode(record, KNOWN)
            self.assertEqual(decoded["format"], fmt)

    def test_normalize_r1_records_filled_fields_and_basis(self):
        decoded = decode(R1, KNOWN)
        normalized = normalize(
            decoded, worker_id="w1", param_version=0,
            param_digest="pd-0",
        )
        self.assertEqual(normalized["source_format"], "r1")
        self.assertEqual(
            normalized["filled_fields"],
            sorted(["param_version", "elapsed_ms", "worker_id", "ranks",
                    "superseded_refs", "param_digest"]),
        )
        self.assertIn("merge_rules.default_param_version",
                      normalized["fill_basis"]["param_version"])
        self.assertEqual(normalized["fields"]["ranks"], [0, 1])

    def test_r3_normalize_fills_nothing(self):
        decoded = decode(R3, KNOWN)
        normalized = normalize(
            decoded, worker_id="w1", param_version=2,
            param_digest="pd-1",
        )
        self.assertEqual(normalized["filled_fields"], [])
        self.assertEqual(normalized["source_format"], "r3")

    def test_encode_projects_normalized_to_any_format(self):
        normalized = normalize(decode(R1, KNOWN), worker_id="w1",
                               param_version=0, param_digest="pd-0")
        self.assertEqual(encode(normalized, "r1"), R1)
        self.assertEqual(set(encode(normalized, "r2")), set(R2))
        self.assertEqual(set(encode(normalized, "r3")), set(R3))

    def test_length_mismatch_is_field_error(self):
        bad = dict(R3, ranks=[0])
        with self.assertRaises(FieldError) as caught:
            decode(bad, KNOWN)
        self.assertEqual(caught.exception.field, "ranks")

    def test_negative_elapsed_is_field_error(self):
        bad = dict(R2, elapsed_ms=-1)
        with self.assertRaises(FieldError) as caught:
            decode(bad, KNOWN)
        self.assertEqual(caught.exception.field, "elapsed_ms")

    def test_duplicate_rank_is_rank_error(self):
        bad = dict(R3, ranks=[0, 0])
        with self.assertRaises(RankError) as caught:
            decode(bad, KNOWN)
        self.assertEqual(caught.exception.code, "E_RANK")

    def test_rank_out_of_bounds_is_rank_error(self):
        bad = dict(R3, ranks=[0, 5])
        with self.assertRaises(RankError):
            decode(bad, KNOWN)

    def test_unknown_superseded_ref_is_ref_error(self):
        bad = dict(R3, superseded_refs=["missing"])
        with self.assertRaises(RefError) as caught:
            decode(bad, KNOWN)
        self.assertEqual(caught.exception.field, "superseded_refs")


if __name__ == "__main__":
    unittest.main()
