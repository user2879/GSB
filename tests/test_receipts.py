import unittest

from batch_receipt_ledger.errors import FieldError, RefError
from batch_receipt_ledger.plan import build_plan
from batch_receipt_ledger.receipts import (
    encode_receipt, normalize_batch, normalize_receipt)
from batch_receipt_ledger.spec import parse_spec

from tests._support import make_spec


def _context(spec, plan):
    ranks = {rid: i for i, rid in enumerate(plan.request_ids)}
    supersession = spec.supersession_map()
    return {
        "batch_size": spec.batch_size,
        "known_request_ids": spec.all_request_ids(),
        "worker_id": plan.worker_id,
        "param_version": 1,
        "rank_of": lambda rid: ranks[rid],
        "dup_ref_of": lambda rid: supersession.get(rid),
    }


class ReceiptTest(unittest.TestCase):
    def setUp(self):
        raw = make_spec(deliveries=6)
        raw["deliveries"].append({
            "request_id": "req-new", "arrival_seq": 7, "value": 9,
            "replaces": "req-002"})
        self.spec = parse_spec(raw)
        self.plan = build_plan(self.spec)[0]
        self.ctx = _context(self.spec, self.plan)
        self.r1 = [
            {"batch_id": self.plan.batch_id, "request_id": rid,
             "output_digest": f"d-{rid}"} for rid in self.plan.request_ids]

    def test_r1_normalize_records_source_and_basis(self):
        normalized = normalize_batch(self.r1, self.ctx)
        self.assertEqual(normalized[0].source_format, 1)
        self.assertIn("rank_in_batch", normalized[0].filled_fields)
        self.assertEqual(normalized[0].fill_basis["rank_in_batch"],
                         "derive:stable_arrival_order")

    def test_roundtrip_three_formats(self):
        normalized = normalize_batch(self.r1, self.ctx)
        for fmt in (1, 2, 3):
            encoded = encode_receipt(normalized[0], fmt)
            back = normalize_receipt(encoded, self.ctx)
            self.assertEqual(back.source_format, fmt)
            self.assertEqual(back.output_digest, normalized[0].output_digest)

    def test_negative_elapsed_rejected(self):
        raw = dict(self.r1[0])
        raw.update({"param_version": 1, "elapsed_ms": -1,
                    "worker_id": self.plan.worker_id})
        with self.assertRaises(FieldError) as ctx:
            normalize_receipt(raw, self.ctx)
        self.assertEqual(ctx.exception.field, "elapsed_ms")

    def test_duplicate_rank_rejected(self):
        raws = []
        for item in self.r1:
            copied = dict(item)
            copied.update({"param_version": 1, "elapsed_ms": 0,
                           "worker_id": self.plan.worker_id,
                           "rank_in_batch": 0, "dup_ref": None})
            raws.append(copied)
        with self.assertRaises(FieldError) as ctx:
            normalize_batch(raws, self.ctx)
        self.assertEqual(ctx.exception.field, "rank_in_batch")

    def test_rank_out_of_range_rejected(self):
        raw = dict(self.r1[0])
        raw.update({"param_version": 1, "elapsed_ms": 0,
                    "worker_id": self.plan.worker_id,
                    "rank_in_batch": self.spec.batch_size + 5,
                    "dup_ref": None})
        with self.assertRaises(FieldError) as ctx:
            normalize_receipt(raw, self.ctx)
        self.assertEqual(ctx.exception.field, "rank_in_batch")

    def test_dangling_dup_ref_rejected(self):
        raw = dict(self.r1[0])
        raw.update({"param_version": 1, "elapsed_ms": 0,
                    "worker_id": self.plan.worker_id,
                    "rank_in_batch": 0, "dup_ref": "ghost"})
        with self.assertRaises(RefError) as ctx:
            normalize_receipt(raw, self.ctx)
        self.assertEqual(ctx.exception.field, "dup_ref")

    def test_existing_dup_ref_accepted(self):
        target = next(rid for rid in self.plan.request_ids if rid == "req-new") \
            if "req-new" in self.plan.request_ids else self.plan.request_ids[-1]
        raw = dict(self.r1[0])
        raw["request_id"] = target
        raw.update({"param_version": 1, "elapsed_ms": 0,
                    "worker_id": self.plan.worker_id,
                    "rank_in_batch": 0, "dup_ref": "req-002"})
        normalized = normalize_receipt(raw, self.ctx)
        self.assertEqual(normalized.dup_ref, "req-002")


if __name__ == "__main__":
    unittest.main()
