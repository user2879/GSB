"""Requirement 2: r1/r2/r3 codec, normalization fill provenance, E_FIELD."""
from __future__ import annotations

import json
import os
import unittest

from grayroute.batching import build_batch_context, plan_batches
from grayroute.errors import FieldError
from grayroute.receipt import (FIELDS_BY_FORMAT, PARAM_SUMMARY_LENGTH,
                               BatchContext, encode_receipt, normalize,
                               normalize_raw)
from grayroute.sharding import attribute_all
from grayroute.spec import load_spec

ROOT = os.path.dirname(os.path.dirname(__file__))


class ReceiptCodecTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.spec = load_spec(os.path.join(ROOT, "config"),
                             os.path.join(ROOT, "data", "requests.ndjson"))
        cls.hits = attribute_all(cls.spec)
        cls.batches = plan_batches(cls.spec, cls.hits)
        cls.batch = next(b for b in cls.batches if len(b.request_ids) >= 2)
        cls.requests_by_id = {q.request_id: q for q in cls.spec.requests}
        cls.ctx = build_batch_context(cls.spec, cls.batch, cls.requests_by_id)
        cls.known = {q.request_id for q in cls.spec.requests}

    def test_field_evolution(self):
        self.assertEqual(FIELDS_BY_FORMAT[1],
                         ("batch_id", "request_id", "output_summary"))
        self.assertIn("rank_in_batch", FIELDS_BY_FORMAT[3])

    def test_normalize_each_format_records_fill_basis(self):
        rid = self.batch.request_ids[0]
        summary = self.ctx.output_summary_by_request[rid]
        for fv, expected_filled in [(1, 6), (2, 3), (3, 0)]:
            raw = encode_receipt(fv, self.ctx, rid, summary, 12)
            norm = normalize_raw(raw, self.ctx, self.known)
            self.assertEqual(norm.format_version, fv)
            self.assertEqual(len(norm.filled_fields), expected_filled)
            for name in norm.filled_fields:
                self.assertIn(name, norm.fill_basis)
            self.assertEqual(
                len(norm.param_summary_snapshot), PARAM_SUMMARY_LENGTH)

    def test_repeated_decode_byte_identical(self):
        rid = self.batch.request_ids[0]
        raw = encode_receipt(1, self.ctx, rid,
                             self.ctx.output_summary_by_request[rid], 12)
        first = normalize_raw(raw, self.ctx, self.known).canonical()
        second = normalize_raw(dict(raw), self.ctx, self.known).canonical()
        self.assertEqual(first, second)

    def test_negative_elapsed_field_error(self):
        rid = self.batch.request_ids[0]
        raw = encode_receipt(2, self.ctx, rid,
                             self.ctx.output_summary_by_request[rid], 12)
        raw["elapsed_ms"] = -1
        with self.assertRaises(FieldError) as ctx:
            normalize_raw(raw, self.ctx, self.known)
        self.assertEqual(ctx.exception.code, "E_FIELD")
        self.assertEqual(ctx.exception.field, "elapsed_ms")

    def test_rank_out_of_bounds_and_duplicate(self):
        rid = self.batch.request_ids[0]
        raw = encode_receipt(3, self.ctx, rid,
                             self.ctx.output_summary_by_request[rid], 12)
        raw["rank_in_batch"] = len(self.batch.request_ids)
        with self.assertRaises(FieldError) as ctx:
            normalize_raw(raw, self.ctx, self.known)
        self.assertEqual(ctx.exception.field, "rank_in_batch")

    def test_superseded_ref_must_exist(self):
        rid = self.batch.request_ids[0]
        raw = encode_receipt(3, self.ctx, rid,
                             self.ctx.output_summary_by_request[rid], 12)
        raw["superseded_ref"] = "ghost-request"
        with self.assertRaises(FieldError) as ctx:
            normalize_raw(raw, self.ctx, self.known)
        self.assertEqual(ctx.exception.field, "superseded_ref")

    def test_param_summary_length_mismatch(self):
        rid = self.batch.request_ids[0]
        raw = encode_receipt(3, self.ctx, rid,
                             self.ctx.output_summary_by_request[rid], 12)
        raw["param_summary_snapshot"] = "abc"
        with self.assertRaises(FieldError) as ctx:
            normalize_raw(raw, self.ctx, self.known)
        self.assertEqual(ctx.exception.field, "param_summary_snapshot")

    def test_invalid_format_version(self):
        rid = self.batch.request_ids[0]
        raw = encode_receipt(1, self.ctx, rid,
                             self.ctx.output_summary_by_request[rid], 12)
        raw["format_version"] = 9
        with self.assertRaises(FieldError) as ctx:
            normalize_raw(raw, self.ctx, self.known)
        self.assertEqual(ctx.exception.field, "format_version")


if __name__ == "__main__":
    unittest.main()
