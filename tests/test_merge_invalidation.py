"""Requirements 5 & 6: version isolation, E_VERSION, merge, invalidation."""
from __future__ import annotations

import os
import unittest

from grayroute.batching import build_batch_context, plan_batches
from grayroute.errors import VersionConflictError
from grayroute.merger import (check_batch_versions, merge_receipts,
                              propagate_invalidation)
from grayroute.receipt import normalize_raw
from grayroute.sharding import attribute_all
from grayroute.spec import MergeRule, load_spec

ROOT = os.path.dirname(os.path.dirname(__file__))


class MergeInvalidationTest(unittest.TestCase):
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

    def _norm(self, fv, rid, pv=None, summary=None):
        import copy
        from grayroute.receipt import encode_receipt
        raw = encode_receipt(fv, self.ctx, rid,
                             self.ctx.output_summary_by_request[rid], 12)
        if pv is not None:
            raw["param_version_seq"] = pv
        if summary is not None and "param_summary_snapshot" in raw:
            raw["param_summary_snapshot"] = summary
        return normalize_raw(raw, self.ctx, self.known)

    def test_hot_update_versions_stay_in_separate_batches(self):
        by_worker = {}
        for batch in self.batches:
            by_worker.setdefault(batch.worker_id, set()).add(
                batch.param_version_seq)
        # every individual batch has exactly one parameter version
        for batch in self.batches:
            seqs = {self.hits[r].param_version_seq for r in batch.request_ids}
            self.assertEqual(seqs, {batch.param_version_seq})

    def test_version_conflict_raises_e_version(self):
        rid = self.batch.request_ids[0]
        receipts = [
            self._norm(3, rid, pv=1,
                       summary=self.ctx.param_summary[:16]),
            self._norm(3, rid, pv=2,
                       summary=self.ctx.param_summary[:16]),
        ]
        with self.assertRaises(VersionConflictError) as ctx:
            check_batch_versions(self.batch, receipts)
        self.assertEqual(ctx.exception.code, "E_VERSION")
        self.assertEqual(ctx.exception.object_id, self.batch.batch_id)

    def test_merge_prefers_higher_param_then_format(self):
        rid = self.batch.request_ids[0]
        snap = self.ctx.param_summary[:16]
        candidates = [
            (0, self._norm(3, rid, pv=1, summary=snap)),
            (1, self._norm(3, rid, pv=1, summary=snap)),
            (2, self._norm(3, rid, pv=2, summary=snap)),
        ]
        decision = merge_receipts(candidates, MergeRule())
        self.assertEqual(decision.adopted.param_version_seq, 2)
        self.assertEqual(len(decision.rejected), 2)
        for rejected, reason in decision.rejected:
            self.assertIn("outranked", reason)

    def test_merge_breakdown_by_merge_seq(self):
        rid = self.batch.request_ids[0]
        a = self._norm(3, rid, pv=2)
        b = self._norm(3, rid, pv=2)
        decision = merge_receipts([(5, a), (0, b)], MergeRule())
        # identical keys except merge_seq; lower merge_seq wins (asc)
        self.assertIs(decision.adopted, b)

    def test_invalidation_propagates_worker_to_requests(self):
        target = self.batch
        invalid = {(target.worker_id, target.param_version_seq)}
        rows = propagate_invalidation(self.batches, {}, invalid)
        affected_batches = {r["batch_id"] for r in rows}
        self.assertIn(target.batch_id, affected_batches)
        affected_requests = {r["request_id"] for r in rows
                             if r["batch_id"] == target.batch_id}
        self.assertEqual(set(target.request_ids), affected_requests)
        # stable ordering: worker, version, batch, request
        keys = [(r["worker_id"], r["param_version_seq"], r["batch_id"],
                 r["request_id"]) for r in rows]
        self.assertEqual(keys, sorted(keys))

    def test_non_invalidated_summary_unchanged(self):
        rid = self.batch.request_ids[0]
        norm = self._norm(3, rid)
        before = norm.content_summary()
        invalid = {("some-other-worker", 99)}
        propagate_invalidation(self.batches, {}, invalid)
        self.assertEqual(norm.content_summary(), before)


if __name__ == "__main__":
    unittest.main()
