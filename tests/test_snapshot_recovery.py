"""Requirement 8: snapshot wholesale replacement, unknown batches, E_REPLAY."""
from __future__ import annotations

import json
import os
import tempfile
import unittest

from grayroute.batching import build_batch_context, plan_batches
from grayroute.errors import ReplayError, SnapshotError
from grayroute.ledger import Ledger
from grayroute.recovery import classify_batches, replay_unfinished
from grayroute.receipt import encode_receipt
from grayroute.sharding import attribute_all
from grayroute.snapshot import (build_snapshot, load_snapshot,
                                restore_or_replace, save_snapshot)
from grayroute.spec import load_spec

ROOT = os.path.dirname(os.path.dirname(__file__))


class SnapshotRecoveryTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.spec = load_spec(os.path.join(ROOT, "config"),
                             os.path.join(ROOT, "data", "requests.ndjson"))
        cls.hits = attribute_all(cls.spec)
        cls.batches = plan_batches(cls.spec, cls.hits)
        cls.requests_by_id = {q.request_id: q for q in cls.spec.requests}

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.ledger_path = os.path.join(self.tmp.name, "ledger.sqlite")
        self.snap_path = os.path.join(self.tmp.name, "snap.json")

    def tearDown(self):
        self.tmp.cleanup()

    def _commit_one(self, ledger, batch):
        context = build_batch_context(self.spec, batch, self.requests_by_id)
        adopted, raws = [], []
        for rid in batch.request_ids:
            raw = encode_receipt(3, context, rid,
                                 context.output_summary_by_request[rid], 12)
            raws.append(raw)
            norm_payload = None
            from grayroute.receipt import normalize_raw
            normalized = normalize_raw(
                raw, context, set(self.requests_by_id))
            adopted.append({
                "batch_id": batch.batch_id, "request_id": rid,
                "source_format": 3,
                "content_summary": normalized.content_summary(),
                "normalized": normalized.to_dict()})
        ledger.commit_batch(batch=batch, hits=self.hits, adopted=adopted,
                            rejected=[], invalidations=[],
                            wait_start_ms=0, wait_end_ms=1)
        return raws

    def test_snapshot_roundtrip_and_tamper_detection(self):
        snapshot = build_snapshot(rule_set_version=1, worker_set_version=1,
                                  param_version_seq=2,
                                  ledger_summary="abc", cursor={"x": 1})
        save_snapshot(self.snap_path, snapshot)
        loaded = load_snapshot(self.snap_path)
        self.assertEqual(loaded.snapshot_summary, snapshot.snapshot_summary)
        with open(self.snap_path) as fh:
            tampered = json.load(fh)
        tampered["param_version_seq"] = 99
        with open(self.snap_path, "w") as fh:
            json.dump(tampered, fh)
        with self.assertRaises(SnapshotError):
            load_snapshot(self.snap_path)

    def test_version_mismatch_replaces_wholesale(self):
        first = build_snapshot(rule_set_version=1, worker_set_version=1,
                               param_version_seq=1, ledger_summary="sum-1",
                               cursor={})
        restore_or_replace(self.snap_path, first)
        second = build_snapshot(rule_set_version=2, worker_set_version=1,
                                param_version_seq=1, ledger_summary="sum-1",
                                cursor={})
        _, action = restore_or_replace(self.snap_path, second)
        self.assertEqual(action, "replaced")
        self.assertEqual(load_snapshot(self.snap_path).rule_set_version, 2)
        _, action = restore_or_replace(
            self.snap_path,
            build_snapshot(rule_set_version=2, worker_set_version=1,
                           param_version_seq=1, ledger_summary="sum-1",
                           cursor={}))
        self.assertEqual(action, "restored")

    def test_unknown_batches_after_force_termination(self):
        with Ledger(self.ledger_path) as ledger:
            self._commit_one(ledger, self.batches[0])
            committed, unknown = classify_batches(ledger, self.batches)
        self.assertEqual([b.batch_id for b in committed],
                         [self.batches[0].batch_id])
        self.assertEqual(
            sorted(b.batch_id for b in unknown),
            sorted(b.batch_id for b in self.batches[1:]))
        with Ledger(self.ledger_path) as ledger:
            rows = ledger.unknown_batches()
        self.assertTrue(rows)
        self.assertIn("rerun", rows[0]["note"])

    def test_replay_matches_ledger_then_detects_first_mismatch(self):
        with Ledger(self.ledger_path) as ledger:
            raws = self._commit_one(ledger, self.batches[0])
            receipts = {self.batches[0].batch_id: raws}
            report = replay_unfinished(
                self.spec, ledger, self.batches[:1], receipts)
        self.assertEqual(report["verified"], [self.batches[0].batch_id])

        # tamper one replay receipt field -> E_REPLAY with precise location
        tampered = dict(raws[0])
        tampered["elapsed_ms"] = 4242
        bad_receipts = {self.batches[0].batch_id: [tampered] + raws[1:]}
        with Ledger(self.ledger_path) as ledger:
            with self.assertRaises(ReplayError) as ctx:
                replay_unfinished(self.spec, ledger, self.batches[:1],
                                  bad_receipts)
        self.assertEqual(ctx.exception.code, "E_REPLAY")
        self.assertEqual(ctx.exception.object_id, self.batches[0].batch_id)
        self.assertEqual(ctx.exception.field, "elapsed_ms")
        self.assertEqual(ctx.exception.extra.get("request_id"),
                         self.batches[0].request_ids[0])


if __name__ == "__main__":
    unittest.main()
