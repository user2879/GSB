import os
import tempfile
import unittest
from pathlib import Path

from rlcb.codes import E_DIVERGENCE, DivergenceError, LedgerError
from rlcb.engine import Engine
from rlcb.ledger import LEDGER_FORMAT_VERSION, Ledger
from rlcb.planner import choose_plan
from rlcb.spec import load_bundle

SERVICES = Path("samples/acceptance/services.json")
REQUESTS = Path("samples/acceptance/requests.json")


def full_plan(bundle):
    batches = sorted({f"{r.service_id}:{r.batch_id}"
                      for r in bundle.requests})
    windows = 0
    for spec in bundle.services:
        reqs = [r for r in bundle.requests if r.service_id == spec.service_id]
        if reqs:
            windows += max(r.arrival_seq for r in reqs) // spec.window_length + 1
    return choose_plan(service_count=len(bundle.services),
                       batch_count=len(batches), window_count=windows,
                       changed_services=[], changed_batches=[],
                       all_batches=batches, force_mode="full")


def commit_run(ledger_path, bundle):
    ledger = Ledger(ledger_path)
    engine = Engine(bundle, use_workers=False)
    result = engine.run(full_plan(bundle))
    by_batch = {}
    for decision in result.decisions:
        by_batch.setdefault(
            f"{decision['service_id']}:{decision['batch_id']}", []
        ).append(decision)
    for batch_key in sorted(by_batch):
        service_id, batch_id = batch_key.split(":", 1)
        spec = bundle.service_by_id[service_id]
        ledger.commit_batch(
            batch={"batch_id": batch_id, "service_id": service_id,
                   "config_version": spec.config_version},
            decisions=by_batch[batch_key],
            lease_ownership=result.lease_ownership,
            lease_events=result.lease_events,
            invalidation=None,
            cursor={"last_batch": batch_key,
                    "format_version": LEDGER_FORMAT_VERSION})
    fingerprint = ledger.snapshot_fingerprint()
    ledger.close()
    return result, fingerprint


class LedgerTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.path = os.path.join(self.tmp, "ledger.db")
        self.bundle = load_bundle(SERVICES, REQUESTS)

    def test_atomic_commit_and_replay_match(self):
        result, fingerprint = commit_run(self.path, self.bundle)
        ledger = Ledger(self.path)
        outcome = ledger.verify_replay(result.decisions)
        self.assertEqual(outcome["matched_requests"], len(result.decisions))
        self.assertIsNone(outcome["divergence"])
        self.assertEqual(ledger.snapshot_fingerprint(), fingerprint)
        ledger.close()

    def test_divergence_detected_at_first_request(self):
        result, _ = commit_run(self.path, self.bundle)
        ledger = Ledger(self.path)
        tampered = [dict(d) for d in result.decisions]
        tampered[5]["failure_rate_bp"] += 1
        with self.assertRaises(DivergenceError) as ctx:
            ledger.verify_replay(tampered)
        self.assertEqual(ctx.exception.code, E_DIVERGENCE)
        self.assertEqual(ctx.exception.context["request_id"],
                         tampered[5]["request_id"])
        ledger.close()

    def test_snapshot_mismatch_replaces_whole_snapshot(self):
        _, fingerprint = commit_run(self.path, self.bundle)
        ledger = Ledger(self.path)
        bogus = {"format_version": LEDGER_FORMAT_VERSION,
                 "digest": "sha256:deadbeef"}
        action = ledger.replace_with(bogus)
        self.assertEqual(action, "replaced")
        self.assertEqual(ledger.load_decisions(), {})
        self.assertEqual(ledger.committed_batch_ids(), set())
        # An identical fingerprint is a no-op match.
        ledger.close()
        _, fingerprint2 = commit_run(self.path, self.bundle)
        ledger = Ledger(self.path)
        self.assertEqual(ledger.replace_with(fingerprint2), "matched")
        ledger.close()

    def test_format_version_mismatch_rejected(self):
        commit_run(self.path, self.bundle)
        ledger = Ledger(self.path)
        ledger._write_meta("format_version", 999)
        ledger.close()
        with self.assertRaises(LedgerError):
            Ledger(self.path)

    def test_crash_recovery_marks_unknown_batches_and_reclaims_quota(self):
        # Commit only the first batch, simulating a forced termination before
        # the remaining batches committed.
        ledger = Ledger(self.path)
        engine = Engine(self.bundle, use_workers=False)
        result = engine.run(full_plan(self.bundle))
        first_key = sorted({f"{d['service_id']}:{d['batch_id']}"
                            for d in result.decisions})[0]
        service_id, batch_id = first_key.split(":", 1)
        decisions = [d for d in result.decisions
                     if f"{d['service_id']}:{d['batch_id']}" == first_key]
        spec = self.bundle.service_by_id[service_id]
        ledger.commit_batch(
            batch={"batch_id": batch_id, "service_id": service_id,
                   "config_version": spec.config_version},
            decisions=decisions, lease_ownership={}, lease_events=[],
            invalidation=None,
            cursor={"last_batch": first_key,
                    "format_version": LEDGER_FORMAT_VERSION})
        ledger.close()

        reopened = Ledger(self.path)
        all_batches = sorted({f"{r.service_id}:{r.batch_id}"
                              for r in self.bundle.requests})
        state = reopened.recover_state(all_batches)
        self.assertIn(first_key, state["committed_batches"])
        self.assertEqual(state["unknown_batches"],
                         [b for b in all_batches if b != first_key])
        # Unknown batches' quota is reclaimed before recompute.
        self.assertEqual(state["reclaimed_quota_for"],
                         state["unknown_batches"])
        # Full replay against committed ledger: committed requests match and
        # the uncommitted ones are reported as the first divergence point.
        with self.assertRaises(DivergenceError) as ctx:
            reopened.verify_replay(result.decisions)
        self.assertEqual(ctx.exception.code, E_DIVERGENCE)
        reopened.close()


if __name__ == "__main__":
    unittest.main()
