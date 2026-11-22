import tempfile
import unittest
from pathlib import Path

from lease_diag.clock import VirtualClock
from lease_diag.engine import Engine
from lease_diag.errors import E_GAP, E_REPLAY, E_SNAPSHOT, LeaseError
from lease_diag.ledger import Ledger
from lease_diag.model import Pool, Request
from lease_diag.replay import (assert_no_gap, compare_lease_ownership,
                               replay_engine)
from lease_diag.runtime import compute_watermarks
from lease_diag.snapshot import (build_snapshot, expected_versions,
                                 verify_snapshot, whole_replace_leases)


def pools():
    return {"P1": Pool("P1", 2, 100, 30, 1, 2),
            "P2": Pool("P2", 1, 80, 20, 1, 1)}


def requests():
    return [
        Request("r1", "k1", "P1", 1, 4, 0),
        Request("r2", "k2", "P1", 1, 3, 10),
        Request("r3", "k1", "P1", 2, 4, 20),
        Request("r4", "k3", "P2", 1, 2, 30),
    ]


def drive(engine, reqs, clock):
    for request in reqs:
        clock.advance(max(0, request.arrive_ms - clock.now_ms()))
        engine.submit(request, clock.now_ms())


class LedgerTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = Path(self.tmp.name) / "ledger.sqlite"

    def tearDown(self):
        self.tmp.cleanup()

    def _committed_scenario(self):
        ledger = Ledger(self.path)
        engine = Engine(pools(), {})
        clock = VirtualClock()
        drive(engine, requests(), clock)
        batch_id = ledger.begin_batch(clock.now_ms())
        events = [event.to_dict() for event in engine.events]
        marks = compute_watermarks(engine, pools(), {}, 10000)
        snapshot = build_snapshot(
            pool_set_version="pool-set/v1", rollback_version="rollback/v1",
            scale=10000, watermarks=marks,
            leases={lid: lease.to_dict()
                    for lid, lease in engine.leases.items()},
            cursor={"last_request_id": "r4",
                    "last_event_seq": engine._seq})
        ledger.commit_batch(
            batch_id, events=events,
            leases={lid: lease.to_dict()
                    for lid, lease in engine.leases.items()},
            watermarks=marks, last_request_id="r4",
            committed_ms=clock.now_ms(), snapshot=snapshot)
        ledger.close()
        return events

    def test_batch_atomic_commit_and_progress_cursor(self):
        events = self._committed_scenario()
        ledger = Ledger(self.path)
        self.assertEqual(ledger.recovery_report()["unknown_open_batches"], [])
        self.assertEqual(ledger.progress()["last_request_id"], "r4")
        self.assertEqual(len(ledger.events()), len(events))
        self.assertTrue(ledger.latest_watermarks())
        ledger.close()

    def test_open_batch_is_unknown_and_requires_rerun(self):
        self._committed_scenario()
        ledger = Ledger(self.path)
        open_id = ledger.begin_batch(999)
        ledger.close()
        # Reopen without commit: the OPEN batch must be treated as unknown.
        ledger = Ledger(self.path)
        report = ledger.recovery_report()
        self.assertEqual(report["unknown_open_batches"], [open_id])
        self.assertTrue(report["rerun_required"])
        ledger.close()

    def test_snapshot_digest_mismatch_raises_e_snapshot(self):
        self._committed_scenario()
        ledger = Ledger(self.path)
        snapshot = ledger.load_snapshot_row()
        expected = expected_versions(
            pool_set_version="pool-set/v1",
            rollback_version="rollback/v1", scale=10000)
        verify_snapshot(snapshot, expected)  # no raise
        tampered = dict(snapshot["payload"])
        tampered["scale"] = 9999
        ledger.replace_snapshot_payload(tampered, snapshot["digest"])
        bad = ledger.load_snapshot_row()
        with self.assertRaises(LeaseError) as ctx:
            verify_snapshot(bad, expected)
        self.assertEqual(ctx.exception.code, E_SNAPSHOT)
        ledger.close()

    def test_whole_state_replacement_recovers_from_ledger(self):
        self._committed_scenario()
        ledger = Ledger(self.path)
        engine = Engine(pools(), {})
        summary = whole_replace_leases(engine, ledger)
        self.assertEqual(summary["active_leases"],
                         len(engine.active_index))
        self.assertIn("L-k1-P1-001", engine.leases)
        ledger.close()


class ReplayTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = Path(self.tmp.name) / "ledger.sqlite"
        self.pools = pools()
        self.requests = requests()

    def tearDown(self):
        self.tmp.cleanup()

    def _ledger_with_committed_run(self):
        ledger = Ledger(self.path)
        engine = Engine(self.pools, {})
        clock = VirtualClock()
        drive(engine, self.requests, clock)
        batch_id = ledger.begin_batch(clock.now_ms())
        marks = compute_watermarks(engine, self.pools, {}, 10000)
        ledger.commit_batch(
            batch_id, events=[e.to_dict() for e in engine.events],
            leases={lid: lease.to_dict()
                    for lid, lease in engine.leases.items()},
            watermarks=marks, last_request_id="r4",
            committed_ms=clock.now_ms())
        ledger.close()

    def test_replay_matches_ledger(self):
        self._ledger_with_committed_run()
        ledger = Ledger(self.path)
        engine, comparisons = replay_engine(
            ledger, self.pools, {}, self.requests)
        self.assertEqual(len(comparisons), len(self.requests))
        self.assertTrue(all(item["match"] for item in comparisons))
        compare_lease_ownership(engine, ledger)
        ledger.close()

    def test_missing_renewal_event_raises_e_gap(self):
        self._ledger_with_committed_run()
        ledger = Ledger(self.path)
        gap_requests = [r for r in self.requests
                        if r.request_id != "r3"]  # remove the renewal
        with self.assertRaises(LeaseError) as ctx:
            assert_no_gap(ledger, gap_requests)
        self.assertEqual(ctx.exception.code, E_GAP)
        self.assertIn("r3", " ".join(ctx.exception.locations))
        ledger.close()

    def test_replay_divergence_reports_first_request_and_field(self):
        self._ledger_with_committed_run()
        ledger = Ledger(self.path)
        # Tamper a ledger lease state so ownership comparison must diverge.
        row = ledger.conn.execute(
            "SELECT lease_id, batch_id, payload FROM lease_states "
            "WHERE lease_id LIKE 'L-k1%' ORDER BY batch_id DESC LIMIT 1"
        ).fetchone()
        import json
        payload = json.loads(row["payload"])
        payload["renew_count"] = 999
        ledger.conn.execute(
            "UPDATE lease_states SET payload=? WHERE lease_id=? AND batch_id=?",
            (json.dumps(payload, sort_keys=True), row["lease_id"],
             row["batch_id"]))
        ledger.conn.commit()
        engine, _ = replay_engine(ledger, self.pools, {}, self.requests)
        with self.assertRaises(LeaseError) as ctx:
            compare_lease_ownership(engine, ledger)
        self.assertEqual(ctx.exception.code, E_REPLAY)
        self.assertIn("renew_count", " ".join(ctx.exception.locations))
        ledger.close()


if __name__ == "__main__":
    unittest.main()
