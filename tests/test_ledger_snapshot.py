import tempfile
import unittest
from pathlib import Path

from distlock.errors import ReplayError, SnapshotError
from distlock.ledger import Ledger, LEDGER_FORMAT_VERSION
from distlock.models import Domain
from distlock.snapshot import (
    compute_digest, read_snapshot, replay_ledger_state, verify_or_replace,
    verify_replay, write_snapshot,
)


def receipt(seq, did="d1", holder="h1", verdict="GRANTED", active=1, queued=0):
    return {"seq": seq, "domain_id": did, "holder_id": holder,
            "verdict": verdict, "remaining_ms": 80, "clock_ms": seq,
            "state": "granted", "queue_length": queued, "active_leases": active,
            "free_capacity": 1 - active, "ratio_fixed": 10000, "scale": 10000,
            "error_bound_fixed": 1, "reason": "x", "unknown": False}


def transition(seq, did="d1", holder="h1", frm=None, to="granted"):
    return {"seq": seq, "domain_id": did, "holder_id": holder,
            "state_from": frm, "state_to": to, "reason": "capacity_available",
            "started_ms": seq, "expires_ms": seq + 100, "domain_version": 1,
            "clock_ms": seq}


def watermark(seq, did="d1", active=1, queued=0):
    return {"seq": seq, "domain_id": did, "active_leases": active,
            "capacity": 1, "free_capacity": 1 - active, "queue_length": queued,
            "ratio_fixed": 10000, "scale": 10000, "truncation": "floor",
            "error_bound_fixed": 1}


class LedgerTests(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.ledger = Ledger(self.tmp / "l.sqlite3")

    def tearDown(self):
        self.ledger.close()

    def test_batch_atomic_and_cursor(self):
        class E:
            def __init__(self, seq):
                self.seq = seq
                self.domain_id = "d1"
                self.holder_id = "h1"
                self.action = "ACQUIRE"
        self.ledger.mark_inflight([E(1), E(2)], 0)
        with self.ledger.batch(1, 2, 0) as btx:
            btx.add_receipt(receipt(1))
            btx.add_receipt(receipt(2))
            btx.add_transitions([transition(1), transition(2)])
            btx.add_watermark(watermark(1))
            btx.add_watermark(watermark(2))
        self.ledger.resolve_inflight([1, 2])
        self.assertEqual(self.ledger.progress_cursor(), 2)
        self.assertEqual(self.ledger.batch_count(), 1)
        self.assertEqual(len(self.ledger.all_receipts()), 2)
        self.assertTrue(self.ledger.latest_digest())

    def test_uncommitted_inflight_becomes_unknown(self):
        class E:
            def __init__(self, seq):
                self.seq = seq
                self.domain_id = "d1"
                self.holder_id = "h1"
                self.action = "ACQUIRE"
        self.ledger.mark_inflight([E(7)], 0)
        # No batch commit -> simulates forced termination; reopen the ledger.
        self.ledger.close()
        ledger = Ledger(self.tmp / "l.sqlite3")
        unknown = ledger.recover_unknown(1)
        self.assertEqual([u["seq"] for u in unknown], [7])
        self.assertEqual(ledger.unknown_events()[0]["seq"], 7)
        ledger.close()

    def test_format_version_present(self):
        row = self.ledger.conn.execute(
            "SELECT value FROM meta WHERE key='format_version'").fetchone()
        self.assertEqual(int(row["value"]), LEDGER_FORMAT_VERSION)


class SnapshotTests(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.path = self.tmp / "snap.json"
        self.domains = {"d1": Domain("d1", 1, 100, 20, 2, 1)}

    def _payload(self, cursor=1):
        return {"domain_set_version": 1, "scale_factor": 10000,
                "ledger_cursor": cursor, "ledger_batch_count": 1,
                "ledger_digest": "abc",
                "domains": {"d1": {
                    "domain_version": 1, "capacity": 1,
                    "active_leases": [{"holder_id": "h1", "state": "granted",
                                       "started_ms": 1, "expires_ms": 101}],
                    "queue": [],
                    "watermark": {"active_leases": 1, "free_capacity": 0,
                                  "queue_length": 0, "ratio_fixed": 10000,
                                  "error_bound_fixed": 1}}}}

    def test_digest_tamper_is_e_snapshot(self):
        write_snapshot(self.path, self._payload())
        import json
        raw = json.loads(self.path.read_text(encoding="utf-8"))
        raw["digest"] = "0" * 64
        self.path.write_text(json.dumps(raw), encoding="utf-8")
        with self.assertRaises(SnapshotError) as ctx:
            read_snapshot(self.path)
        self.assertEqual(ctx.exception.code, "E_SNAPSHOT")
        self.assertEqual(ctx.exception.context["field"], "digest")

    def test_version_mismatch_replaces_with_diff_fields(self):
        payload = self._payload()
        payload["domain_set_version"] = 9
        write_snapshot(self.path, payload)
        snap, diffs = verify_or_replace(
            self.path, domain_set_version=1, scale_factor=10000,
            current_domain_ids=["d1"])
        self.assertIsNone(snap)  # whole replacement, no incremental merge
        self.assertIn("domain_set_version", diffs)

    def test_replay_mismatch_locates_seq_domain_field(self):
        write_snapshot(self.path, self._payload())
        snap = read_snapshot(self.path)
        # Ledger says no active lease, snapshot says h1 holds it.
        rows = replay_ledger_state(
            receipts=[receipt(1, verdict="RELEASED", active=0)],
            transitions=[transition(1, to="released")],
            watermarks=[watermark(1, active=0)],
            domains=self.domains, holder_priority={"h1": 5}, up_to_seq=1)
        with self.assertRaises(ReplayError) as ctx:
            verify_replay(snap, rows, self.domains, up_to_seq=1)
        self.assertEqual(ctx.exception.code, "E_REPLAY")
        self.assertEqual(ctx.exception.context["seq"], 1)
        self.assertEqual(ctx.exception.context["domain_id"], "d1")
        self.assertEqual(ctx.exception.context["field"], "active_leases")

    def test_replay_match_passes(self):
        write_snapshot(self.path, self._payload())
        snap = read_snapshot(self.path)
        rows = replay_ledger_state(
            receipts=[receipt(1)], transitions=[transition(1)],
            watermarks=[watermark(1)], domains=self.domains,
            holder_priority={"h1": 5}, up_to_seq=1)
        # Cursor must align for a fully matching snapshot.
        verify_replay(snap, rows, self.domains, up_to_seq=1)  # no raise


if __name__ == "__main__":
    unittest.main()
