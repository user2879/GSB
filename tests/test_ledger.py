from __future__ import annotations

import sqlite3
import tempfile
import threading
import unittest
from pathlib import Path

from gsbcache.errors import LedgerError
from gsbcache.ledger import Ledger, replace_ledger_from_snapshot
from gsbcache.verify import verify_ledger

from tests.helpers import commit_manual_lease, make_sample


class LedgerCustodyTests(unittest.TestCase):
    def test_expiry_and_quota_race_has_one_winner_and_one_release(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            spec_path, spec, _ = make_sample(root)
            ledger_path = root / "ledger.sqlite3"
            ledger = Ledger(ledger_path, spec)
            session = ledger.begin_session(1)
            commit_manual_lease(ledger, spec, session)
            results = []
            barrier = threading.Barrier(2)

            def close(kind: str) -> None:
                barrier.wait()
                results.append((kind, ledger.close_lease("lease-manual-1", kind, "race")))

            threads = [
                threading.Thread(target=close, args=("expiry",)),
                threading.Thread(target=close, args=("quota_recycle",)),
            ]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join()
            closed = [result for _kind, result in results if result.get("closed")]
            already = [result for _kind, result in results if result.get("already_closed")]
            self.assertEqual(len(closed), 1)
            self.assertEqual(len(already), 1)
            node = ledger.get_node("node-a")
            self.assertEqual(node["capacity_used"], 0)
            self.assertEqual(node["capacity_remaining"], node["capacity_slots"])
            self.assertEqual(node["active_leases"], 0)
            self.assertEqual(node["quota_remaining"], node["lease_limit"])
            releases = [
                event
                for event in ledger.get_events()
                if event["event_type"] in {"lease_expired", "quota_recycled"}
            ]
            self.assertEqual(len(releases), 1)
            self.assertEqual(releases[0]["details"]["capacity_slots_released"], 1)
            self.assertEqual(releases[0]["details"]["quota_units_released"], 1)
            self.assertEqual(releases[0]["details"]["custody_token"], "lease-manual-1")
            ledger.close_session(session)
            ledger.close()
            self.assertTrue(verify_ledger(spec_path, ledger_path)["ok"])

    def test_renewal_is_transactional_and_replayable(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            spec_path, spec, _ = make_sample(root)
            ledger_path = root / "ledger.sqlite3"
            ledger = Ledger(ledger_path, spec)
            session = ledger.begin_session(1)
            commit_manual_lease(ledger, spec, session, lease_ms=500)
            renewed_expiry = __import__("time").time_ns() // 1_000_000 + 2500
            event = ledger.renew_lease(
                "lease-manual-1",
                extend_ms=2500,
                expires_at_ms=renewed_expiry,
                deadline_mono_ns=__import__("time").monotonic_ns() + 2_500_000_000,
                details={"test": "renewal"},
            )
            self.assertEqual(event["event_type"], "lease_renewed")
            self.assertEqual(event["expires_at_ms"], renewed_expiry)
            self.assertEqual(event["details"]["capacity_slots_changed"], 0)
            self.assertEqual(event["details"]["quota_units_changed"], 0)
            node = ledger.get_node("node-a")
            self.assertEqual(node["capacity_used"], 1)
            self.assertEqual(node["active_leases"], 1)
            lease = ledger.get_lease("lease-manual-1")
            self.assertEqual(lease["expires_at_ms"], renewed_expiry)
            self.assertEqual(lease["version"], 2)
            ledger.close_lease("lease-manual-1", "expiry", "test-expiry")
            ledger.close_session(session)
            ledger.close()
            self.assertTrue(verify_ledger(spec_path, ledger_path)["ok"])

    def test_recovery_marks_pending_and_active_as_unknown(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            spec_path, spec, _ = make_sample(root)
            ledger_path = root / "ledger.sqlite3"
            ledger = Ledger(ledger_path, spec)
            session = ledger.begin_session(12345)
            commit_manual_lease(ledger, spec, session)
            pending_attempt = ledger.begin_attempt("pending-request", "src:util", "node-b", owner_session=session)
            ledger.close()
            # Reopen exactly as a restarted controller would.
            reopened = Ledger(ledger_path, spec)
            recovery = reopened.recover(current_session=None)
            self.assertFalse(recovery["exit_report"])
            self.assertTrue(recovery["requires_reapply"])
            self.assertIn(session, recovery["aborted_sessions"])
            self.assertIn("lease-manual-1", recovery["stale_leases"])
            self.assertIn(pending_attempt, recovery["unknown_attempts"])
            sessions = {item["session_id"]: item for item in reopened.get_sessions()}
            self.assertEqual(sessions[session]["state"], "aborted")
            self.assertEqual(sessions[session]["exit_report"], 0)
            attempts = {item["attempt_id"]: item for item in reopened.get_attempts()}
            self.assertEqual(attempts[pending_attempt]["state"], "unknown")
            self.assertTrue(attempts[pending_attempt]["details"]["requires_reapply"])
            lease = reopened.get_lease("lease-manual-1")
            self.assertEqual(lease["state"], "closed")
            self.assertEqual(lease["release_kind"], "unknown")
            node = reopened.get_node("node-a")
            self.assertEqual(node["capacity_used"], 0)
            self.assertEqual(node["active_leases"], 0)
            reopened.close()
            self.assertTrue(verify_ledger(spec_path, ledger_path)["ok"])

    def test_snapshot_replacement_is_whole_file_and_reports_differences(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            spec_path, spec, _ = make_sample(root)
            ledger_path = root / "ledger.sqlite3"
            snapshot_path = root / "snapshot.json"
            ledger = Ledger(ledger_path, spec)
            session = ledger.begin_session(1)
            commit_manual_lease(ledger, spec, session)
            ledger.close_lease("lease-manual-1", "expiry", "test-expiry")
            ledger.close_session(session)
            exported = ledger.export_snapshot(snapshot_path)
            ledger.close()
            original_digest = exported["content_digest"]

            conn = sqlite3.connect(ledger_path)
            conn.execute("UPDATE meta SET value='corrupted-digest' WHERE key='content_digest'")
            conn.commit()
            conn.close()
            with self.assertRaises(LedgerError) as ctx:
                Ledger(ledger_path, spec, read_only=True)
            self.assertEqual(ctx.exception.code, "E_LEDGER")
            self.assertTrue(any("content_digest" in item for item in ctx.exception.details["differences"]))

            replacement = replace_ledger_from_snapshot(ledger_path, snapshot_path)
            self.assertTrue(replacement["ok"])
            self.assertFalse(replacement["incremental_merge"])
            self.assertEqual(replacement["mode"], "whole_snapshot_replace")
            self.assertTrue(any(item.startswith("content_digest:") for item in replacement["differences"]))
            with Ledger(ledger_path, spec, read_only=True) as reopened:
                info = reopened.info()
                self.assertEqual(info["content_digest"], original_digest)
                self.assertEqual(info["cursor"], 2)
            self.assertTrue(verify_ledger(spec_path, ledger_path)["ok"])

    def test_snapshot_rejects_digest_tampering_before_replacement(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            _spec_path, spec, _ = make_sample(root)
            ledger_path = root / "ledger.sqlite3"
            snapshot_path = root / "snapshot.json"
            ledger = Ledger(ledger_path, spec)
            session = ledger.begin_session(1)
            commit_manual_lease(ledger, spec, session)
            ledger.close_session(session)
            ledger.export_snapshot(snapshot_path)
            ledger.close()
            raw = __import__("json").loads(snapshot_path.read_text())
            raw["events"][0]["node_id"] = "node-c"
            snapshot_path.write_text(__import__("json").dumps(raw), encoding="utf-8")
            with self.assertRaises(LedgerError) as ctx:
                replace_ledger_from_snapshot(ledger_path, snapshot_path)
            self.assertEqual(ctx.exception.details["field"], "content_digest")


if __name__ == "__main__":
    unittest.main()
