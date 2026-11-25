import json
import os
import shutil
import tempfile
import unittest

from sharding_diag.clock import ManualClock
from sharding_diag.ledger import Ledger
from sharding_diag.leases import LeaseManager
from sharding_diag.model import parse_declaration


def declaration(capacity_a=1):
    return parse_declaration({
        "nodes": [
            {"node_id": "n1", "host": "127.0.0.1", "port": 19001, "concurrency": 1,
             "lease_limit": 1, "seed": 1, "capacity_version": capacity_a},
        ],
        "shards": [
            {"shard_id": "a", "rows": 5, "demand": 2, "priority": 1},
        ],
        "global": {"budget": 10, "freeze_threshold_ms": 100, "revoke_tolerance": 5},
    })


class LedgerTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.db = os.path.join(self.tmp, "diag.db")

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_atomic_outcome_writes_all_views(self):
        decl = declaration()
        clock = ManualClock(10)
        ledger = Ledger(self.db, decl).open()
        leases = LeaseManager(clock)
        lease = leases.grant("n1", "a", 2, 100)
        ledger.record_lease_grant(lease, 10)
        ledger.record_freeze_migration({
            "shard_id": "a", "from_state": "FROZEN", "to_state": "UNFROZEN",
            "missing_upstreams": [], "at_ms": 10, "frozen_elapsed_ms": 0,
            "threshold_ms": 100, "threshold_breached": False, "reason": "round_evaluation",
        })
        release = leases.release_on_complete("n1", "a")
        ledger.record_lease_release(release)
        summary = {"groups": {}, "total_sum_fp": 1234}
        ledger.record_outcome({
            "shard_id": "a", "status": "COMMITTED", "node_id": "n1",
            "lease_id": lease.lease_id, "lease_version": 1, "capacity_version": 1,
            "summary": summary, "summary_digest": "sha256:abc", "fixed_total": 1234,
            "demand": 2, "at_ms": 20,
        })
        self.assertTrue(ledger.verify_chain())
        outcomes = ledger.outcomes()
        self.assertEqual(outcomes["a"]["status"], "COMMITTED")
        self.assertEqual(ledger.progress()["a"]["state"], "COMMITTED")
        snap = ledger.lease_snapshot()
        # resource_usage keeps the full delta trail; active occupancy is zero
        # because the only lease is released.
        self.assertTrue(snap["leases"][0]["released"])
        active = [row for row in snap["leases"] if not row["released"]]
        self.assertEqual(active, [])
        self.assertEqual(ledger.freeze_snapshot()["state"]["a"], "UNFROZEN")
        identity = ledger.identity()
        self.assertEqual(identity["format_version"], 1)
        ledger.close()

    def test_snapshot_identity_drift_triggers_whole_replace(self):
        decl = declaration(capacity_a=1)
        ledger = Ledger(self.db, decl).open()
        ledger.record_outcome({
            "shard_id": "a", "status": "COMMITTED", "node_id": "n1",
            "lease_id": "L1", "lease_version": 1, "capacity_version": 1,
            "summary": {}, "summary_digest": "sha256:x", "fixed_total": 1,
            "demand": 2, "at_ms": 1,
        })
        ledger.close()

        drifted = declaration(capacity_a=9)
        reopened = Ledger(self.db, drifted).open()
        self.assertTrue(reopened.replaced)
        self.assertEqual(reopened.replace_reason["reason"], "capacity_versions")
        self.assertTrue(os.path.exists(reopened.backup_path))
        self.assertEqual(os.stat(reopened.backup_path).st_mode & 0o200, 0)  # read-only
        # Replacement yields a fresh empty ledger rather than merging rows.
        self.assertEqual(reopened.outcomes(), {})
        reopened.close()

    def test_tampered_digest_is_detected_and_replaced(self):
        decl = declaration()
        ledger = Ledger(self.db, decl).open()
        ledger.record_outcome({
            "shard_id": "a", "status": "COMMITTED", "node_id": "n1",
            "lease_id": "L1", "lease_version": 1, "capacity_version": 1,
            "summary": {}, "summary_digest": "sha256:x", "fixed_total": 1,
            "demand": 2, "at_ms": 1,
        })
        ledger.close()
        import sqlite3
        conn = sqlite3.connect(self.db)
        conn.execute("UPDATE meta SET value='tampered' WHERE key='head_chain_hash'")
        conn.commit()
        conn.close()
        reopened = Ledger(self.db, decl).open()
        self.assertTrue(reopened.replaced)
        self.assertEqual(reopened.replace_reason["reason"], "content_digest")
        reopened.close()


if __name__ == "__main__":
    unittest.main()
