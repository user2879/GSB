from __future__ import annotations

import json
import threading
import time
import unittest
from pathlib import Path

from shard_diag.canonical import canonical_json, content_digest
from shard_diag.diagnostics import verify_ledger_rebuild
from shard_diag.errors import ReplayError
from shard_diag.invalidation import ConclusionStore, plan_change
from shard_diag.ledger import Ledger, verify_replay_view
from shard_diag.leases import LeaseManager
from shard_diag.node_protocol import NodeHandle
from shard_diag.scheduler import DispatchScheduler
from shard_diag.spec import parse_spec
from shard_diag.topology import stable_topological_order

from tests.factories import free_port, sample_spec_dict


def committed_payload(spec, shard_id: str) -> dict:
    from shard_diag.diagnostics import build_replay_view

    rank = stable_topological_order(spec.shards).index(shard_id)
    view = build_replay_view(spec, shard_id)
    return {
        "shard_id": shard_id,
        **view,
        "status": "COMMITTED",
        "lease": {
            "lease_id": f"lease-{shard_id}",
            "shard_id": shard_id,
            "node_id": view["node_id"],
            "granted_at_ms": 1,
            "expires_at_ms": 2,
            "renewal_count": 0,
            "lease_version": 1,
            "capacity_version": view["capacity_version"],
            "status": "RELEASED",
        },
        "wait": {"wait_started_ms": 1, "wait_ended_ms": 2, "wait_ms": 1},
        "cursor": {"shard_id": shard_id, "topo_index": rank, "position": rank + 1},
    }


class LedgerSnapshotTests(unittest.TestCase):
    def test_same_snapshot_reopens_and_digest_mismatch_replaces_all(self) -> None:
        raw = sample_spec_dict()
        spec = parse_spec(raw)
        db_path = Path(self._tmp()) / "ledger.sqlite3"

        ledger = Ledger(db_path, spec)
        ledger.commit_outcome(committed_payload(spec, stable_topological_order(spec.shards)[0]))
        cursor = ledger.latest_cursor()
        self.assertIsNotNone(cursor)
        self.assertEqual(cursor["last_shard_id"], stable_topological_order(spec.shards)[0])
        ledger.close()

        reopened = Ledger(db_path, spec)
        self.assertEqual(len(reopened.committed_ids()), 1)
        reopened.close()

        changed = json.loads(json.dumps(raw))
        changed["scale_power"] = 2
        changed_spec = parse_spec(changed)
        replaced = Ledger(db_path, changed_spec)
        self.assertEqual(replaced.committed_ids(), set())
        self.assertIsNone(replaced.get_outcome(stable_topological_order(spec.shards)[0]))
        self.assertEqual(
            replaced.conn.execute("SELECT value FROM meta WHERE key='scale'").fetchone(),
            None,
        )
        replaced.close()

    def test_format_version_mismatch_also_replaces_snapshot(self) -> None:
        raw = sample_spec_dict()
        spec = parse_spec(raw)
        db_path = Path(self._tmp()) / "version.sqlite3"
        ledger = Ledger(db_path, spec)
        ledger.commit_outcome(committed_payload(spec, stable_topological_order(spec.shards)[0]))
        ledger.close()

        import sqlite3

        conn = sqlite3.connect(db_path)
        conn.execute("PRAGMA user_version = 999")
        conn.commit()
        conn.close()

        reopened = Ledger(db_path, spec)
        self.assertEqual(reopened.committed_ids(), set())
        reopened.close()

    def test_replay_difference_raises_e_replay_with_field(self) -> None:
        spec = parse_spec(sample_spec_dict())
        shard_id = stable_topological_order(spec.shards)[0]
        expected = {shard_id: committed_payload(spec, shard_id)}
        actual = {shard_id: json.loads(json.dumps(expected[shard_id]))}
        actual[shard_id]["summary"]["participating_rows"] += 1
        with self.assertRaises(ReplayError) as raised:
            verify_replay_view(expected, actual, [shard_id])
        self.assertEqual(raised.exception.code, "E_REPLAY")
        self.assertEqual(raised.exception.object, shard_id)
        self.assertIn("participating_rows", raised.exception.field)

    def test_tampered_ledger_is_located_by_rebuild(self) -> None:
        spec = parse_spec(sample_spec_dict())
        db_path = Path(self._tmp()) / "tamper.sqlite3"
        ledger = Ledger(db_path, spec)
        shard_id = stable_topological_order(spec.shards)[0]
        ledger.commit_outcome(committed_payload(spec, shard_id))
        row = ledger.conn.execute(
            "SELECT payload_json FROM outcomes WHERE shard_id=?", (shard_id,)
        ).fetchone()
        payload = json.loads(row[0])
        payload["resource_units"] += 1
        payload["content_digest"] = content_digest(
            {key: value for key, value in payload.items() if key != "content_digest"}
        )
        with ledger.lock, ledger.conn:
            ledger.conn.execute(
                "UPDATE outcomes SET payload_json=?, content_digest=? WHERE shard_id=?",
                (canonical_json(payload), payload["content_digest"], shard_id),
            )
        with self.assertRaises(ReplayError) as raised:
            verify_ledger_rebuild(spec, ledger)
        self.assertEqual(raised.exception.object, shard_id)
        self.assertIn("resource_units", raised.exception.field)
        ledger.close()

    def test_bad_content_digest_is_e_replay(self) -> None:
        spec = parse_spec(sample_spec_dict())
        db_path = Path(self._tmp()) / "digest.sqlite3"
        ledger = Ledger(db_path, spec)
        shard_id = stable_topological_order(spec.shards)[0]
        payload = committed_payload(spec, shard_id)
        ledger.commit_outcome(payload)
        row = ledger.conn.execute(
            "SELECT payload_json FROM outcomes WHERE shard_id=?", (shard_id,)
        ).fetchone()
        stored_payload = json.loads(row[0])
        stored_payload["content_digest"] = "0" * 64
        with ledger.lock, ledger.conn:
            ledger.conn.execute(
                "UPDATE outcomes SET payload_json=? WHERE shard_id=?",
                (canonical_json(stored_payload), shard_id),
            )
        with self.assertRaises(ReplayError) as raised:
            verify_ledger_rebuild(spec, ledger)
        self.assertEqual(raised.exception.field, "content_digest")
        ledger.close()

    def _tmp(self) -> str:
        import tempfile

        return tempfile.mkdtemp(prefix="shard-ledger-")

    def test_capacity_generation_revokes_leases_occupancy_and_views(self) -> None:
        spec = parse_spec(sample_spec_dict())
        db_path = Path(self._tmp()) / "generation.sqlite3"
        ledger = Ledger(db_path, spec)
        manager = LeaseManager({node.node_id: node for node in spec.nodes})

        first = manager.acquire("s1-extract-a", "node-a", 3)
        second = manager.acquire("s5-boundary-allow", "node-a", 3)
        unaffected = manager.acquire("s2-extract-b", "node-b", 7)
        ledger.record_occupancy(first, 2)
        ledger.record_occupancy(second, 1)
        unaffected_bytes = canonical_json(unaffected.to_dict()).encode("utf-8")
        self.assertEqual(ledger.active_resource_units(), 3)
        self.assertEqual(manager.active_count("node-a"), 2)
        self.assertEqual(manager.active_count("node-b"), 1)

        revoked = manager.revoke_for_capacity("node-a", 4)
        ledger.revoke_active_occupancy_for_node("node-a")
        self.assertEqual(len(revoked), 2)
        self.assertEqual(manager.active_count("node-a"), 0)
        self.assertEqual(
            canonical_json(manager.get(unaffected.lease_id).to_dict()).encode("utf-8"),
            unaffected_bytes,
        )
        self.assertEqual(ledger.active_resource_units(), 0)

        plan = plan_change(
            spec, {"type": "capacity_version", "node_id": "node-a", "new_version": 4}
        )
        store = ConclusionStore()
        store.put_current("s1-extract-a", {"water": {"generation": 3}, "decision": {}})
        marked = store.mark_stale(plan, new_generation=4)
        self.assertIn("s1-extract-a", marked)
        self.assertNotIn("s1-extract-a", store.current_view())
        self.assertEqual(store.stale["s1-extract-a"][-1]["status"], "stale")
        ledger.close()


def crash_spec_dict() -> dict:
    return {
        "format_version": 1,
        "session_id": "kill-recovery",
        "scale_power": 0,
        "global_resource_budget": 4,
        "lease_ttl_ms": 5_000,
        "max_revision_lag": 5,
        "dispatch_delay_limit_ms": 10_000,
        "revision_chain": [900, 901],
        "nodes": [
            {
                "node_id": "kill-node",
                "host": "127.0.0.1",
                "port": free_port(),
                "concurrency_quota": 2,
                "lease_limit": 2,
                "seed": 31337,
                "capacity_version": 11,
            }
        ],
        "shards": [
            {
                "shard_id": "committed-shard",
                "node_id": "kill-node",
                "dependencies": [],
                "sample_rows": 2048,
                "resource_units": 1,
                "priority": 10,
            },
            {
                "shard_id": "unknown-shard",
                "node_id": "kill-node",
                "dependencies": [],
                "sample_rows": 2048,
                "resource_units": 1,
                "priority": 1,
            },
        ],
        "delay_batches": [
            {
                "batch_id": "kill-obs",
                "node_id": "kill-node",
                "primary_revision": 901,
                "replayed_revision": 901,
                "delay_samples_ms": [1],
            }
        ],
    }


class ForceKillRecoveryTests(unittest.TestCase):
    def test_killed_child_uncommitted_is_unknown_then_replays(self) -> None:
        spec = parse_spec(crash_spec_dict())
        node = spec.nodes[0]
        handle = NodeHandle(node).start()
        db_dir = Path(__file__).resolve().parent.parent / "artifacts" / "test-runtime"
        db_dir.mkdir(parents=True, exist_ok=True)
        db_path = db_dir / f"kill-recovery-{node.port}.sqlite3"
        if db_path.exists():
            db_path.unlink()

        ledger = Ledger(db_path, spec)
        clients = {node.node_id: handle.client}
        try:
            first = DispatchScheduler(spec, clients=clients, ledger=ledger)
            first.run(target_ids={"committed-shard"})
            self.assertIn("committed-shard", ledger.committed_ids())
            committed_digest = ledger.get_outcome("committed-shard")["content_digest"]
            verify_ledger_rebuild(spec, ledger)

            killer = DispatchScheduler(
                spec, clients=clients, ledger=ledger, processing_delay_ms=2_000
            )
            kill_thread = threading.Thread(
                target=killer.run, kwargs={"target_ids": {"unknown-shard"}}, daemon=True
            )
            kill_thread.start()

            deadline = time.monotonic() + 3.0
            while time.monotonic() < deadline and not ledger.active_occupancy():
                time.sleep(0.02)
            self.assertTrue(ledger.active_occupancy())
            time.sleep(0.25)
            handle.kill()
            kill_thread.join(timeout=6.0)
            self.assertFalse(kill_thread.is_alive())
            self.assertEqual(killer.states["unknown-shard"].status, "UNKNOWN")
            self.assertNotIn("unknown-shard", ledger.committed_ids())
            self.assertEqual(ledger.active_occupancy(), [])

            orphaned = ledger.recover_orphaned_occupancy()
            self.assertEqual(orphaned, [])
            reclaimed_rows = ledger.conn.execute(
                "SELECT COUNT(*) AS count FROM occupancy WHERE status='reclaimed'"
            ).fetchone()
            self.assertGreaterEqual(reclaimed_rows["count"], 1)

            handle.start()
            recovery = DispatchScheduler(spec, clients=clients, ledger=ledger)
            recovery.preload_committed()
            missing = set(stable_topological_order(spec.shards)) - ledger.committed_ids()
            self.assertEqual(missing, {"unknown-shard"})
            recovery.reset_for_rerun(missing)
            result = recovery.run(target_ids=missing)
            self.assertEqual(recovery.states["unknown-shard"].status, "SUCCEEDED")
            self.assertEqual(result["counts"].get("SUCCEEDED"), 2)
            self.assertIn("unknown-shard", ledger.committed_ids())

            verify_ledger_rebuild(spec, ledger)
            self.assertEqual(
                ledger.get_outcome("committed-shard")["content_digest"], committed_digest
            )
            self.assertEqual(ledger.recover_orphaned_occupancy(), [])
        finally:
            handle.stop()
            ledger.close()
            try:
                db_path.unlink()
            except FileNotFoundError:
                pass


if __name__ == "__main__":
    unittest.main()
