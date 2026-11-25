from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from edgecache.errors import ReplayError
from edgecache.replay import replay_manifest
from edgecache.scheduler import Coordinator
from edgecache.spec import load_manifest

from tests.helpers import free_ports, manifest_data, write_manifest


class IntegrationTests(unittest.TestCase):
    def _run(self, path: Path, db: Path, **kwargs: object) -> dict[str, object]:
        manifest = load_manifest(path)
        return Coordinator(manifest, db, sample_path=path, **kwargs).run()

    def test_real_expiry_frees_slot_and_grants_queued_application(self) -> None:
        ports = free_ports(1)
        events = [
            {"seq": 1, "type": "apply", "node_id": "n1", "content_key": "clip", "shards": [0]},
            {"seq": 2, "type": "apply", "node_id": "n1", "content_key": "clip", "shards": [1]},
            {"seq": 3, "type": "release", "lease_id": "L000001-S0000", "delay_ms": 1000},
            {"seq": 4, "type": "release", "lease_id": "L000002-S0001"},
        ]
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            data = manifest_data(ports, ttl_ms=800, events=events, node_count=1, slots=1, lease_limit=1)
            path = write_manifest(root / "expiry.json", data)
            report = self._run(path, root / "expiry.sqlite")
            self.assertEqual(report["status"], "OK")
            replay = replay_manifest(root / "expiry.sqlite", load_manifest(path))
            self.assertTrue(replay["ok"])
            claims = {claim["lease_id"]: claim["claim_kind"] for claim in report["claims"]}
            self.assertEqual(claims["L000001-S0000"], "EXPIRED")
            self.assertEqual(claims["L000002-S0001"], "RELEASED")
            import sqlite3

            conn = sqlite3.connect(root / "expiry.sqlite")
            wait = conn.execute(
                "SELECT wait_start_ms,wait_end_ms,status FROM queue_waits WHERE queue_id='Q000002-S0001'"
            ).fetchone()
            conn.close()
            self.assertEqual(wait[2], "GRANTED")
            self.assertIsNotNone(wait[1])
            self.assertGreaterEqual(wait[1], wait[0])

    def test_quota_replacement_revokes_old_leases_and_cancels_queue(self) -> None:
        ports = free_ports(2)
        events = [
            {"seq": 1, "type": "apply", "node_id": "n1", "content_key": "clip", "shards": [0]},
            {"seq": 2, "type": "apply", "node_id": "n2", "content_key": "clip", "shards": [0]},
            {"seq": 3, "type": "apply", "node_id": "n1", "content_key": "clip", "shards": [1]},
            {"seq": 4, "type": "renew", "lease_id": "L000001-S0000", "expected_version": 1, "extend_ms": 3000},
            {"seq": 5, "type": "quota", "quota_version": 2, "node_ids": ["n1", "n2"]},
        ]
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            data = manifest_data(ports, ttl_ms=5000, events=events, node_count=2, slots=1, lease_limit=1)
            path = write_manifest(root / "quota.json", data)
            report = self._run(path, root / "quota.sqlite")
            kinds = {claim["lease_id"]: claim["claim_kind"] for claim in report["claims"]}
            self.assertEqual(kinds["L000001-S0000"], "QUOTA_REVOKED")
            self.assertEqual(kinds["L000002-S0000"], "QUOTA_REVOKED")
            self.assertEqual(report["current_leases"], [])
            self.assertEqual(report["stale_view_count"], 1)
            self.assertTrue(any(item["lease_id"] == "Q000003-S0001" for item in report["rejections"]))
            transition = report["quota_transitions"][-1]
            self.assertEqual(transition["new_leases"], [])
            self.assertEqual(transition["retained_leases"], [])
            self.assertEqual(
                transition["revoked_leases"], ["L000001-S0000", "L000002-S0000"]
            )
            self.assertTrue(replay_manifest(root / "quota.sqlite", load_manifest(path))["ok"])

    def test_optimistic_renew_version_mismatch_does_not_extend(self) -> None:
        ports = free_ports(1)
        events = [
            {"seq": 1, "type": "apply", "node_id": "n1", "content_key": "clip", "shards": [0]},
            {"seq": 2, "type": "renew", "lease_id": "L000001-S0000", "expected_version": 2, "extend_ms": 4000},
            {"seq": 3, "type": "release", "lease_id": "L000001-S0000"},
        ]
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            path = write_manifest(
                root / "renew.json",
                manifest_data(ports, ttl_ms=5000, events=events, node_count=1, slots=1, lease_limit=1),
            )
            report = self._run(path, root / "renew.sqlite")
            self.assertTrue(
                any(item["code"] == "E_LEASE" and item["event_seq"] == 2 for item in report["rejections"])
            )
            import sqlite3

            conn = sqlite3.connect(root / "renew.sqlite")
            renew_count = conn.execute("SELECT COUNT(*) FROM journal WHERE kind='LEASE_RENEW'").fetchone()[0]
            conn.close()
            self.assertEqual(renew_count, 0)
            self.assertTrue(replay_manifest(root / "renew.sqlite", load_manifest(path))["ok"])

    def test_reopen_resumes_without_duplicating_completed_events(self) -> None:
        ports = free_ports(1)
        events = [
            {"seq": 1, "type": "apply", "node_id": "n1", "content_key": "clip", "shards": [0]},
        ]
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            path = write_manifest(
                root / "resume.json",
                manifest_data(ports, ttl_ms=5000, events=events, node_count=1, slots=1, lease_limit=1),
            )
            db = root / "resume.sqlite"
            first = self._run(path, db)
            second = self._run(path, db)
            self.assertEqual(len(first["current_leases"]), 1)
            self.assertEqual(len(second["current_leases"]), 1)
            self.assertEqual(second["status"], "OK")
            import sqlite3

            conn = sqlite3.connect(db)
            grant_count = conn.execute("SELECT COUNT(*) FROM journal WHERE kind='LEASE_GRANT'").fetchone()[0]
            conn.close()
            self.assertEqual(grant_count, 1)
            self.assertTrue(replay_manifest(db, load_manifest(path))["ok"])

    def test_forced_node_kill_recovers_to_same_conclusion_and_replays(self) -> None:
        ports = free_ports(2)
        events = [
            {"seq": 1, "type": "apply", "node_id": "n1", "content_key": "clip", "shards": [0, 1]},
            {"seq": 2, "type": "apply", "node_id": "n2", "content_key": "clip", "shards": [2, 3]},
        ]
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            data = manifest_data(ports, ttl_ms=5000, events=events, node_count=2, slots=4, lease_limit=4)
            path = write_manifest(root / "crash.json", data)
            clean = self._run(path, root / "clean.sqlite")
            partial = self._run(
                path,
                root / "crashed.sqlite",
                crash_node="n1",
                crash_after_grants=2,
            )
            self.assertEqual(partial["status"], "PARTIAL")
            self.assertTrue(partial["requires_rerun"])
            self.assertEqual(len(partial["unknown_leases"]), 2)
            crashed = self._run(path, root / "crashed.sqlite", retry_unknown=True)

            def conclusion(report: dict[str, object]) -> list[tuple[object, ...]]:
                leases = report["current_leases"]
                assert isinstance(leases, list)
                return sorted(
                    (
                        lease["node_id"],
                        lease["content_key"],
                        int(lease["shard_index"]),
                        int(lease["slot_index"]),
                        lease["digest"],
                        int(lease["quota_version"]),
                    )
                    for lease in leases
                )

            self.assertEqual(conclusion(clean), conclusion(crashed))
            self.assertEqual(len(crashed["unknown_leases"]), 2)
            self.assertFalse(crashed["requires_rerun"])
            self.assertTrue(replay_manifest(root / "crashed.sqlite", load_manifest(path))["ok"])

    def test_tampered_current_view_is_reported_as_e_replay(self) -> None:
        ports = free_ports(1)
        events = [
            {"seq": 1, "type": "apply", "node_id": "n1", "content_key": "clip", "shards": [0]},
        ]
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            path = write_manifest(
                root / "tamper.json",
                manifest_data(ports, ttl_ms=5000, events=events, node_count=1, slots=1, lease_limit=1),
            )
            self._run(path, root / "tamper.sqlite")
            import sqlite3

            conn = sqlite3.connect(root / "tamper.sqlite")
            conn.execute("UPDATE current_view SET digest='tampered'")
            conn.commit()
            conn.close()
            with self.assertRaises(ReplayError) as caught:
                replay_manifest(root / "tamper.sqlite", load_manifest(path))
            self.assertEqual(caught.exception.code, "E_REPLAY")
            self.assertEqual(caught.exception.event_seq, 1)
            self.assertIn("digest", caught.exception.field or "")


if __name__ == "__main__":
    unittest.main()
