from __future__ import annotations

import json
import sqlite3
import tempfile
import unittest
from pathlib import Path

from cdn_edge.errors import PipelineError
from cdn_edge.fixed import FixedGrid
from cdn_edge.ledger import Ledger
from cdn_edge.netproto import send_json_line
from cdn_edge.orchestrator import Cluster, run_session
from cdn_edge.replay import replay_snapshot
from cdn_edge.spec import parse_spec
from cdn_edge.verify import verify_ledger

from tests.helpers import free_port_base, remap_ports, sample_raw


class IntegrationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory(prefix="cdn-edge-test-")
        self.root = Path(self.temp.name)
        base = free_port_base()
        self.raw = remap_ports(sample_raw(), base)
        self.spec = parse_spec(self.raw)
        self.ledger_path = self.root / "ledger.sqlite3"

    def tearDown(self) -> None:
        self.temp.cleanup()

    def test_full_session_verify_snapshot_replacement_and_replay(self) -> None:
        result = run_session(self.spec, self.ledger_path, work_dir=self.root / "run")
        self.assertEqual(result["batch_count"], 3)
        batch2 = next(batch for batch in result["batches"] if batch["batch_id"] == 2)
        self.assertEqual(batch2["metrics"]["aggregate"]["metrics"]["version_consistency_rate"]["value_units"], 499)
        self.assertEqual(batch2["metrics"]["aggregate"]["metrics"]["version_consistency_rate"]["error_bound_units"], 2)
        self.assertEqual(sum(1 for item in batch2["requests"] if not item["version_consistent"]), 3)
        batch1 = next(batch for batch in result["batches"] if batch["batch_id"] == 1)
        batch3 = next(batch for batch in result["batches"] if batch["batch_id"] == 3)
        self.assertNotEqual(batch1["content_digest"], batch3["content_digest"])
        self.assertTrue(any(item["hit_source"] == "expired_revalidated" for item in batch2["requests"]))

        verified = verify_ledger(str(self.ledger_path))
        self.assertTrue(verified["ok"])
        self.assertEqual(verified["batch_count"], 3)
        self.assertEqual(verified["snapshot_count"], 4)

        ledger = Ledger(self.ledger_path, create=False)
        snapshots = ledger.list_snapshots("sample-session")
        active = ledger.active_state()
        ledger.close()
        self.assertEqual(len(snapshots), 4)
        replay_all = replay_snapshot(str(self.ledger_path), snapshots[0]["snapshot_id"])
        self.assertEqual(replay_all["checked_batches"], [1, 2, 3])
        replay_tail = replay_snapshot(str(self.ledger_path), snapshots[1]["snapshot_id"])
        self.assertEqual(replay_tail["checked_batches"], [2, 3])

        ledger = Ledger(self.ledger_path, create=True)
        replacement = ledger.restore_snapshot(
            snapshots[1]["snapshot_id"],
            current_content_digest=active["content_digest"],
            current_grid=FixedGrid(int(active["grid_power"])),
            mode="replace",
        )
        self.assertTrue(replacement["whole_replacement"])
        self.assertFalse(replacement["incremental_merge"])
        self.assertIn("content_digest", replacement["mismatches"])
        self.assertEqual(ledger.active_state()["batch_id"], 1)
        with self.assertRaises(PipelineError) as strict_error:
            ledger.restore_snapshot(
                snapshots[1]["snapshot_id"],
                current_content_digest=active["content_digest"],
                current_grid=FixedGrid(int(active["grid_power"])),
                mode="strict",
            )
        self.assertEqual(strict_error.exception.code, "E_SNAPSHOT")
        with self.assertRaises(PipelineError) as digest_error:
            ledger.restore_snapshot(
                snapshots[1]["snapshot_id"],
                current_content_digest=None,
                current_grid=None,
                expected_content_digest="0" * 64,
            )
        self.assertEqual(digest_error.exception.code, "E_SNAPSHOT")
        with self.assertRaises(sqlite3.IntegrityError):
            ledger.conn.execute(
                "UPDATE snapshots SET content_digest='tampered' WHERE snapshot_id=?",
                (snapshots[1]["snapshot_id"],),
            )
        ledger.conn.rollback()
        original_metrics = ledger.get_batch("sample-session", 2)["metrics"]
        tampered = json.loads(json.dumps(original_metrics))
        tampered["nodes"]["edge-a"]["metrics"]["hit_rate"]["value_units"] += 1
        ledger.conn.execute(
            "UPDATE batches SET metrics_json=? WHERE session_id='sample-session' AND batch_id=2",
            (json.dumps(tampered, sort_keys=True, separators=(",", ":")),),
        )
        ledger.conn.commit()
        ledger.close()
        with self.assertRaises(PipelineError) as replay_error:
            replay_snapshot(str(self.ledger_path), snapshots[0]["snapshot_id"])
        self.assertEqual(replay_error.exception.code, "E_REPLAY")
        self.assertEqual(replay_error.exception.details["first_inconsistency"]["batch_id"], 2)
        with self.assertRaises(PipelineError) as verify_error:
            verify_ledger(str(self.ledger_path))
        self.assertEqual(verify_error.exception.code, "E_REPLAY")

    def test_replay_content_change_has_specific_basis(self) -> None:
        run_session(self.spec, self.ledger_path, work_dir=self.root / "run")
        ledger = Ledger(self.ledger_path, create=False)
        snapshot = ledger.list_snapshots("sample-session")[0]
        ledger.close()
        changed = json.loads(json.dumps(self.raw))
        changed["content"][0]["body"] = "changed-after-snapshot"
        changed_path = self.root / "changed_spec.json"
        changed_path.write_text(json.dumps(changed, ensure_ascii=False), encoding="utf-8")
        with self.assertRaises(PipelineError) as raised:
            replay_snapshot(
                str(self.ledger_path),
                snapshot["snapshot_id"],
                current_spec_path=str(changed_path),
            )
        self.assertEqual(raised.exception.code, "E_REPLAY")
        self.assertEqual(raised.exception.details["reason"], "content_digest_mismatch")
        self.assertIn("not_reproducible_basis", raised.exception.details)

    def test_node_forced_restart_recovers_only_persistent_state(self) -> None:
        work = self.root / "restart-work"
        source_spec = work / "spec.json"
        work.mkdir(parents=True)
        source_spec.write_text(json.dumps(self.raw), encoding="utf-8")
        with Cluster(self.spec, work) as cluster:
            cluster.start_all(source_spec, work / "nodes", work / "source-state.json")
            request = self.spec.requests[0]
            key_values = dict(request.cache_key_values(self.spec.nodes[request.node_id].cache_key_fields))
            first = cluster.node_request(
                request.node_id,
                {
                    "op": "fetch",
                    "seq": request.seq,
                    "request_id": request.request_id,
                    "path": request.path,
                    "at_ms": request.at_ms,
                    "key": key_values,
                    "allow_stale": True,
                },
            )
            self.assertEqual(first["hit_source"], "miss_origin")
            state_db = work / "nodes" / "edge-a.sqlite3"
            handle = cluster.nodes["edge-a"]
            handle.process.kill()
            handle.process.wait(timeout=5)
            cluster.stop_edge("edge-a", force=True)
            self.assertFalse(any(path.name.endswith("exit-report.json") for path in work.rglob("*")))
            cluster.start_edge("edge-a", state_db)
            cluster.wait_edge("edge-a")
            dump = cluster.node_request("edge-a", {"op": "dump"})
            self.assertEqual(len(dump["entries"]), 1)
            self.assertEqual(dump["counters"]["requests"], 1)
            second = cluster.node_request(
                request.node_id,
                {
                    "op": "fetch",
                    "seq": 100,
                    "request_id": "after-restart",
                    "path": request.path,
                    "at_ms": 500,
                    "key": key_values,
                    "allow_stale": True,
                },
            )
            self.assertEqual(second["hit_source"], "cache_hit")
            self.assertEqual(second["entry_version"], 1)

    def test_stale_hit_is_e_stale_and_port_conflict_is_e_port(self) -> None:
        work = self.root / "stale-work"
        source_spec = work / "spec.json"
        work.mkdir(parents=True)
        source_spec.write_text(json.dumps(self.raw), encoding="utf-8")
        with Cluster(self.spec, work) as cluster:
            cluster.start_all(source_spec, work / "nodes", work / "source-state.json")
            request = self.spec.requests[0]
            policy = self.spec.nodes[request.node_id]
            key_values = dict(request.cache_key_values(policy.cache_key_fields))
            cluster.node_request(
                request.node_id,
                {
                    "op": "fetch",
                    "seq": 0,
                    "request_id": "fill",
                    "path": request.path,
                    "at_ms": 0,
                    "key": key_values,
                    "allow_stale": True,
                },
            )
            assert cluster.source is not None
            cluster.source.process.kill()
            cluster.source.process.wait(timeout=5)
            stale = cluster.node_request(
                request.node_id,
                {
                    "op": "fetch",
                    "seq": 1,
                    "request_id": "stale",
                    "path": request.path,
                    "at_ms": 2500,
                    "key": key_values,
                    "allow_stale": True,
                },
            )
            self.assertEqual(stale["hit_source"], "stale_hit")
            self.assertTrue(stale["stale"])
            self.assertEqual(stale["error_code"], "E_STALE")
            self.assertEqual(stale["remaining_ttl_ms"], 0)

        import socket

        conflict_spec = parse_spec(remap_ports(sample_raw(), free_port_base(start=27000)))
        conflict = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            conflict.bind(("127.0.0.1", conflict_spec.session.source_port))
            conflict.listen(1)
            with self.assertRaises(PipelineError) as raised:
                with Cluster(conflict_spec, self.root / "port-work") as cluster:
                    cluster.start_all(
                        self.root / "never-started-spec.json",
                        self.root / "port-nodes",
                        self.root / "port-source.json",
                    )
            self.assertEqual(raised.exception.code, "E_PORT")
            self.assertEqual(raised.exception.field, "port")
        finally:
            conflict.close()


if __name__ == "__main__":
    unittest.main()
