from __future__ import annotations

import json
import socket
import stat
import tempfile
import threading
import time
import unittest
from pathlib import Path

from edgecache.errors import EdgeError
from edgecache.gui import build_view_model
from edgecache.orchestrator import EdgeCluster
from edgecache.protocol import handshake
from edgecache.spec import load_spec

from tests.helpers import SAMPLE_V1, SAMPLE_V2, SAMPLE_V3, free_port_base, load_raw, write_raw


class ProtocolTimeoutTests(unittest.TestCase):
    def test_handshake_timeout_comes_from_socket_timeout(self) -> None:
        listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        listener.bind(("127.0.0.1", 0))
        listener.listen(1)
        port = listener.getsockname()[1]
        release = threading.Event()

        def blackhole() -> None:
            try:
                connection, _ = listener.accept()
                release.wait(2.0)
                connection.close()
            finally:
                listener.close()

        thread = threading.Thread(target=blackhole, daemon=True)
        thread.start()
        started = time.monotonic()
        try:
            with self.assertRaises(EdgeError) as context:
                handshake(
                    "127.0.0.1",
                    port,
                    client="test",
                    timeout=0.2,
                    expected_service="content-source",
                )
            elapsed = time.monotonic() - started
            self.assertEqual("E_TIMEOUT", context.exception.code)
            self.assertGreaterEqual(elapsed, 0.15)
            self.assertLess(elapsed, 1.5)
        finally:
            release.set()
            thread.join(timeout=2.0)


class ClusterIntegrationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.directory = tempfile.TemporaryDirectory()
        self.root = Path(self.directory.name)

    def tearDown(self) -> None:
        self.directory.cleanup()

    def test_port_conflict_reports_e_port(self) -> None:
        base = free_port_base(4, 45200)
        occupied = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        occupied.bind(("127.0.0.1", base))
        occupied.listen(1)
        cluster = EdgeCluster(
            spec_path=SAMPLE_V1,
            db_path=self.root / "port.sqlite",
            runtime_dir=self.root / "port-runtime",
            port_base=base,
        )
        try:
            with self.assertRaises(EdgeError) as context:
                cluster.start()
            self.assertEqual("E_PORT", context.exception.code)
            self.assertEqual(base, context.exception.details["port"])
        finally:
            occupied.close()
            cluster.close()

    def test_propagation_snapshot_restore_replay_and_read_only_view(self) -> None:
        base = free_port_base(4, 45300)
        db_path = self.root / "ledger.sqlite"
        cluster = EdgeCluster(
            spec_path=SAMPLE_V1,
            db_path=db_path,
            runtime_dir=self.root / "runtime",
            port_base=base,
        )
        cluster_closed = False
        try:
            started = cluster.start()
            self.assertEqual(3, len(started["nodes"]))
            self.assertEqual(1, started["bootstrap"]["ledger_version"])

            v2 = load_spec(SAMPLE_V2)
            propagated = cluster.evolve_content(v2, trigger_path="/data/root")
            self.assertEqual(2, propagated["ledger_version"])
            self.assertEqual(["/data/left", "/data/right", "/page/home"], propagated["affected_paths"])
            self.assertEqual([1, 2, 3], [row["seq"] for row in propagated["propagation"]])
            self.assertEqual(1, len({row["content_version"] for row in propagated["propagation"]}))

            first_snapshot = cluster.save_snapshot()
            second_snapshot = cluster.save_snapshot()
            first_mode = stat.S_IMODE(Path(first_snapshot["snapshot_path"]).stat().st_mode)
            self.assertEqual(0, first_mode & stat.S_IWUSR)
            snapshot_document = json.loads(Path(second_snapshot["snapshot_path"]).read_text(encoding="utf-8"))

            replay_document = json.loads(json.dumps(snapshot_document))
            replay_document["propagation"][0]["affected_path"] = "/data/not-a-dependent"
            replay_path = write_raw(self.root / "bad-replay.json", replay_document)
            with self.assertRaises(EdgeError) as replay_error:
                cluster.load_snapshot(replay_path)
            self.assertEqual("E_REPLAY", replay_error.exception.code)
            self.assertEqual("/data/root", replay_error.exception.details["trigger_path"])
            self.assertEqual("/data/not-a-dependent", replay_error.exception.details["affected_path"])

            missing_document = json.loads(json.dumps(snapshot_document))
            missing_document["propagation"].pop(0)
            missing_path = write_raw(self.root / "missing-propagation-row.json", missing_document)
            with self.assertRaises(EdgeError) as missing_error:
                cluster.load_snapshot(missing_path)
            self.assertEqual("E_REPLAY", missing_error.exception.code)
            self.assertEqual("/data/root", missing_error.exception.details["trigger_path"])
            self.assertEqual("/data/left", missing_error.exception.details["expected_affected_path"])
            self.assertEqual("/data/right", missing_error.exception.details["actual_affected_path"])

            digest_document = json.loads(json.dumps(snapshot_document))
            digest_document["content_digest"] = "0" * 64
            digest_path = write_raw(self.root / "bad-digest.json", digest_document)
            with self.assertRaises(EdgeError) as digest_error:
                cluster.load_snapshot(digest_path)
            self.assertEqual("E_SNAPSHOT", digest_error.exception.code)
            self.assertEqual("content_digest", digest_error.exception.field_path)

            v3 = load_spec(SAMPLE_V3)
            newer = cluster.evolve_content(v3, trigger_path="/data/root")
            self.assertEqual(3, newer["ledger_version"])
            restored = cluster.restore_snapshot(second_snapshot["snapshot_path"])
            self.assertEqual(4, restored["restored_ledger_version"])
            self.assertTrue(restored["comparison"]["requires_replace"])
            self.assertTrue(restored["replay"]["ok"])
            self.assertEqual(v2.content_version, restored["content_version"])
            current_entries = cluster.ledger.current_entries()
            self.assertTrue(current_entries)
            self.assertEqual({v2.content_version}, {entry["content_version"] for entry in current_entries})
            self.assertTrue(all(not entry["stale"] for entry in current_entries))
            self.assertEqual({"/api/config": "config-v1", "/data/left": "left-v2", "/data/root": "root-v2", "/data/right": "right-v2", "/page/home": "page-v2"}, {
                entry["path"]: entry["etag"] for entry in current_entries
            })
            view = cluster.ledger.view_model()
            self.assertEqual([True, True, True, False], [item["stale"] for item in view["versions"]])
            cluster.close()
            cluster_closed = True

            first_view = build_view_model(db_path)
            second_view = build_view_model(db_path)
            self.assertEqual(first_view, second_view)
            self.assertEqual(4, first_view["ledger_version"])
            self.assertEqual(v2.content_version, first_view["content_version"])
            self.assertEqual(
                sorted(first_view["entries"], key=lambda item: (item["node_id"], item["path"], item["etag"], item["cache_key"])),
                first_view["entries"],
            )
        finally:
            if not cluster_closed:
                cluster.close()

    def test_forced_termination_recovery_and_persisted_restart(self) -> None:
        base = free_port_base(4, 45400)
        db_path = self.root / "kill.sqlite"
        runtime = self.root / "kill-runtime"
        cluster = EdgeCluster(spec_path=SAMPLE_V1, db_path=db_path, runtime_dir=runtime, port_base=base)
        try:
            cluster.start()
            cluster.evolve_content(load_spec(SAMPLE_V2), trigger_path="/data/root")
            kill_report = cluster.terminate_node("edge-b", force=True)
            self.assertTrue(kill_report["force_killed"])
            self.assertIsNone(kill_report["exit_report"])
            self.assertFalse(kill_report["exit_report_available"])

            recovered = cluster.recover()
            self.assertEqual(load_spec(SAMPLE_V2).content_version, recovered["recovered"]["content_version"])
            self.assertEqual(
                len(cluster.ledger.current_entries()),
                sum(
                    len(cluster.ledger.entries_for_version(cluster.ledger.current_version(), node_id=node_id))
                    for node_id in ("edge-a", "edge-b", "edge-c")
                ),
            )
            self.assertTrue(all(node["alive"] for node in recovered["nodes"]))

            source_kill = cluster.terminate_source(force=True)
            self.assertTrue(source_kill["force_killed"])
            self.assertIsNone(source_kill["exit_report"])
            recovered_source = cluster.recover()
            self.assertTrue(any(node["role"] == "content-source" and node["alive"] for node in [recovered_source["source"]]))

            current_spec = cluster.ledger.current_spec()
            assert current_spec is not None
            pending_version = cluster.ledger.begin_round(current_spec, round_kind="forced_pending")
            recovered_pending = cluster.recover()
            self.assertIn(pending_version, recovered_pending["recovered_unknown_rounds"])
            self.assertEqual(pending_version, cluster.ledger.unknown_rounds()[-1]["ledger_version"])
        finally:
            cluster.close()

        reopened = EdgeCluster(spec_path=SAMPLE_V1, db_path=db_path, runtime_dir=runtime, port_base=base)
        try:
            restarted = reopened.start()
            self.assertTrue(restarted["external_spec_ignored"])
            self.assertEqual(2, restarted["bootstrap"]["recovered_ledger_version"])
            self.assertEqual(load_spec(SAMPLE_V2).content_version, restarted["bootstrap"]["content_version"])
        finally:
            reopened.close()

    def test_policy_change_uses_a_new_monotonic_ledger_version(self) -> None:
        base = free_port_base(5, 45500)
        document = load_raw(SAMPLE_V1)
        document["policy"]["version"] = "policy-v2"
        document["policy"]["nodes"].append(
            {"node_id": "edge-d", "capacity": 3, "eviction_rule": "nearest_expiry"}
        )
        for node in document["policy"]["nodes"]:
            if node["node_id"] == "edge-a":
                node["capacity"] = 2
        policy_path = write_raw(self.root / "policy-v2.json", document)
        cluster = EdgeCluster(
            spec_path=SAMPLE_V1,
            db_path=self.root / "policy.sqlite",
            runtime_dir=self.root / "policy-runtime",
            port_base=base,
        )
        try:
            cluster.start()
            result = cluster.apply_policy(load_spec(policy_path))
            self.assertEqual(2, result["ledger_version"])
            self.assertEqual("policy-v2", result["policy_version"])
            self.assertLessEqual(len(result["entries_by_node"]["edge-a"]), 2)
            self.assertEqual(["edge-a", "edge-b", "edge-c", "edge-d"], sorted(result["entries_by_node"]))
            view = cluster.ledger.view_model()
            self.assertEqual("policy-v2", view["policy_version"])
            self.assertEqual({"edge-a", "edge-b", "edge-c", "edge-d"}, {entry["node_id"] for entry in view["entries"]})
            self.assertEqual([True, False], [item["stale"] for item in view["versions"]])
        finally:
            cluster.close()


if __name__ == "__main__":
    unittest.main()
