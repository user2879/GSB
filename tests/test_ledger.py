from __future__ import annotations

import socket
import tempfile
import threading
import unittest
from pathlib import Path

from edgecache.content import digest_for
from edgecache.errors import HandshakeTimeout, PortError
from edgecache.ledger import Ledger
from edgecache.model import Config, NodeSpec
from edgecache.protocol import NodeClient
from edgecache.scheduler import Coordinator
from edgecache.spec import load_manifest

from tests.helpers import free_ports, manifest_data, write_manifest


class LedgerRecoveryTests(unittest.TestCase):
    def test_open_batch_becomes_unknown_and_reclaims_capacity(self) -> None:
        ports = free_ports(1)
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            path = write_manifest(
                root / "spec.json",
                manifest_data(ports, ttl_ms=5000, events=[], node_count=1, slots=2, lease_limit=2),
            )
            manifest = load_manifest(path)
            ledger = Ledger(root / "ledger.sqlite", manifest)
            node = manifest.nodes[0]
            content = manifest.contents[0]
            shard = digest_for(manifest, node.node_id, content.content_key, 0)
            ledger.record_batch_open("batch-open", 0, node.node_id, {"items": []}, 0)
            ledger.record_grant(
                {
                    "lease_id": "L-open",
                    "queue_id": "Q-open",
                    "batch_id": "batch-open",
                    "node_id": node.node_id,
                    "content_key": content.content_key,
                    "shard_index": 0,
                    "slot_index": 0,
                    "bytes": content.shard_bytes,
                    "granted_ms": 1,
                    "expires_ms": 999999999999,
                    "lease_version": 1,
                    "quota_version": 1,
                    "digest": shard.digest,
                    "checksum": shard.checksum,
                    "event_seq": 1,
                    "attempt": 0,
                }
            )
            self.assertEqual(ledger.capacity()["global_active_leases"], 1)
            ledger.close()

            reopened = Ledger(root / "ledger.sqlite", manifest)
            self.assertTrue(reopened.state()["requires_rerun"])
            self.assertEqual(reopened.capacity()["global_active_leases"], 0)
            self.assertEqual(reopened.current_leases(), [])
            unknown = reopened.leases_by_status("UNKNOWN")
            self.assertEqual([lease["lease_id"] for lease in unknown], ["L-open"])
            self.assertEqual(len(reopened.recovered_unknown), 1)
            reopened.close()

    def test_manifest_digest_change_replaces_snapshot(self) -> None:
        ports = free_ports(1)
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            data = manifest_data(ports, ttl_ms=5000, events=[], node_count=1, slots=1, lease_limit=1)
            path1 = write_manifest(root / "one.json", data)
            ledger = Ledger(root / "ledger.sqlite", load_manifest(path1))
            ledger.close()
            data["nodes"][0]["seed"] = 999999
            path2 = write_manifest(root / "two.json", data)
            changed = Ledger(root / "ledger.sqlite", load_manifest(path2))
            self.assertTrue(changed.reinitialized)
            self.assertEqual(changed.current_leases(), [])
            changed.close()
            self.assertTrue(list(root.glob("ledger.sqlite.stale-*")))

    def test_occupied_loopback_port_reports_e_port(self) -> None:
        ports = free_ports(1)
        occupied = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        occupied.bind(("127.0.0.1", ports[0]))
        occupied.listen(1)
        try:
            with tempfile.TemporaryDirectory() as temp:
                root = Path(temp)
                path = write_manifest(
                    root / "port.json",
                    manifest_data(ports, ttl_ms=5000, events=[], node_count=1, slots=1, lease_limit=1),
                )
                manifest = load_manifest(path)
                with self.assertRaises(PortError) as caught:
                    Coordinator(manifest, root / "port.sqlite", sample_path=path).run()
                self.assertEqual(caught.exception.code, "E_PORT")
        finally:
            occupied.close()

    def test_silent_server_reports_e_timeout(self) -> None:
        listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        listener.bind(("127.0.0.1", 0))
        listener.listen(1)
        port = int(listener.getsockname()[1])
        stop = threading.Event()

        def silent_server() -> None:
            try:
                connection, _ = listener.accept()
                with connection:
                    stop.wait(0.5)
            finally:
                listener.close()

        thread = threading.Thread(target=silent_server, daemon=True)
        thread.start()
        node = NodeSpec("silent", "127.0.0.1", port, 1, 1, 1, 1)
        config = Config(8, 1000, 1, 1000, socket_timeout_ms=100, handshake_timeout_ms=100)
        try:
            with self.assertRaises(HandshakeTimeout) as caught:
                NodeClient(node, config).hello(1)
            self.assertEqual(caught.exception.code, "E_TIMEOUT")
        finally:
            stop.set()
            thread.join(timeout=1)


if __name__ == "__main__":
    unittest.main()
