import json
import os
import sqlite3
import tempfile
import unittest

from shardflow.errors import HandshakeError, LedgerError, PortError, ProtocolFieldError
from shardflow.ledger import Ledger
from shardflow.protocol import NodeClient
from shardflow.spec import load_spec
from shardflow.viewmodel import build_view_model, render_text

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SPEC_PATH = os.path.join(ROOT, "demo", "demo.spec.json")
EVENTS_PATH = os.path.join(ROOT, "demo", "demo.events.ndjson")


class ProtocolErrorTests(unittest.TestCase):
    def test_unused_port_raises_handshake_after_retries(self):
        spec = load_spec(SPEC_PATH, EVENTS_PATH)
        node = spec.nodes["node-a"]
        node.port = 49999
        client = NodeClient(node, 300, 1000)
        with self.assertRaises(HandshakeError) as ctx:
            client.handshake()
        self.assertEqual(ctx.exception.code, "E_HANDSHAKE")

    def test_missing_receipt_field_classified_e_field(self):
        import socket
        import threading
        from shardflow.protocol import NodeClient
        from shardflow.spec import Node
        server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server.bind(("127.0.0.1", 0))
        server.listen(1)
        port = server.getsockname()[1]

        def serve_once():
            conn, _ = server.accept()
            conn.recv(65536)
            conn.sendall(b'{"type":"receipt"}\n')
            conn.close()

        thread = threading.Thread(target=serve_once)
        thread.start()
        node = Node(
            {"node_id": "x", "host": "127.0.0.1", "port": port,
             "concurrency": 1, "lease_limit": 1, "max_renewals": 0,
             "seed": 1, "capacity_version": 1, "fixed_scale": 2,
             "compute_iters": 0}
        )
        client = NodeClient(node, 1000, 1000)
        with self.assertRaises(ProtocolFieldError) as ctx:
            client._request(
                {"type": "dispatch", "shard_id": "s1"}, 1.0,
                ("type", "shard_id", "summary"),
            )
        self.assertEqual(ctx.exception.code, "E_FIELD")
        thread.join()
        server.close()


class LedgerTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = os.path.join(self.tmp.name, "ledger.sqlite")

    def tearDown(self):
        self.tmp.cleanup()

    def test_format_and_digest_round_trip(self):
        ledger = Ledger(self.path, spec_digest="abc")
        digest = ledger.record_lease_event("grant", {"shard_id": "s1"})
        self.assertEqual(ledger.format_version(), 1)
        self.assertEqual(ledger.content_digest(), digest)
        ledger.close()
        reopened = Ledger(self.path, replace_on_mismatch=False)
        self.assertEqual(reopened.content_digest(), digest)
        reopened.close()

    def test_tampered_digest_triggers_whole_file_replacement(self):
        ledger = Ledger(self.path, spec_digest="abc")
        ledger.record_lease_event("grant", {"shard_id": "s1"})
        ledger.close()
        conn = sqlite3.connect(self.path)
        conn.execute("UPDATE meta SET value='tampered' WHERE key='content_digest'")
        conn.commit()
        conn.close()
        replaced = Ledger(self.path, spec_digest="abc")
        self.assertTrue(replaced.replaced)
        self.assertEqual(replaced.fetch_lease_events(), [])
        replaced.close()

    def test_unsupported_format_version_replaces(self):
        ledger = Ledger(self.path, spec_digest="abc")
        ledger.close()
        conn = sqlite3.connect(self.path)
        conn.execute("UPDATE meta SET value='999' WHERE key='format_version'")
        conn.commit()
        conn.close()
        replaced = Ledger(self.path, spec_digest="abc")
        self.assertTrue(replaced.replaced)
        self.assertEqual(replaced.format_version(), 1)
        replaced.close()

    def test_strict_open_rejects_tampering(self):
        ledger = Ledger(self.path, spec_digest="abc")
        ledger.close()
        conn = sqlite3.connect(self.path)
        conn.execute("UPDATE meta SET value='bad' WHERE key='content_digest'")
        conn.commit()
        conn.close()
        with self.assertRaises(LedgerError):
            Ledger(self.path, replace_on_mismatch=False)

    def test_unknown_marker_persists_and_view_model_renders_stably(self):
        ledger = Ledger(self.path, spec_digest="abc")
        ledger.mark_unknown(
            "s1", 1, "node-a",
            {"granted_ms": 10, "config_version": 1, "capacity_version": 1,
             "resource_units": 2, "wait_start_ms": 1, "wait_end_ms": 10},
            0,
        )
        ledger.close()
        model = build_view_model(SPEC_PATH, self.path, EVENTS_PATH)
        self.assertIn("s1", model["unknown_shards"])
        text_a = render_text(model)
        text_b = render_text(build_view_model(SPEC_PATH, self.path,
                                               EVENTS_PATH))
        self.assertEqual(text_a, text_b)


if __name__ == "__main__":
    unittest.main()
