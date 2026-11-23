from __future__ import annotations

import socket
import tempfile
import unittest
from pathlib import Path

from gsbcache.errors import HandshakeError, PortError, ReplyError
from gsbcache.processes import NodeCluster
from gsbcache.protocol import FakeServer, NodeClient

from tests.helpers import make_sample


class ProtocolErrorTests(unittest.TestCase):
    def test_handshake_timeout_is_stable(self) -> None:
        server = FakeServer("silent").start()
        try:
            with self.assertRaises(HandshakeError) as ctx:
                NodeClient("127.0.0.1", server.port, timeout_s=0.12).hello("fake")
            self.assertEqual(ctx.exception.code, "E_HANDSHAKE")
            self.assertEqual(ctx.exception.details["node_id"], "fake")
        finally:
            server.close()

    def test_missing_reply_fields_is_stable(self) -> None:
        server = FakeServer("missing").start()
        try:
            client = NodeClient("127.0.0.1", server.port, timeout_s=0.4)
            self.assertEqual(client.hello("fake")["node_id"], "fake")
            with self.assertRaises(ReplyError) as ctx:
                client.lease(
                    node_id="fake",
                    request_id="request-1",
                    artifact_keys=["artifact"],
                    lease_ms=10,
                    sequence=1,
                )
            self.assertEqual(ctx.exception.code, "E_REPLY")
            self.assertIn("lease_id", ctx.exception.details["fields"])
            self.assertIn("granted_keys", ctx.exception.details["fields"])
            self.assertIn("expires_at_ms", ctx.exception.details["fields"])
        finally:
            server.close()

    def test_port_in_use_is_reported_as_e_port(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            spec_path, spec, _ = make_sample(root)
            node = spec.nodes[0]
            occupied = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            occupied.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            occupied.bind((node.host, node.port))
            occupied.listen(1)
            cluster = NodeCluster(spec, spec_path, root / "nodes")
            try:
                with self.assertRaises(PortError) as ctx:
                    cluster.start_all()
                self.assertEqual(ctx.exception.code, "E_PORT")
                self.assertEqual(ctx.exception.details["node_id"], node.node_id)
                self.assertEqual(ctx.exception.details["port"], node.port)
            finally:
                cluster.stop_all()
                occupied.close()


if __name__ == "__main__":
    unittest.main()
