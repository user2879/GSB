from __future__ import annotations

import socket
import threading
import time
import unittest

from gsb.errors import DiagnosticError
from gsb.params import compute_batch_summary
from gsb.process import WorkerPool
from gsb.protocol import TcpShardClient, recv_json_line, send_json_line
from tests.helpers import make_session, make_shard


class WorkerIntegrationTests(unittest.TestCase):
    def test_real_tcp_batch_and_lease_version(self) -> None:
        shard = make_shard()
        session = make_session((shard,))
        with WorkerPool(session, startup_timeout_s=5, handshake_timeout_s=2) as pool:
            client = pool.client(shard.shard_id)
            response = client.run_batch(
                shard_id=shard.shard_id,
                batch_id="b",
                sample_ids=[3, 1],
                parameter_version=2,
            )
            expected = compute_batch_summary(shard, "b", [1, 3], 2)
            self.assertEqual(response["loss_digest"], expected["loss_digest"])
            self.assertEqual(response["parameter_digest"], expected["parameter_digest"])
            issued = client.lease_issue(shard.shard_id, "lease:test", 1000)
            self.assertEqual(issued["lease_version"], 1)
            with self.assertRaises(DiagnosticError) as caught:
                client.lease_renew(shard.shard_id, "lease:test", 2, 1000)
            self.assertEqual(caught.exception.code, "E_LEASE")
            renewed = client.lease_renew(shard.shard_id, "lease:test", 1, 1000)
            self.assertEqual(renewed["lease_version"], 2)

    def test_port_occupied_is_e_port(self) -> None:
        listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        listener.bind(("127.0.0.1", 0))
        listener.listen(1)
        port = listener.getsockname()[1]
        session = make_session((make_shard(port=port),))
        try:
            with self.assertRaises(DiagnosticError) as caught:
                WorkerPool(session, startup_timeout_s=3).start()
            self.assertEqual(caught.exception.code, "E_PORT")
        finally:
            listener.close()

    def test_handshake_timeout_is_e_timeout(self) -> None:
        listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        listener.bind(("127.0.0.1", 0))
        listener.listen(1)
        port = listener.getsockname()[1]

        def accept_and_wait() -> None:
            conn, _ = listener.accept()
            try:
                recv_json_line(conn, 1.0)
                time.sleep(0.3)
            finally:
                conn.close()
                listener.close()

        thread = threading.Thread(target=accept_and_wait, daemon=True)
        thread.start()
        client = TcpShardClient("127.0.0.1", port, handshake_timeout_s=0.05)
        with self.assertRaises(DiagnosticError) as caught:
            client.handshake("a")
        self.assertEqual(caught.exception.code, "E_TIMEOUT")
        thread.join(timeout=1)

    def test_missing_response_field_is_e_field(self) -> None:
        listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        listener.bind(("127.0.0.1", 0))
        listener.listen(1)
        port = listener.getsockname()[1]

        def serve_once() -> None:
            conn, _ = listener.accept()
            try:
                recv_json_line(conn, 1.0)
                send_json_line(conn, {"ok": True, "shard_id": "a"})
            finally:
                conn.close()
                listener.close()

        thread = threading.Thread(target=serve_once, daemon=True)
        thread.start()
        client = TcpShardClient("127.0.0.1", port, handshake_timeout_s=1)
        with self.assertRaises(DiagnosticError) as caught:
            client.handshake("a")
        self.assertEqual(caught.exception.code, "E_FIELD")
        self.assertEqual(caught.exception.field, "response.pid")
        thread.join(timeout=1)


if __name__ == "__main__":
    unittest.main()
