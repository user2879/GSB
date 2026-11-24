from __future__ import annotations

import json
import socket
import threading
import time
import unittest

from leasecache.errors import LeaseError
from leasecache.node_manager import NodeManager
from leasecache.protocol import NodeClient, validate_lease_ack
from leasecache.spec import load_spec
from leasecache.util import StructuredLogger

from .helpers import free_ports, make_spec, test_tempdir, write_spec


class SilentServer:
    def __init__(self) -> None:
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.socket.bind(("127.0.0.1", 0))
        self.port = int(self.socket.getsockname()[1])
        self.socket.listen()
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.clients: list[socket.socket] = []

    def _run(self) -> None:
        self.socket.settimeout(0.1)
        while not self.stop.is_set():
            try:
                client, _ = self.socket.accept()
                self.clients.append(client)
            except socket.timeout:
                continue
            except OSError:
                break

    def __enter__(self) -> "SilentServer":
        self.thread.start()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.stop.set()
        for client in self.clients:
            try:
                client.close()
            except OSError:
                pass
        self.socket.close()
        self.thread.join(timeout=1)


class ProtocolTests(unittest.TestCase):
    def test_missing_ack_field_is_e_ack(self) -> None:
        response = {
            "ok": True,
            "ack_version": 1,
            "node_id": "node-a",
            "request_seq": 1,
            "lease_id": "lease",
            "artifact_keys": ["src/a"],
            # digests intentionally missing
            "capacity": {
                "used_slots": 1,
                "total_slots": 2,
                "free_slots": 1,
                "active_leases": 1,
                "lease_limit": 2,
            },
            "expires_mono_ms": 10,
            "elapsed_mono_ms": 1,
            "mono_ms": 10,
        }
        with self.assertRaises(LeaseError) as caught:
            validate_lease_ack(response)
        self.assertEqual(caught.exception.code, "E_ACK")
        self.assertEqual(caught.exception.field, "digests")

    def test_handshake_uses_real_socket_timeout(self) -> None:
        with SilentServer() as server:
            client = NodeClient("node-a", "127.0.0.1", server.port, timeout_ms=80)
            started = time.monotonic()
            with self.assertRaises(LeaseError) as caught:
                client.handshake(timeout_ms=80)
            elapsed = time.monotonic() - started
        self.assertEqual(caught.exception.code, "E_HANDSHAKE")
        self.assertGreaterEqual(elapsed, 0.06)


class NodeSubprocessTests(unittest.TestCase):
    def _manager(self, spec_path, workspace, logger):
        spec = load_spec(spec_path)
        return spec, NodeManager(
            spec,
            workspace,
            workspace / "runtime",
            logger,
            start_timeout_ms=4000,
            request_timeout_ms=2000,
        )

    def test_occupied_port_is_e_port_and_structured_log(self) -> None:
        directory = test_tempdir()
        ports = free_ports(1)
        spec_path = write_spec(
            directory,
            make_spec(ports, stages=[
                {
                    "id": "s",
                    "concurrency": 1,
                    "artifacts": [{"key": "a", "references": []}],
                }
            ]),
        )
        blocker = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        blocker.bind(("127.0.0.1", ports[0]))
        blocker.listen()
        logger = StructuredLogger(directory / "runtime" / "logs" / "engine.jsonl", "test")
        try:
            _, manager = self._manager(spec_path, directory / "workspace", logger)
            with self.assertRaises(LeaseError) as caught:
                manager.start_all()
        finally:
            blocker.close()
        self.assertEqual(caught.exception.code, "E_PORT")
        records = [
            json.loads(line)
            for line in (directory / "runtime" / "logs" / "engine.jsonl")
            .read_text(encoding="utf-8")
            .splitlines()
        ]
        self.assertTrue(any(record.get("code") == "E_PORT" for record in records))

    def test_lease_limit_capacity_release_and_node_restart_digest(self) -> None:
        directory = test_tempdir()
        stages = [
            {
                "id": "s",
                "concurrency": 1,
                "artifacts": [
                    {"key": "a", "references": []},
                    {"key": "b", "references": []},
                ],
            }
        ]
        spec_path = write_spec(
            directory,
            make_spec(
                free_ports(1),
                stages=stages,
                capacities=[2],
                limits=[1],
                lease_ms=100,
            ),
        )
        logger = StructuredLogger(directory / "runtime" / "logs" / "engine.jsonl", "test")
        spec, manager = self._manager(spec_path, directory / "workspace", logger)
        manager.start_all()
        try:
            client = manager.client("node-a")
            first = client.lease(["a"], 100, 1, lease_id="node-a:lease:test:1")
            self.assertEqual(first["digests"]["a"], generate_digest("a", spec.nodes[0].random_seed))
            self.assertEqual(first["capacity"]["used_slots"], 1)
            with self.assertRaises(LeaseError) as caught:
                client.lease(["b"], 100, 2, lease_id="node-a:lease:test:2")
            self.assertEqual(caught.exception.code, "E_LEASE_LIMIT")
            renewed = client.renew(first["lease_id"], 150)
            self.assertGreater(renewed["expires_mono_ms"], first["expires_mono_ms"])
            # Wait for the real local monotonic deadline instead of forcing it.
            time.sleep(0.17)
            third = client.lease(["b"], 100, 3, lease_id="node-a:lease:test:3")
            self.assertEqual(third["capacity"]["used_slots"], 1)
            before = manager.digest_snapshot()["node-a"]
            manager.restart_node("node-a")
            after = manager.digest_snapshot()["node-a"]
            self.assertEqual(before["digests"], after["digests"])
            self.assertEqual(before["refs"], after["refs"])
        finally:
            manager.stop_all()


def generate_digest(key: str, seed: int) -> str:
    from leasecache.digest import generate_artifact
    from leasecache.spec import NodeSpec

    node = NodeSpec(
        node_id="node-a",
        host="127.0.0.1",
        port=1,
        capacity_slots=1,
        lease_limit=1,
        random_seed=seed,
    )
    return generate_artifact(node, key).digest


if __name__ == "__main__":
    unittest.main()
