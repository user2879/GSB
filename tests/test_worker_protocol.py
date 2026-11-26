import socket
import unittest

from shardcast.errors import FieldError, PortError, TimeoutError
from shardcast.leases import Lease
from shardcast.clock import now_ms
from shardcast.protocol import request_dispatch, request_renew
from shardcast.sample import aggregate_rows, generate_sample
from shardcast.spec import parse_spec
from shardcast.worker_pool import WorkerPool, port_in_use


def small_spec(port: int, ttl_ms: int = 2000) -> object:
    raw = {
        "format_version": 1,
        "lease_ttl_ms": ttl_ms,
        "global_budget": {1: 10},
        "nodes": [
            {
                "node_id": "n1",
                "host": "127.0.0.1",
                "port": port,
                "seed": 77,
                "initial_capacity_version": 1,
                "capacities": [{"version": 1, "concurrency": 2, "lease_limit": 2}],
            }
        ],
        "shards": [
            {"shard_id": "a", "depends_on": [], "sample_rows": 40,
             "resource_units": 1, "priority": 0}
        ],
        "events": [{"kind": "arrive", "shard_id": "a"}],
    }
    return parse_spec(raw)


class WorkerProtocolTests(unittest.TestCase):
    PORT = 19401

    def setUp(self) -> None:
        self.spec = small_spec(self.PORT)
        self.pool = WorkerPool(self.spec)
        self.pool.start_all()

    def tearDown(self) -> None:
        self.pool.stop_all()

    def test_real_listener_handshake_and_dispatch_receipt(self) -> None:
        node = self.spec.nodes[0]
        lease = Lease(
            lease_id="lease-test",
            shard_id="a",
            node_id="n1",
            granted_ms=now_ms(),
            expire_ms=now_ms() + 5000,
            renewals=0,
            lease_version=1,
            capacity_version=1,
            state="active",
        )
        self.pool.install_lease("n1", lease.to_dict())
        data = generate_sample(node.seed, "a", 40)
        request = {
            "type": "dispatch",
            "lease_id": "lease-test",
            "lease_version": 1,
            "node_id": "n1",
            "shard_id": "a",
            "capacity_version": 1,
            "sample_rows": 40,
            "batches": [data[:20].tolist(), data[20:].tolist()],
        }
        receipt = request_dispatch(node.host, node.port, request, timeout=2.0)
        self.assertTrue(receipt["accepted"])
        expected = aggregate_rows(data)
        self.assertEqual(receipt["summary"]["groups"], expected["groups"])
        self.assertEqual(
            receipt["summary"]["participating_rows"], expected["participating_rows"]
        )
        self.assertIn("elapsed_ms", receipt)

    def test_stale_lease_version_rejected(self) -> None:
        node = self.spec.nodes[0]
        lease = Lease(
            lease_id="lease-v",
            shard_id="a",
            node_id="n1",
            granted_ms=now_ms(),
            expire_ms=now_ms() + 5000,
            renewals=0,
            lease_version=3,
            capacity_version=1,
            state="active",
        )
        self.pool.install_lease("n1", lease.to_dict())
        data = generate_sample(node.seed, "a", 10)
        request = {
            "type": "dispatch",
            "lease_id": "lease-v",
            "lease_version": 2,
            "node_id": "n1",
            "shard_id": "a",
            "capacity_version": 1,
            "sample_rows": 10,
            "batches": [data.tolist()],
        }
        receipt = request_dispatch(node.host, node.port, request, timeout=2.0)
        self.assertFalse(receipt["accepted"])
        self.assertEqual(receipt["error"], "stale_lease_version")

    def test_renew_optimistic_check_on_worker(self) -> None:
        node = self.spec.nodes[0]
        lease = Lease(
            lease_id="lease-r",
            shard_id="a",
            node_id="n1",
            granted_ms=now_ms(),
            expire_ms=now_ms() + 5000,
            renewals=0,
            lease_version=1,
            capacity_version=1,
            state="active",
        )
        self.pool.install_lease("n1", lease.to_dict())
        bad = request_renew(
            node.host,
            node.port,
            {
                "type": "renew",
                "lease_id": "lease-r",
                "lease_version": 9,
                "node_id": "n1",
                "shard_id": "a",
                "capacity_version": 1,
            },
            timeout=2.0,
        )
        self.assertFalse(bad["accepted"])
        good = request_renew(
            node.host,
            node.port,
            {
                "type": "renew",
                "lease_id": "lease-r",
                "lease_version": 1,
                "node_id": "n1",
                "shard_id": "a",
                "capacity_version": 1,
            },
            timeout=2.0,
        )
        self.assertTrue(good["accepted"])
        self.assertEqual(good["lease_version"], 2)

    def test_missing_receipt_field_raises_e_field(self) -> None:
        import threading

        listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        listener.bind(("127.0.0.1", 19499))
        listener.listen(1)
        listener.settimeout(3.0)
        errors: list[Exception] = []

        def client() -> None:
            try:
                request_dispatch(
                    "127.0.0.1",
                    19499,
                    {
                        "type": "dispatch",
                        "lease_id": "l",
                        "lease_version": 1,
                        "node_id": "n1",
                        "shard_id": "a",
                        "capacity_version": 1,
                        "sample_rows": 1,
                        "batches": [[[0, 1]]],
                    },
                    timeout=2.0,
                )
            except FieldError as exc:
                errors.append(exc)

        thread = threading.Thread(target=client)
        thread.start()
        try:
            conn, _ = listener.accept()
            conn.settimeout(2.0)
            conn.recv(4096)
            conn.sendall(b'{"type":"ready","protocol":"shardcast-jsonl/1"}\n')
            conn.recv(4096)
            conn.sendall(b'{"type":"dispatch_receipt","node_id":"n1"}\n')
            thread.join(timeout=3.0)
            conn.close()
        finally:
            listener.close()
        self.assertEqual(len(errors), 1)
        self.assertEqual(errors[0].code, "E_FIELD")

    def test_port_in_use_raises_e_port(self) -> None:
        self.assertTrue(port_in_use("127.0.0.1", self.PORT))
        second = WorkerPool(self.spec)
        with self.assertRaises(PortError):
            second.start_all()

    def test_handshake_timeout_raises_e_timeout(self) -> None:
        listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        listener.bind(("127.0.0.1", 19498))
        listener.listen(1)
        listener.settimeout(2.0)
        try:
            with self.assertRaises(TimeoutError):
                from shardcast.protocol import connect_and_handshake
                connect_and_handshake("127.0.0.1", 19498, timeout=0.2)
        finally:
            listener.close()

    def test_unknown_lease_rejected_with_e_field_path(self) -> None:
        node = self.spec.nodes[0]
        data = generate_sample(node.seed, "a", 5)
        request = {
            "type": "dispatch",
            "lease_id": "never-installed",
            "lease_version": 1,
            "node_id": "n1",
            "shard_id": "a",
            "capacity_version": 1,
            "sample_rows": 5,
            "batches": [data.tolist()],
        }
        receipt = request_dispatch(node.host, node.port, request, timeout=2.0)
        self.assertFalse(receipt["accepted"])
        self.assertEqual(receipt["error"], "unknown_lease")


if __name__ == "__main__":
    unittest.main()
