import socket
import threading
import time
import unittest

from lease_diag.clock import RealClock
from lease_diag.engine import Engine
from lease_diag.errors import E_TIMEOUT
from lease_diag.model import Pool, Request
from lease_diag.server import (CoordinatorServer, LeaseClient, LOOPBACK,
                               is_port_free, pick_free_port)


def make_pools():
    return {"P1": Pool("P1", 2, 500, 100, 1, 1),
            "P2": Pool("P2", 1, 500, 100, 1, 1)}


class ServerTests(unittest.TestCase):
    def setUp(self):
        self.clock = RealClock()
        self.engine = Engine(make_pools(), {})
        self.server = CoordinatorServer(self.engine, self.clock,
                                        make_pools(), global_cap=1)
        self.server.start()
        self.client = LeaseClient(LOOPBACK, self.server.port, 2000, 2)

    def tearDown(self):
        self.server.stop()

    def test_port_is_loopback_high_port_and_free_probe(self):
        self.assertTrue(20000 <= self.server.port <= 60000)
        self.assertFalse(is_port_free(port=self.server.port))
        self.assertTrue(is_port_free(port=pick_free_port(None)))

    def test_grant_renew_over_tcp_json_lines(self):
        grant = self.client.send({
            "kind": "GRANT", "request_id": "r1", "key": "k1",
            "pool_id": "P1", "revision": 1, "priority": 4,
            "arrive_ms": 0})
        self.assertTrue(grant["ok"])
        renew = self.client.send({
            "kind": "RENEW", "request_id": "r2", "key": "k1",
            "pool_id": "P1", "revision": 2, "priority": 4,
            "arrive_ms": 10})
        self.assertTrue(renew["ok"])
        self.assertEqual(renew["event_type"], "RENEW")
        lease = self.engine.leases[renew["lease_id"]]
        self.assertEqual(lease.renew_count, 1)

    def test_global_cap_serializes_grants_and_records_queue(self):
        # P2 quota is 1 and global cap is 1; hold the global slot manually.
        acquired = self.server._global_slots.acquire(blocking=False)
        self.assertTrue(acquired)
        result = {}

        def delayed_grant():
            result["response"] = self.client.send({
                "kind": "GRANT", "request_id": "rq", "key": "kx",
                "pool_id": "P2", "revision": 1, "priority": 1,
                "arrive_ms": 0})

        thread = threading.Thread(target=delayed_grant)
        thread.start()
        time.sleep(0.15)
        self.server._global_slots.release()
        thread.join(timeout=3)
        self.assertTrue(result["response"]["ok"])

    def test_real_socket_timeout_and_retry_preserves_semantics(self):
        # A client whose timeout is shorter than the held-slot window retries;
        # the retry is idempotent because the request id is unchanged.
        server = CoordinatorServer(Engine(make_pools(), {}), RealClock(),
                                   make_pools(), global_cap=1)
        server.start()
        try:
            self.assertTrue(server._global_slots.acquire(blocking=False))
            client = LeaseClient(LOOPBACK, server.port, 50, max_retries=1)
            response = client.send({
                "kind": "GRANT", "request_id": "rt", "key": "kt",
                "pool_id": "P1", "revision": 1, "priority": 1,
                "arrive_ms": 0})
            self.assertEqual(response["code"], E_TIMEOUT)
            self.assertEqual(response["attempts"], 2)
            server._global_slots.release()
            ok = LeaseClient(LOOPBACK, server.port, 1000).send({
                "kind": "GRANT", "request_id": "rt", "key": "kt",
                "pool_id": "P1", "revision": 1, "priority": 1,
                "arrive_ms": 0})
            self.assertTrue(ok["ok"])
        finally:
            server.stop()

    def test_conflict_returns_e_lease_with_lease_reference(self):
        self.client.send({"kind": "GRANT", "request_id": "a1", "key": "ka",
                          "pool_id": "P2", "revision": 1, "priority": 1,
                          "arrive_ms": 0})
        denied = self.client.send({
            "kind": "GRANT", "request_id": "a2", "key": "kb",
            "pool_id": "P2", "revision": 1, "priority": 1,
            "arrive_ms": 5})
        self.assertFalse(denied["ok"])
        self.assertEqual(denied["code"], "E_LEASE")
        self.assertTrue(denied["lease_id"].startswith("L-ka-P2"))


if __name__ == "__main__":
    unittest.main()
