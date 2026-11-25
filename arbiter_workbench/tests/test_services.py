import socket
import unittest
from pathlib import Path

from arbiter.errors import PortError, TimeoutErrorW
from arbiter.protocol import (
    connect_loopback,
    pick_free_port,
    recv_line,
)
from arbiter.sample import DEFAULT_KEYS, generate_all
from arbiter.service_manager import ServiceManager
from arbiter.spec import EnvSpec, load_declarations

DECLS = Path(__file__).resolve().parents[1] / "decls"


class SocketServiceTest(unittest.TestCase):
    def setUp(self):
        self.decls = load_declarations(DECLS, DEFAULT_KEYS)

    def _fresh_manager(self) -> ServiceManager:
        envs = [
            EnvSpec(e.env_id, e.host, pick_free_port(), e.lease_capacity, e.seed, e.weight)
            for e in self.decls.environments.values()
        ]
        return ServiceManager(envs)

    def test_handshake_batch_arbitration_receipts(self):
        manager = self._fresh_manager()
        try:
            manager.start_all()
            observations = generate_all(self.decls)
            for env_id, env_obs in observations.items():
                receipts = manager.push_in_batches(env_id, env_obs, 5)
                self.assertTrue(all(r["status"] == "OK" for r in receipts))
                arb = manager.arbitrate(env_id)
                self.assertEqual(arb["env_id"], env_id)
                self.assertIn("revision_min", arb)
                self.assertIn("revision_max", arb)
                self.assertIn("authority_digest", arb)
                self.assertIsInstance(arb["elapsed_ms"], int)
                self.assertGreaterEqual(arb["elapsed_ms"], 0)
        finally:
            manager.shutdown_all()

    def test_port_collision_raises_e_port(self):
        # Pin the retry candidate list to two ports and occupy both, so the
        # manager deterministically exhausts every option and reports E_PORT.
        from arbiter import service_manager

        candidates = [pick_free_port(), pick_free_port()]
        blocked_env = EnvSpec(
            self.decls.environments["env-prod-a"].env_id,
            "127.0.0.1", candidates[0], 8, 1, 1,
        )
        holders = []
        original = service_manager.RESERVED_PORT_BAND
        try:
            for port in candidates:
                holder = socket.socket()
                holder.bind(("127.0.0.1", port))
                holder.listen(1)
                holders.append(holder)
            service_manager.RESERVED_PORT_BAND = candidates
            with self.assertRaises(PortError) as ctx:
                ServiceManager([blocked_env]).start_all()
            self.assertEqual(ctx.exception.code, "E_PORT")
        finally:
            service_manager.RESERVED_PORT_BAND = original
            for holder in holders:
                holder.close()

    def test_timeout_on_silent_peer_raises_e_timeout(self):
        silent = socket.socket()
        silent.bind(("127.0.0.1", 0))
        silent.listen(1)
        client = socket.socket()
        client.settimeout(0.3)
        client.connect(silent.getsockname())
        try:
            with self.assertRaises(TimeoutErrorW) as ctx:
                recv_line(client, timeout_s=0.3)
            self.assertEqual(ctx.exception.code, "E_TIMEOUT")
        finally:
            client.close()
            silent.close()

    def test_connect_refused_is_e_port(self):
        listener = socket.socket()
        listener.bind(("127.0.0.1", 0))
        host, port = listener.getsockname()
        listener.close()
        with self.assertRaises((PortError, TimeoutErrorW)) as ctx:
            connect_loopback(host, port, timeout_s=1.0)
        self.assertIn(ctx.exception.code, {"E_PORT", "E_TIMEOUT"})


if __name__ == "__main__":
    unittest.main()
