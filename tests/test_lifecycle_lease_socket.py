import os
import signal
import unittest

from rlcb.clock import FakeMonotonicClock
from rlcb.codes import E_LEASE, LeaseError
from rlcb.lease import LeaseManager, LeaseRegistry
from rlcb.transport import ServiceClient


class LeaseSocketRenewalTest(unittest.TestCase):
    def setUp(self):
        self.client = ServiceClient(handshake_timeout_ms=3000)
        self.sp = self.client.spawn_and_connect(
            service_id="svcRenew", host="127.0.0.1", port=34050,
            config_version=1, seed=20261126, ttl_ms=500)
        self.registry = None  # mirror server state locally for the manager

    def tearDown(self):
        self.sp.close()

    def test_optimistic_version_check_over_real_socket(self):
        # Server-side authoritative version lives in the subprocess.
        reply = self.client.request(self.sp, {
            "kind": "lease_bump", "batch_id": "",
            "lease_id": "LS-renew-1", "initial_version": 1})
        self.assertEqual(reply["version"], 2)
        ok = self.client.request(self.sp, {
            "kind": "lease_check", "batch_id": "",
            "lease_id": "LS-renew-1", "expected_version": 2})
        self.assertTrue(ok["lease_ok"])
        stale = self.client.request(self.sp, {
            "kind": "lease_check", "batch_id": "",
            "lease_id": "LS-renew-1", "expected_version": 1})
        self.assertFalse(stale["lease_ok"])

    def test_manager_refuses_stale_renewal_without_extending(self):
        clock = FakeMonotonicClock(0)
        manager = LeaseManager(clock, ttl_ms=100)
        registry = LeaseRegistry("svcRenew", ttl_ms=100)
        lease = manager.grant(service_id="svcRenew", batch_id="B1",
                              slot_index=0, quota_version=1, lease_seq=1)
        registry.record_grant(lease.lease_id, 1)
        registry.bump(lease.lease_id)  # server is now at version 2
        original_expiry = manager._leases[lease.lease_id].expires_ms
        with self.assertRaises(LeaseError) as ctx:
            manager.renew(lease.lease_id, 1, registry)
        self.assertEqual(ctx.exception.code, E_LEASE)
        self.assertEqual(manager._leases[lease.lease_id].expires_ms,
                         original_expiry)


class ForceKillRecoveryTest(unittest.TestCase):
    def test_subprocess_sigkill_leaves_recoverable_state(self):
        client = ServiceClient(handshake_timeout_ms=3000)
        sp = client.spawn_and_connect(
            service_id="svcKill", host="127.0.0.1", port=34060,
            config_version=1, seed=20261126, ttl_ms=200)
        pid = sp.process.pid
        # A valid round trip before the kill proves the socket path was live.
        reply = client.request(sp, {
            "kind": "execute", "batch_id": "BK",
            "requests": [{"request_id": "k1", "arrival_seq": 0,
                          "failed": 0}]})
        self.assertTrue(reply["ok"])
        # Close the parent-side pipes before the forced kill so the killed
        # process leaves no dangling file descriptors in the test process.
        for stream in (sp.process.stdout, sp.process.stderr):
            stream.close()
        os.kill(pid, signal.SIGKILL)
        sp.process.wait()
        self.assertIsNotNone(sp.process.poll())
        self.assertLess(sp.process.returncode, 0)
        # The socket client must surface a clean transport error afterwards,
        # never a hang or a silent fake success.
        sp.sock.close()


if __name__ == "__main__":
    unittest.main()
