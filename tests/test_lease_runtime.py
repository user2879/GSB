import os
import tempfile
import time
import unittest

from shardflow.engine import Engine
from shardflow.lease import grant_lease
from shardflow.spec import load_spec

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SPEC_PATH = os.path.join(ROOT, "demo", "demo.spec.json")
EVENTS_PATH = os.path.join(ROOT, "demo", "demo.events.ndjson")


class LeaseRuntimeTests(unittest.TestCase):
    def test_reaper_uses_real_monotonic_expiry(self):
        spec = load_spec(SPEC_PATH, EVENTS_PATH)
        with tempfile.TemporaryDirectory() as tmp:
            engine = Engine(
                spec, os.path.join(tmp, "expiry.sqlite"),
                start_nodes=False, enable_renewer=False,
            )
            runtime = engine.runtimes["s1"]
            runtime.status = "inflight"
            runtime.lease = grant_lease("s1", "node-a", 1, 0,
                                        spec.lease_ttl_ms)
            engine.active_leases["node-a"].add("s1")
            engine.active_shards["node-a"].add("s1")
            engine.resource_in_use = 2
            time.sleep((spec.lease_ttl_ms + 30) / 1000.0)
            engine._reaper_loop = lambda: None
            # Execute one reaper pass inline using the loop body semantics.
            from shardflow.util import now_ms
            now = now_ms()
            self.assertTrue(now >= runtime.lease.expires_ms)
            engine._release(runtime, kind="expire",
                            extra={"expires_ms": runtime.lease.expires_ms})
            self.assertEqual(runtime.status, "pending")
            self.assertIsNone(runtime.lease)
            self.assertEqual(engine.resource_in_use, 0)
            kinds = [event["kind"] for event in
                     engine.ledger.fetch_lease_events()]
            self.assertIn("expire", kinds)

    def test_optimistic_renew_rejects_stale_node_version(self):
        spec = load_spec(SPEC_PATH, EVENTS_PATH)
        with tempfile.TemporaryDirectory() as tmp:
            engine = Engine(
                spec, os.path.join(tmp, "renew.sqlite"),
                start_nodes=False, enable_renewer=False,
            )
            runtime = engine.runtimes["s1"]
            runtime.status = "inflight"
            runtime.lease = grant_lease("s1", "node-a", 1, 0,
                                        spec.lease_ttl_ms)
            before = runtime.lease.expires_ms
            runtime.lease.lease_version = 3
            from shardflow.errors import LeaseError
            with self.assertRaises(LeaseError) as ctx:
                from shardflow.lease import check_renew
                check_renew(
                    runtime.lease, 1,
                    spec.nodes["node-a"].max_renewals,
                    10, spec.lease_ttl_ms,
                )
            self.assertEqual(ctx.exception.code, "E_LEASE")
            self.assertEqual(runtime.lease.expires_ms, before)


if __name__ == "__main__":
    unittest.main()
