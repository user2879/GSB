from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from quota_ledger.errors import BusinessError
from quota_ledger.ledger import Ledger
from quota_ledger.spec import load_spec
from quota_ledger.util import monotonic_ms


class LedgerQuotaTests(unittest.TestCase):
    def test_cancel_releases_reserved_quota_once_and_lease_is_globally_unique(self) -> None:
        project = Path(__file__).resolve().parent.parent
        spec = load_spec(project / "samples" / "demo")
        with tempfile.TemporaryDirectory() as temp:
            ledger = Ledger(Path(temp) / "ledger.sqlite3")
            try:
                nodes = {node_id: dict(values) for node_id, values in spec.revisions[1].node_changes.items()}
                ledger.begin_run("test-input", nodes, 4)
                now = monotonic_ms()
                ledger.reserve_dispatch(
                    event_seq=1,
                    shard_id="s01",
                    generation=1,
                    node_id="node-a",
                    lease_id="lease:s01:g1",
                    granted_ms=now,
                    expires_ms=now + 5000,
                    config_version=1,
                    units=2,
                )
                self.assertEqual(ledger.budget_state()["used_units"], 2)
                with self.assertRaises(BusinessError) as ctx:
                    ledger.reserve_dispatch(
                        event_seq=1,
                        shard_id="s01",
                        generation=1,
                        node_id="node-b",
                        lease_id="lease:s01:other",
                        granted_ms=now,
                        expires_ms=now + 5000,
                        config_version=1,
                        units=2,
                    )
                self.assertEqual(ctx.exception.code, "E_LEASE")
                result = ledger.apply_cancel_event(
                    {"seq": 2, "type": "cancel", "shard_id": "s01", "reason": "interrupt"}
                )
                self.assertIsNotNone(result["released"])
                self.assertEqual(ledger.budget_state()["used_units"], 0)
                for row in ledger.node_quotas():
                    self.assertEqual(row["active_shards"], 0)
                    self.assertEqual(row["active_leases"], 0)
                lease = ledger.leases()[0]
                self.assertEqual(lease["state"], "cancelled")
                self.assertEqual(ledger.cancelled_shards()["s01"]["reason"], "interrupt")
                ledger.apply_cancel_event(
                    {"seq": 3, "type": "cancel", "shard_id": "s01", "reason": "interrupt-again"}
                )
                self.assertEqual(ledger.budget_state()["used_units"], 0)
            finally:
                ledger.close()


if __name__ == "__main__":
    unittest.main()
