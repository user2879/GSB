import time
import unittest
from pathlib import Path
import tempfile

from lease_middleware.clock import Clock
from lease_middleware.engine import EXPIRED, RENEWED, LeaseEngine
from lease_middleware.errors import E_VERSION, MiddlewareError
from lease_middleware.spec import load_spec
from lease_middleware.storage import Storage

SAMPLES = Path(__file__).resolve().parent.parent / "samples"


class EngineTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.spec = load_spec(SAMPLES / "config.json")
        self.storage = Storage(Path(self.tmp.name) / "db.sqlite")
        self.engine = LeaseEngine(self.spec, self.storage, Clock())
        self.engine.rebuild_subscription_view("test_bootstrap")

    def tearDown(self):
        self.storage.close()
        self.tmp.cleanup()

    def test_grant_renew_and_optimistic_version_conflict(self):
        rec = self.engine.grant("client-r2", "cache.policy", config_version=1)
        renewed = self.engine.renew(rec.lease_id, 1)
        self.assertEqual(renewed.renewal_count, 1)
        with self.assertRaises(MiddlewareError) as ctx:
            self.engine.renew(rec.lease_id, 99)
        self.assertEqual(ctx.exception.code, E_VERSION)
        # failed renewal must not mutate state/count
        row = self.storage.get_lease_row(rec.lease_id)
        self.assertEqual(row["state"], RENEWED)

    def test_real_monotonic_expiry_and_reclaim(self):
        rec = self.engine.grant("client-r1", "feature.flags", config_version=1)
        time.sleep(0.45)  # client-r1 duration is 400ms
        expired = self.engine.sweep_expired()
        self.assertIn(rec.lease_id, expired)
        view = [r for r in self.storage.list_subscription_view()
                if r["client_id"] == "client-r1" and r["config_key"] == "feature.flags"][0]
        self.assertEqual(view["state"], EXPIRED)
        self.assertIsNone(view["lease_id"])
        # expired lease cannot be pushed to or renewed
        with self.assertRaises(MiddlewareError):
            self.engine.renew(rec.lease_id, 1)

    def test_push_and_cursor_atomic_and_ignores_unsubscribed(self):
        self.engine.grant("client-r1", "feature.flags", config_version=1)
        result = self.engine.publish_config(
            "feature.flags", 2,
            self.spec.configs["feature.flags"].value_at(2))
        clients = {p["client_id"] for p in result["pushed"]}
        self.assertIn("client-r1", clients)
        # client-r3 does not subscribe to feature.flags
        self.assertNotIn("client-r3", clients)
        self.assertEqual(self.storage.get_cursor("client-r1"), 2)
        self.assertEqual(self.storage.get_cursor("client-r3"), 0)

    def test_uncommitted_renewal_marked_unknown_on_rebuild(self):
        rec = self.engine.grant("client-r3", "service.endpoint", config_version=1)
        self.engine.renew(rec.lease_id, 1, commit=False)
        self.assertTrue(self.engine.uncommitted_renewals())
        diff = self.engine.rebuild_subscription_view("test")
        self.assertEqual(diff.unknown_renewals[0]["lease_id"], rec.lease_id)
        self.assertEqual(diff.unknown_renewals[0]["resolution"], "re_grant_required")
        self.assertEqual(self.engine.uncommitted_renewals(), [])


if __name__ == "__main__":
    unittest.main()
