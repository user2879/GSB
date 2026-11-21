import tempfile
import unittest
from pathlib import Path

from lease_middleware import inject
from lease_middleware.errors import E_CURSOR, E_REPLAY, MiddlewareError
from lease_middleware.replay import Replayer
from lease_middleware.scenario import build_fresh
from lease_middleware.spec import load_spec
from lease_middleware.storage import Storage
from lease_middleware.verifier import verify

SAMPLES = Path(__file__).resolve().parent.parent / "samples"
LEGACY = ["leases_r1.json", "leases_r2.json", "leases_r3.json"]


class ReplayVerifierTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)
        self.db = self.dir / "ledger.sqlite"
        build_fresh(SAMPLES / "config.json", self.db,
                    [SAMPLES / n for n in LEGACY], SAMPLES / "changes.json")

    def tearDown(self):
        self.tmp.cleanup()

    def test_replay_and_independent_verifier_pass(self):
        storage = Storage(self.db)
        summary = Replayer(storage, load_spec(SAMPLES / "config.json")).run()
        self.assertTrue(summary["leases"])
        storage.close()
        report = verify(self.db, SAMPLES / "config.json")
        self.assertEqual(report["status"], "OK")
        for cid, info in report["cursors"].items():
            self.assertEqual(info["stored_cursor"], info["ledger_max_push"])

    def test_cursor_ahead_of_ledger_is_e_cursor(self):
        inject.inject_cursor_ahead(self.db, "client-r3", jump=50)
        with self.assertRaises(MiddlewareError) as ctx:
            verify(self.db, SAMPLES / "config.json")
        self.assertEqual(ctx.exception.code, E_CURSOR)

    def test_illegal_transition_detected_as_e_replay(self):
        conn = Storage(self.db).conn
        # Insert an impossible transition: revoke an already expired lease.
        conn.execute(
            "INSERT INTO ledger(event_type,client_id,lease_id,config_key,"
            "payload_json,created_at_ms) VALUES('lease_revoked',?,?,?,'{}',0)",
            ("client-r1", "L:client-r1:feature.flags:0001", "feature.flags"))
        conn.commit()
        conn.close()
        with self.assertRaises(MiddlewareError) as ctx:
            verify(self.db, SAMPLES / "config.json")
        self.assertEqual(ctx.exception.code, E_REPLAY)
        self.assertEqual(ctx.exception.field_path, "state")


if __name__ == "__main__":
    unittest.main()
