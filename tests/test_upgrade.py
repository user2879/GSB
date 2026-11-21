import json
import tempfile
import unittest
from pathlib import Path

from lease_middleware.codec import decode_records
from lease_middleware.errors import E_UPGRADE, MiddlewareError
from lease_middleware.scenario import import_legacy_leases
from lease_middleware.spec import load_spec
from lease_middleware.storage import Storage
from lease_middleware.upgrade import UpgradeManager

SAMPLES = Path(__file__).resolve().parent.parent / "samples"


class UpgradeTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)
        self.spec = load_spec(SAMPLES / "config.json")
        self.db = self.dir / "db.sqlite"
        self.storage = Storage(self.db)
        import_legacy_leases(self.storage, self.spec,
                             [SAMPLES / "leases_r1.json", SAMPLES / "leases_r2.json"])
        self.storage.close()

    def tearDown(self):
        self.tmp.cleanup()

    def test_upgrade_one_by_one_with_checkpoint(self):
        storage = Storage(self.db)
        manager = UpgradeManager(storage, self.spec)
        first = manager.upgrade_next()
        self.assertIsNotNone(first)
        checkpoint = json.loads(self.db.with_suffix(".upgrade.json").read_text())
        self.assertEqual(checkpoint["upgraded"], [first])
        second = manager.upgrade_next()
        self.assertNotEqual(first, second)
        # resume: already upgraded rows skipped, no duplicate processing
        report = UpgradeManager(storage, self.spec).run()
        self.assertNotIn(first, report.upgraded_ids)
        rows = storage.list_lease_rows()
        self.assertTrue(all(r["format"] == "r3" for r in rows))
        self.assertTrue(all(r["backup_json"] is not None for r in rows))
        storage.close()

    def test_checkpoint_hash_mismatch_aborts(self):
        storage = Storage(self.db)
        UpgradeManager(storage, self.spec).upgrade_next()
        storage.close()
        # Tamper the upgraded lease row after the checkpoint was written.
        storage = Storage(self.db)
        row = storage.list_lease_rows()[0]
        rec = json.loads(row["record_json"])
        rec["renewal_count"] = 999
        storage.conn.execute(
            "UPDATE lease_records SET record_json=? WHERE lease_id=?",
            (json.dumps(rec, sort_keys=True), row["lease_id"]))
        storage.conn.commit()
        storage.close()
        storage = Storage(self.db)
        with self.assertRaises(MiddlewareError) as ctx:
            UpgradeManager(storage, self.spec).run()
        self.assertEqual(ctx.exception.code, E_UPGRADE)
        storage.close()

    def test_legacy_decode_rejects_bad_record_without_overwriting(self):
        bad = json.loads((SAMPLES / "leases_r3.json").read_text())
        bad[0]["value_digest"] = "tampered"
        path = self.dir / "bad.json"
        path.write_text(json.dumps(bad))
        storage = Storage(self.dir / "fresh.sqlite")
        with self.assertRaises(MiddlewareError):
            import_legacy_leases(storage, self.spec, [path])
        self.assertEqual(storage.list_lease_rows(), [])
        storage.close()


if __name__ == "__main__":
    unittest.main()
