import copy
import tempfile
import unittest
from pathlib import Path

from arbworkbench.errors import ReplayError, VersionError
from arbworkbench.ledger import LEDGER_FORMAT_VERSION, Ledger
from arbworkbench.session import Session
from arbworkbench.spec import load_specification
from arbworkbench.versioning import VersionRegistry, diff_keys

from tests import SAMPLE_ENV, SAMPLE_VIEWS


class PersistenceTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.tmp_path = Path(self.tmp.name)
        self.ledger_path = self.tmp_path / "ledger.sqlite"
        self.snapshot_path = self.tmp_path / "snapshot.json"

    def tearDown(self):
        self.tmp.cleanup()

    def _session(self) -> Session:
        spec = load_specification(SAMPLE_ENV, SAMPLE_VIEWS)
        return Session(spec, Ledger(self.ledger_path))

    def test_full_run_persists_and_reloads(self):
        session = self._session()
        session.run_all()
        self.assertEqual(session.unknown_views(), [])
        session.ledger.close()

        reopened = self._session()
        self.assertEqual(sorted(reopened.conclusions),
                         ["view-all-fleets", "view-production"])
        for view_id, vc in reopened.conclusions.items():
            self.assertGreaterEqual(vc.version, 1)

    def test_forced_termination_leaves_unknown_then_rerun(self):
        session = self._session()
        with self.assertRaises(RuntimeError):
            session.run_all(crash_view_id="view-production")
        session.ledger.close()

        # Reopen with a fresh ledger handle; orphan row must be cleaned.
        recovered = self._session()
        self.assertIn("view-production", recovered.unknown_views())
        rerun = recovered.rerun_unknown()
        self.assertIn("view-production", rerun)
        self.assertEqual(recovered.unknown_views(), [])
        recovered.ledger.close()

        # A third open finds everything committed and nothing unknown.
        final = self._session()
        self.assertEqual(final.unknown_views(), [])

    def test_view_edit_monotonic_version_and_stale_replacement(self):
        session = self._session()
        session.run_all()
        old_version = session.registry.version_of("view-production")
        session.ledger.close()

        # Edit the view: change a policy and participating env set.
        import json
        view_doc = json.loads(Path(SAMPLE_VIEWS).read_text(encoding="utf-8"))
        view_doc["views"][0]["policies"]["timeout_ms"] = "weighted_majority"
        edited_view = self.tmp_path / "views_edited.json"
        edited_view.write_text(json.dumps(view_doc), encoding="utf-8")

        spec = load_specification(SAMPLE_ENV, edited_view)
        registry_snapshot = session.registry.snapshot()
        registry = VersionRegistry.from_snapshot(registry_snapshot)
        edited = Session(spec, Ledger(self.ledger_path), registry=registry)
        new_version = edited.registry.version_of("view-production")
        self.assertEqual(new_version, old_version + 1)
        # Old ledger rows carry the old version and must not be displayed.
        self.assertNotIn("view-production", edited.conclusions)
        conclusion = edited.run_view("view-production")
        self.assertEqual(conclusion.version, new_version)
        self.assertIn("timeout_ms", conclusion.changed_keys)

    def test_diff_keys_reports_added_removed_changed(self):
        old = {"a": "latest_wins", "b": "latest_wins"}
        new = {"b": "weighted_majority", "c": "interval_intersection"}
        diff = diff_keys(old, new)
        self.assertEqual(diff.added_keys, ["c"])
        self.assertEqual(diff.removed_keys, ["a"])
        self.assertEqual(diff.changed_keys, ["b"])
        self.assertTrue(diff.version_bumped)

    def test_snapshot_roundtrip_and_replay(self):
        session = self._session()
        session.run_all()
        session.save_snapshot(self.snapshot_path)
        session.ledger.close()

        import json
        snapshot = json.loads(self.snapshot_path.read_text(encoding="utf-8"))
        fresh = self._session()
        fresh.replay_against_snapshot(snapshot)  # no exception == match

    def test_snapshot_version_mismatch_raises_e_version(self):
        session = self._session()
        session.run_all()
        session.save_snapshot(self.snapshot_path)
        session.ledger.close()

        import json
        snapshot = json.loads(self.snapshot_path.read_text(encoding="utf-8"))
        snapshot["environment_version"] = "tampered"
        fresh = self._session()
        with self.assertRaises(VersionError) as ctx:
            fresh.verify_snapshot(snapshot)
        self.assertEqual(ctx.exception.code, "E_VERSION")

    def test_snapshot_digest_tampering_raises_e_version(self):
        session = self._session()
        session.run_all()
        session.save_snapshot(self.snapshot_path)
        session.ledger.close()

        import json
        snapshot = json.loads(self.snapshot_path.read_text(encoding="utf-8"))
        snapshot["scale_exponent"] = 3  # body changes, digest now stale
        fresh = self._session()
        with self.assertRaises(VersionError) as ctx:
            fresh.verify_snapshot(snapshot)
        self.assertEqual(ctx.exception.code, "E_VERSION")

    def test_replay_mismatch_locates_view_key_field(self):
        session = self._session()
        session.run_all()
        snapshot = session.build_snapshot()
        # Tamper a stored authority to simulate divergent persisted state.
        stored_kc = snapshot["view_conclusions"]["view-production"]["keys"]["pool_size"]
        stored_kc["authority_q"] = stored_kc["authority_q"] + 999
        # Recompute only the outer content digest wrapper bypass by removing it
        # so the integrity check reports the digest mismatch path first;
        # instead tamper the body AND its digest to reach the field comparator.
        from arbworkbench.canonical import content_digest
        body = {k: v for k, v in snapshot.items() if k != "content_digest"}
        snapshot["content_digest"] = content_digest(body)
        fresh = self._session()
        with self.assertRaises(ReplayError) as ctx:
            fresh.replay_against_snapshot(snapshot)
        self.assertEqual(ctx.exception.code, "E_REPLAY")
        self.assertIn("view-production", ctx.exception.location)
        self.assertIn("pool_size", ctx.exception.location)
        self.assertIn("authority_q", ctx.exception.location)

    def test_ledger_format_version_present(self):
        ledger = Ledger(self.ledger_path)
        row = ledger.conn.execute(
            "SELECT meta_value FROM ledger_meta WHERE meta_key='format_version'"
        ).fetchone()
        self.assertEqual(int(row[0]), LEDGER_FORMAT_VERSION)
        ledger.close()


if __name__ == "__main__":
    unittest.main()
