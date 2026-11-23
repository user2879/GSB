"""Snapshot save / whole-replace restore / replay-verification tests."""

import json
import os
import tempfile
import unittest

from tests.helpers import make_spec_files
from workbench import E_REPLAY
from workbench.engine import Engine
from workbench.errors import DiagError
from workbench.ledger import Ledger
from workbench.snapshot import (build_snapshot, restore_snapshot,
                                save_snapshot, verify_replay_after_restore)
from workbench.spec import load_spec
from workbench.upstream_mgr import UpstreamManager


class SnapshotTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="wbsnap-")
        make_spec_files(self.tmp)
        self.spec = load_spec(self.tmp)
        self.ledger_path = os.path.join(self.tmp, "ledger.db")
        self.snap_path = os.path.join(self.tmp, "snapshot.json")
        ledger = Ledger(self.ledger_path, scale=3,
                        services_digest=self.spec.services_digest,
                        keys_digest=self.spec.keys_digest)
        engine = Engine(self.spec, ledger, global_limit=2,
                        per_service_quota=1, upstream_timeout_s=1.0)
        with UpstreamManager(self.spec) as mgr:
            mgr.start()
            engine.run_all(batch_size=4)
        ledger.close()

    def _open_ledger(self):
        return Ledger(self.ledger_path)

    def test_save_and_accept(self):
        ledger = self._open_ledger()
        snap = build_snapshot(self.spec, ledger)
        save_snapshot(self.snap_path, snap)
        restored, diffs, replaced = restore_snapshot(
            self.snap_path, self.spec, ledger.scale)
        self.assertFalse(replaced)
        self.assertEqual(diffs, [])
        self.assertEqual(restored["ledger_cursor"], 8)
        self.assertEqual(len(restored["conclusions"]), 8)
        self.assertIn("svc-a|alpha", restored["token_balances"])
        n = verify_replay_after_restore(self.spec, ledger)
        self.assertEqual(n, 8)
        ledger.close()

    def test_tampered_digest_whole_replace(self):
        ledger = self._open_ledger()
        snap = build_snapshot(self.spec, ledger)
        save_snapshot(self.snap_path, snap)
        with open(self.snap_path, encoding="utf-8") as fh:
            tampered = json.load(fh)
        tampered["token_balances"]["svc-a|alpha"] = 12345
        with open(self.snap_path, "w", encoding="utf-8") as fh:
            json.dump(tampered, fh)
        restored, diffs, replaced = restore_snapshot(
            self.snap_path, self.spec, ledger.scale)
        self.assertTrue(replaced)
        self.assertIsNone(restored)
        self.assertEqual(diffs, ["content_digest"])
        ledger.close()

    def test_version_mismatch_replace_stable_diffs(self):
        ledger = self._open_ledger()
        snap = build_snapshot(self.spec, ledger)
        save_snapshot(self.snap_path, snap)
        with open(self.snap_path, encoding="utf-8") as fh:
            tampered = json.load(fh)
        tampered["scale_factor"] = 5
        tampered["key_versions"]["svc-a"] = [1, 2, 3]
        with open(self.snap_path, "w", encoding="utf-8") as fh:
            json.dump(tampered, fh)
        restored, diffs, replaced = restore_snapshot(
            self.snap_path, self.spec, ledger.scale)
        self.assertTrue(replaced)
        self.assertEqual(diffs, sorted(diffs))
        self.assertIn("scale_factor", diffs)
        self.assertIn("key_versions.svc-a", diffs)
        self.assertIn("content_digest", diffs)
        ledger.close()

    def test_replay_detects_ledger_tamper(self):
        ledger = self._open_ledger()
        snap = build_snapshot(self.spec, ledger)
        save_snapshot(self.snap_path, snap)
        ledger.conn.execute(
            "UPDATE requests SET token_after=token_after+1 WHERE seq=3")
        ledger.conn.commit()
        with self.assertRaises(DiagError) as ctx:
            verify_replay_after_restore(self.spec, ledger)
        self.assertEqual(ctx.exception.code, E_REPLAY)
        self.assertEqual(ctx.exception.seq, 3)
        self.assertEqual(ctx.exception.field, "token_after")
        ledger.close()


if __name__ == "__main__":
    unittest.main()
