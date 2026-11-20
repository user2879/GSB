"""Snapshot recovery and mismatch-localization tests."""
from __future__ import annotations

import os
import shutil
import sys
import tempfile
import unittest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))

from featflow.common import CODE_INCREMENT_MISMATCH, write_jsonl  # noqa: E402
from featflow.run import run_pipeline, run_resume  # noqa: E402
from featflow.snapshot import SnapshotStore  # noqa: E402
from featflow.verifier import IncrementalVerifier  # noqa: E402

sys.path.insert(0, os.path.join(ROOT, "tests"))
from test_e2e import make_tree, make_events, base_config  # noqa: E402


class RecoveryTests(unittest.TestCase):
    def setUp(self) -> None:
        self.data = tempfile.mkdtemp()
        self.out = tempfile.mkdtemp()
        make_tree(self.data, events=make_events(16),
                  config=base_config(snapshot_commit_every=5))

    def tearDown(self) -> None:
        shutil.rmtree(self.data)
        shutil.rmtree(self.out)

    def test_resume_matches_fresh_run(self) -> None:
        run_pipeline(self.data, self.out, use_injections=False, tag="full")
        report = run_resume(self.data, self.out)
        self.assertEqual(report["status"], "resumed")
        self.assertIsNone(report["first_diff"])
        self.assertGreaterEqual(report["resumed_from_seq"], 0)

    def test_stale_snapshot_rejected(self) -> None:
        run_pipeline(self.data, self.out, use_injections=False, tag="full")
        snap = SnapshotStore(os.path.join(self.out, "snapshot", "full"))
        manifest = snap.load_manifest()
        manifest["definition_hash"] = "tampered"
        snap.save_manifest(manifest)
        report = run_resume(self.data, self.out)
        self.assertEqual(report["status"], "stale")
        self.assertEqual(report["failure"]["code"], "E_SNAPSHOT_STALE")


class MismatchLocalizationTests(unittest.TestCase):
    def test_first_diff_key_is_reported(self) -> None:
        verifier = IncrementalVerifier(None)  # type: ignore[arg-type]
        full = {"features": [{"a": 1}], "samples": [], "ledger": [],
                "failures": [], "cache": {"registry": {"0": ["k"]}}}
        broken = {"features": [{"a": 2}], "samples": [], "ledger": [],
                  "failures": [], "cache": {"registry": {"0": ["k"]}}}
        path = os.path.join(tempfile.mkdtemp(), "eq.json")
        ok, diff, failures = verifier.compare_runs(full, broken, path)
        self.assertFalse(ok)
        self.assertEqual(diff, "[0].a")
        self.assertEqual(failures[0].code, CODE_INCREMENT_MISMATCH)


if __name__ == "__main__":
    unittest.main(verbosity=2)
