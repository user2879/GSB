import json
import os
import shutil
import sqlite3
import tempfile
import unittest

from layerbench.app import Workbench
from layerbench.cache import CacheIndex, INDEX_FILENAME
from layerbench.errors import E_MANIFEST, E_REPLAY, E_SNAPSHOT
from layerbench import replay as replay_mod
from layerbench import snapshot as snapshot_mod
from layerbench.ledger import Ledger
from layerbench.manifest import load_manifest
from layerbench.scheduler import build_dag

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "samples", "workspace")
MANIFEST = os.path.join(ROOT, "samples", "manifests", "sample_manifest.json")


class SnapshotReplayTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="lb-snap-")
        self.workdir = os.path.join(self.tmp, "wd")
        self.bench = Workbench(self.workdir)
        self.result = self.bench.build(MANIFEST, source_root=SRC)
        self.snap = self.result["snapshot_path"]

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_snapshot_meta_is_self_consistent(self):
        meta = snapshot_mod.load_snapshot_meta(self.snap)
        self.assertEqual(meta["manifest_version"], "sample-v1")
        self.assertEqual(len(meta["completed_node_uids"]), 10)
        self.assertTrue(meta["content_digest"])

    def test_replay_matches_snapshot(self):
        outcome = replay_mod.replay_snapshot(self.snap, SRC)
        self.assertTrue(outcome["ok"])
        self.assertEqual(outcome["nodes_compared"], 10)
        self.assertLessEqual(outcome["occupancy"]["peak_global"], 2)

    def test_cache_mismatch_forces_full_replace(self):
        cache_path = os.path.join(self.bench.cache_dir, INDEX_FILENAME)
        before = self.bench.current_cache_fingerprint()
        replay_mod.inject_cache_mismatch(cache_path)
        after = self.bench.current_cache_fingerprint()
        self.assertNotEqual(before, after)
        active = self.bench.stage_manifest(MANIFEST)
        manifest = load_manifest(active, workspace=SRC)
        decision = snapshot_mod.restore_snapshot(
            self.snap, self.workdir, active, self.bench.cache_dir,
            self.bench.ledger_path,
            snapshot_mod.manifest_fingerprint(manifest), after,
        )
        self.assertTrue(decision["replaced"])
        self.assertIn("cache_index", decision["mismatch"])
        self.assertTrue(os.path.isdir(decision["archive_dir"]))

    def test_corrupt_manifest_and_zero_slots_rejected(self):
        inject_dir = os.path.join(self.workdir, "injections")
        os.makedirs(inject_dir, exist_ok=True)
        for kind in ("corrupt-manifest", "zero-slots"):
            target = os.path.join(inject_dir, f"_inject_{kind}.json")
            shutil.copy2(MANIFEST, target)
            outcome = replay_mod.assert_injection_rejected(kind, target, SRC)
            self.assertTrue(outcome["rejected"])
            self.assertEqual(outcome["code"], E_MANIFEST)

    def test_tampered_snapshot_meta_detected(self):
        meta_path = os.path.join(self.snap, "meta.json")
        os.chmod(meta_path, 0o644)
        with open(meta_path, encoding="utf-8") as handle:
            meta = json.load(handle)
        meta["manifest_version"] = "tampered"
        with open(meta_path, "w", encoding="utf-8") as handle:
            handle.write(json.dumps(meta, sort_keys=True))
        with self.assertRaises(Exception) as ctx:
            snapshot_mod.load_snapshot_meta(self.snap)
        self.assertEqual(ctx.exception.code, E_SNAPSHOT)


class CrashRecoveryTests(unittest.TestCase):
    def test_unknown_in_flight_requires_rerun(self):
        tmp = tempfile.mkdtemp(prefix="lb-crash-")
        try:
            workdir = os.path.join(tmp, "wd")
            bench = Workbench(workdir)
            proc_exit = os.system(
                f'python3 -m layerbench.cli build --manifest "{MANIFEST}" '
                f'--workdir "{workdir}" --source-root "{SRC}" '
                f'--crash-at builder/2 >/dev/null 2>&1'
            )
            self.assertEqual(proc_exit, 137 * 256)
            ledger = Ledger(workdir)
            manifest = load_manifest(MANIFEST, workspace=SRC)
            uids = set(build_dag(manifest))
            build_id = ledger._conn.execute(
                "select build_id from builds order by started_ms desc limit 1"
            ).fetchone()[0]
            info = snapshot_mod.detect_in_flight(ledger, build_id, uids)
            ledger.close()
            self.assertIn("builder#2:bd-compute", info["unknown"])
            self.assertTrue(info["completed"])
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
