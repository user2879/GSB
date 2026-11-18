import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


from scripts.generate_samples import write_bundle
from texmat.errors import ConfigMismatch
from texmat.snapshot import SnapshotStore
from texmat.testkit import make_engine, run_incremental


class SnapshotRecoveryTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)
        self.sample = self.tmp / "sample"
        write_bundle(self.sample, size=16, workers=2, budget=100000)

    def tearDown(self):
        self._tmp.cleanup()

    def test_manifest_roundtrip_and_versions(self):
        snapshot = self.tmp / "snap"
        result = run_incremental(self.sample, snapshot)
        store = SnapshotStore(snapshot)
        manifest = store.load_manifest()
        self.assertIsNotNone(manifest)
        self.assertGreater(manifest.global_version, 0)
        self.assertTrue(manifest.completed_cursor)
        self.assertEqual(manifest.config_hash, result["engine"].config_hash_value)
        # every artifact file exists and is reachable from the manifest
        for node_id, state in manifest.artifacts.items():
            self.assertTrue(store.artifact_exists(state.payload_path), node_id)

    def test_strict_resume_rejects_config_change(self):
        snapshot = self.tmp / "snap"
        run_incremental(self.sample, snapshot)
        raw = json.loads((self.sample / "generation_config.json").read_text())
        raw["fractional_bits"] = 11
        (self.sample / "generation_config.json").write_text(json.dumps(raw, sort_keys=True))
        engine = make_engine(self.sample, snapshot, strict=True)
        with self.assertRaises(ConfigMismatch):
            engine.execute_run()

    def test_non_strict_reports_mismatch_and_rebuilds(self):
        snapshot = self.tmp / "snap"
        first = run_incremental(self.sample, snapshot)
        old_hash = first["engine"].config_hash_value
        raw = json.loads((self.sample / "generation_config.json").read_text())
        raw["fractional_bits"] = 11
        (self.sample / "generation_config.json").write_text(json.dumps(raw, sort_keys=True))
        engine = make_engine(self.sample, snapshot)
        result = engine.execute_run()
        self.assertIsNotNone(result.mismatch)
        self.assertEqual(result.mismatch["code"], "E_CONFIG_MISMATCH")
        self.assertEqual(result.mismatch["stored"], old_hash)

    def test_graceful_signal_exit_code_zero(self):
        snapshot = self.tmp / "snap"
        report = self.tmp / "report.json"
        # Deterministic graceful stop after two committed tasks, independent
        # of wall-clock timing or thread scheduling.
        cmd = [
            sys.executable, "-m", "texmat", "run",
            "--sample-dir", str(self.sample),
            "--snapshot-dir", str(snapshot),
            "--report", str(report),
            "--stop-after-batches", "2",
        ]
        proc = subprocess.run(cmd, cwd=Path(__file__).resolve().parents[1],
                              capture_output=True, text=True, timeout=30)
        self.assertEqual(proc.returncode, 0, proc.stderr)
        payload = json.loads(report.read_text())
        self.assertEqual(payload["status"], "signal_stopped")
        # committed manifest must still be internally consistent
        store = SnapshotStore(snapshot)
        manifest = store.load_manifest()
        self.assertIsNotNone(manifest)

    def test_force_kill_recovers_only_committed_artifacts(self):
        snapshot = self.tmp / "snap"
        # First establish a valid committed prefix with a tight budget.
        engine = make_engine(self.sample, snapshot)
        object.__setattr__(engine.config, "point_budget", 300)
        engine.execute_run()
        store = SnapshotStore(snapshot)
        manifest_before = store.load_manifest()
        committed = set(manifest_before.artifacts)
        # Simulate a hard kill: leave a temp file behind that must be ignored.
        stray = snapshot / ".tmp-stray.npz"
        stray.write_bytes(b"not a real artifact")
        engine2 = make_engine(self.sample, snapshot)
        object.__setattr__(engine2.config, "point_budget", 300)
        result = engine2.execute_run()
        manifest_after = engine2.store.load_manifest()
        self.assertTrue(set(manifest_after.artifacts) >= committed)
        # recovery only references manifest-listed payloads, never the stray
        for state in manifest_after.artifacts.values():
            self.assertNotIn("stray", state.payload_path)
        self.assertTrue(result.reused or result.recomputed)


if __name__ == "__main__":
    unittest.main()
