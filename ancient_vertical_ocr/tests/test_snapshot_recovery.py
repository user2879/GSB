import json
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from ancient_ocr.config import Config, write_default_config
from ancient_ocr.pipeline import Pipeline
from ancient_ocr.sample_gen import build_default_corpus
from ancient_ocr.snapshot import SnapshotStore


def _workspace():
    root = Path(tempfile.mkdtemp())
    (root / "config").mkdir()
    write_default_config(root / "config" / "config.json")
    cfg = Config.load(root / "config" / "config.json")
    samples = root / "samples"
    samples.mkdir()
    build_default_corpus(samples, cfg.frac_bits)
    return root, cfg


class SnapshotTest(unittest.TestCase):
    def test_commit_uses_single_current_directory(self):
        root, cfg = _workspace()
        pipeline = Pipeline(root / "samples", cfg, root / "snapshots")
        pipeline.execute()
        store = SnapshotStore(root / "snapshots")
        loaded = store.latest()
        self.assertIsNotNone(loaded)
        self.assertTrue((store.current / "COMMITTED").exists())
        tmp_leftovers = [p for p in (root / "snapshots").iterdir()
                         if p.name.startswith("tmp-")]
        self.assertEqual(tmp_leftovers, [])
        shutil.rmtree(root)

    def test_config_hash_mismatch_forces_full_rebuild(self):
        root, cfg = _workspace()
        Pipeline(root / "samples", cfg, root / "snapshots").execute()
        store = SnapshotStore(root / "snapshots")
        info = store.recover("wrong-hash")
        self.assertFalse(info.recoverable)
        self.assertEqual(info.issue.code, "E_SNAPSHOT_VERSION_MISMATCH")
        shutil.rmtree(root)

    def test_real_sigkill_then_recovery(self):
        root, cfg = _workspace()
        marker = root / ".crash_marker"
        marker.write_text("armed", encoding="utf-8")
        env = dict(os.environ)
        proc = subprocess.run(
            [sys.executable, "-m", "ancient_ocr.cli", "crash-demo", "--root", str(root)],
            cwd=Path(__file__).resolve().parents[1],
            env=env, capture_output=True, text=True)
        self.assertEqual(proc.returncode, -signal.SIGKILL)
        # 第二次运行：基于已提交快照恢复并继续
        proc2 = subprocess.run(
            [sys.executable, "-m", "ancient_ocr.cli", "crash-demo", "--root", str(root)],
            cwd=Path(__file__).resolve().parents[1],
            env=env, capture_output=True, text=True)
        self.assertEqual(proc2.returncode, 0, proc2.stderr)
        report = json.loads((root / "reports" / "crash_recovery_report.json").read_text())
        self.assertTrue(report["recovery"]["recoverable"])
        self.assertEqual(report["recovery"]["stage"], "merge")
        self.assertGreater(len(report["recovery"]["completed_units"]), 0)
        self.assertFalse(marker.exists())
        shutil.rmtree(root)


if __name__ == "__main__":
    unittest.main()
