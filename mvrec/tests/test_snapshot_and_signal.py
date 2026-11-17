import json
import os
import signal
import shutil
import tempfile
import unittest

from mvrec.cli import run_pipeline
from mvrec.samplegen import generate_sample
from mvrec.snapshot import atomic_write_json


class SnapshotTests(unittest.TestCase):
    def setUp(self):
        self.workdir = tempfile.mkdtemp(prefix="mvrec_snap_")
        self.sample_dir = os.path.join(self.workdir, "sample")
        generate_sample(self.sample_dir, 8)
        self.base = json.load(open(os.path.join(os.path.dirname(__file__), "..",
                                                 "configs", "recon_config.json"),
                                   encoding="utf-8"))

    def tearDown(self):
        shutil.rmtree(self.workdir, ignore_errors=True)

    def _config(self, **overrides):
        config = dict(self.base)
        tag = overrides.pop("tag", "x")
        config.update(overrides)
        path = os.path.join(self.workdir, "config_%s.json" % tag)
        json.dump(config, open(path, "w", encoding="utf-8"))
        return path

    def test_snapshot_uses_whole_directory_replacement(self):
        config_path = self._config(tag="snap")
        snap = os.path.join(self.workdir, "snap")
        run_pipeline(self.sample_dir, config_path, None, snap,
                     os.path.join(self.workdir, "report.json"))
        manifest = json.load(open(os.path.join(snap, "manifest.json"), encoding="utf-8"))
        self.assertGreater(manifest["global_version"], 0)
        self.assertEqual(len(manifest["config_hash"]), 64)
        self.assertEqual(len(manifest["camera_hash"]), 64)
        for stage in ("features", "matching", "depth", "merge"):
            version = manifest["stage_versions"][stage]
            stage_dir = os.path.join(snap, "%s_v%06d" % (stage, version))
            self.assertTrue(os.path.isdir(stage_dir))
            self.assertEqual(manifest["stage_paths"][stage], stage_dir)
        leftovers = [name for name in os.listdir(snap) if name.startswith(".tmp_")]
        self.assertEqual(leftovers, [])

    def test_force_kill_recovers_only_from_committed_manifest(self):
        config_path = self._config(tag="kill", point_budget=120, thread_count=2)
        snap = os.path.join(self.workdir, "snap_kill")
        first_report, _ = run_pipeline(
            self.sample_dir, config_path, None, snap,
            os.path.join(self.workdir, "r1.json"),
        )
        self.assertTrue(first_report["pending_queue"])
        # Simulate a forced kill: leave an untracked temp directory behind and
        # confirm recovery only trusts paths listed in the manifest.
        rogue_dir = os.path.join(snap, ".tmp_depth_v999999")
        os.makedirs(rogue_dir, exist_ok=True)
        atomic_write_json(os.path.join(rogue_dir, "rogue.json"), {"trust": False})
        second_config = self._config(tag="kill2", point_budget=100000, thread_count=4)
        second_report, _ = run_pipeline(
            self.sample_dir, second_config, None, snap,
            os.path.join(self.workdir, "r2.json"),
        )
        self.assertEqual(second_report["pending_queue"], [])
        resume = second_report["resume_point"]
        self.assertTrue(resume["committed_groups"])
        self.assertGreater(second_report["point_cloud_size"], 0)

    def test_config_hash_mismatch_rejects_stale_snapshot(self):
        config_path = self._config(tag="h1")
        snap = os.path.join(self.workdir, "snap_hash")
        run_pipeline(self.sample_dir, config_path, None, snap,
                     os.path.join(self.workdir, "rh1.json"))
        stale_config = dict(self.base)
        stale_config["min_parallax_deg"] = 4.0
        stale_path = os.path.join(self.workdir, "stale.json")
        json.dump(stale_config, open(stale_path, "w", encoding="utf-8"))
        from mvrec.errors import PipelineError
        with self.assertRaises(PipelineError) as context:
            run_pipeline(self.sample_dir, stale_path, None, snap,
                         os.path.join(self.workdir, "rh2.json"))
        self.assertEqual(context.exception.code, "E_SNAPSHOT_INVALID")

    def test_sigterm_finishes_inflight_group_then_exits_zero(self):
        config_path = self._config(tag="sig", thread_count=2,
                                   snapshot_commit_interval=1, point_budget=100000)
        snap = os.path.join(self.workdir, "snap_sig")
        import subprocess
        import sys
        report_path = os.path.join(self.workdir, "sig_report.json")
        runner = os.path.join(os.path.dirname(__file__), "..", "scripts", "signal_runner.py")
        proc = subprocess.Popen(
            [sys.executable, os.path.abspath(runner),
             "--sample", self.sample_dir, "--config", config_path,
             "--snapshot", snap, "--report", report_path],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        )
        proc.wait(timeout=60)
        self.assertEqual(proc.returncode, 0)
        report = json.load(open(report_path, encoding="utf-8"))
        self.assertTrue(report["shutdown_observed"])
        manifest = json.load(open(os.path.join(snap, "manifest.json"), encoding="utf-8"))
        committed = [key for key, value in manifest["group_progress"].items()
                     if value["status"] == "committed"]
        self.assertGreater(len(committed), 0)
        self.assertNotIn(".tmp_", " ".join(os.listdir(snap)))


if __name__ == "__main__":
    unittest.main()
