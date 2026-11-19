import dataclasses
import json
import unittest
from pathlib import Path

from tests._helpers import load_test_config, make_workspace
from workbench.errors import E_CONFIG_MISMATCH, ConfigMismatchError
from workbench.pipeline import InterruptFlag, run_pipeline
from workbench.report import build_report, render_strip, write_run_artifacts
from workbench.store import SnapshotStore


class SnapshotResumeTests(unittest.TestCase):
    def setUp(self):
        self.workspace = make_workspace()
        self.addCleanup(self.workspace.cleanup)
        self.config = load_test_config(self.workspace)

    def test_resume_reuses_and_persistent_change_rejected(self):
        first = run_pipeline(self.config, reset=True)
        second = run_pipeline(self.config)
        self.assertEqual(second["final_version"], first["final_version"])
        self.assertTrue(all(second["reuse_counts"][stage] > 0 for stage in second["reuse_counts"]))

        store = SnapshotStore(self.config.snapshots_dir)
        manifest = store.load_manifest()
        self.assertEqual(manifest["config_fingerprint"], self.config.persistent_fingerprint())
        self.assertIn("global_version", manifest)
        self.assertIn("completed_samples", manifest)

        changed = dataclasses.replace(
            self.config,
            mask_threshold=self.config.mask_threshold + 10,
            raw={**self.config.raw, "mask_threshold": self.config.mask_threshold + 10},
        )
        with self.assertRaises(ConfigMismatchError) as ctx:
            run_pipeline(changed)
        self.assertEqual(ctx.exception.code, E_CONFIG_MISMATCH)

    def test_runtime_fields_do_not_break_resume(self):
        run_pipeline(self.config, reset=True)
        # active_tier and workers are runtime-only resume keys
        shifted = dataclasses.replace(
            self.config,
            active_tier="medium",
            workers_override=2,
            raw={**self.config.raw, "active_tier": "medium", "workers_override": 2},
        )
        report = run_pipeline(shifted)
        self.assertEqual(len(report["completed_samples"]), 12)

    def test_manifest_version_bumps_only_on_change(self):
        run_pipeline(self.config, reset=True)
        store = SnapshotStore(self.config.snapshots_dir)
        version_after_build = store.load_manifest()["global_version"]
        run_pipeline(self.config)
        self.assertEqual(store.load_manifest()["global_version"], version_after_build)


class SignalHandlingTests(unittest.TestCase):
    def setUp(self):
        self.workspace = make_workspace()
        self.addCleanup(self.workspace.cleanup)
        self.config = load_test_config(self.workspace)

    def test_interrupt_after_first_sample_commits_progress(self):
        interrupt = InterruptFlag()
        # Trigger the cooperative flag once the first sample reaches a stage
        # boundary, emulating SIGTERM arriving mid-run.
        from workbench import pipeline as pipeline_module

        real_process = pipeline_module.process_sample
        calls = {"n": 0}

        def wrapped(config, tier, name, store, decision, manifest=None, flag=None):
            calls["n"] += 1
            if calls["n"] == 1:
                flag.trigger("SIGTERM")
            return real_process(config, tier, name, store, decision, manifest, flag)

        pipeline_module.process_sample = wrapped
        try:
            report = run_pipeline(self.config, run_mode="serial", interrupt=interrupt)
        finally:
            pipeline_module.process_sample = real_process

        self.assertTrue(report["interrupted"])
        self.assertEqual(report["samples"][0]["status"], "interrupted")
        store = SnapshotStore(self.config.snapshots_dir)
        manifest = store.load_manifest()
        # No partial sample entry committed; progress bookkeeping stays consistent.
        self.assertEqual(manifest["completed_samples"], [])
        self.assertIn("SIGTERM", [r.get("signal") for r in manifest["runs"]])


class ReportArtifactTests(unittest.TestCase):
    def setUp(self):
        self.workspace = make_workspace()
        self.addCleanup(self.workspace.cleanup)
        self.config = load_test_config(self.workspace)

    def test_report_and_visualization_artifacts(self):
        report = run_pipeline(self.config, reset=True)
        doc = build_report(report)
        self.assertEqual(doc["schema"], "workbench-report-v1")
        self.assertIn("strategy_decision", doc)
        self.assertIn("reuse_audit", doc)
        self.assertIn("reproducibility", doc)
        self.assertEqual(doc["failures"]["count"], 0)

        import numpy as np

        strip = render_strip(
            [np.zeros((8, 8, 3), dtype=np.uint8), np.ones((8, 8, 3), dtype=np.uint8)],
            ["a", "b"],
        )
        self.assertEqual(strip.ndim, 3)

        out_dir = Path(self.workspace.name) / "outputs" / "run"
        artifacts = write_run_artifacts(
            out_dir, doc, strip, {"sample_0000": b"\x89PNG\r\n\x1a\n"}
        )
        for key in ("report", "dashboard", "strip", "html"):
            self.assertTrue(Path(artifacts[key]).exists())
        saved = json.loads(Path(artifacts["report"]).read_text())
        self.assertEqual(saved["run_id"], doc["run_id"])


if __name__ == "__main__":
    unittest.main()
