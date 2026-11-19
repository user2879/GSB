from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from forensic_workbench.config import ScopedRect
from forensic_workbench.pipeline import execute
from forensic_workbench.runcontrol import inspect_run, record_recovery
from forensic_workbench.state import SnapshotStore
from forensic_workbench.util import atomic_write_json
from tests.support import TemporaryWorkspace, make_workspace_source


class RecoveryTests(unittest.TestCase):
    def test_controlled_exit_writes_manifest_and_close_marker(self):
        with TemporaryWorkspace() as workspace:
            source = make_workspace_source(workspace)
            run_dir = workspace / "runs" / "controlled"
            result = execute(
                workspace,
                source.default_pipeline_config(),
                verify_full=True,
                run_dir=run_dir,
            )
            self.assertTrue((run_dir / "exit_manifest.json").exists())
            self.assertTrue((run_dir / "close.json").exists())
            status = inspect_run(run_dir)
            self.assertTrue(status["clean_exit"])
            self.assertEqual(status["status"], "controlled_exit")
            manifest = __import__("json").loads(
                (run_dir / "exit_manifest.json").read_text(encoding="utf-8")
            )
            self.assertEqual(
                set(manifest["node_hashes"]),
                set(result.graph.nodes),
            )
            self.assertEqual(manifest["conclusion_hash"], result.conclusion_hash)

    def test_forced_termination_recovers_committed_graph_only(self):
        with TemporaryWorkspace() as workspace:
            source = make_workspace_source(workspace)
            config = source.default_pipeline_config()
            execute(workspace, config, verify_full=True)
            store = SnapshotStore(workspace)
            before = store.load()
            self.assertIsNotNone(before)
            crash_root = workspace / "recovery-runs"
            crash_root.mkdir(parents=True, exist_ok=True)
            crash_run_dir = Path(
                tempfile.mkdtemp(prefix="forced-", dir=str(crash_root))
            )
            variant = config.with_regions(
                mask=list(config.watermark_mask.regions)
                + [ScopedRect(3, 3, 6, 6, frame_start=10, frame_end=20)]
            )
            config_path = crash_run_dir / "candidate_config.json"
            atomic_write_json(config_path, variant.to_dict())
            project_root = Path(__file__).resolve().parent.parent
            environment = dict(os.environ)
            environment["PYTHONPATH"] = str(project_root) + os.pathsep + environment.get(
                "PYTHONPATH", ""
            )
            process = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "forensic_workbench",
                    "_crash-child",
                    "--workspace",
                    str(workspace),
                    "--run-dir",
                    str(crash_run_dir),
                    "--config",
                    str(config_path),
                    "--expected-base-version",
                    str(before.version),
                    "--expected-base-hash",
                    before.snapshot_hash,
                ],
                cwd=str(project_root),
                env=environment,
                capture_output=True,
                text=True,
                timeout=120,
            )
            self.assertEqual(process.returncode, 17, process.stderr)
            status = inspect_run(crash_run_dir)
            self.assertFalse(status["clean_exit"])
            self.assertEqual(status["status"], "unclean_termination")
            self.assertFalse((crash_run_dir / "close.json").exists())
            recovery = store.recover_orphans()
            self.assertTrue(recovery["report"]["issues"])
            self.assertTrue(
                all(
                    issue["code"] == "E_SNAPSHOT_VERSION"
                    for issue in recovery["report"]["issues"]
                )
            )
            recovered = SnapshotStore(workspace).load()
            self.assertEqual(recovered.snapshot_hash, before.snapshot_hash)
            self.assertEqual(recovered.graph.node_hashes(), before.graph.node_hashes())
            recovery_record = record_recovery(
                crash_run_dir,
                {
                    "clean_exit": False,
                    "state_recovery": recovery["report"],
                    "run_status": status,
                },
            )
            self.assertTrue(recovery_record.exists())
            post = execute(workspace, config, verify_full=False)
            self.assertEqual(post.audit["recomputed_nodes"], [])
            self.assertEqual(post.conclusion_hash, before.graph.nodes["conclusion"].content_hash)
            self.assertEqual(post.graph.node_hashes(), before.graph.node_hashes())


if __name__ == "__main__":
    unittest.main()
