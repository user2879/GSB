from __future__ import annotations

import json
import tempfile
import time
import unittest
from dataclasses import replace
from pathlib import Path

from streamparse.engine import full_decode, semantic_payload
from streamparse.errors import E_RESPONSE_BUDGET_PENDING, E_SNAPSHOT_STALE
from streamparse.loader import load_dataset
from streamparse.snapshot import SnapshotManager

from tests.helpers import make_app, make_config, make_sample


class BudgetSnapshotConcurrencyTests(unittest.TestCase):
    def test_budget_returns_pending_then_background_completes(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            sample_dir = make_sample(root)
            config = make_config(
                sample_dir,
                root,
                response_budget_ms=20,
                test_compute_delay_ms=180,
            )
            from streamparse.app import IncrementalApp

            app = IncrementalApp(sample_dir, workspace_root=root, config=config)
            try:
                started = time.monotonic()
                response = app.execute(
                    {"cmd": "set_strategy", "session_id": "S01", "strategy": "greedy"}
                )
                elapsed_ms = (time.monotonic() - started) * 1000.0
                self.assertEqual(response["status"], "pending")
                self.assertIn(E_RESPONSE_BUDGET_PENDING, response["business_codes"])
                self.assertLess(elapsed_ms, 80.0)
                self.assertEqual(response["global_version"], 1)
                self.assertEqual(response["latest_completed_version"], 0)
                self.assertTrue(response["summary"]["sessions"])
                completed = app.execute({"cmd": "wait"})
                self.assertEqual(completed["status"], "ok")
                self.assertEqual(app.latest_version, 1)
                self.assertEqual(app.latest_report["session_order"], sorted(app.dataset.sessions))
            finally:
                app.close()

    def test_concurrency_does_not_change_serial_semantics(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            sample_dir = make_sample(root)
            dataset, config, injections, load_failures = load_dataset(sample_dir)
            serial = full_decode(
                dataset,
                replace(config, thread_pool_concurrency=1),
                injections=injections,
                initial_failures=load_failures,
            )
            parallel = full_decode(
                dataset,
                replace(config, thread_pool_concurrency=4),
                injections=injections,
                initial_failures=load_failures,
            )
        self.assertEqual(semantic_payload(serial), semantic_payload(parallel))
        self.assertEqual(serial["session_order"], parallel["session_order"])
        self.assertEqual(serial["overlap_ownership"], parallel["overlap_ownership"])
        for left, right in zip(serial["sessions"], parallel["sessions"]):
            self.assertEqual(left["fingerprints"], right["fingerprints"])
        self.assertEqual(serial["failures"], parallel["failures"])

    def test_snapshot_atomic_commit_and_stale_version_rejection(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            version = [1]
            manager = SnapshotManager(root / "snapshots", version_provider=lambda: version[0])
            first = manager.commit_stage(
                stage="unit-test-1",
                expected_version=1,
                config_hash="config-a",
                completed_frame_cursor={"S01": {"completed_frames": 10, "last_global_frame": 9}},
                payload={"kind": "test", "value": 1},
                extra_files={"ownership.json": {"frames": []}},
            )
            self.assertEqual(first["status"], "committed")
            manifest_path = root / "snapshots" / "current" / "manifest.json"
            self.assertTrue(manifest_path.exists())
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            self.assertEqual(manifest["global_version"], 1)
            self.assertEqual(manifest["config_hash"], "config-a")
            self.assertIn("stage.json", manifest["files"])

            version[0] = 2
            stale = manager.commit_stage(
                stage="unit-test-2",
                expected_version=1,
                config_hash="config-b",
                completed_frame_cursor={},
                payload={"kind": "test", "value": 2},
            )
            self.assertEqual(stale["status"], "stale")
            self.assertEqual(stale["business_code"], E_SNAPSHOT_STALE)
            manifest_after = json.loads(manifest_path.read_text(encoding="utf-8"))
            self.assertEqual(manifest_after["global_version"], 1)
            self.assertFalse(any(path.name.startswith(".tmp-snapshot-") for path in (root / "snapshots").iterdir()))

    def test_final_snapshot_contains_cursor_and_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            sample_dir = make_sample(root)
            dataset, config, injections, load_failures = load_dataset(sample_dir)
            manager = SnapshotManager(root / "snapshots", version_provider=lambda: 7)
            report = full_decode(
                dataset,
                config,
                injections=injections,
                version=7,
                initial_failures=load_failures,
                snapshot_manager=manager,
            )
            self.assertIn("snapshot", report)
            manifest = json.loads(
                (root / "snapshots" / "current" / "manifest.json").read_text(encoding="utf-8")
            )
            self.assertEqual(manifest["global_version"], 7)
            self.assertIn("S01", manifest["completed_frame_cursor"])
            self.assertEqual(manifest["stage"], "final")
            self.assertTrue((root / "snapshots" / "current" / "decode.json").exists())


if __name__ == "__main__":
    unittest.main()
