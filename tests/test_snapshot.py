from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from search_intent_workbench.canonical import digest
from search_intent_workbench.loader import load_dataset
from search_intent_workbench.pipeline import Workbench
from search_intent_workbench.snapshot import (
    SimulatedCrash,
    SnapshotStore,
    SnapshotVersionExists,
)
from search_intent_workbench.state import WorkbenchState


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "data" / "base"


def build_state_and_run():  # type: ignore[no-untyped-def]
    dataset = load_dataset(
        BASE / "queries.jsonl",
        BASE / "intent_config.json",
        BASE / "normalization_rules.json",
    )
    state = WorkbenchState(
        queries=dataset.queries,
        config=dataset.config,
        rules=dataset.rules,
        failures=list(dataset.failures),
    )
    run = Workbench(state.config, state.rules, max_workers=4).run(
        state.sorted_queries(),
        input_failures=state.failures,
        global_version=1,
    )
    return state, run


class SnapshotTests(unittest.TestCase):
    def test_commit_is_atomic_and_manifest_complete(self) -> None:
        state, run = build_state_and_run()
        with tempfile.TemporaryDirectory() as temp:
            store = SnapshotStore(temp)
            manifest = store.commit(version=1, state=state, run=run)
            self.assertEqual(manifest["status"], "committed")
            self.assertEqual(manifest["global_version"], 1)
            self.assertEqual(manifest["completed_stage"], "rewrite")
            self.assertEqual(manifest["completed_query_cursor"], len(run["queries"]))
            self.assertEqual(manifest["config_hash"], run["metadata"]["config_hash"])
            self.assertEqual(manifest["dictionary_hash"], run["metadata"]["dictionary_hash"])
            snapshot_dir = Path(temp) / "versions" / "v000001"
            self.assertTrue(snapshot_dir.is_dir())
            for descriptor in manifest["stage_files"].values():
                self.assertTrue((snapshot_dir / descriptor["file"]).is_file())
            self.assertEqual(list(Path(temp).glob(".staging-*")), [])
            current = json.loads((Path(temp) / "CURRENT").read_text(encoding="utf-8"))
            self.assertEqual(current["version"], 1)
            with self.assertRaises(SnapshotVersionExists):
                store.commit(version=1, state=state, run=run)

    def test_recovery_resume_and_stale_no_reuse(self) -> None:
        state, run = build_state_and_run()
        config_hash = run["metadata"]["config_hash"]
        with tempfile.TemporaryDirectory() as temp:
            store = SnapshotStore(temp)
            store.commit(version=3, state=state, run=run)
            matching = store.inspect(expected_version=3, expected_config_hash=config_hash)
            self.assertEqual(matching["status"], "RESUME")
            self.assertTrue(matching["reusable"])
            self.assertEqual(matching["completed_query_cursor"], 16)
            self.assertEqual(len(matching["reusable_artifacts"]), 4)

            cache = store.load_stage_cache(matching)
            self.assertEqual(len(cache), 16 * 4)

            hash_mismatch = store.inspect(
                expected_version=3,
                expected_config_hash=digest({"other": config_hash}),
            )
            self.assertEqual(hash_mismatch["status"], "E_SNAPSHOT_STALE")
            self.assertFalse(hash_mismatch["reusable"])
            self.assertEqual(hash_mismatch["reusable_artifacts"], [])
            self.assertEqual(store.load_stage_cache(hash_mismatch), {})

            version_mismatch = store.inspect(
                expected_version=4,
                expected_config_hash=config_hash,
            )
            self.assertEqual(version_mismatch["status"], "E_SNAPSHOT_STALE")
            self.assertFalse(version_mismatch["reusable"])
            self.assertEqual(version_mismatch["reusable_artifacts"], [])

    def test_simulated_crash_after_slots_is_resumable(self) -> None:
        state, run = build_state_and_run()
        config_hash = run["metadata"]["config_hash"]
        with tempfile.TemporaryDirectory() as temp:
            store = SnapshotStore(temp)
            with self.assertRaises(SimulatedCrash):
                store.commit(
                    version=2,
                    state=state,
                    run=run,
                    crash_after_stage="slots",
                )
            inspection = store.inspect(
                expected_version=2, expected_config_hash=config_hash
            )
            self.assertEqual(inspection["status"], "RESUME_STAGING")
            self.assertEqual(inspection["completed_stage"], "slots")
            self.assertEqual(inspection["completed_query_cursor"], 16)
            cache = store.load_stage_cache(inspection)
            self.assertEqual({stage for _, stage in cache}, {"normalization", "slots"})
            self.assertEqual(len(cache), 32)
            stale = store.inspect(
                expected_version=2,
                expected_config_hash=digest({"different": True}),
            )
            self.assertEqual(stale["status"], "E_SNAPSHOT_STALE")
            self.assertEqual(stale["reusable_artifacts"], [])

    def test_real_child_process_hard_exit_then_resume(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            code = f"""
from pathlib import Path
from search_intent_workbench.loader import load_dataset
from search_intent_workbench.pipeline import Workbench
from search_intent_workbench.snapshot import SnapshotStore
from search_intent_workbench.state import WorkbenchState
base = Path(r'{BASE}')
d = load_dataset(base/'queries.jsonl', base/'intent_config.json', base/'normalization_rules.json')
s = WorkbenchState(queries=d.queries, config=d.config, rules=d.rules, failures=list(d.failures))
r = Workbench(s.config, s.rules, max_workers=4).run(s.sorted_queries(), input_failures=s.failures, global_version=1)
Path(r'{root}').mkdir(parents=True, exist_ok=True)
Path(r'{root}/expected_hash.txt').write_text(r['metadata']['config_hash'], encoding='utf-8')
SnapshotStore(r'{root}/snapshots').commit(version=1, state=s, run=r, crash_after_stage='slots', hard_exit=True)
"""
            process = subprocess.run(
                [sys.executable, "-c", code],
                cwd=ROOT,
                text=True,
                capture_output=True,
                timeout=30,
            )
            self.assertEqual(process.returncode, 87, process.stderr)
            config_hash = (root / "expected_hash.txt").read_text(encoding="utf-8")
            inspection = SnapshotStore(root / "snapshots").inspect(
                expected_version=1, expected_config_hash=config_hash
            )
            self.assertEqual(inspection["status"], "RESUME_STAGING")
            self.assertEqual(inspection["completed_stage"], "slots")
            self.assertTrue(inspection["reusable"])


if __name__ == "__main__":
    unittest.main()
