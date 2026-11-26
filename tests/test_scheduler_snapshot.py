from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path

from intentlab.errors import Codes
from intentlab.scheduler import RecomputeTask, execute_tasks, select_tasks
from intentlab.snapshot import SnapshotError, SnapshotStore

from tests.helpers import ROOT, conclusion_signature, prepare_project, read_json, run, write_json


class SchedulerTests(unittest.TestCase):
    def test_pending_jobs_are_scheduled_first_and_constraints_are_all_enforced(self) -> None:
        tasks = [
            RecomputeTask("a", "source-a", 10, True, True, pending=False, priority=0),
            RecomputeTask("b", "source-b", 20, True, True, pending=True, priority=5),
            RecomputeTask("c", "source-a", 5, True, True, pending=False, priority=0),
        ]
        plan = select_tasks(
            tasks,
            match_budget_points=25,
            source_quotas={"source-a": 1, "source-b": 1},
            batch_query_limit=3,
        )
        self.assertEqual([task.query_id for task in plan.selected], ["b", "c"])
        self.assertEqual(plan.deferred, ({"query_id": "a", "code": Codes.PENDING_BUDGET},))
        self.assertEqual(plan.reserved_points, 25)

    def test_workers_do_not_change_value_or_failure_order(self) -> None:
        tasks = [
            RecomputeTask(f"q{index}", "source-a", 1, True, True, priority=index)
            for index in range(1, 9)
        ]

        def worker(task: RecomputeTask):
            time.sleep((8 - int(task.query_id[1:])) * 0.002)
            if task.query_id == "q7":
                raise RuntimeError("expected failure")
            return {"query_id": task.query_id, "value": int(task.query_id[1:]) * 10}

        serial_values, serial_failures = execute_tasks(tasks, worker, workers=1)
        parallel_values, parallel_failures = execute_tasks(tasks, worker, workers=4)
        self.assertEqual(serial_values, parallel_values)
        self.assertEqual(serial_failures, parallel_failures)
        self.assertEqual([item["query_id"] for item in parallel_failures], ["q7"])

    def test_pipeline_output_is_identical_with_one_or_eight_workers(self) -> None:
        with tempfile.TemporaryDirectory() as one_tmp, tempfile.TemporaryDirectory() as many_tmp:
            one_paths = prepare_project(
                one_tmp,
                include_edits=False,
                mutate_config=lambda data: data["scheduler"].__setitem__("workers", 1),
            )
            many_paths = prepare_project(
                many_tmp,
                include_edits=False,
                mutate_config=lambda data: data["scheduler"].__setitem__("workers", 8),
            )
            one = run(one_paths)
            many = run(many_paths)
            self.assertEqual(conclusion_signature(one.report), conclusion_signature(many.report))

    def test_budget_exhaustion_persists_pending_and_next_batch_prioritizes_it(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            paths = prepare_project(
                tmp,
                include_edits=False,
                mutate_config=lambda data: data["scheduler"].update(
                    {"match_budget_points": 0, "batch_query_limit": 5}
                ),
            )
            first = run(paths)
            self.assertEqual(first.report["summary"]["queries_processed"], 0)
            self.assertEqual(first.report["summary"]["queries_pending"], 24)
            self.assertEqual(first.report["summary"]["business_code_counts"][Codes.PENDING_BUDGET], 24)
            first_pending = {item["query_id"] for item in first.report["pending"]}

            queries_doc = read_json(paths.queries_path)
            new_query = json.loads(json.dumps(next(item for item in queries_doc["queries"] if item["query_id"] == "q001")))
            new_query["query_id"] = "q000"
            new_query["session_id"] = "s00"
            new_query["turn"] = 0
            queries_doc["queries"].append(new_query)
            write_json(paths.queries_path, queries_doc)
            config = read_json(paths.config_path)
            config["scheduler"]["match_budget_points"] = 100_000
            write_json(paths.config_path, config)
            second = run(paths)
            processed = {item["query_id"] for item in second.report["queries"]}
            self.assertEqual(len(processed), 5)
            self.assertTrue(processed.issubset(first_pending))
            self.assertNotIn("q000", processed)
            self.assertEqual(processed, {"q001", "q004", "q007", "q010", "q013"})
            self.assertEqual(second.report["summary"]["business_code_counts"][Codes.PENDING_BATCH_LIMIT], 20)


class SnapshotRecoveryTests(unittest.TestCase):
    def _cli(self, paths) -> list[str]:
        return [
            sys.executable,
            "-m",
            "intentlab",
            "run",
            "--config",
            str(paths.config_path),
            "--vocabulary",
            str(paths.vocabulary_path),
            "--queries",
            str(paths.queries_path),
            "--rules",
            str(paths.rules_path),
            "--edits",
            "none",
            "--snapshots",
            str(paths.snapshot_root),
            "--reports",
            str(paths.report_root),
        ]

    def test_sigkill_during_staging_never_replaces_committed_generation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            paths = prepare_project(tmp, include_edits=False)
            baseline = run(paths)
            baseline_generation = baseline.report["snapshot"]["generation"]
            current_path = Path(paths.snapshot_root) / "current.json"
            current_before = current_path.read_bytes()
            env = os.environ.copy()
            env["PYTHONPATH"] = str(ROOT)

            for stage in ("rules", "score"):
                crash_env = dict(env)
                crash_env["INTENTLAB_CRASH_AFTER_STAGE"] = stage
                completed = subprocess.run(
                    self._cli(paths),
                    cwd=ROOT,
                    env=crash_env,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    timeout=60,
                    check=False,
                )
                self.assertNotEqual(completed.returncode, 0)
                self.assertEqual(current_path.read_bytes(), current_before)
                self.assertEqual(json.loads(current_before)["generation"], baseline_generation)

            recovered = run(paths)
            self.assertEqual(recovered.report["recovery"]["committed_generation"], baseline_generation)
            self.assertTrue(recovered.report["recovery"]["resumed_from_committed_snapshot"])
            ignored = recovered.report["recovery"]["ignored_staging"]
            self.assertEqual(len(ignored), 2)
            self.assertEqual(
                [item["completed_stages"] for item in ignored],
                [["rules"], ["rules", "match", "score"]],
            )
            self.assertNotEqual(recovered.report["snapshot"]["generation"], baseline_generation)
            self.assertEqual(recovered.report["summary"]["match_reuse_hits"], 24)
            self.assertEqual(recovered.report["summary"]["score_reuse_hits"], 24)
            self.assertTrue((Path(paths.snapshot_root) / "generations" / baseline_generation).is_dir())

    def test_tampered_committed_artifact_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            paths = prepare_project(tmp, include_edits=False)
            result = run(paths)
            generation = result.report["snapshot"]["generation"]
            artifact = (
                Path(paths.snapshot_root)
                / "generations"
                / generation
                / "matches"
                / "q001.json"
            )
            artifact.write_text("{}", encoding="utf-8")
            with self.assertRaises(SnapshotError) as raised:
                SnapshotStore(paths.snapshot_root).load_current()
            self.assertEqual(raised.exception.code, Codes.SNAPSHOT_INTEGRITY)


if __name__ == "__main__":
    unittest.main()
