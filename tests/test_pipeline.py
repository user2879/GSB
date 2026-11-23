from __future__ import annotations

import dataclasses
import json
import tempfile
import threading
import unittest
from pathlib import Path

from posdiag.cache import stage_cache_key
from posdiag.models import load_config, load_edit_plan, load_samples
from posdiag.pipeline import run_pipeline
from posdiag.samples import generate_sample_directory
from tests.helpers import make_sample, report_without_metadata, write_custom_inputs


def _load_generated(root: Path):
    generated = generate_sample_directory(root / "samples")
    config = load_config(generated["config"])
    loaded = load_samples(generated["samples"], dimension=config.dimension)
    return generated, config, loaded


def _core_result(report: dict) -> dict:
    keys = (
        "summary",
        "samples",
        "layer_statistics",
        "error_distributions",
        "over_tolerance_samples",
        "boundary_samples",
        "budget_usage",
        "validation",
    )
    return {key: report[key] for key in keys}


class PipelineIntegrationTests(unittest.TestCase):
    def test_all_three_schemes_change_outputs_and_are_repeatable(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _, config, loaded = _load_generated(root)
            first = run_pipeline(
                config,
                loaded.samples,
                root / "state-a",
                schemes_override=("rope_standard", "rope_linear", "rope_ntk"),
            )
            second = run_pipeline(
                config,
                loaded.samples,
                root / "state-b",
                schemes_override=("rope_standard", "rope_linear", "rope_ntk"),
            )
            self.assertEqual(first.report["status"], "complete")
            self.assertEqual(report_without_metadata(first.report), report_without_metadata(second.report))
            self.assertEqual(first.report["summary"]["successful_task_count"], 42)
            sample = first.report["samples"][6]
            digests = {item["scheme"]: item["position_summary"]["digest"] for item in sample["schemes"]}
            self.assertEqual(len(set(digests.values())), 3)
            score_vectors = {
                item["scheme"]: item["attention"]["scores"] for item in sample["schemes"]
            }
            self.assertNotEqual(score_vectors["rope_standard"], score_vectors["rope_linear"])
            self.assertNotEqual(score_vectors["rope_standard"], score_vectors["rope_ntk"])

    def test_concurrency_does_not_change_diagnostic_result_or_budget_ledger(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _, config, loaded = _load_generated(root)
            serial = run_pipeline(
                dataclasses.replace(config, thread_pool_concurrency=1),
                loaded.samples,
                root / "serial",
            )
            parallel = run_pipeline(
                dataclasses.replace(config, thread_pool_concurrency=4),
                loaded.samples,
                root / "parallel",
            )
            self.assertEqual(_core_result(serial.report), _core_result(parallel.report))
            self.assertLessEqual(parallel.report["run_metadata"]["execution"]["observed_max_concurrency"], 4)
            self.assertEqual(
                serial.report["budget_usage"]["deduction_sequence"],
                parallel.report["budget_usage"]["deduction_sequence"],
            )

    def test_sample_replacement_reuses_downstream_cache_for_unaffected_samples(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            generated, config, loaded = _load_generated(root)
            incremental_workspace = root / "incremental"
            run_pipeline(config, loaded.samples, incremental_workspace)
            edited = load_edit_plan(
                config,
                loaded.samples,
                generated["edit_plan_sample_only.json"],
            )
            incremental = run_pipeline(
                edited.config,
                edited.samples,
                incremental_workspace,
                edit_issues=edited.issues,
                edit_plan_fingerprint=edited.plan_fingerprint,
                applied_edit_operations=edited.applied_operations,
            )
            full = run_pipeline(edited.config, edited.samples, root / "full")
            self.assertEqual(report_without_metadata(incremental.report), report_without_metadata(full.report))
            for stage in ("encoding", "attention", "error_trace"):
                self.assertEqual(incremental.report["run_metadata"]["cache"][stage]["computed"], 1)
                self.assertEqual(incremental.report["run_metadata"]["cache"][stage]["reused"], 13)
            self.assertTrue(incremental.report["run_metadata"]["resumed"])
            self.assertFalse(incremental.report["run_metadata"]["stale"])

    def test_edit_sequences_equal_blank_full_rebuild(self) -> None:
        plan_names = (
            "edit_plan_sample_only.json",
            "edit_plan_scheme_only.json",
            "edit_plan_frequency_only.json",
            "edit_plan_precision_only.json",
            "edit_plan.json",
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            generated, config, loaded = _load_generated(root)
            for index, plan_name in enumerate(plan_names):
                with self.subTest(plan=plan_name):
                    incremental_workspace = root / f"incremental-{index}"
                    baseline = run_pipeline(config, loaded.samples, incremental_workspace)
                    self.assertEqual(baseline.report["status"], "complete")
                    edited = load_edit_plan(config, loaded.samples, generated[plan_name])
                    incremental = run_pipeline(
                        edited.config,
                        edited.samples,
                        incremental_workspace,
                        edit_issues=edited.issues,
                        edit_plan_fingerprint=edited.plan_fingerprint,
                        applied_edit_operations=edited.applied_operations,
                    )
                    full = run_pipeline(edited.config, edited.samples, root / f"full-{index}")
                    self.assertEqual(
                        report_without_metadata(incremental.report),
                        report_without_metadata(full.report),
                    )
                    if plan_name == "edit_plan_sample_only.json":
                        self.assertFalse(incremental.report["run_metadata"]["stale"])
                    else:
                        self.assertTrue(incremental.report["run_metadata"]["stale"])
                        self.assertIn(
                            "E_SNAPSHOT_STALE",
                            incremental.report["run_metadata"]["diagnostic_codes"],
                        )

    def test_interrupted_stage_snapshot_resumes_and_matches_clean_run(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _, config, loaded = _load_generated(root)
            workspace = root / "resume"
            stop = threading.Event()

            def stop_after_encoding(stage: str) -> None:
                if stage == "encoding":
                    stop.set()

            interrupted = run_pipeline(
                config,
                loaded.samples,
                workspace,
                stop_event=stop,
                stage_complete_callback=stop_after_encoding,
            )
            self.assertEqual(interrupted.report["status"], "interrupted")
            self.assertEqual(interrupted.report["run_metadata"]["cache"]["encoding"]["computed"], 14)
            self.assertEqual(interrupted.report["run_metadata"]["cache"]["attention"]["computed"], 0)
            resumed = run_pipeline(config, loaded.samples, workspace)
            clean = run_pipeline(config, loaded.samples, root / "clean")
            self.assertEqual(resumed.report["status"], "complete")
            self.assertEqual(resumed.report["run_metadata"]["cache"]["encoding"]["reused"], 14)
            self.assertEqual(report_without_metadata(resumed.report), report_without_metadata(clean.report))
            manifest = json.loads((workspace / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["status"], "complete")
            self.assertEqual(manifest["config_hash"], config.fingerprint())
            self.assertEqual(set(manifest["stages"]), {"loading", "encoding", "attention", "error_trace"})
            self.assertEqual(list((workspace / "tmp").iterdir()), [])

    def test_cache_key_invalidation_scope(self) -> None:
        config = load_config(Path("samples/position_config.json"))
        sample = make_sample("s0001", dimension=config.dimension)
        other = make_sample("s0002", dimension=config.dimension)
        replacement = make_sample("s0001", dimension=config.dimension, query_position=41, key_positions=(1, 20))

        baseline = {stage: stage_cache_key(stage, config, sample, config.scheme) for stage in ("encoding", "attention", "error_trace")}
        replacement_keys = {
            stage: stage_cache_key(stage, config, replacement, config.scheme)
            for stage in ("encoding", "attention", "error_trace")
        }
        untouched_keys = {
            stage: stage_cache_key(stage, config, other, config.scheme)
            for stage in ("encoding", "attention", "error_trace")
        }
        self.assertEqual(
            untouched_keys,
            {stage: stage_cache_key(stage, config, other, config.scheme) for stage in untouched_keys},
        )
        self.assertTrue(all(baseline[stage] != replacement_keys[stage] for stage in baseline))

        scheme_config = dataclasses.replace(config, scheme="rope_ntk")
        scheme_keys = {
            stage: stage_cache_key(stage, scheme_config, sample, "rope_ntk")
            for stage in ("encoding", "attention", "error_trace")
        }
        self.assertTrue(all(baseline[stage] != scheme_keys[stage] for stage in baseline))

        tolerance_config = dataclasses.replace(config, tolerance=config.tolerance * 2.0)
        tolerance_keys = {
            stage: stage_cache_key(stage, tolerance_config, sample, config.scheme)
            for stage in ("encoding", "attention", "error_trace")
        }
        self.assertEqual(baseline["encoding"], tolerance_keys["encoding"])
        self.assertEqual(baseline["attention"], tolerance_keys["attention"])
        self.assertNotEqual(baseline["error_trace"], tolerance_keys["error_trace"])

        precision_config = dataclasses.replace(config, precision="compact")
        precision_keys = {
            stage: stage_cache_key(stage, precision_config, sample, config.scheme)
            for stage in ("encoding", "attention", "error_trace")
        }
        self.assertTrue(all(baseline[stage] != precision_keys[stage] for stage in baseline))

        budget_config = dataclasses.replace(config, batch_point_budget=config.batch_point_budget + 1)
        budget_keys = {
            stage: stage_cache_key(stage, budget_config, sample, config.scheme)
            for stage in ("encoding", "attention", "error_trace")
        }
        self.assertEqual(baseline, budget_keys)

    def test_budget_exhaustion_and_pending_are_reported(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            samples = [
                {
                    "sample_id": "s0001",
                    "query_position": 2,
                    "key_positions": [],
                    "query_vector": [0.1, 0.2, 0.3, 0.4],
                    "key_vectors": [],
                },
                {
                    "sample_id": "s0002",
                    "query_position": 2,
                    "key_positions": [1],
                    "query_vector": [0.1, 0.2, 0.3, 0.4],
                    "key_vectors": [[0.4, 0.3, 0.2, 0.1]],
                },
            ]
            _, _, config, loaded_samples = write_custom_inputs(
                root / "pending",
                {
                    "dimension": 4,
                    "base_frequency": 100.0,
                    "training_context_length": 8,
                    "extrapolation_target_length": 16,
                    "batch_point_budget": 3,
                    "global_mac_budget": 4,
                    "thread_pool_concurrency": 1,
                },
                samples,
            )
            pending = run_pipeline(config, loaded_samples, root / "pending-state")
            self.assertEqual(pending.report["summary"]["pending_task_count"], 1)
            self.assertEqual(pending.report["budget_usage"]["pending_queue"][0]["sample_id"], "s0002")
            self.assertEqual(pending.report["budget_usage"]["pending_queue"][0]["error_code"], "E_POINT_BUDGET_PENDING")
            pending_sample = next(item for item in pending.report["samples"] if item["sample_id"] == "s0002")
            self.assertEqual(pending_sample["schemes"][0]["status"], "budget_pending")

            _, _, quota_config, quota_samples = write_custom_inputs(
                root / "quota",
                {
                    "dimension": 4,
                    "base_frequency": 100.0,
                    "training_context_length": 8,
                    "extrapolation_target_length": 16,
                    "batch_point_budget": 2,
                    "global_mac_budget": 1000,
                    "thread_pool_concurrency": 1,
                },
                samples[1:],
            )
            quota = run_pipeline(quota_config, quota_samples, root / "quota-state")
            self.assertEqual(quota.report["summary"]["budget_exhausted_task_count"], 1)
            exhausted = quota.report["budget_usage"]["budget_exhausted_samples"][0]
            self.assertEqual(exhausted["sample_id"], "s0002")
            self.assertEqual(exhausted["error_code"], "E_SAMPLE_BUDGET_EXHAUSTED")

    def test_invalid_sample_errors_do_not_stop_valid_samples(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config_path = root / "config.json"
            samples_path = root / "samples.jsonl"
            config_path.write_text(
                json.dumps(
                    {
                        "scheme": "rope_standard",
                        "base_frequency": 100.0,
                        "training_context_length": 8,
                        "extrapolation_target_length": 16,
                        "dimension": 4,
                        "scaling_factor": 2.0,
                        "precision": "strict",
                        "batch_point_budget": 16,
                        "global_mac_budget": 10000,
                        "thread_pool_concurrency": 1,
                        "snapshot_commit_interval": 1,
                        "tolerance": 0.35,
                    }
                ),
                encoding="utf-8",
            )
            samples_path.write_text(
                json.dumps(
                    {
                        "sample_id": "s0001",
                        "query_position": 2,
                        "key_positions": [1],
                        "query_vector": [0.1, 0.2, 0.3, 0.4],
                        "key_vectors": [[0.4, 0.3, 0.2, 0.1]],
                    }
                )
                + "\n"
                + json.dumps(
                    {
                        "sample_id": "s0002",
                        "query_position": -1,
                        "key_positions": [],
                        "query_vector": [0.1, 0.2, 0.3, 0.4],
                        "key_vectors": [],
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            config = load_config(config_path)
            loaded = load_samples(samples_path, dimension=config.dimension)
            run = run_pipeline(config, loaded.samples, root / "state", sample_issues=loaded.issues)
            self.assertEqual(run.report["summary"]["valid_sample_count"], 1)
            self.assertEqual(run.report["summary"]["invalid_sample_count"], 1)
            self.assertEqual(run.report["validation"]["sample_errors"][0]["code"], "E_SAMPLE_INVALID")
            self.assertEqual(run.report["samples"][0]["schemes"][0]["status"], "ok")


if __name__ == "__main__":
    unittest.main()
