from __future__ import annotations

import json
import sys
import tempfile
import threading
import unittest
from dataclasses import replace
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from recall_workbench.diagnostics import STAGE_ORDER, compare_precisions
from recall_workbench.errors import ConfigMismatch
from recall_workbench.evaluation import evaluate_user, summarize_evaluation
from recall_workbench.fixed import FixedFormat, round_ratio
from recall_workbench.fusion import run_fusion_for_user
from recall_workbench.models import Dataset
from recall_workbench.recall import RecallContext, run_recall_for_user
from recall_workbench.report import build_report
from recall_workbench.runner import ExperimentRunner
from recall_workbench.samplegen import DEFAULT_SEED, generate_samples
from recall_workbench.snapshot import SnapshotStore
from recall_workbench.validation import load_config, load_dataset


def subset_dataset(dataset: Dataset, user_ids: list[str]) -> Dataset:
    selected = set(user_ids)
    return Dataset(
        interactions=[record for record in dataset.interactions if record.user_id in selected],
        items=dict(dataset.items),
        truth={user_id: dataset.truth[user_id] for user_id in user_ids},
        failures=list(dataset.failures),
        file_hashes=dict(dataset.file_hashes),
    )


class StopAfterFirstRecallCommit(SnapshotStore):
    def __init__(self, *args, stop_event: threading.Event, **kwargs):
        self.stop_event = stop_event
        super().__init__(*args, **kwargs)

    def commit_stage(self, stage, payload, completed_users, complete):
        artifact = super().commit_stage(stage, payload, completed_users, complete)
        if stage == "recall" and not complete and len(completed_users) > 0:
            self.stop_event.set()
        return artifact


class WorkbenchTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.temp_context = tempfile.TemporaryDirectory(prefix="recall-workbench-tests-")
        cls.temp_root = Path(cls.temp_context.name)
        cls.data_dir = cls.temp_root / "data"
        cls.sample_manifest = generate_samples(cls.data_dir, seed=DEFAULT_SEED)
        cls.config, config_failures = load_config(cls.data_dir / "config.json")
        assert cls.config is not None, [failure.to_dict() for failure in config_failures]
        cls.dataset = load_dataset(cls.data_dir, cls.config)
        cls.context = RecallContext.build(cls.dataset.items, cls.dataset.histories, cls.config)

    @classmethod
    def tearDownClass(cls) -> None:
        cls.temp_context.cleanup()

    def test_sample_is_deterministic_and_invalid_rows_are_retained(self) -> None:
        second_dir = self.temp_root / "data-second"
        second_manifest = generate_samples(second_dir, seed=DEFAULT_SEED)
        self.assertEqual(self.sample_manifest["deterministic_digest"], second_manifest["deterministic_digest"])
        self.assertEqual(self.sample_manifest["files"], second_manifest["files"])
        self.assertEqual(len(self.dataset.interactions), self.sample_manifest["counts"]["interactions_valid"])
        self.assertEqual(len(self.dataset.items), self.sample_manifest["counts"]["catalog_valid"])
        self.assertEqual(len(self.dataset.user_ids), 48)
        self.assertEqual(len(self.dataset.failures), 11)
        self.assertEqual({failure.code for failure in self.dataset.failures}, {"E_RECORD_INVALID"})
        keys = [failure.stable_key for failure in self.dataset.failures]
        self.assertEqual(keys, sorted(keys))

    def test_invalid_configuration_uses_business_code(self) -> None:
        raw = json.loads((self.data_dir / "config.json").read_text(encoding="utf-8"))
        raw["channels"]["lexical"]["candidate_limit"] = 0
        raw["channels"]["collaborative"]["weight"] = -1
        raw["fixed_point"]["reference_bits"] = raw["fixed_point"]["primary_bits"]
        path = self.temp_root / "invalid-config.json"
        path.write_text(json.dumps(raw), encoding="utf-8")
        config, failures = load_config(path)
        self.assertIsNone(config)
        self.assertTrue(failures)
        self.assertEqual({failure.code for failure in failures}, {"E_CONFIG_INVALID"})
        fields = {failure.field for failure in failures}
        self.assertIn("channels.lexical.candidate_limit", fields)
        self.assertIn("channels.collaborative.weight", fields)
        self.assertIn("fixed_point.reference_bits", fields)

    def test_half_even_fixed_arithmetic(self) -> None:
        self.assertEqual(round_ratio(1, 2), 0)
        self.assertEqual(round_ratio(3, 2), 2)
        self.assertEqual(round_ratio(5, 2), 2)
        self.assertEqual(round_ratio(-3, 2), -2)
        fmt = FixedFormat(5)
        self.assertEqual(fmt.quantize(0.015625), 0)
        self.assertEqual(fmt.quantize(0.046875), 2)
        self.assertEqual(fmt.quantize(1.0), 32)

    def test_recall_order_ties_and_cold_start(self) -> None:
        first = run_recall_for_user("u0000", self.context, self.config)
        second = run_recall_for_user("u0000", self.context, self.config)
        self.assertEqual(first, second)
        for channel_name, channel in first["channels"].items():
            candidates = channel["candidates"]
            ordered = sorted(candidates, key=lambda candidate: (-candidate["raw_score"], candidate["item_id"]))
            self.assertEqual(candidates, ordered)
            self.assertEqual([candidate["rank"] for candidate in candidates], list(range(1, len(candidates) + 1)))
            self.assertLessEqual(len(candidates), self.config.channels[channel_name].candidate_limit)
            self.assertEqual(channel["stable_task_key"], f"u0000|{channel_name}")

        cold = run_recall_for_user("u0046", self.context, self.config)
        self.assertEqual(cold["channels"]["lexical"]["candidate_count"], 0)
        self.assertEqual(cold["channels"]["collaborative"]["candidate_count"], 0)
        self.assertGreater(cold["channels"]["popularity"]["candidate_count"], 0)
        fusion = run_fusion_for_user(cold, self.config, self.config.primary_bits)
        self.assertEqual(fusion["zero_channels"], ["lexical", "collaborative"])
        weights = {entry["channel"]: entry["effective_weight_fixed"] for entry in fusion["weight_decomposition"]}
        self.assertEqual(weights["lexical"], 0)
        self.assertEqual(weights["collaborative"], 0)
        self.assertEqual(weights["popularity"], fusion["scale"])
        self.assertEqual(
            [entry["channel"] for entry in fusion["candidates"][0]["channel_contributions"]],
            ["lexical", "collaborative", "popularity"],
        )

    def test_dual_precision_error_attribution_and_evaluation_change(self) -> None:
        rank_change_causes: set[str] = set()
        primary_evaluations = []
        reference_evaluations = []
        for user_id in self.dataset.user_ids:
            recall = run_recall_for_user(user_id, self.context, self.config)
            primary = run_fusion_for_user(recall, self.config, self.config.primary_bits)
            reference = run_fusion_for_user(recall, self.config, self.config.reference_bits)
            comparison = compare_precisions(primary, reference, self.config)
            for candidate in comparison["candidates"]:
                self.assertEqual(list(candidate["stage_differences"]), list(STAGE_ORDER))
                if candidate["first_exceeds_tolerance"] is not None:
                    self.assertIn(candidate["first_exceeds_tolerance"], STAGE_ORDER)
            rank_change_causes.update(change["cause"] for change in comparison["rank_changes"])
            ordered = sorted(
                primary["candidates"],
                key=lambda candidate: (-candidate["final_fixed"], tuple(candidate["rank_vector"]), candidate["item_id"]),
            )
            self.assertEqual(primary["final_order"], [candidate["item_id"] for candidate in ordered])
            primary_evaluations.append(evaluate_user(primary, self.dataset, self.config))
            reference_evaluations.append(evaluate_user(reference, self.dataset, self.config))
        self.assertEqual(rank_change_causes, {"normalization_residual", "feature_rounding_accumulation"})
        primary_summary = summarize_evaluation(primary_evaluations, self.config, self.config.primary_bits)
        reference_summary = summarize_evaluation(reference_evaluations, self.config, self.config.reference_bits)
        self.assertNotEqual(primary_summary, reference_summary)
        self.assertLess(primary_summary["mean_hit_rate"], reference_summary["mean_hit_rate"])

    def _run_case(self, dataset: Dataset, config, run_dir: Path, *, serial: bool):
        snapshot = SnapshotStore(run_dir, config, 1)
        runner = ExperimentRunner(dataset, config, snapshot, serial=serial)
        outcome = runner.run()
        self.assertEqual(outcome.exit_code, 0)
        self.assertEqual(outcome.status, "completed")
        return snapshot, outcome

    def test_concurrent_and_serial_payloads_are_identical(self) -> None:
        users = ["u0000", "u0001", "u0002", "u0003", "u0045", "u0046", "u0047"]
        dataset = subset_dataset(self.dataset, users)
        config = replace(self.config, max_workers=4, snapshot_commit_interval=2, top_k=5)
        serial_dir = self.temp_root / "serial-run"
        concurrent_dir = self.temp_root / "concurrent-run"
        serial_snapshot, serial_outcome = self._run_case(dataset, config, serial_dir, serial=True)
        concurrent_snapshot, concurrent_outcome = self._run_case(dataset, config, concurrent_dir, serial=False)
        for stage in ("recall", "fusion", "evaluation"):
            serial_artifact = serial_snapshot.load_stage(stage)
            concurrent_artifact = concurrent_snapshot.load_stage(stage)
            self.assertEqual(serial_artifact["payload"], concurrent_artifact["payload"])
        serial_report = build_report(dataset, config, serial_snapshot, serial_outcome, execution_mode="serial")
        concurrent_report = build_report(
            dataset, config, concurrent_snapshot, concurrent_outcome, execution_mode="thread_pool"
        )
        self.assertEqual(serial_report["result_digest"], concurrent_report["result_digest"])
        self.assertEqual(serial_report["evaluation"], concurrent_report["evaluation"])

    def test_interrupted_stage_resumes_and_finishes_identically(self) -> None:
        users = ["u0000", "u0001", "u0002", "u0003", "u0045", "u0046", "u0047"]
        dataset = subset_dataset(self.dataset, users)
        config = replace(self.config, max_workers=2, snapshot_commit_interval=2, top_k=5)
        baseline_snapshot, _ = self._run_case(dataset, config, self.temp_root / "resume-baseline", serial=True)

        stop_event = threading.Event()
        interrupted_dir = self.temp_root / "resume-interrupted"
        interrupted_snapshot = StopAfterFirstRecallCommit(
            interrupted_dir, config, 1, stop_event=stop_event
        )
        interrupted_runner = ExperimentRunner(dataset, config, interrupted_snapshot, serial=False, stop_event=stop_event)
        interrupted_outcome = interrupted_runner.run()
        self.assertEqual(interrupted_outcome.exit_code, 0)
        self.assertEqual(interrupted_outcome.status, "interrupted")
        interrupted_manifest = json.loads((interrupted_dir / "manifest.json").read_text(encoding="utf-8"))
        self.assertFalse(interrupted_manifest["stages"]["recall"]["complete"])
        self.assertGreater(len(interrupted_manifest["stages"]["recall"]["completed_users"]), 0)

        resumed_snapshot = SnapshotStore(interrupted_dir, config, 1)
        resumed_runner = ExperimentRunner(dataset, config, resumed_snapshot, serial=True)
        resumed_outcome = resumed_runner.run()
        self.assertEqual(resumed_outcome.status, "completed")
        for stage in ("recall", "fusion", "evaluation"):
            self.assertEqual(
                resumed_snapshot.load_stage(stage)["payload"],
                baseline_snapshot.load_stage(stage)["payload"],
            )

        changed_raw = json.loads((self.data_dir / "config.json").read_text(encoding="utf-8"))
        changed_raw["seed"] += 1
        changed_path = self.temp_root / "changed-config.json"
        changed_path.write_text(json.dumps(changed_raw), encoding="utf-8")
        changed_config, failures = load_config(changed_path)
        self.assertIsNotNone(changed_config)
        self.assertFalse(failures)
        with self.assertRaises(ConfigMismatch) as context:
            SnapshotStore(interrupted_dir, changed_config, 1)
        self.assertEqual(context.exception.code, "E_CONFIG_MISMATCH")
        with self.assertRaises(ConfigMismatch):
            SnapshotStore(interrupted_dir, config, 2)

    def test_atomic_commit_leaves_complete_json_and_no_temp_dirs(self) -> None:
        serial_snapshot, _ = self._run_case(
            subset_dataset(self.dataset, ["u0000", "u0046"]),
            replace(self.config, max_workers=2, snapshot_commit_interval=1, top_k=3),
            self.temp_root / "atomic-run",
            serial=True,
        )
        for stage in ("recall", "fusion", "evaluation"):
            payload = json.loads(serial_snapshot.stage_path(stage).read_text(encoding="utf-8"))
            self.assertEqual(payload["config_hash"], serial_snapshot.config.config_hash)
            self.assertTrue(payload["complete"])
        self.assertEqual(list(serial_snapshot.temp_dir.glob(".atomic-*")), [])


if __name__ == "__main__":
    unittest.main()
