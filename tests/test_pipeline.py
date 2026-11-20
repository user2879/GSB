from __future__ import annotations

import copy
import tempfile
import unittest
from pathlib import Path

from transfer_pipeline.canonical import materialized_projection
from transfer_pipeline.encoder import run_shared_forward
from transfer_pipeline.errors import PipelineConfigMismatchError, SimulatedCrash
from transfer_pipeline.inputs import InputState, parse_encoder
from transfer_pipeline.pipeline import run_pipeline
from transfer_pipeline.samplegen import generate_sample
from transfer_pipeline.snapshot import SnapshotManager
from transfer_pipeline.workflow import _task_projection, compare_reports


def task_result(report: dict, name: str) -> dict:
    return next(item for item in report["task_results"] if item["task_name"] == name)


class PipelineRegressionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.temporary = tempfile.TemporaryDirectory(prefix="transfer-tests-")
        cls.root = Path(cls.temporary.name)
        cls.sample = cls.root / "sample"
        generate_sample(cls.sample, seed=20261120)
        cls.baseline = run_pipeline(
            cls.sample,
            cls.root / "baseline-workspace",
            mode="full",
            fresh=True,
        ).report

    @classmethod
    def tearDownClass(cls) -> None:
        cls.temporary.cleanup()

    def test_validation_and_injection_codes(self) -> None:
        failures = {item["task_name"]: item for item in self.baseline["failures"]}
        expected = {
            "inj_missing_sample": "E_TASK_INVALID",
            "inj_duplicate_classes": "E_TASK_INVALID",
            "inj_target_not_whitelisted": "E_TASK_INVALID",
            "inj_zero_rank": "E_RANK_INVALID",
            "inj_empty_validation": "E_TASK_INVALID",
            "vision_gamma": "E_TRAINING_EXCEPTION",
        }
        self.assertEqual(
            {name: failures[name]["business_code"] for name in expected},
            expected,
        )
        self.assertEqual(failures["vision_gamma"]["exception_type"], "InjectedTrainingError")
        self.assertEqual(
            failures["inj_empty_validation"]["reason_code"], "E_VALIDATION_EMPTY"
        )
        self.assertTrue(self.baseline["injection_summary"]["all_hit"])
        self.assertEqual(self.baseline["injection_summary"]["unmatched_ids"], [])
        validated = {item["task_name"] for item in self.baseline["validation_metrics"]}
        self.assertNotIn("inj_zero_rank", validated)
        self.assertNotIn("vision_gamma", validated)
        self.assertEqual(validated, {"vision_alpha", "signals_mix", "signals_abg"})

    def test_shared_forward_is_repeatable_and_does_not_mutate_base(self) -> None:
        state = InputState.from_directory(self.sample)
        original = copy.deepcopy(state.encoder)
        encoder = parse_encoder(state.encoder)
        first = run_shared_forward(encoder, state.features, "tanh")
        second = run_shared_forward(encoder, state.features, "tanh")
        self.assertEqual(first["representation_fingerprint"], second["representation_fingerprint"])
        self.assertEqual(first["samples"], second["samples"])
        self.assertEqual(state.encoder, original)
        self.assertTrue(all(not layer.weights.flags.writeable for layer in encoder.layers))

    def test_encoder_invalid_is_isolated_at_load(self) -> None:
        state = InputState.from_directory(self.sample)
        state.encoder["layers"][1]["weights"] = [[1.0, 2.0], [3.0, 4.0]]
        with tempfile.TemporaryDirectory(prefix="encoder-invalid-") as temporary:
            result = run_pipeline(
                state,
                Path(temporary) / "workspace",
                mode="full",
                fresh=True,
            )
        self.assertEqual(len(result.report["failures"]), 1)
        self.assertEqual(result.report["failures"][0]["business_code"], "E_ENCODER_INVALID")
        self.assertEqual(result.report["validation_metrics"], [])
        dimension_state = InputState.from_directory(self.sample)
        dimension_state.tasks["shared_feature_dim"] = 7
        with tempfile.TemporaryDirectory(prefix="dimension-invalid-") as temporary:
            dimension_result = run_pipeline(
                dimension_state,
                Path(temporary) / "workspace",
                mode="full",
                fresh=True,
            )
        self.assertEqual(
            dimension_result.report["failures"][0]["business_code"],
            "E_ENCODER_INVALID",
        )

    def test_selective_reuse_and_base_layer_precision(self) -> None:
        workspace = self.root / "selective-workspace"
        run_pipeline(self.sample, workspace, mode="full", fresh=True)
        state = InputState.from_directory(self.sample)
        state.apply_batch(state.edit_batches[0])
        incremental = run_pipeline(
            state,
            workspace,
            mode="incremental",
            allow_config_migration=False,
        ).report
        reused = {item["task_name"] for item in incremental["lifecycle"]["reused"]}
        invalidated = {
            item["task_name"] for item in incremental["lifecycle"]["invalidated"]
        }
        retrained = {item["task_name"] for item in incremental["lifecycle"]["retrained"]}
        self.assertIn("inj_zero_rank", reused)
        self.assertTrue({"vision_alpha", "vision_gamma", "signals_abg"} <= invalidated)
        self.assertFalse(reused & retrained)
        self.assertFalse(reused & invalidated)
        self.assertIn("signals_mix", incremental["lifecycle"]["removed"])
        self.assertIn(
            "referenced_base_layer_changed:layer_2",
            {item["reason"] for item in incremental["lifecycle"]["invalidated"]},
        )

    def test_adapter_type_and_rank_change_trajectory(self) -> None:
        base_state = InputState.from_directory(self.sample)
        reports = {}
        with tempfile.TemporaryDirectory(prefix="adapter-switch-") as temporary:
            temporary_path = Path(temporary)
            variants = {
                "lr1": {"adapter_type": "low_rank", "rank": 1, "scale": 0.9},
                "lr2": {"adapter_type": "low_rank", "rank": 2, "scale": 0.9},
                "linear": {"adapter_type": "linear_probe", "rank": 2, "scale": 0.9},
            }
            for name, changes in variants.items():
                state = copy.deepcopy(base_state)
                state.config.update(changes)
                reports[name] = run_pipeline(
                    state,
                    temporary_path / f"workspace-{name}",
                    mode="full",
                    fresh=True,
                ).report
        low_rank_1 = task_result(reports["lr1"], "signals_mix")
        low_rank_2 = task_result(reports["lr2"], "signals_mix")
        linear = task_result(reports["linear"], "signals_mix")
        self.assertNotEqual(
            low_rank_1["parameter_fingerprint"], low_rank_2["parameter_fingerprint"]
        )
        self.assertNotEqual(low_rank_1["loss_trajectory"], low_rank_2["loss_trajectory"])
        self.assertEqual(low_rank_2["parameter_shapes"]["right"], [2, 2])
        self.assertIsNone(linear["parameter_shapes"]["right"])
        self.assertNotEqual(low_rank_2["parameter_fingerprint"], linear["parameter_fingerprint"])
        self.assertNotEqual(low_rank_2["loss_trajectory"], linear["loss_trajectory"])

    def test_incremental_batches_match_clean_rebuild(self) -> None:
        state = InputState.from_directory(self.sample)
        workspace = self.root / "sequence-workspace"
        run_pipeline(self.sample, workspace, mode="full", fresh=True)
        for index, batch in enumerate(state.edit_batches, 1):
            state.apply_batch(batch)
            incremental = run_pipeline(
                state,
                workspace,
                mode="incremental",
                allow_config_migration=True,
            ).report
            with tempfile.TemporaryDirectory(prefix=f"rebuild-{index}-") as temporary:
                full = run_pipeline(
                    state,
                    Path(temporary) / "workspace",
                    mode="full",
                    fresh=True,
                ).report
            self.assertEqual(
                materialized_projection(incremental),
                materialized_projection(full),
            )
        idempotent = run_pipeline(state, workspace, mode="incremental").report
        self.assertEqual(idempotent["lifecycle"]["retrained"], [])
        self.assertTrue(idempotent["lifecycle"]["reused"])

    def test_concurrency_does_not_change_task_materialization(self) -> None:
        state_one = InputState.from_directory(self.sample)
        state_five = InputState.from_directory(self.sample)
        state_one.config["max_concurrency"] = 1
        state_five.config["max_concurrency"] = 5
        with tempfile.TemporaryDirectory(prefix="concurrency-test-") as temporary:
            root = Path(temporary)
            one = run_pipeline(
                state_one, root / "one", mode="full", fresh=True
            ).report
            five = run_pipeline(
                state_five, root / "five", mode="full", fresh=True
            ).report
        self.assertEqual(_task_projection(one), _task_projection(five))
        self.assertLessEqual(one["execution"]["observed_max_inflight"], 1)
        self.assertLessEqual(five["execution"]["observed_max_inflight"], 5)

    def test_crash_recovery_uses_committed_stable_prefix(self) -> None:
        workspace = self.root / "crash-workspace"
        with self.assertRaises(SimulatedCrash) as context:
            run_pipeline(
                self.sample,
                workspace,
                mode="incremental",
                crash_after_completed=2,
            )
        crashed_generation = context.exception.details["generation"]
        manager = SnapshotManager(workspace)
        committed = manager.load_current()
        self.assertIsNotNone(committed)
        self.assertEqual(committed.generation, crashed_generation)
        ordered_names = [item["task_name"] for item in self.baseline["task_results"]]
        self.assertEqual(
            [item["task_name"] for item in committed.completed_tasks],
            ordered_names[:2],
        )
        resumed = run_pipeline(self.sample, workspace, mode="incremental").report
        self.assertTrue(resumed["recovery"]["resumed"])
        self.assertEqual(
            resumed["recovery"]["recovered_completed_tasks"], ordered_names[:2]
        )
        self.assertTrue(resumed["recovery"]["shared_forward_reused"])
        self.assertTrue(set(ordered_names[:2]) <= {
            item["task_name"] for item in resumed["lifecycle"]["reused"]
        })
        self.assertEqual(compare_reports(resumed, self.baseline)["first_difference"], None)

    def test_config_and_version_mismatch_refuse_resume(self) -> None:
        workspace = self.root / "mismatch-workspace"
        first = run_pipeline(self.sample, workspace, mode="incremental").report
        pointer_before = (workspace / "CURRENT.json").read_text(encoding="utf-8")
        changed = InputState.from_directory(self.sample)
        changed.config["validation_threshold"] = 0.99
        with self.assertRaises(PipelineConfigMismatchError) as context:
            run_pipeline(changed, workspace, mode="incremental")
        self.assertEqual(context.exception.business_code, "E_CONFIG_MISMATCH")
        self.assertEqual(context.exception.details["reason"], "config_hash_mismatch")
        self.assertEqual(
            (workspace / "CURRENT.json").read_text(encoding="utf-8"), pointer_before
        )
        with self.assertRaises(PipelineConfigMismatchError) as version_context:
            run_pipeline(
                self.sample,
                workspace,
                mode="incremental",
                expected_global_version=first["snapshot"]["global_version"] + 1,
            )
        self.assertEqual(version_context.exception.details["reason"], "global_version_mismatch")


if __name__ == "__main__":
    unittest.main()
