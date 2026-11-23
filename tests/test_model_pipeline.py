from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from ctr_cvr_pipeline.cache import ArtifactCache
from ctr_cvr_pipeline.commands import apply_command
from ctr_cvr_pipeline.config import feature_config_from_dict, training_config_from_dict
from ctr_cvr_pipeline.fixed import fraction_scale
from ctr_cvr_pipeline.fusion import compare_rankings
from ctr_cvr_pipeline.pipeline import analyze_state, report_core
from ctr_cvr_pipeline.visibility import evaluate_cvr_visibility

from tests.helpers import build_state, feature_config_dict, make_raw, training_config_dict


def _analyze(state, config, training, cache, *, workers=None):
    return analyze_state(
        state,
        config,
        training,
        cache,
        mode="test",
        command_index=0,
        command_type="test",
        workers=workers,
    )


class ModelPipelineTests(unittest.TestCase):
    def test_visibility_pending_visible_truncated_and_missing(self) -> None:
        state, config, training = build_state()
        samples = {sample.sample_id: sample for sample in state.sorted_samples()}
        scale = fraction_scale(8)
        pending = evaluate_cvr_visibility(samples["R0003:I0003_1"], 0, 5000, training, scale)
        visible = evaluate_cvr_visibility(samples["R0003:I0003_1"], 1000, 5000, training, scale)
        truncated = evaluate_cvr_visibility(samples["R0002:I0002_0"], 20000, 1000, training, scale)
        self.assertEqual(pending.reason, "LABEL_PENDING")
        self.assertEqual(pending.effective_target, 0)
        self.assertEqual(visible.reason, "VISIBLE")
        self.assertEqual(visible.effective_target, 1)
        self.assertEqual(truncated.reason, "DELAY_EXCEEDS_CUTOFF")
        self.assertEqual(truncated.effective_target, 0)

    def test_training_is_bitwise_repeatable(self) -> None:
        state, config, training = build_state()
        with tempfile.TemporaryDirectory() as left_temp, tempfile.TemporaryDirectory() as right_temp:
            left = _analyze(state, config, training, ArtifactCache(left_temp), workers=2)
            right = _analyze(state, config, training, ArtifactCache(right_temp), workers=1)
        self.assertEqual(left["deterministic_core_digest"], right["deterministic_core_digest"])
        self.assertEqual(left["training"]["loss_trace"], right["training"]["loss_trace"])
        self.assertEqual(
            left["training"]["final_fixed_parameters"], right["training"]["final_fixed_parameters"]
        )
        self.assertGreater(left["training"]["arithmetic_trace_count"], 0)
        self.assertEqual(
            left["training"]["parameter_fingerprint"],
            right["training"]["parameter_fingerprint"],
        )

    def test_visibility_change_invalidates_training_and_prevents_leakage(self) -> None:
        state, config, training = build_state()
        with tempfile.TemporaryDirectory() as temp:
            cache = ArtifactCache(temp)
            early = _analyze(state, config, training, cache)
            early_shared = early["cache"]["stage_keys"]["shared_hidden"]
            state.current_event_time_ms = 300
            no_visibility_change = _analyze(state, config, training, cache)
            self.assertEqual(
                no_visibility_change["cache"]["stage_keys"]["shared_hidden"], early_shared
            )
            shared_entries = [
                item
                for item in no_visibility_change["cache"]["ledger"]
                if item["stage"] == "shared_hidden"
            ]
            self.assertEqual(shared_entries[0]["status"], "REUSED")

            state.current_event_time_ms = 1000
            visible = _analyze(state, config, training, cache)
        self.assertNotEqual(visible["cache"]["stage_keys"]["shared_hidden"], early_shared)

        def cvr_targets(report, sample_id):
            values = []
            for batch in report["training"]["loss_trace"]:
                for detail in batch["cvr_details"]:
                    if detail["sample"] == sample_id:
                        values.append((detail["target"], detail["visibility_reason"]))
            return values

        self.assertTrue(all(target == 0 for target, _ in cvr_targets(early, "R0003:I0003_1")))
        self.assertTrue(all(target == 1 for target, _ in cvr_targets(visible, "R0003:I0003_1")))
        self.assertNotEqual(
            early["training"]["parameter_fingerprint"],
            visible["training"]["parameter_fingerprint"],
        )

    def test_label_and_edge_invalidation_is_scoped(self) -> None:
        state, config, training = build_state()
        with tempfile.TemporaryDirectory() as temp:
            cache = ArtifactCache(temp)
            _analyze(state, config, training, cache)
            apply_command(
                state,
                {
                    "type": "correct_label",
                    "request_id": "R0001",
                    "item_id": "I0001_0",
                    "click": 0,
                    "conversion": 0,
                    "conversion_delay_ms": 1000,
                },
                0,
                config,
            )
            label_report = _analyze(state, config, training, cache)
            label_quant = [
                item for item in label_report["cache"]["ledger"] if item["stage"] == "feature_quantization"
            ]
            self.assertTrue(all(item["status"] == "REUSED" for item in label_quant))
            self.assertEqual(
                [item["status"] for item in label_report["cache"]["ledger"] if item["stage"] == "shared_hidden"],
                ["COMPUTED"],
            )

            apply_command(
                state,
                {
                    "type": "modify_bin_edges",
                    "feature": "price",
                    "edges": [0, 20, 45, 200],
                },
                1,
                config,
            )
            edge_report = _analyze(state, config, training, cache)
        price_entries = [
            item
            for item in edge_report["cache"]["ledger"]
            if item["stage"] == "feature_quantization" and item["features"] == ["price"]
        ]
        unrelated_entries = [
            item
            for item in edge_report["cache"]["ledger"]
            if item["stage"] == "feature_quantization" and item["features"] != ["price"]
        ]
        self.assertTrue(price_entries)
        self.assertTrue(all(item["status"] == "COMPUTED" for item in price_entries))
        self.assertTrue(all(item["status"] == "REUSED" for item in unrelated_entries))

    def test_report_contains_five_level_errors_and_budget_pending(self) -> None:
        training_raw = dict(training_config_dict())
        training_raw["global_train_sample_budget"] = 3
        training = training_config_from_dict(training_raw)
        raws = [
            make_raw(1, 0, click=1, conversion=1, delay=0, category=0, brand=0),
            make_raw(1, 1, click=0, conversion=0, delay=4000),
            make_raw(2, 0, click=1, conversion=1, delay=2000),
            make_raw(2, 1, click=1, conversion=0, delay=0),
            make_raw(3, 0, click=0, conversion=1, delay=9000),
            make_raw(3, 1, click=1, conversion=1, delay=1000),
        ]
        state, config, training = build_state(raws, training_config=training)
        with tempfile.TemporaryDirectory() as temp:
            report = _analyze(state, config, training, ArtifactCache(temp))
        self.assertEqual(len(report["training"]["selected_samples"]), 3)
        self.assertEqual(len(report["training"]["pending_samples"]), 3)
        self.assertEqual(
            sum(1 for failure in report["failures"] if failure["code"] == "E_TRAIN_BUDGET_PENDING"),
            3,
        )
        for point in report["pointwise"]:
            self.assertEqual(
                set(point["stage_errors"]),
                {"feature_quantization", "shared_hidden", "ctr_tower", "cvr_tower", "fusion"},
            )
            self.assertIn("first_exceeded", point)
            for stage in point["stage_errors"].values():
                self.assertIn(stage["comparison_branch"], {"RELATIVE", "ABSOLUTE_ZERO_REFERENCE"})
        zero_branches = [
            element
            for point in report["pointwise"]
            for element in point["stage_errors"]["feature_quantization"]["elements"]
            if element["comparison_branch"] == "ABSOLUTE_ZERO_REFERENCE"
        ]
        self.assertTrue(zero_branches)
        for stage_summary in report["stage_summary"].values():
            self.assertIn("exceeded_count", stage_summary)
            self.assertIn("max_deviation", stage_summary)

    def test_incremental_full_and_worker_consistency(self) -> None:
        commands = [
            {
                "type": "append_batch",
                "batch_id": "b0001",
                "event_time_ms": 1000,
                "samples": [make_raw(4, 0, click=1, conversion=1, delay=0), make_raw(4, 1, click=0)],
            },
            {
                "type": "correct_label",
                "request_id": "R0002",
                "item_id": "I0002_0",
                "click": 0,
                "conversion": 0,
                "conversion_delay_ms": 0,
                "event_time_ms": 1000,
            },
            {
                "type": "modify_bin_edges",
                "feature": "price",
                "edges": [0, 20, 50, 200],
                "event_time_ms": 1000,
            },
            {
                "type": "switch_quantization",
                "tier": "Q12",
                "fractional_bits": 6,
                "event_time_ms": 1000,
            },
        ]

        def run_incremental(root: Path, workers: int):
            state, config, training = build_state()
            cache = ArtifactCache(root / "cache")
            reports = []
            for index, command in enumerate(commands):
                apply_command(state, command, index, config)
                reports.append(_analyze(state, config, training, cache, workers=workers))
            return reports

        def run_full(root: Path):
            reports = []
            for step in range(len(commands)):
                state, config, training = build_state()
                for index in range(step + 1):
                    apply_command(state, commands[index], index, config)
                reports.append(
                    _analyze(
                        state,
                        config,
                        training,
                        ArtifactCache(root / f"cache_{step}"),
                    )
                )
            return reports

        with tempfile.TemporaryDirectory() as temp:
            base = Path(temp)
            incremental = run_incremental(base / "incremental", 2)
            full = run_full(base / "full")
            serial = run_incremental(base / "serial", 1)
            concurrent = run_incremental(base / "concurrent", 2)
        for index in range(len(commands)):
            self.assertEqual(report_core(incremental[index]), report_core(full[index]))
            self.assertEqual(
                incremental[index]["deterministic_core_digest"],
                serial[index]["deterministic_core_digest"],
            )
            self.assertEqual(
                serial[index]["deterministic_core_digest"],
                concurrent[index]["deterministic_core_digest"],
            )
        self.assertGreater(incremental[-1]["cache"]["entry_count"], 0)

    def test_quantization_flip_attribution(self) -> None:
        def row(item, fusion, ctr, cvr):
            return {
                "request_id": "R1",
                "item_id": item,
                "fusion_score": fusion,
                "ctr_score": ctr,
                "cvr_score": cvr,
            }

        before = [row("A", 0.80, 0.90, 0.70), row("B", 0.70, 0.60, 0.80)]
        after = [row("A", 0.60, 0.80, 0.50), row("B", 0.75, 0.60, 0.95)]
        flips = compare_rankings(
            before,
            after,
            mode="weighted_sum",
            ctr_weight=0.6,
            cvr_weight=0.4,
            comparison={"comparison_kind": "test"},
        )
        self.assertEqual(len(flips), 1)
        flip = flips[0]
        self.assertEqual((flip["rank_before"], flip["rank_after"]), (1, 2))
        self.assertIn(flip["cause"], {"CTR_TOWER", "CVR_TOWER", "FUSION_QUANTIZATION"})
        self.assertIn("ctr_contribution_delta", flip)
        self.assertIn("cvr_contribution_delta", flip)

    def test_product_fusion_mode(self) -> None:
        raw = feature_config_dict()
        raw["fusion"] = {"mode": "product", "ctr_weight": 1.0, "cvr_weight": 1.0}
        config = feature_config_from_dict(raw)
        state, config, training = build_state(feature_config=config)
        with tempfile.TemporaryDirectory() as temp:
            report = _analyze(state, config, training, ArtifactCache(temp))
        self.assertTrue(report["pointwise"])
        for point in report["pointwise"]:
            self.assertAlmostEqual(
                point["reference"]["fusion_score"],
                point["reference"]["ctr_score"] * point["reference"]["cvr_score"],
                places=14,
            )
            self.assertIn("fusion_rounding", point)


if __name__ == "__main__":
    unittest.main()
