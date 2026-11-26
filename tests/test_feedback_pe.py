from __future__ import annotations

import copy
import json
import shutil
import tempfile
import unittest
from pathlib import Path

from feedback_pe.analysis import build_error_trace
from feedback_pe.attribution import analyze_tier_flips
from feedback_pe.commands import apply_command
from feedback_pe.config import load_config
from feedback_pe.demo import generate_project_files
from feedback_pe.engine import Workspace
from feedback_pe.hashing import canonical_json
from feedback_pe.pipeline import compute_encoding_and_scores
from feedback_pe.reporting import build_report
from feedback_pe.runner import fresh_full_run, run_batch
from feedback_pe.samples import load_samples_jsonl
from feedback_pe.tkview import view_model


ROOT = Path(__file__).resolve().parents[1]


class FixtureMixin:
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.base = Path(self.temp.name)
        generate_project_files(self.base)
        self.config, errors = load_config(self.base / "configs" / "demo_config.json")
        self.assertEqual(errors, [])
        loaded = load_samples_jsonl(self.base / "samples" / "attention_samples.jsonl", 8)
        self.samples = loaded.samples
        self.failures = loaded.failures
        self.assertEqual(len(self.samples), 12)

    def tearDown(self) -> None:
        self.temp.cleanup()

    def unlimited_config(self) -> dict:
        config = copy.deepcopy(self.config)
        config["batch_point_budget"] = 10**9
        config["sample_point_quota"] = 10**9
        config["workers"] = 1
        return config


class ValidationTests(FixtureMixin, unittest.TestCase):
    def test_business_codes_and_failure_order(self) -> None:
        ids = [item["sample_id"] for item in self.failures]
        self.assertEqual(ids[:2], ["<line:13>", "s001"])
        self.assertGreaterEqual(len(self.failures), 6)
        codes = {code for failure in self.failures for error in failure["errors"]
                 for code in [error["code"]]}
        for expected in (
            "E_SAMPLE_PARSE", "E_SAMPLE_ID_DUPLICATE", "E_QUERY_POSITION",
            "E_KEY_POSITION_DUPLICATE", "E_FEATURE_VECTOR", "E_DIMENSION",
            "E_LENGTH_ORDER",
        ):
            self.assertIn(expected, codes)

    def test_invalid_config_enums(self) -> None:
        from feedback_pe.config import validate_config
        raw = copy.deepcopy(self.config)
        raw["frac_bits"] = 7
        raw["quant_tier"] = "BF16"
        errors = validate_config(raw)
        codes = {item["code"] for item in errors}
        self.assertIn("E_ENUM_FRAC_BITS", codes)
        self.assertIn("E_ENUM_QUANT_TIER", codes)


class DeterminismTests(FixtureMixin, unittest.TestCase):
    def test_repeat_serial_and_thread_are_identical(self) -> None:
        config = self.unlimited_config()
        serial_root = self.base / "serial"
        thread_root = self.base / "thread"
        serial_ws = Workspace(serial_root)
        thread_ws = Workspace(thread_root)
        serial = run_batch(serial_ws, config, self.samples, self.failures, parallel=False)
        threaded = run_batch(thread_ws, config, self.samples, self.failures, parallel=True)
        serial_report = build_report(config, self.samples, self.failures, serial_ws, serial)
        thread_report = build_report(config, self.samples, self.failures, thread_ws, threaded)
        self.assertEqual(
            canonical_json(serial_report["business"]),
            canonical_json(thread_report["business"]),
        )

    def test_full_repeat_is_identical(self) -> None:
        config = self.unlimited_config()
        first = fresh_full_run(self.base / "full1", config, self.samples, self.failures)
        second = fresh_full_run(self.base / "full2", config, self.samples, self.failures)
        self.assertEqual(
            canonical_json(first["report"]["business"]),
            canonical_json(second["report"]["business"]),
        )


class IncrementalTests(FixtureMixin, unittest.TestCase):
    def _incremental_report(self, commands: list[dict]) -> dict:
        config = copy.deepcopy(self.config)
        config["batch_point_budget"] = 10**9
        config["sample_point_quota"] = 10**9
        samples = {item["id"]: copy.deepcopy(item) for item in self.samples}
        ws = Workspace(self.base / "incremental")
        last = None
        for command in commands:
            result = apply_command(config, samples, command)
            self.assertNotIn("status", result.get("error", {}))
            last = run_batch(
                ws, config, list(samples.values()), self.failures,
                batch_budget=10**9, sample_quota=10**9,
            )
        return build_report(config, list(samples.values()), self.failures, ws, last)

    def test_length_change_reuses_position_and_score(self) -> None:
        config = self.unlimited_config()
        ws = Workspace(self.base / "length-cache")
        run_batch(ws, config, self.samples, self.failures)
        samples = {item["id"]: item for item in copy.deepcopy(self.samples)}
        apply_command(config, samples, {
            "op": "set_lengths", "train_length": 20, "target_length": 64,
        })
        second = run_batch(ws, config, list(samples.values()), self.failures)
        reused = {item["stage"] for item in second["reuse"]}
        recomputed = {item["stage"] for item in second["recompute"]}
        self.assertEqual(reused, {"position", "score"})
        self.assertEqual(recomputed, {"stats", "error"})

    def test_base_freq_change_is_scheme_scoped(self) -> None:
        config = self.unlimited_config()
        ws = Workspace(self.base / "base-cache")
        run_batch(ws, config, self.samples, self.failures)
        samples = {item["id"]: item for item in copy.deepcopy(self.samples)}
        apply_command(config, samples, {
            "op": "set_base_freq", "scheme": "sinusoidal", "base_freq": 4321.0,
        })
        second = run_batch(ws, config, list(samples.values()), self.failures)
        sinusoidal = [x for x in second["recompute"] if x["scheme"] == "sinusoidal"]
        other_reuse = [x for x in second["reuse"] if x["scheme"] != "sinusoidal"]
        self.assertEqual({x["stage"] for x in sinusoidal},
                         {"position", "score", "stats", "error"})
        self.assertTrue(other_reuse)

    def test_command_sequences_match_blank_full_rebuild(self) -> None:
        replacement = copy.deepcopy(next(x for x in self.samples if x["id"] == "s003"))
        replacement["query_vector"][0] += 0.002
        extra = {
            "name": "extra", "kind": "linear_ramp", "base_freq": 777.0,
            "scale": 0.5, "block_size": 2, "enabled": True,
        }
        sequences = [
            [
                {"op": "switch_scheme", "name": "blockwave"},
                {"op": "set_base_freq", "scheme": "sinusoidal", "base_freq": 5000.0},
                {"op": "set_frac_bits", "frac_bits": 12},
                {"op": "set_quant_tier", "quant_tier": "Q16"},
                {"op": "replace_sample", "sample": replacement},
                {"op": "delete_sample", "id": "s010"},
                {"op": "set_lengths", "train_length": 12, "target_length": 80},
            ],
            [
                {"op": "add_scheme", "scheme": extra},
                {"op": "set_scale", "scheme": "extra", "scale": 1.75},
                {"op": "set_block_size", "scheme": "extra", "block_size": 2},
                {"op": "set_quant_tier", "quant_tier": "Q8"},
                {"op": "set_frac_bits", "frac_bits": 16},
                {"op": "switch_scheme", "name": "extra"},
                {"op": "remove_scheme", "name": "linear_ramp"},
                {"op": "delete_sample", "id": "s004"},
                {"op": "replace_sample", "sample": copy.deepcopy(self.samples[4])},
            ],
        ]
        for index, commands in enumerate(sequences):
            with self.subTest(sequence=index):
                incremental = self._incremental_report(commands)
                final_config = incremental["business"]["config"]
                final_samples = incremental["business"]["samples"]
                full = fresh_full_run(
                    self.base / f"full-seq-{index}",
                    copy.deepcopy(final_config),
                    copy.deepcopy(final_samples),
                    self.failures,
                )
                self.assertEqual(
                    canonical_json(incremental["business"]),
                    canonical_json(full["report"]["business"]),
                )


class BudgetTests(FixtureMixin, unittest.TestCase):
    def test_batch_budget_and_next_batch_priority(self) -> None:
        config = copy.deepcopy(self.config)
        config["workers"] = 1
        ws = Workspace(self.base / "budget")
        first = run_batch(ws, config, self.samples, self.failures,
                          batch_budget=104, sample_quota=80)
        self.assertEqual(first["ready"], ["s001", "s002"])
        self.assertEqual(first["pending"][0], "s003")
        self.assertEqual(first["scheduler"]["pending_count"], 10)
        self.assertTrue(all(
            item["code"] == "E_POINT_BUDGET_PENDING"
            for item in first["scheduler"]["pending_entries"]
        ))
        second = run_batch(ws, config, self.samples, self.failures,
                           batch_budget=104, sample_quota=80)
        self.assertEqual(second["ready"], ["s003", "s004"])

    def test_sample_quota_pending_then_recovers(self) -> None:
        config = self.unlimited_config()
        ws = Workspace(self.base / "quota")
        first = run_batch(ws, config, self.samples, self.failures,
                          batch_budget=10**9, sample_quota=80)
        self.assertEqual(first["pending"], ["s012"])
        self.assertEqual(first["scheduler"]["pending_entries"][0]["code"],
                         "E_POINT_BUDGET_PENDING")
        second = run_batch(ws, config, self.samples, self.failures,
                           batch_budget=10**9, sample_quota=128)
        self.assertEqual(second["ready"], ["s012"])
        self.assertEqual(second["pending"], [])


class SnapshotTests(FixtureMixin, unittest.TestCase):
    def test_committed_snapshot_recovery_and_temp_ignored(self) -> None:
        config = self.unlimited_config()
        ws = Workspace(self.base / "snap")
        run_batch(ws, config, self.samples, self.failures)
        version = ws.global_version
        killed = Path(tempfile.mkdtemp(prefix="tx-killed-", dir=ws.root))
        (killed / "partial.json").write_text("{}", encoding="utf-8")
        reloaded = Workspace(ws.root)
        self.assertEqual(reloaded.global_version, version)
        self.assertEqual(reloaded.recovery, "resumed_committed_snapshot")
        self.assertTrue(reloaded.stores["position"])


class AttributionTests(FixtureMixin, unittest.TestCase):
    def test_scheme_parameter_and_tier_changes_are_numeric(self) -> None:
        sample = self.samples[0]
        scheme = copy.deepcopy(self.config["schemes"][0])
        original = compute_encoding_and_scores(sample, scheme, 16, "Q32")

        changed_scheme = copy.deepcopy(scheme)
        changed_scheme["base_freq"] = scheme["base_freq"] / 2.0
        changed_scheme["scale"] = scheme["scale"] + 0.37
        changed = compute_encoding_and_scores(sample, changed_scheme, 16, "Q32")
        self.assertNotEqual(
            original["position"]["query"]["pe_int"],
            changed["position"]["query"]["pe_int"],
        )
        self.assertNotEqual(
            canonical_json([q["residual"] for q in original["position"]["query"]["pe_quantization"]]),
            canonical_json([q["residual"] for q in changed["position"]["query"]["pe_quantization"]]),
        )

        q8 = compute_encoding_and_scores(sample, scheme, 16, "Q8")
        q32 = compute_encoding_and_scores(sample, scheme, 16, "Q32")
        self.assertNotEqual(
            q8["position"]["query"]["pe_int"],
            q32["position"]["query"]["pe_int"],
        )
        self.assertNotEqual(
            canonical_json([q["int_value"] for q in q8["position"]["query"]["pe_quantization"]]),
            canonical_json([q["int_value"] for q in q32["position"]["query"]["pe_quantization"]]),
        )

    def test_zero_reference_branch_and_tier_flips(self) -> None:
        scheme = self.config["schemes"][0]
        bundle = compute_encoding_and_scores(self.samples[0], scheme, 16, "Q32")
        trace = build_error_trace(self.samples[0], scheme, self.config, bundle)
        self.assertGreater(
            trace["stages"]["position"]["absolute_zero_reference_count"], 0,
        )
        flips = []
        for sample in self.samples:
            for scheme in self.config["schemes"]:
                result = analyze_tier_flips(sample, scheme, 16)
                flips.extend(result["flips"])
        self.assertTrue(flips)
        allowed = {
            "position_vector_quantization", "accumulation_rounding",
            "normalization_max_subtraction", "combined_or_unresolved",
        }
        for flip in flips:
            self.assertIn(flip["attributed_source"], allowed)
            for key in (
                "position_vector_quantization", "accumulation_rounding",
                "normalization_max_subtraction",
            ):
                self.assertIn(key, flip["contributions"])

    def test_tk_view_model_matches_report(self) -> None:
        full = fresh_full_run(self.base / "tk-full", self.unlimited_config(),
                              self.samples, self.failures)
        model = view_model(full["report"])
        self.assertEqual(
            len(model["error_rows"]),
            len(self.samples) * 3,
        )
        self.assertEqual(model["code_counts"], full["report"]["business"]["business_code_counts"])
        try:
            from feedback_pe.tkview import DiagnosticView
            view = DiagnosticView(full["report"])
            view.smoke()
        except Exception as exc:  # pragma: no cover - depends on WSL display/packages
            self.skipTest(f"Tk smoke unavailable: {type(exc).__name__}: {exc}")


if __name__ == "__main__":
    unittest.main(verbosity=2)
