from __future__ import annotations

import copy
import tempfile
import unittest
from pathlib import Path

from rope_lab.codes import (
    E_BLOCK_INCOMPATIBLE,
    E_DIM_MISMATCH,
    E_SAMPLE_DUPLICATE,
    E_SCHEMA_UNKNOWN,
)
from rope_lab.commands import apply_command
from rope_lab.engine import LabEngine
from rope_lab.generator import (
    generate_commands,
    generate_injections,
    generate_samples,
    default_config,
    make_flip_attribution_cases,
)
from rope_lab.hashing import canonical_json
from rope_lab.injections import run_injections
from rope_lab.snapshot import SnapshotStore
from rope_lab.validation import validate_workspace


class LabTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.config = default_config()
        cls.samples = generate_samples(cls.config)
        cls.commands = generate_commands(cls.config, cls.samples)
        cls.injections = generate_injections()

    def test_generated_workspace_is_valid(self) -> None:
        self.assertEqual(validate_workspace(self.config, self.samples), [])

    def test_required_schema_codes(self) -> None:
        duplicate_samples = copy.deepcopy(self.samples) + [copy.deepcopy(self.samples[0])]
        errors = validate_workspace(self.config, duplicate_samples)
        self.assertIn(E_SAMPLE_DUPLICATE, {error["code"] for error in errors})

        bad_dimension = copy.deepcopy(self.samples)
        bad_dimension[0]["query_vector"].pop()
        errors = validate_workspace(self.config, bad_dimension)
        self.assertIn(E_DIM_MISMATCH, {error["code"] for error in errors})

        bad_block = copy.deepcopy(self.config)
        bad_block["schemes"][0]["block_size"] = 3
        errors = validate_workspace(bad_block, self.samples)
        self.assertIn(E_BLOCK_INCOMPATIBLE, {error["code"] for error in errors})

        bad_scheme_ref = copy.deepcopy(self.samples)
        bad_scheme_ref[0]["schemes"] = ["not_declared"]
        errors = validate_workspace(self.config, bad_scheme_ref)
        self.assertIn(E_SCHEMA_UNKNOWN, {error["code"] for error in errors})

    def test_width_and_quant_level_change_integer_outputs(self) -> None:
        coarse_config = copy.deepcopy(self.config)
        coarse_config["fixed_point"].update({"fraction_bits": 4, "signed_bits": 32, "quant_level": "truncate"})
        fine_config = copy.deepcopy(self.config)
        fine_config["fixed_point"].update({"fraction_bits": 16, "signed_bits": 32, "quant_level": "ties_even"})

        coarse_engine = LabEngine(coarse_config, self.samples[:2])
        fine_engine = LabEngine(fine_config, self.samples[:2])
        coarse_engine.run(parallel=False)
        fine_engine.run(parallel=False)
        coarse_rows = coarse_engine.row_outputs()[0]["payload"]["rows"]
        fine_rows = fine_engine.row_outputs()[0]["payload"]["rows"]
        self.assertNotEqual(
            [row["q_score"] for row in coarse_rows],
            [row["q_score"] for row in fine_rows],
        )
        coarse_rotations = [
            item["payload"]["query"]["q_rotated"]
            for item in coarse_engine.dimension_outputs()
            if ":rotation" in item["stable_key"]
        ]
        fine_rotations = [
            item["payload"]["query"]["q_rotated"]
            for item in fine_engine.dimension_outputs()
            if ":rotation" in item["stable_key"]
        ]
        self.assertTrue(any(left != right for left, right in zip(coarse_rotations, fine_rotations)))

    def test_serial_and_threads_are_field_identical(self) -> None:
        serial_engine = LabEngine(copy.deepcopy(self.config), copy.deepcopy(self.samples))
        parallel_engine = LabEngine(copy.deepcopy(self.config), copy.deepcopy(self.samples))
        serial_engine.run(parallel=False)
        parallel_engine.run(parallel=True)
        self.assertEqual(
            canonical_json(serial_engine.semantic_report()),
            canonical_json(parallel_engine.semantic_report()),
        )

    def test_quantization_gear_alone_changes_integers(self) -> None:
        observed = {}
        for level in ("truncate", "nearest", "ties_even"):
            config = copy.deepcopy(self.config)
            config["fixed_point"]["quant_level"] = level
            engine = LabEngine(config, self.samples[:2])
            engine.run(parallel=False)
            observed[level] = [
                item["payload"]["query"]["q_rotated"]
                for item in engine.dimension_outputs()
                if ":rotation" in item["stable_key"]
            ]
        self.assertNotEqual(observed["truncate"], observed["nearest"])
        self.assertNotEqual(observed["nearest"], observed["ties_even"])

    def test_command_sequence_incremental_matches_blank(self) -> None:
        incremental = LabEngine(copy.deepcopy(self.config), copy.deepcopy(self.samples))
        for command in self.commands:
            manifest = incremental.apply_command(command)
            incremental.run(manifest, parallel=True)
        incremental_semantic = incremental.semantic_report()

        final_config = copy.deepcopy(self.config)
        final_samples = copy.deepcopy(self.samples)
        for command in self.commands:
            final_config, final_samples, _ = apply_command(final_config, final_samples, command)
        blank = LabEngine(final_config, final_samples, global_version=incremental.global_version)
        blank_report = blank.run(parallel=False)
        self.assertEqual(canonical_json(incremental_semantic), canonical_json(blank.semantic_report()))
        self.assertEqual(
            [entry["stable_key"] for entry in incremental.last_report["ledger"]],
            [entry["stable_key"] for entry in blank_report["ledger"]],
        )
        self.assertEqual(
            [(entry["stable_key"], entry["cache_key"]) for entry in incremental.last_report["ledger"]],
            [(entry["stable_key"], entry["cache_key"]) for entry in blank_report["ledger"]],
        )

    def test_train_length_only_invalidates_statistics(self) -> None:
        engine = LabEngine(copy.deepcopy(self.config), copy.deepcopy(self.samples))
        engine.run(parallel=False)
        manifest = engine.apply_command({"command_id": "T", "op": "set_train_length", "train_length": 11})
        report = engine.run(manifest, parallel=False)
        for entry in report["ledger"]:
            if any(token in entry["stable_key"] for token in (":position", ":rotation", ":dot_term", ":score", ":normalization")):
                self.assertEqual(entry["cache_status"], "reused", entry["stable_key"])
            if ":stats" in entry["stable_key"]:
                self.assertEqual(entry["cache_status"], "recomputed")

    def test_scheme_base_change_isolates_named_scheme(self) -> None:
        engine = LabEngine(copy.deepcopy(self.config), copy.deepcopy(self.samples[:2]))
        engine.run(parallel=False)
        manifest = engine.apply_command(
            {"command_id": "B", "op": "update_scheme", "name": "rope_standard", "params": {"base_frequency": 12345.0}}
        )
        report = engine.run(manifest, parallel=False)
        for entry in report["ledger"]:
            if "scheme=rope_standard" in entry["stable_key"] and ":stats" not in entry["stable_key"]:
                self.assertEqual(entry["cache_status"], "recomputed")
            if "scheme=scaled_ntk" in entry["stable_key"]:
                self.assertEqual(entry["cache_status"], "reused")

    def test_budget_pending_and_resume(self) -> None:
        config = copy.deepcopy(self.config)
        config["point_budget"] = 140
        config["sample_quota"] = 80
        engine = LabEngine(config, copy.deepcopy(self.samples))
        first = engine.run(parallel=False)
        self.assertEqual(first["newly_admitted_sample_ids"], ["S000", "S001", "S002"])
        self.assertEqual(first["pending_sample_ids"], ["S003", "S004", "S005"])
        self.assertIn("E_BUDGET_PENDING", first["code_counts"])

        second = engine.run({}, parallel=False)
        self.assertEqual(second["newly_admitted_sample_ids"], ["S003", "S004", "S005"])
        self.assertEqual(second["pending_sample_ids"], [])

    def test_snapshot_commit_and_recovery(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = SnapshotStore(Path(tmp) / "state")
            engine = LabEngine(self.config, self.samples[:2])
            report = engine.run(parallel=False)
            meta = store.commit({"report": report}, global_version=1, config_hash=engine.config_hash())
            decision = store.recovery_decision(1, engine.config_hash())
            self.assertTrue(decision["use_snapshot"])
            self.assertFalse(store.recovery_decision(2, engine.config_hash())["use_snapshot"])
            loaded = store.load_current()
            self.assertEqual(loaded["meta"]["commit_name"], meta["commit_name"])

    def test_injections_all_hit_and_isolate(self) -> None:
        report = run_injections(self.config, self.samples, self.injections)
        self.assertTrue(report["all_hit"], report)
        nan_result = next(item for item in report["results"] if item["injection_id"] == "INJ-NAN-04")
        self.assertGreater(nan_result["isolation"]["ledger_failed"], 0)
        self.assertEqual(len(nan_result["isolation"]["stats_schemes"]), len(self.config["schemes"]))

    def test_ranking_flip_source_attribution(self) -> None:
        observed = set()
        for case in make_flip_attribution_cases():
            config = copy.deepcopy(self.config)
            config["schemes"] = [scheme for scheme in config["schemes"] if scheme["name"] == "rope_standard"]
            config["train_length"] = 0
            config["target_length"] = 1000
            config["extrapolation_bins"] = [100000]
            config["fixed_point"].update(
                {
                    "signed_bits": 64,
                    "fraction_bits": case["fraction_bits"],
                    "accumulator_bits": case["accumulator_bits"],
                    "quant_level": "nearest",
                }
            )
            report = LabEngine(config, [case["sample"]]).run(parallel=False)
            sources = {flip["source"] for flip in report["extrapolation_flips"]}
            self.assertIn(case["expected_source"], sources, case["case_id"])
            observed.update(sources)
        self.assertIn("angle_quantization", observed)
        self.assertIn("rotation_rounding", observed)
        self.assertIn("accumulator_overflow", observed)


def _semantic_from_report(report):
    """Build the same stable semantic mapping without requiring engine state."""
    return {
        "global_version": report.get("global_version"),
        "config_hash": report.get("config_hash"),
        "workspace_hash": report.get("workspace_hash"),
        "complete": report.get("complete"),
        "sample_ids": report.get("sample_ids"),
        "processed_sample_ids": report.get("processed_sample_ids"),
        "pending_sample_ids": report.get("pending_sample_ids"),
        "config": report.get("config"),
        "error_records": report.get("error_records"),
        "scheme_stage_matrix": report.get("scheme_stage_matrix"),
        "stats_records": report.get("stats_records"),
        "extrapolation_flips": report.get("extrapolation_flips"),
    }


if __name__ == "__main__":
    unittest.main()
