from __future__ import annotations

import json
import copy
import shutil
import tempfile
import unittest
from pathlib import Path

from depworkbench import Workbench
from depworkbench.canonical import canonical_json
from depworkbench.errors import SimulatedCrash, ValidationError
from depworkbench.io import load_project, write_json, write_jsonl_atomic
from depworkbench.models import model_from_dict
from depworkbench.samplegen import generate_samples
from depworkbench.snapshot import SnapshotStore


class WorkbenchTestBase(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.root = Path(tempfile.mkdtemp(prefix="depworkbench-tests-"))
        cls.base_samples = cls.root / "samples"
        generate_samples(cls.base_samples, count=5, seed=20261123)

    @classmethod
    def tearDownClass(cls) -> None:
        shutil.rmtree(cls.root, ignore_errors=True)

    def project_copy(self, name: str) -> Path:
        target = self.root / name / "samples"
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(self.base_samples, target)
        return target

    def snapshot_path(self, name: str) -> Path:
        return self.root / name / "snapshots"

    def full_report(self, project: Path, label: str, *, injection_file: str | None = None):
        workbench = Workbench(
            project,
            snapshot_dir=self.root / f"{label}-full-snapshot",
            injection_file=injection_file,
        )
        return workbench.run(fresh=True)

    def sentence(self, report: dict, sid: str) -> dict:
        return next(item for item in report["sentences"] if item["sentence_id"] == sid)


class LoaderValidationTests(WorkbenchTestBase):
    def test_invalid_rows_are_sorted_and_duplicate_ids_are_closed(self) -> None:
        project_dir = self.project_copy("validation")
        first = next(
            line for line in (project_dir / "sentences.jsonl").read_text(encoding="utf-8").splitlines() if line
        )
        row = json.loads(first)
        rows = [
            {
                "sentence_id": "DUP",
                "words": row["words"],
                "pos_tags": row["pos_tags"],
            },
            {
                "sentence_id": "DUP",
                "words": row["words"],
                "pos_tags": row["pos_tags"],
            },
            {
                "sentence_id": "BAD-WORD",
                "words": "不存在的词 " + " ".join(row["words"].split()[1:]),
                "pos_tags": row["pos_tags"],
            },
            {
                "sentence_id": "BAD-LEN",
                "words": row["words"],
                "pos_tags": row["pos_tags"].split()[0],
            },
            {
                "sentence_id": "OK",
                "words": row["words"],
                "pos_tags": row["pos_tags"],
            },
        ]
        write_jsonl_atomic(project_dir / "sentences.jsonl", rows)
        project = load_project(project_dir)
        self.assertEqual([sentence.sentence_id for sentence in project.sentences], ["OK"])
        codes = [failure.code for failure in project.failures]
        self.assertEqual(codes, ["E_SENT_INVALID"] * 4)
        keys = [failure.sort_key for failure in project.failures]
        self.assertEqual(keys, sorted(keys))

    def test_duplicate_id_with_an_invalid_row_closes_the_valid_row_too(self) -> None:
        project_dir = self.project_copy("duplicate-invalid")
        first = next(
            line for line in (project_dir / "sentences.jsonl").read_text(encoding="utf-8").splitlines() if line
        )
        row = json.loads(first)
        rows = [
            {
                "sentence_id": "DUP2",
                "words": "不存在的词",
                "pos_tags": row["pos_tags"],
            },
            {
                "sentence_id": "DUP2",
                "words": row["words"],
                "pos_tags": row["pos_tags"],
            },
        ]
        write_jsonl_atomic(project_dir / "sentences.jsonl", rows)
        project = load_project(project_dir)
        self.assertFalse(project.sentences)
        self.assertEqual(len(project.failures), 2)
        self.assertTrue(all(item.code == "E_SENT_INVALID" for item in project.failures))

    def test_model_and_config_validation(self) -> None:
        project_dir = self.project_copy("model-invalid")
        model = json.loads((project_dir / "model.json").read_text(encoding="utf-8"))
        model["feature_dim"] = 0
        write_json(project_dir / "model.json", model)
        with self.assertRaises(ValidationError) as context:
            load_project(project_dir)
        self.assertEqual(context.exception.code, "E_MODEL_INVALID")

        project_dir = self.project_copy("config-invalid")
        config = json.loads((project_dir / "config.json").read_text(encoding="utf-8"))
        config["max_threads"] = 999
        write_json(project_dir / "config.json", config)
        with self.assertRaises(ValidationError) as context:
            load_project(project_dir)
        self.assertEqual(context.exception.code, "E_CONFIG_INVALID")

        config["max_threads"] = 2
        config["quantization_level"] = 7
        write_json(project_dir / "config.json", config)
        with self.assertRaises(ValidationError) as context:
            load_project(project_dir)
        self.assertEqual(context.exception.code, "E_CONFIG_INVALID")


class ParsingTests(WorkbenchTestBase):
    def test_repeated_parse_is_identical(self) -> None:
        project = load_project(self.base_samples)
        from depworkbench.parser import parse_sentence
        from depworkbench.quantize import quantize_model

        quantized = quantize_model(project.model, project.config)
        sentence = project.sentences[0]
        first = parse_sentence(sentence, project.model, quantized, project.config)
        second = parse_sentence(sentence, project.model, quantized, project.config)
        self.assertTrue(first.completed)
        self.assertEqual(first.final_state.actions, second.final_state.actions)
        self.assertEqual(
            [step["score"] for step in first.steps],
            [step["score"] for step in second.steps],
        )
        self.assertEqual(
            [step["feasible_action_fingerprint"] for step in first.steps],
            [step["feasible_action_fingerprint"] for step in second.steps],
        )
        self.assertEqual(
            [step["step_index"] for step in first.steps],
            list(range(len(first.steps))),
        )

    def test_tie_break_and_exclusion_metadata_are_deterministic(self) -> None:
        project = load_project(self.base_samples)
        from depworkbench.parser import parse_sentence
        from depworkbench.quantize import quantize_model

        quantized = quantize_model(project.model, project.config)
        outcome = parse_sentence(project.sentences[1], project.model, quantized, project.config)
        for step in outcome.steps:
            selected_candidates = sorted(
                step["feasible_actions"],
                key=lambda item: (-item["score"], item["code"]),
            )
            self.assertEqual(step["action_code"], selected_candidates[0]["code"])
            self.assertTrue(step["excluded_actions"])
            for excluded in step["excluded_actions"]:
                self.assertTrue(excluded["reason"].startswith("E_"))

    def test_equal_scores_choose_smaller_action_code(self) -> None:
        project = load_project(self.base_samples)
        from depworkbench.parser import parse_sentence
        from depworkbench.quantize import quantize_model

        raw = copy.deepcopy(project.model.raw)
        raw["feature_dim"] = 1
        raw["num_buckets"] = 1
        raw["action_weights"] = {action: [1.0] for action in ("SHIFT", "LEFT_ARC", "RIGHT_ARC", "FINISH")}
        raw["feature_hashes"] = {key: 0 for key in raw["feature_hashes"]}
        model = model_from_dict(
            raw,
            vocabulary=list(project.model.vocabulary),
            pos_tags=list(project.model.pos_tags),
        )
        quantized = quantize_model(model, project.config)
        outcome = parse_sentence(project.sentences[0], model, quantized, project.config)
        tie_steps = [step for step in outcome.steps if {"SHIFT", "LEFT_ARC"}.issubset(step["tied_best_actions"])]
        self.assertTrue(tie_steps)
        for step in tie_steps:
            self.assertEqual(step["action"], "SHIFT")

    def test_exhaustive_node_limit_is_an_explicit_failure(self) -> None:
        project = load_project(self.base_samples)
        from depworkbench.models import config_from_dict
        from depworkbench.parser import parse_sentence
        from depworkbench.quantize import quantize_model

        raw = project.config.to_dict()
        raw["mode"] = "exhaustive"
        raw["max_exhaustive_nodes"] = 1
        config = config_from_dict(raw)
        quantized = quantize_model(project.model, config)
        outcome = parse_sentence(project.sentences[0], project.model, quantized, config)
        self.assertFalse(outcome.completed)
        self.assertEqual(outcome.error_code, "E_EXHAUSTIVE_NODE_LIMIT")
        self.assertTrue(outcome.search_stats["node_limit_hit"])

    def test_exhaustive_mode_has_pruning_and_differs_from_greedy(self) -> None:
        project_dir = self.project_copy("mode")
        workbench = Workbench(project_dir, snapshot_dir=self.snapshot_path("mode"))
        report = workbench.run(fresh=True)
        self.assertGreater(report["mode_differences"]["different_sentence_count"], 0)
        differences = report["mode_differences"]["differences"]
        self.assertTrue(any(not item["same"] for item in differences))
        self.assertTrue(
            any(
                item.get("greedy", {}).get("evaluated_transition_steps")
                != item.get("exhaustive", {}).get("evaluated_transition_steps")
                for item in differences
                if item.get("comparison_status") == "OK"
            )
        )

        exhaustive = workbench.run(mode_override="exhaustive", fresh=True)
        exhaustive_sentences = [item for item in exhaustive["sentences"] if item["status"] == "OK"]
        self.assertTrue(exhaustive_sentences)
        self.assertTrue(
            any(item["pruned_count_by_reason"] for item in exhaustive_sentences)
        )
        self.assertTrue(
            all(item["search_stats"].get("nodes", 0) >= item["action_sequence_length"] for item in exhaustive_sentences)
        )
        for sentence in exhaustive_sentences:
            self.assertEqual(
                [step["step_index"] for step in sentence["steps"]],
                list(range(sentence["action_sequence_length"])),
            )


class IncrementalTests(WorkbenchTestBase):
    def test_no_change_reuses_all_steps(self) -> None:
        project_dir = self.project_copy("no-change")
        workbench = Workbench(project_dir, snapshot_dir=self.snapshot_path("no-change"))
        first = workbench.run(fresh=True)
        second = workbench.run()
        self.assertEqual(first["semantic_digest"], second["semantic_digest"])
        for sentence in second["sentences"]:
            if sentence["status"] == "OK":
                self.assertEqual(sentence["cache_ledger"]["recomputed_count"], 0)
                self.assertEqual(
                    sentence["cache_ledger"]["reused_count"],
                    sentence["action_sequence_length"],
                )

    def test_pos_edit_invalidates_from_exact_step_and_matches_full(self) -> None:
        project_dir = self.project_copy("pos-edit")
        workbench = Workbench(project_dir, snapshot_dir=self.snapshot_path("pos-edit"))
        baseline = workbench.run(fresh=True)
        target = self.sentence(baseline, "S001")
        words = list(target["words"])
        pos_tags = list(target["pos_tags"])
        pos_tags[-1] = "a" if pos_tags[-1] != "a" else "n"
        workbench.replace_sentence("S001", words, pos_tags)
        incremental = workbench.run()
        edited = self.sentence(incremental, "S001")
        ledger = edited["cache_ledger"]
        self.assertEqual(ledger["reason"], "TOKEN_CHANGED")
        self.assertIsNotNone(ledger["first_invalid_step"])
        self.assertGreater(ledger["first_invalid_step"], 0)
        self.assertEqual(ledger["reused_count"], ledger["first_invalid_step"])
        self.assertGreater(ledger["recomputed_count"], 0)
        for sid in ("S002", "S003", "S004", "S005"):
            unaffected = self.sentence(incremental, sid)
            self.assertEqual(unaffected["cache_ledger"]["recomputed_count"], 0)
        full = Workbench(
            project_dir,
            snapshot_dir=self.root / "pos-edit-full-snapshot",
        ).run(fresh=True)
        self.assertEqual(incremental["semantic_digest"], full["semantic_digest"])

    def test_weight_edit_is_step_precise_and_matches_full(self) -> None:
        project_dir = self.project_copy("weight-edit")
        workbench = Workbench(project_dir, snapshot_dir=self.snapshot_path("weight-edit"))
        baseline = workbench.run(fresh=True)
        target = self.sentence(baseline, "S001")
        usage: dict[str, list[int]] = {}
        for step in target["steps"]:
            for cell in step["cache"]["weight_cells"]:
                usage.setdefault(cell, []).append(step["step_index"])
        cell = max((max(indexes), key) for key, indexes in usage.items())[1]
        action, raw_index = cell.rsplit(":", 1)
        model = json.loads((project_dir / "model.json").read_text(encoding="utf-8"))
        old_value = model["action_weights"][action][int(raw_index)]
        new_value = old_value + 0.5 if old_value >= 0 else old_value - 0.5
        workbench.set_action_weight(action, int(raw_index), new_value)
        incremental = workbench.run()
        edited = self.sentence(incremental, "S001")
        self.assertEqual(edited["cache_ledger"]["reason"], "MODEL_WEIGHT_CHANGED")
        self.assertGreater(edited["cache_ledger"]["first_invalid_step"], 0)
        self.assertEqual(
            edited["cache_ledger"]["reused_count"],
            edited["cache_ledger"]["first_invalid_step"],
        )
        full = Workbench(project_dir, snapshot_dir=self.root / "weight-full-snapshot").run(fresh=True)
        self.assertEqual(incremental["semantic_digest"], full["semantic_digest"])

    def test_edit_sequence_always_matches_blank_rebuild(self) -> None:
        project_dir = self.project_copy("sequence")
        workbench = Workbench(project_dir, snapshot_dir=self.snapshot_path("sequence"))
        workbench.run(fresh=True)
        baseline = load_project(self.base_samples)
        words = baseline.sentences[0].words
        pos_tags = baseline.sentences[0].pos_tags
        operations = [
            ("add", lambda: workbench.add_sentence("S006", words, pos_tags)),
            ("delete", lambda: workbench.delete_sentence("S002")),
            (
                "constraint",
                lambda: workbench.set_constraint("forbid_left_dependent_pos", ["d"]),
            ),
            (
                "weight",
                lambda: workbench.set_action_weight("LEFT_ARC", 4, 0.75),
            ),
            ("mode", lambda: workbench.set_mode("exhaustive")),
        ]
        for index, (name, operation) in enumerate(operations):
            operation()
            incremental = workbench.run()
            full = Workbench(
                project_dir,
                snapshot_dir=self.root / f"sequence-full-{index}",
            ).run(fresh=True)
            self.assertEqual(
                incremental["semantic_digest"],
                full["semantic_digest"],
                msg=f"semantic mismatch after {name}",
            )
            self.assertEqual(
                canonical_json(incremental["failures"]),
                canonical_json(full["failures"]),
            )
            self.assertEqual(incremental["arcs"], full["arcs"])
            self.assertEqual(incremental["trees"], full["trees"])

    def test_exhaustive_incremental_environment_change_matches_full(self) -> None:
        project_dir = self.project_copy("exhaustive-edit")
        workbench = Workbench(project_dir, snapshot_dir=self.snapshot_path("exhaustive-edit"))
        workbench.run(mode_override="exhaustive", fresh=True)
        workbench.set_action_weight("FINISH", 3, -0.875)
        incremental = workbench.run(mode_override="exhaustive")
        full = Workbench(
            project_dir,
            snapshot_dir=self.root / "exhaustive-edit-full",
        ).run(mode_override="exhaustive", fresh=True)
        self.assertEqual(incremental["semantic_digest"], full["semantic_digest"])
        for sentence in incremental["sentences"]:
            if sentence["status"] == "OK":
                self.assertEqual(
                    sentence["cache_ledger"]["reason"],
                    "EXHAUSTIVE_SEARCH_INPUT_CHANGED",
                )


class ConcurrencyAndBudgetTests(WorkbenchTestBase):
    def test_thread_count_does_not_change_semantics(self) -> None:
        serial_dir = self.project_copy("serial")
        parallel_dir = self.project_copy("parallel")
        serial_wb = Workbench(serial_dir, snapshot_dir=self.snapshot_path("serial"), report_path=None)
        parallel_wb = Workbench(parallel_dir, snapshot_dir=self.snapshot_path("parallel"), report_path=None)
        serial_wb.set_config("max_threads", 1)
        parallel_wb.set_config("max_threads", 4)
        serial = serial_wb.run(fresh=True)
        parallel = parallel_wb.run(fresh=True)
        self.assertEqual(serial["semantic_digest"], parallel["semantic_digest"])
        self.assertEqual(serial["arcs"], parallel["arcs"])
        self.assertEqual(serial["trees"], parallel["trees"])
        self.assertEqual(serial["failures"], parallel["failures"])

    def test_budget_defers_with_business_code_and_next_batch_priority(self) -> None:
        project_dir = self.project_copy("budget")
        workbench = Workbench(project_dir, snapshot_dir=self.snapshot_path("budget"))
        workbench.set_config("batch_parse_point_budget", 6)
        report = workbench.run(fresh=True)
        self.assertTrue(report["pending"])
        self.assertTrue(all(item["code"] == "E_PARSE_BUDGET_PENDING" for item in report["pending"]))
        self.assertGreater(len(report["budget"]["batches"]), 1)
        pending_ids = {item["sentence_id"] for item in report["pending"]}
        self.assertTrue(any(item["status"] == "PENDING" for item in report["sentences"]))
        self.assertTrue(any(item["status"] == "OK" for item in report["sentences"]))
        self.assertTrue(pending_ids)

    def test_fault_injections_are_isolated_and_mapped(self) -> None:
        project_dir = self.project_copy("injections")
        workbench = Workbench(
            project_dir,
            snapshot_dir=self.snapshot_path("injections"),
            injection_file="injections_demo.json",
        )
        report = workbench.run(fresh=True)
        codes = {failure["sentence_id"]: failure["code"] for failure in report["failures"]}
        self.assertEqual(codes.get("S002"), "E_SENT_INVALID")
        self.assertEqual(codes.get("S003"), "E_FEATURE_UNDECLARED")
        self.assertEqual(codes.get("S004"), "E_MODEL_INVALID")
        self.assertEqual(codes.get("S005"), "E_CONFIG_INVALID")
        self.assertEqual(self.sentence(report, "S001")["status"], "OK")
        items = {item["injection_id"]: item for item in report["injections"]}
        self.assertTrue(all(item["hit"] for item in items.values()))
        self.assertEqual(items["INJ-001"]["business_code"], "E_SENT_INVALID")
        self.assertEqual(items["INJ-002"]["business_code"], "E_FEATURE_UNDECLARED")
        self.assertEqual(items["INJ-003"]["business_code"], "E_MODEL_INVALID")
        self.assertEqual(items["INJ-004"]["business_code"], "E_CONFIG_INVALID")


class SnapshotRecoveryTests(WorkbenchTestBase):
    def test_crash_after_sentence_and_step_cursor_resumes_identically(self) -> None:
        project_dir = self.project_copy("crash")
        reference = Workbench(
            project_dir,
            snapshot_dir=self.root / "crash-reference-snapshot",
        ).run(fresh=True)
        workbench = Workbench(project_dir, snapshot_dir=self.snapshot_path("crash"))
        with self.assertRaises(SimulatedCrash):
            workbench.run(
                fresh=True,
                crash_after_sentences=3,
                crash_partial_steps=2,
            )
        store = SnapshotStore(self.snapshot_path("crash"))
        manifest = store.latest_manifest()
        self.assertIsNotNone(manifest)
        assert manifest is not None
        cursors = manifest["sentence_cursors"]
        partial = [sid for sid, cursor in cursors.items() if not cursor["completed"] and cursor["steps"] == 2]
        self.assertTrue(partial)
        loaded = store.load_latest()
        assert loaded is not None
        partial_sid = partial[0]
        self.assertEqual(
            len(loaded["parse"]["caches"][partial_sid]["greedy"]["steps"]),
            2,
        )
        resumed = workbench.run(
            resume=True,
            expected_version=int(manifest["global_version"]),
        )
        self.assertEqual(resumed["semantic_digest"], reference["semantic_digest"])
        resumed_partial = self.sentence(resumed, partial_sid)
        self.assertEqual(resumed_partial["cache_ledger"]["reason"], "RESUME_CURSOR")
        self.assertEqual(resumed_partial["cache_ledger"]["reused_count"], 2)
        self.assertGreater(resumed_partial["cache_ledger"]["recomputed_count"], 0)

    def test_stale_snapshot_is_rejected_and_reinitialized(self) -> None:
        project_dir = self.project_copy("stale")
        workbench = Workbench(project_dir, snapshot_dir=self.snapshot_path("stale"))
        first = workbench.run(fresh=True)
        old_version = int(first["snapshot"]["global_version"])
        workbench.set_mode("exhaustive")
        report = workbench.run(resume=True, expected_version=old_version)
        self.assertEqual(report["snapshot_stale"]["code"], "E_SNAPSHOT_STALE")
        self.assertEqual(report["mode"], "exhaustive")
        full = Workbench(
            project_dir,
            snapshot_dir=self.root / "stale-full-snapshot",
        ).run(fresh=True)
        self.assertEqual(report["semantic_digest"], full["semantic_digest"])
        self.assertGreater(report["snapshot"]["global_version"], old_version)


if __name__ == "__main__":
    unittest.main()
