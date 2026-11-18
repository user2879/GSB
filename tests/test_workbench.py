from __future__ import annotations

import json
import tempfile
import unittest
from decimal import Decimal
from pathlib import Path

from kg_workbench.entity import recognize_entities
from kg_workbench.errors import (
    ConfigError,
    ConfigMismatchError,
    ModelShapeError,
    SnapshotError,
)
from kg_workbench.loader import (
    EvidenceCandidate,
    EvidenceTable,
    ModelPackage,
    Sentence,
    load_config,
    load_model,
    load_sentences,
)
from kg_workbench.pipeline import CheckpointHook, run_workbench
from kg_workbench.relation import classify_relations
from kg_workbench.snapshot import SnapshotStore


ROOT = Path(__file__).resolve().parents[1]
SAMPLE_CONFIG = ROOT / "config.json"
CORE_KEYS = (
    "triples",
    "diagnostics",
    "boundary_sensitive_triples",
    "injection_hits",
    "failures",
    "score_histograms",
    "statistics",
    "sentences",
    "snapshot",
    "config_hash",
    "model_hash",
    "input_hash",
)


def core_projection(report: dict) -> dict:
    return {key: report[key] for key in CORE_KEYS}


def load_report(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, value: object) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")


def absolute_config(
    directory: Path,
    *,
    snapshot_dir: Path,
    report_path: Path,
    model_path: Path | None = None,
    threshold: float = 0.5,
) -> Path:
    raw = load_report(SAMPLE_CONFIG)
    raw["paths"] = {
        "sentences": str((ROOT / "samples/sentences.jsonl").resolve()),
        "model": str((model_path or (ROOT / "samples/model.json")).resolve()),
        "evidence": str((ROOT / "samples/evidence.json").resolve()),
        "injections": str((ROOT / "samples/injections.json").resolve()),
        "snapshot_dir": str(snapshot_dir.resolve()),
        "report": str(report_path.resolve()),
    }
    raw["entity"]["threshold"] = threshold
    path = directory / "config.json"
    write_json(path, raw)
    return path


def zero_model(relation_types: tuple[str, ...] = ("A", "B")) -> ModelPackage:
    size = 7
    zeros = tuple(Decimal(0) for _ in range(size))
    labels = ("B", "I", "E", "S", "O")
    return ModelPackage(
        raw={},
        model_hash="zero",
        feature_dim=size,
        hash_buckets=size,
        labels=labels,
        transitions={source: {target: Decimal(0) for target in labels} for source in labels},
        entity_bias={label: Decimal(0) for label in labels},
        entity_weights={label: zeros for label in labels},
        entity_types=("MISC",),
        entity_type_bias={"MISC": Decimal(0)},
        entity_type_weights={"MISC": zeros},
        relation_types=relation_types,
        relation_bias={relation_type: Decimal(0) for relation_type in relation_types},
        relation_weights={relation_type: zeros for relation_type in relation_types},
        feature_hashes={},
        lexicon=(),
    )


class WorkbenchIntegrationTests(unittest.TestCase):
    def test_sample_contract_and_injections(self) -> None:
        with tempfile.TemporaryDirectory() as temp_name:
            temp = Path(temp_name)
            report_path = temp / "report.json"
            snapshot = temp / "snapshot"
            report = run_workbench(
                SAMPLE_CONFIG,
                snapshot_dir=snapshot,
                report_path=report_path,
                fresh=True,
            )
            self.assertEqual(report["status"], "completed")
            self.assertEqual(report_path.exists(), True)
            self.assertEqual(report["statistics"]["input_sentences"], 8)
            self.assertEqual(report["statistics"]["output_triplets"], 2)
            self.assertEqual(report["statistics"]["boundary_sensitive_triplets"], 1)
            self.assertEqual(report["statistics"]["injections_hit"], 4)
            self.assertEqual(
                [(triple["sentence_id"], triple["subject"], triple["predicate"], triple["object"]) for triple in report["triples"]],
                [
                    ("S006", "孙八", "任职于", "杭州公司"),
                    ("S005", "陈七", "任职于", "南京大学"),
                ],
            )
            failure_tuples = {
                (failure["sentence_id"], failure["stage"], failure["code"])
                for failure in report["failures"]
            }
            self.assertEqual(
                failure_tuples,
                {
                    ("S002", "entity", "E_CONFIG_INVALID"),
                    ("S003", "relation", "E_MODEL_SHAPE_MISMATCH"),
                    ("S004", "load", "E_ANNOTATION_RANGE"),
                    ("S007", "load", "E_CONFIG_INVALID"),
                    ("S008", "load", "E_ANNOTATION_RANGE"),
                },
            )
            self.assertTrue(all(item["hit"] for item in report["injection_hits"]))
            boundary = report["boundary_sensitive_triples"][0]
            self.assertEqual(boundary["sentence_id"], "S001")
            self.assertEqual(boundary["configured"]["accepted"], False)
            self.assertEqual(boundary["reference"]["accepted"], True)
            self.assertEqual(boundary["primary_diagnostic_source"], "fusion_quantization")
            diagnostics = {item["sentence_id"]: item for item in report["diagnostics"]}
            self.assertEqual(diagnostics["S001"]["first_stage_over_tolerance"], "fusion_weighting")
            self.assertEqual(diagnostics["S005"]["primary_source"], "none")
            self.assertEqual(diagnostics["S006"]["primary_source"], "none")
            self.assertEqual(report["score_histograms"]["configured"]["candidates"]["total"], 3)
            self.assertEqual(report["score_histograms"]["reference"]["output_triples"]["total"], 3)

            state = SnapshotStore(snapshot).read_current()
            self.assertIsNotNone(state)
            assert state is not None
            self.assertEqual(state.manifest["stages_done"], ["load", "entity", "relation", "fusion", "diagnosis"])
            entity_records = state.records("entity")
            self.assertTrue(all(record.get("task_key") for record in entity_records))
            self.assertEqual(
                [record["sentence_id"] for record in entity_records],
                sorted(record["sentence_id"] for record in entity_records),
            )

    def test_concurrency_is_field_identical(self) -> None:
        with tempfile.TemporaryDirectory() as temp_name:
            temp = Path(temp_name)
            reports = []
            for workers in (1, 5):
                reports.append(
                    run_workbench(
                        SAMPLE_CONFIG,
                        snapshot_dir=temp / f"snapshot-{workers}",
                        report_path=temp / f"report-{workers}.json",
                        workers=workers,
                        fresh=True,
                    )
                )
            self.assertEqual(core_projection(reports[0]), core_projection(reports[1]))

    def test_graceful_stop_at_each_stage_then_resume(self) -> None:
        with tempfile.TemporaryDirectory() as temp_name:
            temp = Path(temp_name)
            baseline = run_workbench(
                SAMPLE_CONFIG,
                snapshot_dir=temp / "baseline-snapshot",
                report_path=temp / "baseline-report.json",
                fresh=True,
            )
            for stage in ("load", "entity", "relation", "fusion", "diagnosis"):
                with self.subTest(stage=stage):
                    snapshot = temp / f"stop-{stage}-snapshot"
                    report_path = temp / f"stop-{stage}-report.json"
                    interrupted = run_workbench(
                        SAMPLE_CONFIG,
                        snapshot_dir=snapshot,
                        report_path=report_path,
                        fresh=True,
                        hooks=CheckpointHook(stop_after_stage=stage),
                    )
                    self.assertEqual(interrupted["status"], "interrupted")
                    self.assertEqual(interrupted["exit_code"], 0)
                    self.assertIn(stage, interrupted["snapshot"]["stages_done"])
                    resumed = run_workbench(
                        SAMPLE_CONFIG,
                        snapshot_dir=snapshot,
                        report_path=report_path,
                        workers=3,
                    )
                    self.assertEqual(core_projection(baseline), core_projection(resumed))

    def test_force_kill_style_partial_stage_resume(self) -> None:
        class SimulatedCrash(RuntimeError):
            pass

        with tempfile.TemporaryDirectory() as temp_name:
            temp = Path(temp_name)
            snapshot = temp / "snapshot"
            report_path = temp / "report.json"
            baseline = run_workbench(
                SAMPLE_CONFIG,
                snapshot_dir=temp / "baseline-snapshot",
                report_path=temp / "baseline-report.json",
                fresh=True,
            )

            def crash_after_partial_relation(state) -> None:
                artifact = state.artifact("relation")
                if artifact is not None and not artifact.get("stage_done"):
                    raise SimulatedCrash("simulated hard termination")

            with self.assertRaises(SimulatedCrash):
                run_workbench(
                    SAMPLE_CONFIG,
                    snapshot_dir=snapshot,
                    report_path=report_path,
                    workers=2,
                    fresh=True,
                    hooks=CheckpointHook(callback=crash_after_partial_relation),
                )

            state = SnapshotStore(snapshot).read_current()
            self.assertIsNotNone(state)
            assert state is not None
            self.assertFalse(state.artifact("relation")["stage_done"])
            self.assertGreater(len(state.records("relation")), 0)
            self.assertLess(len(state.records("relation")), 4)

            resumed = run_workbench(
                SAMPLE_CONFIG,
                snapshot_dir=snapshot,
                report_path=report_path,
                workers=4,
            )
            self.assertEqual(core_projection(baseline), core_projection(resumed))
            self.assertTrue(resumed["execution"]["resumed"])

    def test_config_hash_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as temp_name:
            temp = Path(temp_name)
            snapshot = temp / "snapshot"
            report_path = temp / "report.json"
            config_path = absolute_config(
                temp, snapshot_dir=snapshot, report_path=report_path, threshold=0.5
            )
            run_workbench(config_path, fresh=True)
            changed_path = absolute_config(
                temp, snapshot_dir=snapshot, report_path=report_path, threshold=0.51
            )
            with self.assertRaises(ConfigMismatchError) as caught:
                run_workbench(changed_path)
            self.assertEqual(caught.exception.code, "E_CONFIG_MISMATCH")


class ValidationAndTieTests(unittest.TestCase):
    def test_config_model_and_sentence_validation_codes(self) -> None:
        with tempfile.TemporaryDirectory() as temp_name:
            temp = Path(temp_name)
            base_config = load_report(SAMPLE_CONFIG)
            base_config["paths"] = {
                "sentences": str(ROOT / "samples/sentences.jsonl"),
                "model": str(ROOT / "samples/model.json"),
                "evidence": str(ROOT / "samples/evidence.json"),
                "injections": str(ROOT / "samples/injections.json"),
                "snapshot_dir": str(temp / "snapshot"),
                "report": str(temp / "report.json"),
            }

            invalid_config = json.loads(json.dumps(base_config))
            invalid_config["fixed"]["decimals"] = 7
            invalid_config_path = temp / "invalid-config.json"
            write_json(invalid_config_path, invalid_config)
            with self.assertRaises(ConfigError) as config_error:
                load_config(invalid_config_path)
            self.assertEqual(config_error.exception.code, "E_CONFIG_INVALID")

            raw_model = load_report(ROOT / "samples/model.json")
            raw_model["feature_dim"] = raw_model["hash_buckets"] + 1
            bad_model_path = temp / "bad-model.json"
            write_json(bad_model_path, raw_model)
            model_config_raw = json.loads(json.dumps(base_config))
            model_config_raw["paths"]["model"] = str(bad_model_path)
            model_config_path = temp / "model-config.json"
            write_json(model_config_path, model_config_raw)
            config = load_config(model_config_path)
            with self.assertRaises(ModelShapeError) as model_error:
                load_model(bad_model_path, config)
            self.assertEqual(model_error.exception.code, "E_MODEL_SHAPE_MISMATCH")

            sentence_path = temp / "sentences.jsonl"
            records = [
                {
                    "id": "T1",
                    "words": "甲 乙",
                    "pos": "N V",
                    "entities": [{"start": 0, "end": 0, "type": "PER"}],
                    "relations": [],
                },
                {
                    "id": "T2",
                    "words": "甲 乙",
                    "pos": "N",
                    "entities": [],
                    "relations": [],
                },
                {
                    "id": "T3",
                    "words": "甲 乙",
                    "pos": "N V",
                    "entities": [{"start": 1, "end": 3, "type": "LOC"}],
                    "relations": [],
                },
                {
                    "id": "T1",
                    "words": "甲 乙",
                    "pos": "N V",
                    "entities": [],
                    "relations": [],
                },
            ]
            with sentence_path.open("w", encoding="utf-8", newline="\n") as handle:
                for record in records:
                    handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            bundle = load_sentences(sentence_path, load_config(SAMPLE_CONFIG), ())
            self.assertEqual([sentence.sentence_id for sentence in bundle.sentences], ["T1"])
            self.assertEqual(
                {(failure["sentence_id"], failure["code"]) for failure in bundle.failures},
                {
                    ("T2", "E_CONFIG_INVALID"),
                    ("T3", "E_ANNOTATION_RANGE"),
                    ("T1", "E_CONFIG_INVALID"),
                },
            )

    def test_bieso_and_relation_tie_breaks_are_fixed(self) -> None:
        config = load_config(SAMPLE_CONFIG)
        model = zero_model()
        sentence = Sentence(
            sentence_id="TIE-ENTITY",
            words=("甲",),
            pos=("N",),
            entities=(),
            relations=(),
            source_line=1,
        )
        entity_result = recognize_entities(sentence, model, config, config.fixed)
        self.assertEqual(entity_result["labels"], ["S"])
        self.assertEqual(entity_result["entities"][0]["type"], "MISC")

        relation_sentence = Sentence(
            sentence_id="TIE-REL",
            words=("甲", "乙"),
            pos=("N", "N"),
            entities=(),
            relations=(),
            source_line=1,
        )
        fake_entities = {
            "entities": [
                {"start": 0, "end": 0, "type": "MISC", "confidence_units": 9000, "confidence": "0.9000"},
                {"start": 1, "end": 1, "type": "MISC", "confidence_units": 9000, "confidence": "0.9000"},
            ]
        }
        evidence = EvidenceTable(
            raw={},
            candidates_by_sentence={
                "TIE-REL": (EvidenceCandidate("TIE-REL", (0, 0), (1, 1), ("A", "B")),)
            },
            exclusive_relations={"A": ("B",), "B": ("A",)},
        )
        relation_result = classify_relations(
            relation_sentence,
            fake_entities,
            evidence,
            model,
            config,
            config.fixed,
        )
        self.assertEqual(relation_result["relations"][0]["type"], "A")
        self.assertEqual(relation_result["relations"][0]["suppressed"], ["B"])

    def test_snapshot_artifact_corruption_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temp_name:
            temp = Path(temp_name)
            snapshot = temp / "snapshot"
            report_path = temp / "report.json"
            run_workbench(
                SAMPLE_CONFIG,
                snapshot_dir=snapshot,
                report_path=report_path,
                fresh=True,
            )
            store = SnapshotStore(snapshot)
            state = store.read_current()
            assert state is not None
            meta = state.manifest["artifacts"]["entity"]
            artifact_path = (
                snapshot / "versions" / f"v{state.version:06d}" / meta["file"]
            )
            with artifact_path.open("a", encoding="utf-8") as handle:
                handle.write(" ")
            with self.assertRaises(SnapshotError) as caught:
                store.read_current()
            self.assertEqual(caught.exception.code, "E_SNAPSHOT_INVALID")


if __name__ == "__main__":
    unittest.main()
