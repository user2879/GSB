"""Concurrent staged execution, checkpointing, recovery and report assembly."""

from __future__ import annotations

import dataclasses
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable

from .diagnostics import diagnose_sentence
from .entity import recognize_entities
from .errors import ConfigError, SnapshotError, failure_from_exception
from .fusion import fuse_sentence
from .loader import (
    STAGE_ORDER,
    EvidenceTable,
    Injection,
    ModelPackage,
    Sentence,
    WorkbenchConfig,
    load_config,
    load_evidence,
    load_injections,
    load_model,
    load_sentences,
    input_hashes,
)
from .relation import classify_relations
from .report import build_report
from .snapshot import SnapshotState, SnapshotStore, snapshot_summary
from .util import atomic_write_json, sha256_bytes


@dataclass(frozen=True)
class CheckpointHook:
    stop_event: threading.Event | None = None
    stop_after_stage: str | None = None
    callback: Callable[[SnapshotState], None] | None = None


def _events_sorted(events: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(
        events,
        key=lambda item: (
            item.get("injection_id", ""),
            item.get("sentence_id", ""),
            item.get("stage", ""),
            item.get("kind", ""),
        ),
    )


def _ok_record(
    sentence_id: str,
    data: dict[str, Any],
    events: Iterable[dict[str, Any]] = (),
) -> dict[str, Any]:
    return {
        "sentence_id": sentence_id,
        "status": "ok",
        "data": data,
        "injection_events": _events_sorted(events),
    }


def _error_record(
    sentence_id: str,
    failure: dict[str, Any],
    events: Iterable[dict[str, Any]] = (),
) -> dict[str, Any]:
    return {
        "sentence_id": sentence_id,
        "status": "error",
        "failure": failure,
        "injection_events": _events_sorted(events),
    }


def _record_map(records: Iterable[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {record["sentence_id"]: record for record in records}


def stable_task_key(stage: str, sentence_id: str) -> str:
    digest = sha256_bytes(f"{stage}\0{sentence_id}".encode("utf-8"))
    return f"{stage}:{digest[:16]}"


def _injection_ids_for(
    injections: Iterable[Injection], sentence_id: str, stage: str, kind: str
) -> tuple[str, ...]:
    return tuple(
        injection.injection_id
        for injection in injections
        if injection.sentence_id == sentence_id
        and injection.stage == stage
        and injection.kind == kind
    )


def _make_injection_events(
    injection_ids: Iterable[str],
    sentence_id: str,
    stage: str,
    kind: str,
    business_code: str | None,
) -> list[dict[str, Any]]:
    return [
        {
            "injection_id": injection_id,
            "sentence_id": sentence_id,
            "stage": stage,
            "kind": kind,
            "applied": True,
            "business_code": business_code,
        }
        for injection_id in injection_ids
    ]


class WorkbenchRunner:
    def __init__(
        self,
        config_path: str | Path,
        *,
        snapshot_dir: str | Path | None = None,
        report_path: str | Path | None = None,
        workers: int | None = None,
        fresh: bool = False,
        hooks: CheckpointHook | None = None,
    ):
        self.config = load_config(config_path)
        if snapshot_dir is not None:
            self.config = dataclasses.replace(
                self.config, snapshot_dir=Path(snapshot_dir).resolve()
            )
        if report_path is not None:
            self.config = dataclasses.replace(
                self.config, report_path=Path(report_path).resolve()
            )
        if workers is not None:
            if workers < 1 or workers > 64:
                raise ConfigError("workers override must be between 1 and 64")
            self.config = dataclasses.replace(self.config, workers=workers)
        self.model: ModelPackage = load_model(self.config.model_path, self.config)
        self.evidence: EvidenceTable = load_evidence(
            self.config.evidence_path, self.config
        )
        self.injections = load_injections(
            self.config.injections_path, self.config
        )
        self.load_bundle = load_sentences(
            self.config.sentences_path, self.config, self.injections
        )
        self.sentences = {
            sentence.sentence_id: sentence for sentence in self.load_bundle.sentences
        }
        self.hashes = input_hashes(self.config, self.evidence, self.injections)
        self.store = SnapshotStore(self.config.snapshot_dir)
        self.fresh = fresh
        self.hooks = hooks or CheckpointHook()
        self.initial_state: SnapshotState | None = None
        self.resumed = False

    def _load_records(self) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        records: list[dict[str, Any]] = []
        for sentence in self.load_bundle.sentences:
            records.append(
                {
                    "sentence_id": sentence.sentence_id,
                    "status": "ok",
                    "data": None,
                    "injection_events": [],
                }
            )
        for failure in self.load_bundle.failures:
            records.append(
                {
                    "sentence_id": failure["sentence_id"],
                    "status": "error",
                    "failure": failure,
                    "injection_events": [],
                }
            )
        return records, _events_sorted(self.load_bundle.injection_events)

    def _initialize(self) -> SnapshotState:
        if self.fresh:
            self.store.reset()
        state = self.store.read_current()
        self.resumed = state is not None
        if state is None:
            records, events = self._load_records()
            state = self.store.commit(
                stage="load",
                records=records,
                stage_done=True,
                config_hash=self.config.config_hash,
                model_hash=self.model.model_hash,
                input_hash=self.hashes["combined"],
                metadata={"injection_events": events},
            )
        else:
            self.store.assert_compatible(
                state,
                config_hash=self.config.config_hash,
                model_hash=self.model.model_hash,
                input_hash=self.hashes["combined"],
            )
            if state.artifact("load") is None:
                raise SnapshotError("snapshot has no load artifact")
            artifact_ids = {
                record["sentence_id"]
                for record in state.records("load")
                if record.get("status") == "ok"
            }
            bundle_ids = set(self.sentences)
            if artifact_ids != bundle_ids:
                raise SnapshotError("load snapshot accepted sentence IDs differ from input")
        self.initial_state = state
        return state

    def _stage_prior(self, state: SnapshotState, stage: str) -> dict[str, dict[str, Any]]:
        artifact = state.artifact(stage)
        if artifact is None:
            return {}
        return _record_map(artifact.get("records", []))

    def _run_stage(
        self,
        state: SnapshotState,
        stage: str,
        target_ids: list[str],
        task: Callable[[str], dict[str, Any]],
    ) -> SnapshotState:
        prior = self._stage_prior(state, stage)
        records = dict(prior)
        pending = [sentence_id for sentence_id in target_ids if sentence_id not in records]
        interval = self.config.snapshot_commit_interval
        if not pending:
            if not state.artifact(stage) or not state.artifact(stage).get("stage_done"):
                state = self.store.commit(
                    stage=stage,
                    records=[records[key] for key in sorted(records)],
                    stage_done=True,
                    config_hash=self.config.config_hash,
                    model_hash=self.model.model_hash,
                    input_hash=self.hashes["combined"],
                )
                if self.hooks.callback:
                    self.hooks.callback(state)
            return state

        for start in range(0, len(pending), interval):
            chunk = pending[start : start + interval]
            with ThreadPoolExecutor(max_workers=self.config.workers) as executor:
                for sentence_id, record in zip(
                    chunk, executor.map(task, chunk, chunksize=1)
                ):
                    record["stage"] = stage
                    record["task_key"] = stable_task_key(stage, sentence_id)
                    records[sentence_id] = record
            done = start + len(chunk) >= len(pending)
            state = self.store.commit(
                stage=stage,
                records=[records[key] for key in sorted(records)],
                stage_done=done,
                config_hash=self.config.config_hash,
                model_hash=self.model.model_hash,
                input_hash=self.hashes["combined"],
            )
            if self.hooks.callback:
                self.hooks.callback(state)
        return state

    def _entity_task(self, sentence: Sentence) -> dict[str, Any]:
        injection_ids = _injection_ids_for(
            self.injections, sentence.sentence_id, "entity", "missing_pos"
        )
        events: list[dict[str, Any]] = []
        try:
            data = recognize_entities(
                sentence,
                self.model,
                self.config,
                self.config.fixed,
                missing_pos_injection=bool(injection_ids),
            )
            events.extend(
                _make_injection_events(
                    injection_ids, sentence.sentence_id, "entity", "missing_pos", None
                )
            )
            return _ok_record(sentence.sentence_id, data, events)
        except BaseException as exc:  # noqa: BLE001 - sentence isolation boundary
            failure = failure_from_exception(
                sentence.sentence_id, "entity", exc, injection_ids=injection_ids
            )
            events.extend(
                _make_injection_events(
                    injection_ids,
                    sentence.sentence_id,
                    "entity",
                    "missing_pos",
                    failure["code"],
                )
            )
            return _error_record(sentence.sentence_id, failure, events)

    def _relation_task(
        self, sentence: Sentence, entity_record: dict[str, Any]
    ) -> dict[str, Any]:
        injection_ids = _injection_ids_for(
            self.injections,
            sentence.sentence_id,
            "relation",
            "feature_hash_bucket_mismatch",
        )
        events: list[dict[str, Any]] = []
        try:
            data = classify_relations(
                sentence,
                entity_record["data"],
                self.evidence,
                self.model,
                self.config,
                self.config.fixed,
                feature_hash_bucket_injection=bool(injection_ids),
            )
            events.extend(
                _make_injection_events(
                    injection_ids,
                    sentence.sentence_id,
                    "relation",
                    "feature_hash_bucket_mismatch",
                    None,
                )
            )
            return _ok_record(sentence.sentence_id, data, events)
        except BaseException as exc:  # noqa: BLE001
            failure = failure_from_exception(
                sentence.sentence_id, "relation", exc, injection_ids=injection_ids
            )
            events.extend(
                _make_injection_events(
                    injection_ids,
                    sentence.sentence_id,
                    "relation",
                    "feature_hash_bucket_mismatch",
                    failure["code"],
                )
            )
            return _error_record(sentence.sentence_id, failure, events)

    def _fusion_task(
        self, sentence: Sentence, relation_record: dict[str, Any]
    ) -> dict[str, Any]:
        injection_ids = _injection_ids_for(
            self.injections, sentence.sentence_id, "fusion", "threshold_equal"
        )
        events: list[dict[str, Any]] = []
        try:
            data = fuse_sentence(
                sentence,
                relation_record["data"],
                self.config,
                self.config.fixed,
                threshold_equal_injection_id=injection_ids[0] if injection_ids else None,
            )
            if injection_ids:
                discovered = data.get("injection_events", [])
                if discovered and len(injection_ids) > 1:
                    template = discovered[0]
                    for extra_id in injection_ids[1:]:
                        extra = dict(template)
                        extra["injection_id"] = extra_id
                        discovered.append(extra)
            events.extend(discovered if injection_ids else [])
            return _ok_record(sentence.sentence_id, data, events)
        except BaseException as exc:  # noqa: BLE001
            failure = failure_from_exception(
                sentence.sentence_id, "fusion", exc, injection_ids=injection_ids
            )
            events.extend(
                _make_injection_events(
                    injection_ids,
                    sentence.sentence_id,
                    "fusion",
                    "threshold_equal",
                    failure["code"],
                )
            )
            return _error_record(sentence.sentence_id, failure, events)

    def _diagnosis_task(
        self,
        sentence: Sentence,
        entity_record: dict[str, Any],
        relation_record: dict[str, Any],
        fusion_record: dict[str, Any],
    ) -> dict[str, Any]:
        try:
            reference_fixed = self.config.fixed.reference()
            high_entity = recognize_entities(
                sentence, self.model, self.config, reference_fixed
            )
            high_relation = classify_relations(
                sentence,
                high_entity,
                self.evidence,
                self.model,
                self.config,
                reference_fixed,
            )
            high_fusion = fuse_sentence(
                sentence,
                high_relation,
                self.config,
                reference_fixed,
            )
            diagnostic = diagnose_sentence(
                sentence,
                entity_record["data"],
                high_entity,
                relation_record["data"],
                high_relation,
                fusion_record["data"],
                high_fusion,
                self.config.fixed,
                reference_fixed,
            )
            data = {
                "diagnostic": diagnostic,
                "reference": {
                    "entity": high_entity,
                    "relation": high_relation,
                    "fusion": high_fusion,
                },
            }
            return _ok_record(sentence.sentence_id, data)
        except BaseException as exc:  # noqa: BLE001
            return _error_record(
                sentence.sentence_id,
                failure_from_exception(sentence.sentence_id, "diagnosis", exc),
            )

    def _check_stop(self, state: SnapshotState, completed_stage: str) -> bool:
        if self.hooks.stop_after_stage == completed_stage:
            return True
        return self.hooks.stop_event is not None and self.hooks.stop_event.is_set()

    def run(self) -> dict[str, Any]:
        state = self._initialize()
        initial_version = (
            self.initial_state.version
            if self.resumed and self.initial_state is not None
            else None
        )
        initial_stages = (
            list(self.initial_state.stages_done)
            if self.resumed and self.initial_state is not None
            else []
        )

        if self.hooks.stop_event is not None and self.hooks.stop_event.is_set():
            return self._interrupted_result(state)

        if self._check_stop(state, "load"):
            return self._interrupted_result(state)

        # Entity stage
        if "entity" not in state.stages_done:
            target_ids = sorted(self.sentences)
            state = self._run_stage(
                state, "entity", target_ids, lambda sid: self._entity_task(self.sentences[sid])
            )
            if self._check_stop(state, "entity"):
                return self._interrupted_result(state)

        # Relation stage
        if "relation" not in state.stages_done:
            entity_records = _record_map(state.records("entity"))
            target_ids = sorted(
                sentence_id
                for sentence_id, record in entity_records.items()
                if record.get("status") == "ok" and sentence_id in self.sentences
            )
            relation_inputs = {
                sentence_id: entity_records[sentence_id] for sentence_id in target_ids
            }
            state = self._run_stage(
                state,
                "relation",
                target_ids,
                lambda sid: self._relation_task(
                    self.sentences[sid], relation_inputs[sid]
                ),
            )
            if self._check_stop(state, "relation"):
                return self._interrupted_result(state)

        # Fusion stage
        if "fusion" not in state.stages_done:
            relation_records = _record_map(state.records("relation"))
            target_ids = sorted(
                sentence_id
                for sentence_id, record in relation_records.items()
                if record.get("status") == "ok" and sentence_id in self.sentences
            )
            fusion_inputs = {
                sentence_id: relation_records[sentence_id] for sentence_id in target_ids
            }
            state = self._run_stage(
                state,
                "fusion",
                target_ids,
                lambda sid: self._fusion_task(
                    self.sentences[sid], fusion_inputs[sid]
                ),
            )
            if self._check_stop(state, "fusion"):
                return self._interrupted_result(state)

        # Diagnosis stage
        if "diagnosis" not in state.stages_done:
            entity_records = _record_map(state.records("entity"))
            relation_records = _record_map(state.records("relation"))
            fusion_records = _record_map(state.records("fusion"))
            target_ids = sorted(
                sentence_id
                for sentence_id in entity_records
                if entity_records.get(sentence_id, {}).get("status") == "ok"
                and relation_records.get(sentence_id, {}).get("status") == "ok"
                and fusion_records.get(sentence_id, {}).get("status") == "ok"
                and sentence_id in self.sentences
            )
            diagnosis_inputs = {
                sentence_id: (
                    entity_records[sentence_id],
                    relation_records[sentence_id],
                    fusion_records[sentence_id],
                )
                for sentence_id in target_ids
            }
            state = self._run_stage(
                state,
                "diagnosis",
                target_ids,
                lambda sid: self._diagnosis_task(
                    self.sentences[sid], *diagnosis_inputs[sid]
                ),
            )
            if self._check_stop(state, "diagnosis"):
                return self._interrupted_result(state)

        report = build_report(
            self.config,
            self.model,
            self.injections,
            self.sentences,
            state,
            initial_version=initial_version,
            initial_stages=initial_stages,
            resumed=self.resumed,
        )
        atomic_write_json(self.config.report_path, report)
        return report

    def _interrupted_result(self, state: SnapshotState) -> dict[str, Any]:
        return {
            "status": "interrupted",
            "exit_code": 0,
            "snapshot": snapshot_summary(state),
            "message": "current stage completed and snapshot committed; rerun to resume",
        }


def run_workbench(
    config_path: str | Path,
    *,
    snapshot_dir: str | Path | None = None,
    report_path: str | Path | None = None,
    workers: int | None = None,
    fresh: bool = False,
    hooks: CheckpointHook | None = None,
) -> dict[str, Any]:
    return WorkbenchRunner(
        config_path,
        snapshot_dir=snapshot_dir,
        report_path=report_path,
        workers=workers,
        fresh=fresh,
        hooks=hooks,
    ).run()
