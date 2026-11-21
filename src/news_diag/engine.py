from __future__ import annotations

import hashlib
import json
import shutil
import sys
import threading
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from decimal import Decimal
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from .errors import (
    E_TEMPLATE_MISSING,
    WorkbenchError,
    failure_from_error,
)
from .io_utils import append_jsonl, atomic_write_json_compact, json_text
from .loader import (
    build_runtime_table,
    load_config,
    load_injections,
    load_raw_news,
    load_synonym_table,
    missing_template_rules,
    preapply_illegal_entity_injections,
    resolve_synonym_table_path,
    validate_raw_documents,
)
from .models import (
    Document,
    InjectionSpec,
    Sentence,
    SentenceScore,
    SynonymTable,
    WorkbenchConfig,
    decimal_string,
)
from .rewrite import rewrite_sentence
from .scoring import score_sentence
from .selection import select_summary
from .splitter import document_frequency_total, split_document


REPORT_SCHEMA_VERSION = "1.0"
SCORE_COMPONENTS = (
    "word_frequency",
    "entity_coverage",
    "position_decay",
    "quantization_residual",
)


@dataclass(frozen=True)
class RunnerOptions:
    news_path: Path
    config_path: Path
    injections_path: Path | None = None
    serial: bool = False
    state_dir: Path | None = None
    resume: bool = False
    stop_event: threading.Event | None = None


@dataclass
class RunResult:
    report: dict[str, Any]
    exit_code: int
    interrupted: bool
    completed_document_count: int
    pending_document_ids: list[str]


@dataclass
class DocTaskResult:
    doc_id: str
    task_key: str
    report: dict[str, Any] | None = None
    failure: dict[str, Any] | None = None
    injection_updates: dict[str, dict[str, Any]] = field(default_factory=dict)
    skipped: bool = False


def _dependency_report() -> dict[str, str]:
    return {
        "python": sys.version.split()[0],
        "numpy": str(np.__version__),
        "pyyaml": str(getattr(yaml, "__version__", "unknown")),
    }


def _file_bytes(path: Path) -> bytes:
    return path.read_bytes()


def compute_input_fingerprint(paths: list[Path]) -> str:
    digest = hashlib.sha256()
    expanded: list[Path] = []
    for path in paths:
        if path.is_dir():
            expanded.extend(sorted((item for item in path.iterdir() if item.is_file()), key=lambda item: item.name))
        else:
            expanded.append(path)
    for path in sorted(expanded, key=lambda item: (item.name, str(item))):
        digest.update(b"FILE\0")
        digest.update(path.name.encode("utf-8"))
        digest.update(b"\0")
        if path.exists() and path.is_file():
            digest.update(_file_bytes(path))
        else:
            digest.update(b"<missing>")
        digest.update(b"\0")
    return digest.hexdigest()


def _failure_key(record: dict[str, Any]) -> str:
    return "|".join(
        [
            str(record.get("stage", "")),
            str(record.get("document_id") or ""),
            str(record.get("sort_key", "")),
            str(record.get("code", "")),
            str(record.get("message", "")),
        ]
    )


def _sorted_failures(failures: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    order = {
        "config": 0,
        "load": 1,
        "split": 2,
        "score": 3,
        "select": 4,
        "rewrite": 5,
        "report": 6,
    }
    return sorted(
        failures.values(),
        key=lambda item: (
            order.get(str(item.get("stage", "")), 99),
            str(item.get("document_id") or ""),
            str(item.get("sort_key", "")),
            str(item.get("code", "")),
            str(item.get("message", "")),
        ),
    )


def _score_to_dict(score: SentenceScore, config: WorkbenchConfig) -> dict[str, Any]:
    fixed_numeric = float(score.fixed_value)
    difference = fixed_numeric - score.reference
    terms: dict[str, Any] = {}
    for term in score.terms:
        terms[term.name] = {
            "normalized": decimal_string(term.normalized),
            "weight": decimal_string(term.weight),
            "weighted": decimal_string(term.weighted),
            "weighted_reference": term.reference,
            "fixed_integer": term.fixed_integer,
            "fixed_value": float(term.fixed_value),
            "fixed_value_text": decimal_string(term.fixed_value, places=config.fixed_decimals),
            "rounding_residual": decimal_string(term.rounding_residual),
            "rounding_residual_numeric": float(term.rounding_residual),
        }
    return {
        "fixed": {
            "score_integer": score.fixed_integer,
            "score": fixed_numeric,
            "score_text": decimal_string(score.fixed_value, places=config.fixed_decimals),
            "scale": config.score_scale,
            "quantization_unit": config.quantization_unit,
        },
        "reference": {"score": score.reference},
        "difference": {
            "fixed_minus_reference": difference,
            "reference_minus_fixed": score.reference - fixed_numeric,
        },
        "terms": terms,
        "rounding": {
            "ideal_total": decimal_string(score.ideal_total),
            "ideal_total_numeric": float(score.ideal_total),
            "fixed_total": decimal_string(score.fixed_value, places=config.fixed_decimals),
            "residual": decimal_string(score.residual),
            "residual_numeric": float(score.residual),
        },
        "inputs": {
            "frequency_hits": decimal_string(score.frequency_hits),
            "frequency_total": decimal_string(score.frequency_total),
            "entity_hits": score.entity_hits,
            "entity_total": score.entity_total,
            "entity_coverage_delta": decimal_string(score.entity_delta),
        },
    }


def _attribution(
    before: SentenceScore,
    after: SentenceScore,
    config: WorkbenchConfig,
    *,
    before_selected: bool,
    after_selected: bool,
) -> dict[str, Any]:
    delta_map: dict[str, Decimal] = {}
    for name in SCORE_COMPONENTS[:-1]:
        delta_map[name] = Decimal(
            after.term(name).fixed_integer - before.term(name).fixed_integer
        )
    delta_map["quantization_residual"] = (after.residual - before.residual) * Decimal(
        config.score_scale
    )
    if not (before_selected != after_selected):
        return {
            "selection_flip": False,
            "dominant_component": None,
            "component_deltas_fixed_units": {
                name: decimal_string(value) for name, value in delta_map.items()
            },
        }
    dominant = max(SCORE_COMPONENTS, key=lambda name: (abs(delta_map[name]), -SCORE_COMPONENTS.index(name)))
    return {
        "selection_flip": True,
        "dominant_component": dominant,
        "component_deltas_fixed_units": {
            name: decimal_string(value) for name, value in delta_map.items()
        },
        "tie_break_order": list(SCORE_COMPONENTS),
        "note": "selection flip is attributed to the largest absolute fixed-unit change",
    }


def _text_offsets(document: Document) -> dict[str, Any]:
    return {
        "document_character_length": len(document.text),
        "paragraph_spans": [
            {"paragraph_index": index, "start": start, "end": end}
            for index, (start, end) in enumerate(document.paragraph_spans)
        ],
    }


def _stable_sentence_key(document: Document, sentence_index: int, stage: str) -> str:
    return f"{document.doc_id}|{sentence_index:08d}|{stage}"


class WorkbenchRunner:
    def __init__(self, options: RunnerOptions) -> None:
        self.options = options
        self.stop_event = options.stop_event or threading.Event()
        self._injection_lock = threading.Lock()
        self._state_path = (
            self.options.state_dir / "run_state.json" if self.options.state_dir else None
        )
        self._state_documents_dir = (
            self.options.state_dir / "documents" if self.options.state_dir else None
        )
        self._state_failures_dir = (
            self.options.state_dir / "failures" if self.options.state_dir else None
        )
        self._progress_path = (
            self.options.state_dir / "progress.jsonl" if self.options.state_dir else None
        )
        self._input_signature = ""
        self._state_docs: dict[str, dict[str, Any]] = {}
        self._state_failures: dict[str, dict[str, Any]] = {}
        self._processed_ids: set[str] = set()
        self._active_injections: list[InjectionSpec] = []
        self._active_config: WorkbenchConfig | None = None

    def run(self) -> RunResult:
        config, config_failures = load_config(self.options.config_path)
        injections, injection_failures = load_injections(self.options.injections_path)
        self._active_injections = injections
        self._active_config = config
        news_paths = [self.options.config_path, self.options.news_path]
        if self.options.injections_path is not None:
            news_paths.append(self.options.injections_path)
        if config is not None:
            news_paths.append(resolve_synonym_table_path(self.options.config_path, config))
        self._input_signature = compute_input_fingerprint(news_paths)
        failures: dict[str, dict[str, Any]] = {}
        for record in config_failures + injection_failures:
            failures[_failure_key(record.to_dict())] = record.to_dict()

        if config is None:
            self._mark_unrunnable_injections(injections, "configuration validation failed")
            report = self._build_report(
                config=None,
                documents=[],
                failures=failures,
                injections=injections,
                rewrite_warnings=[],
                rewrite_errors=[],
                pending_document_ids=[],
            )
            return RunResult(
                report=report,
                exit_code=2,
                interrupted=False,
                completed_document_count=0,
                pending_document_ids=[],
            )

        table_path = resolve_synonym_table_path(self.options.config_path, config)
        table, table_failures = load_synonym_table(table_path, config)
        for record in table_failures:
            failures[_failure_key(record.to_dict())] = record.to_dict()
        if table is None:
            self._mark_unrunnable_injections(injections, "synonym table validation failed")
            report = self._build_report(
                config=config,
                documents=[],
                failures=failures,
                injections=injections,
                rewrite_warnings=[],
                rewrite_errors=[],
                pending_document_ids=[],
            )
            return RunResult(
                report=report,
                exit_code=2,
                interrupted=False,
                completed_document_count=0,
                pending_document_ids=[],
            )

        excluded_news_paths = {table_path}
        if self.options.injections_path is not None:
            excluded_news_paths.add(self.options.injections_path)
        raw_documents, raw_failures = load_raw_news(
            self.options.news_path, excluded_news_paths
        )
        for record in raw_failures:
            failures[_failure_key(record.to_dict())] = record.to_dict()

        preapply_illegal_entity_injections(raw_documents, injections)
        runtime_table, _ = build_runtime_table(table, injections)
        documents, document_load_failures = validate_raw_documents(raw_documents, injections)
        for record in document_load_failures:
            failures[_failure_key(record.to_dict())] = record.to_dict()

        rewrite_warnings = self._rewrite_warnings(runtime_table)
        self._prepare_state()
        self._load_state_if_resumable()
        failures.update(self._state_failures)

        valid_ids = {document.doc_id for document in documents}
        reports: dict[str, dict[str, Any]] = {
            doc_id: report for doc_id, report in self._state_docs.items() if doc_id in valid_ids
        }
        processed_ids = set(self._state_docs) | set(self._processed_ids)
        pending_documents = [
            document
            for document in sorted(documents, key=lambda item: item.doc_id)
            if document.doc_id not in processed_ids
        ]

        skipped_ids: list[str] = []
        if pending_documents:
            max_workers = 1 if self.options.serial else min(config.max_workers, len(pending_documents))
            max_workers = max(1, max_workers)
            if max_workers == 1:
                for document in pending_documents:
                    task_result = self._run_document_task(document, config, runtime_table, injections)
                    self._integrate_task_result(task_result, config, reports, failures, processed_ids)
                    if task_result.skipped:
                        skipped_ids.append(document.doc_id)
            else:
                with ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="news-diag") as executor:
                    future_map: dict[Future[DocTaskResult], Document] = {
                        executor.submit(
                            self._run_document_task, document, config, runtime_table, injections
                        ): document
                        for document in pending_documents
                    }
                    for future in as_completed(future_map):
                        document = future_map[future]
                        try:
                            task_result = future.result()
                        except Exception as exc:  # noqa: BLE001 - task boundary
                            failure = failure_from_error(
                                exc,
                                document_id=document.doc_id,
                                stage="report",
                                sort_key=f"{document.doc_id}|report|task",
                            ).to_dict()
                            task_result = DocTaskResult(
                                doc_id=document.doc_id,
                                task_key=f"{document.doc_id}|document|task",
                                failure=failure,
                            )
                        self._integrate_task_result(
                            task_result, config, reports, failures, processed_ids
                        )
                        if task_result.skipped:
                            skipped_ids.append(document.doc_id)

        self._finalize_scoring_injections(injections, reports, failures, valid_ids, skipped_ids)
        self._finalize_template_injections(injections, reports)
        self._finalize_illegal_injections(injections, failures)

        all_pending = sorted(
            document.doc_id
            for document in documents
            if document.doc_id not in reports and document.doc_id not in processed_ids
        )
        rewrite_errors = self._collect_rewrite_errors(reports)
        report = self._build_report(
            config=config,
            documents=[reports[doc_id] for doc_id in sorted(reports)],
            failures=failures,
            injections=injections,
            rewrite_warnings=rewrite_warnings,
            rewrite_errors=rewrite_errors,
            pending_document_ids=all_pending,
        )
        interrupted = bool(skipped_ids or all_pending)
        return RunResult(
            report=report,
            exit_code=0,
            interrupted=interrupted,
            completed_document_count=len(reports),
            pending_document_ids=all_pending,
        )

    def _prepare_state(self) -> None:
        if self.options.state_dir is None:
            return
        self.options.state_dir.mkdir(parents=True, exist_ok=True)
        if not self.options.resume:
            if self._state_path and self._state_path.exists():
                self._state_path.unlink()
            if self._progress_path and self._progress_path.exists():
                self._progress_path.unlink()
            for artifact_dir in (self._state_documents_dir, self._state_failures_dir):
                if artifact_dir and artifact_dir.exists():
                    shutil.rmtree(artifact_dir)
        if self._state_documents_dir is not None:
            self._state_documents_dir.mkdir(parents=True, exist_ok=True)
        if self._state_failures_dir is not None:
            self._state_failures_dir.mkdir(parents=True, exist_ok=True)
        if self._state_path is not None and not self._state_path.exists():
            self._save_state_manifest({}, {}, set(), self._active_injections)

    def _load_state_if_resumable(self) -> None:
        if not self.options.resume or self._state_path is None or not self._state_path.exists():
            return
        try:
            payload = json.loads(self._state_path.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001 - corrupt state is ignored rather than trusted
            return
        if payload.get("schema_version") != REPORT_SCHEMA_VERSION:
            return
        if payload.get("input_signature") != self._input_signature:
            return
        self._state_docs = {}
        for doc_id, reference in dict(payload.get("documents", {})).items():
            if not isinstance(reference, dict) or not isinstance(reference.get("file"), str):
                continue
            artifact_path = self.options.state_dir / reference["file"]
            try:
                value = json.loads(artifact_path.read_text(encoding="utf-8"))
            except Exception:  # noqa: BLE001 - incomplete artifacts are ignored
                continue
            if isinstance(value, dict):
                self._state_docs[str(doc_id)] = value
        self._state_failures = {}
        for key, reference in dict(payload.get("failures", {})).items():
            if not isinstance(reference, dict) or not isinstance(reference.get("file"), str):
                continue
            artifact_path = self.options.state_dir / reference["file"]
            try:
                value = json.loads(artifact_path.read_text(encoding="utf-8"))
            except Exception:  # noqa: BLE001 - incomplete artifacts are ignored
                continue
            if isinstance(value, dict):
                self._state_failures[str(key)] = value
        self._processed_ids = {
            str(doc_id) for doc_id in payload.get("processed_document_ids", [])
        }
        injection_state = payload.get("injections", {})
        if isinstance(injection_state, dict):
            for injection in self._pending_injection_objects():
                saved = injection_state.get(injection.injection_id)
                if not isinstance(saved, dict):
                    continue
                status = saved.get("status")
                if isinstance(status, str) and status in {"applied", "not_applied"}:
                    injection.status = status
                    injection.reason = saved.get("reason")
                    effect = saved.get("effect")
                    if isinstance(effect, dict):
                        injection.effect = effect

    def _pending_injection_objects(self) -> list[InjectionSpec]:
        return list(self._active_injections)

    def _artifact_reference(self, directory: Path, key: str) -> dict[str, str]:
        digest = hashlib.sha256(key.encode("utf-8")).hexdigest()[:32]
        relative = f"{directory.name}/{digest}.json"
        return {"file": relative}

    def _persist_artifact(self, directory: Path, key: str, payload: dict[str, Any]) -> dict[str, str]:
        reference = self._artifact_reference(directory, key)
        atomic_write_json_compact(self.options.state_dir / reference["file"], payload)
        return reference

    def _save_state_manifest(
        self,
        reports: dict[str, dict[str, Any]],
        failures: dict[str, dict[str, Any]],
        processed_ids: set[str],
        injections: list[InjectionSpec],
    ) -> None:
        if self._state_path is None:
            return
        payload = {
            "schema_version": REPORT_SCHEMA_VERSION,
            "input_signature": self._input_signature,
            "documents": {
                doc_id: self._artifact_reference(self._state_documents_dir, doc_id)
                for doc_id in sorted(reports)
            },
            "failures": {
                key: self._artifact_reference(self._state_failures_dir, key)
                for key in sorted(failures)
            },
            "processed_document_ids": sorted(processed_ids),
            "injections": {
                injection.injection_id: {
                    "status": injection.status,
                    "reason": injection.reason,
                    "effect": injection.effect,
                }
                for injection in sorted(injections, key=lambda item: item.ordinal)
            },
        }
        atomic_write_json_compact(self._state_path, payload)

    def _write_progress(self, task_result: DocTaskResult, status: str, completed_count: int) -> None:
        if self._progress_path is None:
            return
        append_jsonl(
            self._progress_path,
            {
                "stage": "document",
                "document_id": task_result.doc_id,
                "sort_key": task_result.task_key,
                "status": status,
                "completed_document_count": completed_count,
            },
        )

    def _integrate_task_result(
        self,
        task_result: DocTaskResult,
        config: WorkbenchConfig,
        reports: dict[str, dict[str, Any]],
        failures: dict[str, dict[str, Any]],
        processed_ids: set[str],
    ) -> None:
        if task_result.skipped:
            self._write_progress(task_result, "skipped_after_graceful_stop", len(reports))
            return
        processed_ids.add(task_result.doc_id)
        if task_result.report is not None:
            reports[task_result.doc_id] = task_result.report
            if self._state_documents_dir is not None:
                self._persist_artifact(
                    self._state_documents_dir, task_result.doc_id, task_result.report
                )
            progress_status = "completed"
        elif task_result.failure is not None:
            failure_key = _failure_key(task_result.failure)
            failures[failure_key] = task_result.failure
            if self._state_failures_dir is not None:
                self._persist_artifact(
                    self._state_failures_dir, failure_key, task_result.failure
                )
            progress_status = "failed"
        else:
            progress_status = "empty"
        with self._injection_lock:
            self._apply_injection_updates(task_result.injection_updates)
        self._save_state_manifest(reports, failures, processed_ids, self._active_injections)
        self._write_progress(task_result, progress_status, len(reports))

    def _apply_injection_updates(self, updates: dict[str, dict[str, Any]]) -> None:
        by_id = {injection.injection_id: injection for injection in self._active_injections}
        for injection_id, state in updates.items():
            injection = by_id.get(injection_id)
            if injection is None:
                continue
            status = state.get("status")
            if isinstance(status, str) and status in {"applied", "not_applied"}:
                injection.status = status
            reason = state.get("reason")
            if isinstance(reason, str):
                injection.reason = reason
            effect = state.get("effect")
            if isinstance(effect, dict):
                injection.effect = effect

    def _run_document_task(
        self,
        document: Document,
        config: WorkbenchConfig,
        table: SynonymTable,
        injections: list[InjectionSpec],
    ) -> DocTaskResult:
        task_key = f"{document.doc_id}|document|pipeline"
        if self.stop_event.is_set():
            return DocTaskResult(doc_id=document.doc_id, task_key=task_key, skipped=True)
        targeted = [
            injection
            for injection in injections
            if injection.doc_id == document.doc_id
            and injection.kind in {"frequency_delta", "entity_coverage_delta"}
        ]
        targeted_ids = [injection.injection_id for injection in targeted]
        current_stage = "split"
        try:
            sentences = split_document(document)
            current_stage = "score"
            frequency_total = document_frequency_total(document)
            before_scores = {
                sentence.sentence_index: score_sentence(
                    document,
                    sentence,
                    len(sentences),
                    frequency_total,
                    config,
                )
                for sentence in sentences
            }
            current_stage = "select"
            before_selection = select_summary(document, sentences, before_scores, config)
            before_reference_selection = select_summary(
                document, sentences, before_scores, config, reference_path=True
            )

            frequency_deltas: dict[int, Decimal] = {}
            entity_deltas: dict[int, Decimal] = {}
            for injection in targeted:
                index = injection.sentence_index
                if index is None or index < 0 or index >= len(sentences):
                    continue
                if injection.kind == "frequency_delta":
                    frequency_deltas[index] = frequency_deltas.get(index, Decimal("0")) + (
                        injection.delta or Decimal("0")
                    )
                elif injection.kind == "entity_coverage_delta":
                    entity_deltas[index] = entity_deltas.get(index, Decimal("0")) + (
                        injection.delta or Decimal("0")
                    )

            current_stage = "score"
            after_scores = {
                sentence.sentence_index: score_sentence(
                    document,
                    sentence,
                    len(sentences),
                    frequency_total,
                    config,
                    frequency_delta=frequency_deltas.get(sentence.sentence_index, Decimal("0")),
                    entity_coverage_delta=entity_deltas.get(
                        sentence.sentence_index, Decimal("0")
                    ),
                )
                for sentence in sentences
            }
            current_stage = "select"
            after_selection = select_summary(document, sentences, after_scores, config)
            after_reference_selection = select_summary(
                document, sentences, after_scores, config, reference_path=True
            )
            before_fixed_set = set(before_selection.selected_indices)
            before_reference_set = set(before_reference_selection.selected_indices)
            after_fixed_set = set(after_selection.selected_indices)
            after_reference_set = set(after_reference_selection.selected_indices)
            after_rank = {
                sentence_index: rank
                for rank, sentence_index in enumerate(after_selection.ordered_indices, start=1)
            }
            after_selection_rank = {
                sentence_index: rank
                for rank, sentence_index in enumerate(after_selection.selected_indices, start=1)
            }
            before_rank = {
                sentence_index: rank
                for rank, sentence_index in enumerate(before_selection.ordered_indices, start=1)
            }

            current_stage = "rewrite"
            priority_selected = [
                sentences[index]
                for index in after_selection.selected_indices
                if 0 <= index < len(sentences)
            ]
            summary_selected = sorted(priority_selected, key=lambda sentence: sentence.start)
            rewritten_by_index: dict[int, str] = {}
            replacements: list[dict[str, Any]] = []
            suppressed: list[dict[str, Any]] = []
            rewrite_errors: list[dict[str, Any]] = []
            for sentence in summary_selected:
                (
                    rewritten,
                    sentence_replacements,
                    sentence_suppressed,
                    sentence_errors,
                ) = rewrite_sentence(sentence, table, config.style_template_id)
                rewritten_by_index[sentence.sentence_index] = rewritten
                replacements.extend(sentence_replacements)
                suppressed.extend(sentence_suppressed)
                rewrite_errors.extend(sentence_errors)

            current_stage = "report"
            sentence_reports: list[dict[str, Any]] = []
            boundary_sentences: list[dict[str, Any]] = []
            for sentence in sentences:
                index = sentence.sentence_index
                before_score = before_scores[index]
                after_score = after_scores[index]
                fixed_selected = index in after_fixed_set
                reference_selected = index in after_reference_set
                fixed_vs_reference_flip = fixed_selected != reference_selected
                before_selected = index in before_fixed_set
                before_reference_selected = index in before_reference_set
                attribution = _attribution(
                    before_score,
                    after_score,
                    config,
                    before_selected=before_selected,
                    after_selected=fixed_selected,
                )
                score_difference = float(after_score.fixed_value) - after_score.reference
                sentence_report = {
                    "sentence_index": index,
                    "paragraph_index": sentence.paragraph_index,
                    "start": sentence.start,
                    "end": sentence.end,
                    "text": sentence.text,
                    "task_keys": {
                        stage: _stable_sentence_key(document, index, stage)
                        for stage in ("split", "score", "select", "rewrite")
                    },
                    "before": {
                        "fixed_score": float(before_score.fixed_value),
                        "fixed_score_integer": before_score.fixed_integer,
                        "reference_score": before_score.reference,
                        "selected_by_fixed": before_selected,
                        "selected_by_reference": before_reference_selected,
                        "competition_rank_fixed": before_rank[index],
                    },
                    "after": {
                        "fixed_score": float(after_score.fixed_value),
                        "fixed_score_integer": after_score.fixed_integer,
                        "reference_score": after_score.reference,
                        "selected_by_fixed": fixed_selected,
                        "selected_by_reference": reference_selected,
                        "competition_rank_fixed": after_rank[index],
                        "selection_rank_fixed": after_selection_rank.get(index),
                    },
                    "fixed_vs_reference": {
                        "fixed_score": float(after_score.fixed_value),
                        "reference_score": after_score.reference,
                        "difference": score_difference,
                        "selection_flip": fixed_vs_reference_flip,
                    },
                    "score_before": _score_to_dict(before_score, config),
                    "score_after": _score_to_dict(after_score, config),
                    "selected_before": before_selected,
                    "selected_after": fixed_selected,
                    "selected_by_reference_after": reference_selected,
                    "selection_flip_before_after": before_selected != fixed_selected,
                    "selection_flip_fixed_vs_reference": fixed_vs_reference_flip,
                    "perturbation_attribution": attribution,
                    "rewritten_text": rewritten_by_index.get(index),
                }
                sentence_reports.append(sentence_report)
                if fixed_vs_reference_flip:
                    boundary_sentences.append(
                        {
                            "sentence_index": index,
                            "start": sentence.start,
                            "end": sentence.end,
                            "fixed_score": float(after_score.fixed_value),
                            "fixed_score_integer": after_score.fixed_integer,
                            "reference_score": after_score.reference,
                            "difference": score_difference,
                            "selected_by_fixed": fixed_selected,
                            "selected_by_reference": reference_selected,
                        }
                    )

            summary_sentence_reports = [
                {
                    "sentence_index": sentence.sentence_index,
                    "selection_rank": after_selection_rank.get(sentence.sentence_index),
                    "start": sentence.start,
                    "end": sentence.end,
                    "original_text": sentence.text,
                    "rewritten_text": rewritten_by_index.get(sentence.sentence_index, sentence.text),
                }
                for sentence in summary_selected
            ]
            summary_text = "".join(item["rewritten_text"] for item in summary_sentence_reports)
            competition_order = [
                {
                    "rank": rank,
                    "sentence_index": index,
                    "start": sentences[index].start,
                    "fixed_score": float(after_scores[index].fixed_value),
                    "fixed_score_integer": after_scores[index].fixed_integer,
                    "reference_score": after_scores[index].reference,
                    "difference": float(after_scores[index].fixed_value) - after_scores[index].reference,
                    "selected_by_fixed": index in after_fixed_set,
                }
                for rank, index in enumerate(after_selection.ordered_indices, start=1)
            ]
            reference_competition_order = [
                {
                    "rank": rank,
                    "sentence_index": index,
                    "reference_score": after_scores[index].reference,
                }
                for rank, index in enumerate(after_reference_selection.ordered_indices, start=1)
            ]
            reference_covered = set(after_reference_selection.covered_entity_ids)
            actual_coverage = (
                Decimal(len(after_selection.covered_entity_ids)) / Decimal(len(document.entities))
            )
            injection_updates = self._injection_updates_for_document(
                targeted,
                sentences,
                before_scores,
                after_scores,
                before_fixed_set,
                after_fixed_set,
                before_reference_set,
                after_reference_set,
            )
            replacement_records = sorted(
                replacements,
                key=lambda item: (
                    int(item["sentence_index"]),
                    int(item["start"]),
                    int(item["end"]),
                    str(item["rule_id"]),
                ),
            )
            suppressed_records = sorted(
                suppressed,
                key=lambda item: (
                    int(item["sentence_index"]),
                    int(item["start"]),
                    int(item["end"]),
                    str(item["rule_id"]),
                ),
            )
            rewrite_errors = sorted(
                rewrite_errors,
                key=lambda item: (
                    int(item.get("sentence_index", -1)),
                    int(item.get("start", -1)),
                    str(item.get("rule_id", "")),
                ),
            )
            boundary_sentences = sorted(
                boundary_sentences, key=lambda item: int(item["sentence_index"])
            )
            report = {
                "doc_id": document.doc_id,
                "title": document.title,
                "source": f"{document.source_name}:{document.source_line}",
                "document_task_key": task_key,
                "text_layout": _text_offsets(document),
                "entities": [
                    {
                        "entity_id": entity.entity_id,
                        "name": entity.name,
                        "aliases": list(entity.aliases),
                        "type": entity.entity_type,
                    }
                    for entity in document.entities
                ],
                "events": [event.to_dict() for event in document.events],
                "sentences": sentence_reports,
                "summary": {
                    "text": summary_text,
                    "summary_order": "original_start_offset_ascending",
                    "selection_order": list(after_selection.selected_indices),
                    "sentences": summary_sentence_reports,
                },
                "selection": {
                    "fixed_selected_indices": list(after_selection.selected_indices),
                    "reference_selected_indices": list(after_reference_selection.selected_indices),
                    "before_fixed_selected_indices": list(before_selection.selected_indices),
                    "before_reference_selected_indices": list(
                        before_reference_selection.selected_indices
                    ),
                    "covered_entity_ids": list(after_selection.covered_entity_ids),
                    "reference_covered_entity_ids": sorted(reference_covered),
                    "coverage_events": [
                        event.to_dict() for event in after_selection.coverage_events
                    ],
                    "reference_coverage_events": [
                        event.to_dict() for event in after_reference_selection.coverage_events
                    ],
                },
                "competition_order": competition_order,
                "reference_competition_order": reference_competition_order,
                "boundary_sentences": boundary_sentences,
                "replacement_records": replacement_records,
                "suppressed_replacements": suppressed_records,
                "diagnostic_errors": rewrite_errors,
                "statistics": {
                    "selected_sentence_count": len(after_selection.selected_indices),
                    "reference_selected_sentence_count": len(
                        after_reference_selection.selected_indices
                    ),
                    "covered_entity_count": len(after_selection.covered_entity_ids),
                    "total_entity_count": len(document.entities),
                    "actual_entity_coverage": decimal_string(actual_coverage),
                    "actual_entity_coverage_numeric": float(actual_coverage),
                    "entity_coverage_min": decimal_string(config.entity_coverage_min),
                    "coverage_satisfied": after_selection.coverage_satisfied,
                    "coverage_boundary_hit_count": len(after_selection.coverage_events),
                    "boundary_sentence_count": len(boundary_sentences),
                    "boundary_sentence_ratio": len(boundary_sentences) / len(sentences),
                    "sentence_count": len(sentences),
                    "applied_replacement_count": len(replacement_records),
                    "suppressed_replacement_count": len(suppressed_records),
                },
                "injection_ids": [
                    injection.injection_id
                    for injection in sorted(targeted, key=lambda item: item.ordinal)
                ],
            }
            return DocTaskResult(
                doc_id=document.doc_id,
                task_key=task_key,
                report=report,
                injection_updates=injection_updates,
            )
        except Exception as exc:  # noqa: BLE001 - one document must not stop the pool
            stage = exc.stage if isinstance(exc, WorkbenchError) else current_stage
            failure = failure_from_error(
                exc,
                document_id=document.doc_id,
                stage=stage,
                sort_key=f"{document.doc_id}|{stage}|pipeline",
                injection_ids=targeted_ids,
                details={
                    "document_task_key": task_key,
                    "failed_stage": stage,
                },
            ).to_dict()
            injection_updates = {
                injection.injection_id: {
                    "status": "not_applied",
                    "reason": f"document failed during {stage}",
                    "effect": {
                        "document_id": document.doc_id,
                        "document_failure_stage": stage,
                    },
                }
                for injection in targeted
            }
            return DocTaskResult(
                doc_id=document.doc_id,
                task_key=task_key,
                failure=failure,
                injection_updates=injection_updates,
            )

    def _injection_updates_for_document(
        self,
        targeted: list[InjectionSpec],
        sentences: list[Sentence],
        before_scores: dict[int, SentenceScore],
        after_scores: dict[int, SentenceScore],
        before_fixed_set: set[int],
        after_fixed_set: set[int],
        before_reference_set: set[int],
        after_reference_set: set[int],
    ) -> dict[str, dict[str, Any]]:
        updates: dict[str, dict[str, Any]] = {}
        for injection in targeted:
            index = injection.sentence_index
            if index is None or index < 0 or index >= len(sentences):
                updates[injection.injection_id] = {
                    "status": "not_applied",
                    "reason": "sentence_index is outside the split sentence list",
                    "effect": {
                        "document_id": injection.doc_id,
                        "requested_sentence_index": index,
                        "sentence_count": len(sentences),
                    },
                }
                continue
            before_score = before_scores[index]
            after_score = after_scores[index]
            selected_before = index in before_fixed_set
            selected_after = index in after_fixed_set
            updates[injection.injection_id] = {
                "status": "applied",
                "effect": {
                    "document_id": injection.doc_id,
                    "sentence_index": index,
                    "delta": decimal_string(injection.delta or Decimal("0")),
                    "before_fixed_score_integer": before_score.fixed_integer,
                    "after_fixed_score_integer": after_score.fixed_integer,
                    "before_reference_score": before_score.reference,
                    "after_reference_score": after_score.reference,
                    "selected_before": selected_before,
                    "selected_after": selected_after,
                    "selected_by_reference_before": index in before_reference_set,
                    "selected_by_reference_after": index in after_reference_set,
                    "selection_flip": selected_before != selected_after,
                    "dominant_component": _attribution(
                        before_score,
                        after_score,
                        self._active_config,
                        before_selected=selected_before,
                        after_selected=selected_after,
                    )["dominant_component"],
                },
            }
        return updates

    def _finalize_scoring_injections(
        self,
        injections: list[InjectionSpec],
        reports: dict[str, dict[str, Any]],
        failures: dict[str, dict[str, Any]],
        valid_ids: set[str],
        skipped_ids: list[str],
    ) -> None:
        failed_doc_stages: dict[str, str] = {}
        for failure in failures.values():
            doc_id = failure.get("document_id")
            stage = failure.get("stage")
            if isinstance(doc_id, str) and isinstance(stage, str) and doc_id not in failed_doc_stages:
                failed_doc_stages[doc_id] = stage
        for injection in injections:
            if injection.kind not in {"frequency_delta", "entity_coverage_delta"}:
                continue
            if injection.status in {"applied", "not_applied"}:
                continue
            doc_id = injection.doc_id or ""
            if doc_id in skipped_ids:
                injection.status = "not_applied"
                injection.reason = "execution stopped gracefully before the target document completed"
            elif doc_id in failed_doc_stages:
                injection.status = "not_applied"
                injection.reason = (
                    f"target document failed during {failed_doc_stages[doc_id]}"
                )
            elif doc_id not in valid_ids:
                injection.status = "not_applied"
                injection.reason = "target document failed load validation or is absent"
            elif doc_id not in reports:
                injection.status = "not_applied"
                injection.reason = "target document did not produce a report"
            else:
                injection.status = "not_applied"
                injection.reason = "target sentence was not scored"

    def _finalize_template_injections(
        self, injections: list[InjectionSpec], reports: dict[str, dict[str, Any]]
    ) -> None:
        for injection in injections:
            if injection.kind != "missing_template_ref" or injection.status != "applied":
                continue
            affected_documents: list[str] = []
            preserved_count = 0
            for doc_id in sorted(reports):
                matching = [
                    error
                    for error in reports[doc_id].get("diagnostic_errors", [])
                    if error.get("code") == E_TEMPLATE_MISSING
                    and error.get("rule_id") == injection.rule_id
                ]
                if matching:
                    affected_documents.append(doc_id)
                    preserved_count += len(matching)
            injection.effect = dict(injection.effect)
            injection.effect.update(
                {
                    "affected_document_ids": affected_documents,
                    "preserved_occurrence_count": preserved_count,
                }
            )

    def _finalize_illegal_injections(
        self, injections: list[InjectionSpec], failures: dict[str, dict[str, Any]]
    ) -> None:
        for injection in injections:
            if injection.kind != "illegal_entity" or injection.status != "applied":
                continue
            matching = [
                failure
                for failure in failures.values()
                if failure.get("document_id") == injection.doc_id
                and failure.get("stage") == "load"
            ]
            injection.effect = dict(injection.effect)
            if matching:
                injection.effect.update(
                    {
                        "document_rejected": True,
                        "document_failure_code": matching[0].get("code"),
                        "document_failure_stage": matching[0].get("stage"),
                    }
                )
            else:
                injection.effect.update({"document_rejected": False})

    def _mark_unrunnable_injections(
        self, injections: list[InjectionSpec], reason: str
    ) -> None:
        for injection in injections:
            if injection.status == "pending":
                injection.status = "not_applied"
                injection.reason = reason

    def _rewrite_warnings(self, table: SynonymTable) -> list[dict[str, Any]]:
        declared = sorted(table.templates)
        return [
            {
                "code": E_TEMPLATE_MISSING,
                "stage": "rewrite",
                "rule_id": rule.rule_id,
                "template_id": rule.template_id,
                "declared_templates": declared,
                "decision": "rule excluded; original fragments preserved",
            }
            for rule in missing_template_rules(table)
        ]

    def _collect_rewrite_errors(
        self, reports: dict[str, dict[str, Any]]
    ) -> list[dict[str, Any]]:
        errors: list[dict[str, Any]] = []
        for doc_id in sorted(reports):
            for error in reports[doc_id].get("diagnostic_errors", []):
                item = dict(error)
                item["document_id"] = doc_id
                errors.append(item)
        return sorted(
            errors,
            key=lambda item: (
                str(item.get("document_id", "")),
                int(item.get("sentence_index", -1)),
                int(item.get("start", -1)),
                str(item.get("rule_id", "")),
            ),
        )

    def _build_report(
        self,
        *,
        config: WorkbenchConfig | None,
        documents: list[dict[str, Any]],
        failures: dict[str, dict[str, Any]],
        injections: list[InjectionSpec],
        rewrite_warnings: list[dict[str, Any]],
        rewrite_errors: list[dict[str, Any]],
        pending_document_ids: list[str],
    ) -> dict[str, Any]:
        sorted_injections = [item.to_dict() for item in sorted(injections, key=lambda item: item.ordinal)]
        applied = [item for item in sorted_injections if item.get("status") == "applied"]
        not_applied = [item for item in sorted_injections if item.get("status") == "not_applied"]
        return {
            "schema_version": REPORT_SCHEMA_VERSION,
            "dependencies": _dependency_report(),
            "input_fingerprint": self._input_signature,
            "config": config.to_report_dict() if config is not None else None,
            "documents": documents,
            "failures": _sorted_failures(failures),
            "rewrite_table_warnings": rewrite_warnings,
            "rewrite_errors": rewrite_errors,
            "injections": {
                "all": sorted_injections,
                "applied": applied,
                "not_applied": not_applied,
            },
            "completion": {
                "complete": not pending_document_ids,
                "completed_document_count": len(documents),
                "pending_document_ids": pending_document_ids,
            },
        }


def run_workbench(options: RunnerOptions) -> RunResult:
    return WorkbenchRunner(options).run()


def report_text(report: dict[str, Any]) -> str:
    return json_text(report)
