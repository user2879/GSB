"""Incremental and full-rebuild orchestration."""

from __future__ import annotations

import threading
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from .commands import CommandEngine
from .config import WorkbenchConfig
from .errors import DocumentInvalid, SnapshotStale, WorkbenchError
from .injection import STAGE_ORDER, InjectionController
from .loader import ParsedRecord, load_commands, load_documents, load_injections, normalize_document
from .snapshot import SnapshotStore, stage_versions
from .stages import (
    rewrite_stage,
    scoring_stage,
    selection_stage,
    split_stage,
    summarize_stage,
)
from .util import canonical_json, content_fingerprint, deep_copy_json


PIPELINE_STAGES = ("split", "scoring", "selection", "rewrite", "summary")
DOCUMENT_STAGES = ("split", "scoring", "selection", "rewrite")


@dataclass
class DocumentTaskResult:
    document_id: str
    success: bool
    entries: dict[str, dict[str, Any]] = field(default_factory=dict)
    dispositions: list[dict[str, Any]] = field(default_factory=list)
    timings: dict[str, float] = field(default_factory=dict)
    failure: dict[str, Any] | None = None
    injection_reports: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class WorkbenchRun:
    canonical: dict[str, Any]
    report: dict[str, Any]
    snapshot_state: dict[str, Any]
    interrupted: bool = False


class Workbench:
    def __init__(self, docs_path: str | Path, config: WorkbenchConfig,
                 commands_path: str | Path | None, snapshot_dir: str | Path,
                 injections_path: str | Path | None = None) -> None:
        self.docs_path = Path(docs_path)
        self.config = config
        self.commands_path = Path(commands_path) if commands_path is not None else None
        self.snapshot_store = SnapshotStore(snapshot_dir)
        self.injection_entries = load_injections(injections_path)
        self.injection_hash = content_fingerprint(self.injection_entries)

    def run(self, *, rebuild: bool = False, apply_through: int | None = None,
            expect_version: int | None = None,
            stop_event: threading.Event | None = None,
            stop_after_docs: int | None = None,
            document_delay_seconds: float = 0.0,
            clean_snapshots: bool = False,
            commit_observer: Callable[[dict[str, Any]], None] | None = None) -> WorkbenchRun:
        started = time.perf_counter()
        stop_event = stop_event or threading.Event()
        if clean_snapshots:
            import shutil

            if self.snapshot_store.root.exists():
                shutil.rmtree(self.snapshot_store.root)
            self.snapshot_store = SnapshotStore(self.snapshot_store.root)

        load_result = load_documents(self.docs_path)
        command_records = load_commands(self.commands_path) if self.commands_path else []
        total_commands = len(command_records)
        target_commands = total_commands if apply_through is None else max(0, min(apply_through, total_commands))

        pre_failures: list[dict[str, Any]] = deep_copy_json(load_result.failures)
        loaded_state = None if rebuild else self.snapshot_store.load()
        stale_error: SnapshotStale | None = None
        stale_reasons: list[str] = []
        if loaded_state is None and not rebuild and self.snapshot_store.last_load_error:
            stale_error = SnapshotStale(
                "快照不可读，拒绝复用",
                details={"reason": self.snapshot_store.last_load_error},
            )
            pre_failures.append(stale_error.to_failure())
        if loaded_state is not None:
            stale_reasons.extend(self._snapshot_stale_reasons(
                loaded_state, command_records, expect_version,
                load_result.base_documents_hash,
            ))
            if stale_reasons:
                stale_error = SnapshotStale(
                    "快照与当前输入不一致，拒绝复用",
                    details={"reasons": stale_reasons,
                             "snapshot_version": loaded_state.get("global_version")},
                )
                pre_failures.append(stale_error.to_failure())
                loaded_state = None

        loaded_cursor = int(loaded_state.get("applied_command_count", 0)) if loaded_state else 0
        if loaded_state is not None and target_commands < loaded_cursor:
            failure = {
                "document_id": "",
                "stage": "command",
                "code": "E_COMMAND_INVALID",
                "exception_type": "CommandInvalid",
                "message": f"apply_through={target_commands} 小于快照命令游标 {loaded_cursor}",
                "command_index": target_commands,
            }
            pre_failures.append(failure)
            target_commands = loaded_cursor

        if loaded_state is not None:
            try:
                documents = self._load_state_documents(loaded_state)
            except SnapshotStale as exc:
                stale_error = exc
                pre_failures.append(exc.to_failure(details=exc.details))
                loaded_state = None
                loaded_cursor = 0
                documents = deep_copy_json(load_result.documents)
                overrides = {}
                batch_history = []
                persistent_command_failures = []
                cache = {}
                mode = "full"
            else:
                overrides = deep_copy_json(loaded_state.get("effective_overrides", {}))
                batch_history = deep_copy_json(loaded_state.get("batch_history", []))
                persistent_command_failures = deep_copy_json(loaded_state.get("command_failures", []))
                cache = deep_copy_json(loaded_state.get("cache", {}))
                mode = "incremental"
        else:
            documents = deep_copy_json(load_result.documents)
            overrides = {}
            batch_history = []
            persistent_command_failures = []
            cache = {}
            mode = "rebuild" if rebuild else "full"

        selected_records = [record for record in command_records if record.index < target_commands]
        engine = CommandEngine(self.config, documents, overrides, batch_history)
        engine.failures = persistent_command_failures
        command_result: dict[str, Any] = engine.apply_records(selected_records, loaded_cursor)
        documents = command_result["documents"]
        overrides = command_result["overrides"]
        batch_history = command_result["batch_history"]
        effective_config: WorkbenchConfig = command_result["effective_config"]
        applied_commands = min(int(command_result["applied_command_count"]), target_commands)

        all_failures: list[dict[str, Any]] = pre_failures + deep_copy_json(engine.failures)
        completed_document_ids: list[str] = []
        last_document_cursor = ""
        all_dispositions: list[dict[str, Any]] = []
        stage_timings = {stage: 0.0 for stage in PIPELINE_STAGES}
        injection_reports: list[dict[str, Any]] = []
        document_results: dict[str, DocumentTaskResult] = {}
        interrupted = False

        sorted_document_ids = sorted(documents)
        state = self._base_state(
            documents=documents,
            overrides=overrides,
            batch_history=batch_history,
            command_failures=engine.failures,
            cache=cache,
            effective_config=effective_config,
            base_documents_hash=load_result.base_documents_hash,
            applied_commands=applied_commands,
            command_records=command_records,
        )

        def commit(reason: str) -> None:
            nonlocal state
            state["cache"] = cache
            state["command_failures"] = deep_copy_json(engine.failures)
            state["effective_overrides"] = deep_copy_json(overrides)
            state["synonym_table_version"] = effective_config.synonym_table_version
            state["applied_command_count"] = applied_commands
            state["command_prefix_hash"] = _command_prefix_hash(command_records, applied_commands)
            state["documents"] = deep_copy_json(documents)
            state["last_completed_document_ids"] = list(completed_document_ids)
            state["last_document_cursor"] = last_document_cursor
            state["last_interrupted"] = interrupted
            state["last_commit_reason"] = reason
            state["stage_versions"] = stage_versions(state)
            state = self.snapshot_store.commit(state)
            if commit_observer is not None:
                commit_observer(state)

        if sorted_document_ids:
            dilation_docs = 0
            window = max(1, effective_config.thread_pool_size)
            with ThreadPoolExecutor(max_workers=window, thread_name_prefix="news-doc") as executor:
                pending: deque[tuple[int, str, Any]] = deque()
                next_index = 0
                while next_index < len(sorted_document_ids) and len(pending) < window and not stop_event.is_set():
                    document_id = sorted_document_ids[next_index]
                    future = executor.submit(
                        self._process_document,
                        documents[document_id], effective_config, cache,
                        self.injection_entries, stop_event, document_delay_seconds,
                    )
                    pending.append((next_index, document_id, future))
                    next_index += 1

                while pending:
                    _, document_id, future = pending.popleft()
                    result = future.result()
                    document_results[document_id] = result
                    completed_document_ids.append(document_id)
                    last_document_cursor = document_id
                    injection_reports.extend(result.injection_reports)
                    all_dispositions.extend(result.dispositions)
                    for stage, elapsed in result.timings.items():
                        stage_timings[stage] += elapsed
                    if result.failure is not None:
                        all_failures.append(result.failure)
                    for stage, entry in result.entries.items():
                        _merge_cache_entry(cache, stage, document_id, entry)
                    dilation_docs += 1

                    if stop_after_docs is not None and len(completed_document_ids) >= stop_after_docs:
                        interrupted = True
                        stop_event.set()
                    if stop_event.is_set():
                        interrupted = True
                    if not stop_event.is_set() and next_index < len(sorted_document_ids):
                        document_id = sorted_document_ids[next_index]
                        future = executor.submit(
                            self._process_document,
                            documents[document_id], effective_config, cache,
                            self.injection_entries, stop_event, document_delay_seconds,
                        )
                        pending.append((next_index, document_id, future))
                        next_index += 1

                    should_commit = (
                        stop_event.is_set()
                        or dilation_docs >= effective_config.snapshot_commit_interval_docs
                        or (next_index >= len(sorted_document_ids) and not pending)
                    )
                    if should_commit:
                        commit("interrupt" if stop_event.is_set() else "interval")
                        dilation_docs = 0

        if stop_event.is_set():
            interrupted = True

        summary_entry: dict[str, Any] | None = None
        summary_disposition: dict[str, Any] | None = None
        if not interrupted:
            rewrite_payloads = []
            for document_id in sorted(document_results):
                result = document_results[document_id]
                if result.success and "rewrite" in result.entries:
                    rewrite_payloads.append(result.entries["rewrite"]["payload"])
            upstream = {
                "rewrites": [{"document_id": item["document_id"],
                              "fingerprint": content_fingerprint(item)} for item in rewrite_payloads],
                "failures": sorted_failures(all_failures),
                "synonym_table_version": effective_config.synonym_table_version,
            }
            summary_key = make_cache_key(
                "summary", content_fingerprint(upstream), effective_config, "__global__",
            )
            started_stage = time.perf_counter()
            cached_summary = _cache_get(cache, "summary", "__global__", summary_key)
            if cached_summary is not None:
                summary_entry = cached_summary
                status = "reused"
                payload = cached_summary["payload"]
            else:
                payload = summarize_stage(rewrite_payloads, sorted_failures(all_failures))
                summary_entry = _make_entry(payload, summary_key)
                _merge_cache_entry(cache, "summary", "__global__", summary_entry)
                status = "recomputed"
            stage_timings["summary"] += time.perf_counter() - started_stage
            summary_disposition = {
                "document_id": "__global__",
                "stage": "summary",
                "cache_key": summary_key,
                "fingerprint": summary_entry["fingerprint"],
                "status": status,
            }
            all_dispositions.append(summary_disposition)

        all_failures = sorted_failures(all_failures)
        canonical = self._build_canonical(
            effective_config, document_results, summary_entry, all_failures,
        )
        report = self._build_report(
            mode=mode,
            effective_config=effective_config,
            canonical=canonical,
            state=state,
            failures=all_failures,
            dispositions=all_dispositions,
            stage_timings=stage_timings,
            injection_reports=injection_reports,
            interrupted=interrupted,
            completed_document_ids=completed_document_ids,
            stale_error=stale_error,
            total_elapsed=time.perf_counter() - started,
        )

        state["last_completed_document_ids"] = list(completed_document_ids)
        state["last_document_cursor"] = last_document_cursor
        state["last_interrupted"] = interrupted
        state["last_commit_reason"] = "final_interrupted" if interrupted else "final"
        state["command_failures"] = deep_copy_json(engine.failures)
        state["cache"] = cache
        state["stage_versions"] = stage_versions(state)
        state["last_canonical_digest"] = report["canonical_digest"]
        state = self.snapshot_store.commit(state)
        if commit_observer is not None:
            commit_observer(state)
        report["snapshot_version"] = state["global_version"]
        report["snapshot_stage_versions"] = state.get("stage_versions", {})
        report["snapshot_generation"] = self.snapshot_store.current_generation()
        report["elapsed_seconds"] = round(time.perf_counter() - started, 6)
        return WorkbenchRun(canonical=canonical, report=report, snapshot_state=state,
                            interrupted=interrupted)

    def _process_document(self, document: dict[str, Any], config: WorkbenchConfig,
                          cache: dict[str, Any], injection_entries: list[dict[str, Any]],
                          stop_event: threading.Event,
                          document_delay_seconds: float = 0.0) -> DocumentTaskResult:
        document_id = document["document_id"]
        task_controller = InjectionController([
            entry for entry in injection_entries
            if entry.get("document_id", entry.get("doc_id")) == document_id
        ])
        result = DocumentTaskResult(document_id=document_id, success=True)
        current_document = document
        current_config = config
        upstream_fingerprint = content_fingerprint(current_document)
        stage_operations = (
            ("split", split_stage),
            ("scoring", scoring_stage),
            ("selection", selection_stage),
            ("rewrite", rewrite_stage),
        )
        payloads: dict[str, dict[str, Any]] = {}
        try:
            if document_delay_seconds > 0:
                time.sleep(document_delay_seconds)
            for stage, stage_function in stage_operations:
                injected_document, injected_config, injection_report = task_controller.apply(
                    current_document, current_config, stage,
                )
                if injection_report is not None:
                    result.injection_reports.append(injection_report)
                current_document, current_config = injected_document, injected_config
                force = injection_report is not None and injection_report.get("hit", False)
                cache_key = make_cache_key(stage, upstream_fingerprint, current_config, document_id)
                stage_started = time.perf_counter()
                cached = None if force else _cache_get(cache, stage, document_id, cache_key)
                if cached is not None:
                    entry = cached
                    payload = entry["payload"]
                    status = "reused"
                else:
                    if stage == "split":
                        payload = stage_function(current_document, current_config)
                    elif stage == "scoring":
                        payload = stage_function(current_document, payloads["split"], current_config)
                    elif stage == "selection":
                        payload = stage_function(current_document, payloads["scoring"], current_config)
                    else:
                        payload = stage_function(current_document, payloads["selection"], current_config)
                    entry = _make_entry(payload, cache_key)
                    status = "recomputed"
                elapsed = time.perf_counter() - stage_started
                result.timings[stage] = elapsed
                result.entries[stage] = entry
                payloads[stage] = payload
                result.dispositions.append({
                    "document_id": document_id,
                    "stage": stage,
                    "cache_key": cache_key,
                    "fingerprint": entry["fingerprint"],
                    "status": status,
                })
                upstream_fingerprint = entry["fingerprint"]
        except WorkbenchError as exc:
            failure = exc.to_failure()
            failure["document_id"] = document_id
            if result.injection_reports:
                active = result.injection_reports[-1]
                if active.get("hit"):
                    failure["injection_id"] = active["id"]
                    active["code"] = failure["code"]
                    active["code_match"] = (
                        active.get("expected_code") is None
                        or active.get("expected_code") == failure["code"]
                    )
            result.success = False
            result.failure = failure
            if not result.dispositions:
                result.dispositions.append({
                    "document_id": document_id,
                    "stage": failure["stage"],
                    "cache_key": "",
                    "fingerprint": "",
                    "status": "failed",
                })
            return result
        except Exception as exc:
            stage = _last_stage(result.dispositions)
            code = "E_DOC_INVALID" if stage in {"split", "scoring"} else "E_CONFIG_INVALID"
            result.success = False
            result.failure = {
                "document_id": document_id,
                "stage": stage,
                "code": code,
                "exception_type": type(exc).__name__,
                "message": str(exc),
            }
            if result.injection_reports:
                active = result.injection_reports[-1]
                if active.get("hit"):
                    result.failure["injection_id"] = active["id"]
                    active["code"] = code
                    active["code_match"] = (
                        active.get("expected_code") is None
                        or active.get("expected_code") == code
                    )
            return result
        return result

    def _snapshot_stale_reasons(self, state: dict[str, Any],
                                command_records: list[ParsedRecord],
                                expect_version: int | None,
                                base_documents_hash: str) -> list[str]:
        reasons: list[str] = []
        if expect_version is not None and int(state.get("global_version", -1)) != expect_version:
            reasons.append("global_version_mismatch")
        if state.get("config_hash") != self.config.base_config_hash:
            reasons.append("config_hash_mismatch")
        if state.get("base_documents_hash") != base_documents_hash:
            reasons.append("documents_hash_mismatch")
        cursor = int(state.get("applied_command_count", 0))
        if state.get("command_prefix_hash") != _command_prefix_hash(command_records, cursor):
            reasons.append("command_prefix_mismatch")
        return reasons

    def _load_state_documents(self, state: dict[str, Any]) -> dict[str, dict[str, Any]]:
        raw_documents = state.get("documents", {})
        if not isinstance(raw_documents, dict):
            raise SnapshotStale("快照 documents 结构非法")
        documents: dict[str, dict[str, Any]] = {}
        for document_id, raw_document in raw_documents.items():
            try:
                document = normalize_document(raw_document, stage="load")
            except DocumentInvalid as exc:
                raise SnapshotStale("快照包含非法文档，拒绝复用",
                                    details={"document_id": document_id, "reason": exc.message}) from exc
            documents[document_id] = document
        return documents

    def _base_state(self, *, documents: dict[str, dict[str, Any]], overrides: dict[str, Any],
                    batch_history: list[dict[str, Any]], command_failures: list[dict[str, Any]],
                    cache: dict[str, Any], effective_config: WorkbenchConfig,
                    base_documents_hash: str, applied_commands: int,
                    command_records: list[ParsedRecord]) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "config_hash": self.config.base_config_hash,
            "base_documents_hash": base_documents_hash,
            "synonym_table_version": effective_config.synonym_table_version,
            "effective_overrides": deep_copy_json(overrides),
            "applied_command_count": applied_commands,
            "command_prefix_hash": _command_prefix_hash(command_records, applied_commands),
            "documents": deep_copy_json(documents),
            "command_failures": deep_copy_json(command_failures),
            "batch_history": deep_copy_json(batch_history),
            "cache": deep_copy_json(cache),
            "stage_versions": {},
            "last_completed_document_ids": [],
            "last_document_cursor": "",
            "last_interrupted": False,
        }

    def _build_canonical(self, config: WorkbenchConfig,
                         document_results: dict[str, DocumentTaskResult],
                         summary_entry: dict[str, Any] | None,
                         failures: list[dict[str, Any]]) -> dict[str, Any]:
        documents = []
        stage_entries: list[dict[str, Any]] = []
        cache_keys: dict[str, dict[str, str]] = {stage: {} for stage in PIPELINE_STAGES}
        for document_id in sorted(document_results):
            result = document_results[document_id]
            for stage in DOCUMENT_STAGES:
                entry = result.entries.get(stage)
                if entry is None:
                    continue
                stage_entries.append({
                    "document_id": document_id,
                    "stage": stage,
                    "cache_key": entry["cache_key"],
                    "fingerprint": entry["fingerprint"],
                })
                cache_keys[stage][document_id] = entry["cache_key"]
            if result.success and all(stage in result.entries for stage in DOCUMENT_STAGES):
                documents.append({
                    "document_id": document_id,
                    "split": result.entries["split"]["payload"],
                    "scoring": result.entries["scoring"]["payload"],
                    "selection": result.entries["selection"]["payload"],
                    "rewrite": result.entries["rewrite"]["payload"],
                })
        if summary_entry is not None:
            stage_entries.append({
                "document_id": "__global__",
                "stage": "summary",
                "cache_key": summary_entry["cache_key"],
                "fingerprint": summary_entry["fingerprint"],
            })
            cache_keys["summary"]["__global__"] = summary_entry["cache_key"]
        summary_payload = summary_entry["payload"] if summary_entry is not None else {
            "documents": [],
            "stats": {
                "document_count": 0,
                "failed_document_count": len({item.get("document_id", "") for item in failures
                                               if item.get("document_id") and item.get("stage") in {
                                                   "load", "split", "scoring", "selection",
                                                   "rewrite", "summary",
                                               }}),
                "summary_sentence_count": 0,
                "replacement_count": 0,
                "truncation_count": 0,
                "summary_character_count": 0,
            },
        }
        canonical = {
            "schema_version": 1,
            "config": {
                "base_config_hash": config.base_config_hash,
                "synonym_table_version": config.synonym_table_version,
                "effective": config.overrides_projection(),
            },
            "documents": documents,
            "summary": summary_payload,
            "failures": failures,
            "stage_entries": stage_entries,
            "invalidation_reuse": stage_entries,
            "cache_keys": cache_keys,
        }
        return canonical

    def _build_report(self, *, mode: str, effective_config: WorkbenchConfig,
                      canonical: dict[str, Any], state: dict[str, Any],
                      failures: list[dict[str, Any]], dispositions: list[dict[str, Any]],
                      stage_timings: dict[str, float],
                      injection_reports: list[dict[str, Any]], interrupted: bool,
                      completed_document_ids: list[str], stale_error: SnapshotStale | None,
                      total_elapsed: float) -> dict[str, Any]:
        ordered_dispositions = sorted(dispositions, key=lambda item: (
            item.get("document_id", ""), STAGE_ORDER.get(item.get("stage", ""), 99),
            item.get("cache_key", ""),
        ))
        reuse_count = sum(1 for item in ordered_dispositions if item["status"] == "reused")
        recompute_count = sum(1 for item in ordered_dispositions if item["status"] == "recomputed")
        failed_stage_count = sum(1 for item in ordered_dispositions if item["status"] == "failed")
        injection_output = self._merge_injection_reports(injection_reports)
        return {
            "schema_version": 1,
            "mode": mode,
            "config_hash": self.config.base_config_hash,
            "synonym_table_version": effective_config.synonym_table_version,
            "snapshot_version": state.get("global_version", 0),
            "snapshot_stage_versions": state.get("stage_versions", {}),
            "stale": stale_error is not None,
            "stale_error": stale_error.to_failure() if stale_error else None,
            "interrupted": interrupted,
            "completed_document_ids": completed_document_ids,
            "document_cursor": completed_document_ids[-1] if completed_document_ids else "",
            "summary": canonical["summary"]["stats"],
            "failures": failures,
            "failure_count": len(failures),
            "invalidation_reuse": ordered_dispositions,
            "reuse_counts": {
                "checked": len(ordered_dispositions),
                "reused": reuse_count,
                "recomputed": recompute_count,
                "failed": failed_stage_count,
            },
            "stage_timings_ms": {stage: round(value * 1000.0, 6)
                                 for stage, value in stage_timings.items()},
            "total_elapsed_ms": round(total_elapsed * 1000.0, 6),
            "injections": injection_output,
            "injection_hit_count": sum(1 for item in injection_output if item.get("hit")),
            "canonical_digest": content_fingerprint(canonical),
            "canonical_json_sha256": content_fingerprint(canonical_json(canonical)),
        }

    def _merge_injection_reports(self, observed: list[dict[str, Any]]) -> list[dict[str, Any]]:
        by_id: dict[str, dict[str, Any]] = {}
        for item in observed:
            by_id[item["id"]] = item
        output = []
        for entry in self.injection_entries:
            injection_id = str(entry.get("id", "INJ-UNKNOWN"))
            if injection_id in by_id:
                output.append(by_id[injection_id])
            else:
                output.append({
                    "id": injection_id,
                    "document_id": entry.get("document_id", entry.get("doc_id", "")),
                    "stage": entry.get("stage", ""),
                    "kind": entry.get("kind", entry.get("inject", "")),
                    "expected_code": entry.get("code", entry.get("expected_code")),
                    "hit": False,
                    "status": "unreached",
                })
        return output


def make_cache_key(stage: str, upstream_fingerprint: str, config: WorkbenchConfig,
                   document_id: str = "") -> str:
    return content_fingerprint({
        "schema": 1,
        "stage": stage,
        "document_id": document_id,
        "upstream": upstream_fingerprint,
        "config": config.stage_config_fingerprint(stage),
        "synonym_table_version": (
            config.synonym_table_version if stage in {"rewrite", "summary"} else None
        ),
    })


def _make_entry(payload: dict[str, Any], cache_key: str) -> dict[str, Any]:
    return {
        "cache_key": cache_key,
        "fingerprint": content_fingerprint(payload),
        "payload": payload,
    }


def _cache_get(cache: dict[str, Any], stage: str, document_id: str,
               cache_key: str) -> dict[str, Any] | None:
    entry = cache.get(stage, {}).get(document_id, {}).get(cache_key)
    if not isinstance(entry, dict):
        return None
    if entry.get("cache_key") != cache_key:
        return None
    payload = entry.get("payload")
    if not isinstance(payload, dict) or content_fingerprint(payload) != entry.get("fingerprint"):
        return None
    return entry


def _merge_cache_entry(cache: dict[str, Any], stage: str, document_id: str,
                       entry: dict[str, Any]) -> None:
    cache.setdefault(stage, {}).setdefault(document_id, {})[entry["cache_key"]] = entry


def sorted_failures(failures: list[dict[str, Any]]) -> list[dict[str, Any]]:
    unique: dict[str, dict[str, Any]] = {}
    for failure in failures:
        key = canonical_json(failure)
        unique[key] = failure
    return sorted(unique.values(), key=lambda item: (
        item.get("document_id", ""),
        STAGE_ORDER.get(item.get("stage", ""), 99),
        item.get("code", ""),
        int(item.get("command_index", -1)) if isinstance(item.get("command_index"), int) else -1,
        item.get("injection_id", "") or "",
        item.get("message", ""),
    ))


def _command_prefix_hash(records: list[ParsedRecord], cursor: int) -> str:
    prefix = []
    for record in records:
        if record.index >= cursor:
            break
        if record.error is not None:
            prefix.append({"index": record.index, "error": record.error})
        else:
            prefix.append({"index": record.index, "value": record.value})
    return content_fingerprint(prefix)


def _last_stage(dispositions: list[dict[str, Any]]) -> str:
    if not dispositions:
        return "split"
    return dispositions[-1].get("stage", "split")
