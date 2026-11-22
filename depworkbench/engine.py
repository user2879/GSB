from __future__ import annotations

import copy
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Iterable

from .canonical import sha256_json, stable_task_key
from .errors import SimulatedCrash, ValidationError
from .io import (
    load_project,
    write_json_atomic,
    write_jsonl_atomic,
)
from .models import (
    Config,
    Failure,
    Injection,
    Model,
    Project,
    Sentence,
    config_from_dict,
    copy_model_with_nonfinite,
    model_from_dict,
    sentence_from_dict,
)
from .output import build_sentence_products, semantic_sentence_payload
from .parser import (
    ParseOutcome,
    initial_state,
    parse_sentence,
    sentence_load_key,
    state_from_dict,
    state_to_dict,
)
from .quantize import quantize_model
from .snapshot import SnapshotStore


class Engine:
    def __init__(self, project: Project, snapshot_dir: str | Path | None = None):
        self.project = project
        self.snapshot = SnapshotStore(snapshot_dir) if snapshot_dir is not None else None

    def run(
        self,
        *,
        mode_override: str | None = None,
        resume: bool = False,
        fresh: bool = False,
        expected_version: int | None = None,
        crash_after_sentences: int | None = None,
        crash_partial_steps: int | None = None,
        commit_snapshots: bool = True,
    ) -> dict[str, Any]:
        total_start = time.perf_counter()
        config = self._config_with_mode(self.project.config, mode_override)
        model = self.project.model
        quantized = quantize_model(model, config)
        caches: dict[str, dict[str, dict[str, Any]]] = {}
        snapshot_data: dict[str, Any] | None = None
        stale_event: dict[str, Any] | None = None
        if not fresh and self.snapshot is not None:
            try:
                snapshot_data = self.snapshot.load_latest()
            except ValidationError as exc:
                stale_event = {
                    "code": "E_SNAPSHOT_STALE",
                    "message": exc.message,
                    "action": "reinitialize",
                }
                self.snapshot.reinitialize()
                snapshot_data = None
            if snapshot_data is not None and resume:
                manifest = snapshot_data["manifest"]
                mismatch = self._resume_mismatch(manifest, model, config, expected_version)
                if mismatch:
                    stale_event = {
                        "code": "E_SNAPSHOT_STALE",
                        "message": mismatch,
                        "action": "reinitialize",
                        "snapshot_version": manifest.get("global_version"),
                    }
                    self.snapshot.reinitialize()
                    snapshot_data = None
            if snapshot_data is not None:
                parse_stage = snapshot_data.get("parse") or {}
                loaded_caches = parse_stage.get("caches", {}) if isinstance(parse_stage, dict) else {}
                if isinstance(loaded_caches, dict):
                    caches = copy.deepcopy(loaded_caches)
        if resume and expected_version is not None and snapshot_data is None and stale_event is None:
            stale_event = {
                "code": "E_SNAPSHOT_STALE",
                "message": f"expected snapshot version {expected_version} but no snapshot exists",
                "action": "reinitialize",
            }

        load_start = time.perf_counter()
        active_sentences: dict[str, Sentence] = {
            sentence.sentence_id: sentence for sentence in self.project.sentences
        }
        failures: list[Failure] = list(self.project.failures)
        injection_report = {
            injection.injection_id: {
                "injection_id": injection.injection_id,
                "sentence_id": injection.sentence_id,
                "stage": injection.stage,
                "kind": injection.kind,
                "hit": False,
                "business_code": None,
                "detail": None,
            }
            for injection in self.project.injections
        }

        for sentence_id in sorted(list(active_sentences)):
            sentence = active_sentences[sentence_id]
            for injection in self._injections_for(sentence_id, "sentence_load"):
                if injection.kind != "empty_words":
                    continue
                empty_sentence = sentence.with_words((), ())
                failure = Failure(
                    sentence_id=sentence_id,
                    code="E_SENT_INVALID",
                    message="injected empty word sequence",
                    line_no=sentence.line_no,
                    injection_id=injection.injection_id,
                    stage="sentence_load",
                )
                failures.append(failure)
                injection_report[injection.injection_id].update(
                    {"hit": True, "business_code": "E_SENT_INVALID", "detail": "empty_words"}
                )
                active_sentences[sentence_id] = empty_sentence
                break
        load_injected_failure_ids = {failure.sentence_id for failure in failures if failure.stage == "sentence_load"}
        load_closed_ids = set(load_injected_failure_ids)
        load_failures = list(failures)

        # Sentences closed by a load-stage injection are handled before scheduling.
        active_for_parse = {
            sid: sentence for sid, sentence in active_sentences.items() if sid not in load_closed_ids
        }
        load_elapsed = (time.perf_counter() - load_start) * 1000.0

        schedule_start = time.perf_counter()
        remaining_ids = set(active_for_parse)
        pending_priority: set[str] = set()
        budgets: list[dict[str, Any]] = []
        main_results: dict[str, dict[str, Any]] = {}
        effective_sentences: dict[str, Sentence] = {}
        products: dict[str, dict[str, Any]] = {}
        processed_ids: set[str] = set()
        batch_index = 0
        total_transition_steps = 0
        while remaining_ids:
            batch_index += 1
            order = sorted(
                remaining_ids,
                key=lambda sid: (
                    0 if sid in pending_priority else 1,
                    sid,
                ),
            )
            budget_left = config.batch_parse_point_budget
            selected: list[str] = []
            deferred: list[str] = []
            entries: list[dict[str, Any]] = []
            for rank, sid in enumerate(order, start=1):
                cost = self._parse_point_cost(active_for_parse[sid])
                if cost <= budget_left:
                    budget_left -= cost
                    selected.append(sid)
                    entries.append(
                        {
                            "sentence_id": sid,
                            "selection_rank": rank,
                            "cost": cost,
                            "decision": "SELECTED",
                            "remaining_after": budget_left,
                        }
                    )
                else:
                    deferred.append(sid)
                    entries.append(
                        {
                            "sentence_id": sid,
                            "selection_rank": rank,
                            "cost": cost,
                            "decision": "DEFERRED",
                            "code": "E_PARSE_BUDGET_PENDING",
                            "remaining_after": budget_left,
                        }
                    )
            if not selected:
                budgets.append(
                    {
                        "batch_index": batch_index,
                        "batch_budget": config.batch_parse_point_budget,
                        "used_points": 0,
                        "remaining_points": budget_left,
                        "selection_order": [],
                        "entries": entries,
                    }
                )
                pending_priority = set(deferred)
                break

            batch_start_processed = len(processed_ids)
            chunk_size = config.snapshot_commit_interval_sentences
            for chunk_start in range(0, len(selected), chunk_size):
                chunk = selected[chunk_start:chunk_start + chunk_size]
                chunk_results = self._run_parse_chunk(
                    chunk,
                    active_for_parse,
                    caches,
                    model,
                    quantized,
                    config,
                )
                for sid in chunk:
                    result = chunk_results[sid]
                    main_results[sid] = result
                    effective_sentences[sid] = result["sentence"]
                    processed_ids.add(sid)
                    total_transition_steps += len(result.get("outcome").steps) if result.get("outcome") else 0
                    self._apply_injection_updates(injection_report, result.get("injection_updates", []))
                    if result["status"] == "OK":
                        outcome = result["outcome"]
                        caches.setdefault(sid, {})[config.mode] = outcome.cache
                        products[sid] = build_sentence_products(
                            outcome,
                            result["sentence"],
                            config,
                            sentence_load_key(result["sentence"], model, config),
                        )
                crash_now = (
                    crash_after_sentences is not None
                    and len(processed_ids) >= crash_after_sentences
                )
                if crash_now and crash_partial_steps is not None:
                    for sid in reversed(sorted(chunk)):
                        mode_cache = caches.get(sid, {}).get(config.mode)
                        if mode_cache is None or not mode_cache.get("steps"):
                            continue
                        keep = max(0, min(crash_partial_steps, len(mode_cache["steps"]) - 1))
                        mode_cache["steps"] = copy.deepcopy(mode_cache["steps"][:keep])
                        mode_cache["completed"] = False
                        mode_cache["error_code"] = None
                        mode_cache["error_message"] = None
                        mode_cache["final_state"] = (
                            state_to_dict(state_from_dict(mode_cache["steps"][-1]["state_after"]))
                            if keep
                            else state_to_dict(initial_state(len(active_for_parse[sid].words)))
                        )
                        products.pop(sid, None)
                        processed_ids.discard(sid)
                        break
                if self.snapshot is not None and commit_snapshots:
                    self._commit_snapshot(
                        caches=caches,
                        products=products,
                        active_sentences=active_sentences,
                        load_failures=load_failures,
                        processed_ids=processed_ids,
                        config=config,
                        mode=config.mode,
                        stage_status="partial" if remaining_ids or len(processed_ids) < len(active_for_parse) else "complete",
                    )
                if crash_now:
                    raise SimulatedCrash(
                        f"simulated crash after {len(processed_ids)} processed sentences"
                    )
            used_points = config.batch_parse_point_budget - budget_left
            budgets.append(
                {
                    "batch_index": batch_index,
                    "batch_budget": config.batch_parse_point_budget,
                    "used_points": used_points,
                    "remaining_points": budget_left,
                    "selection_order": selected,
                    "entries": entries,
                    "processed_until": sorted(processed_ids),
                }
            )
            remaining_ids.difference_update(selected)
            pending_priority = set(deferred)
            if len(processed_ids) == batch_start_processed:
                break
        schedule_elapsed = (time.perf_counter() - schedule_start) * 1000.0

        pending_ids = sorted(remaining_ids)
        pending = [
            {
                "sentence_id": sid,
                "code": "E_PARSE_BUDGET_PENDING",
                "message": "sentence remains in the next-batch priority queue",
                "cost": self._parse_point_cost(active_for_parse[sid]),
            }
            for sid in pending_ids
        ]

        alternate_start = time.perf_counter()
        alternate_mode = "exhaustive" if config.mode == "greedy" else "greedy"
        alternate_config = self._config_with_mode(config, alternate_mode)
        alternate_quantized = quantize_model(model, alternate_config)
        alternating_ids = [
            sid for sid in sorted(main_results) if main_results[sid]["status"] == "OK"
        ]
        alternate_results: dict[str, dict[str, Any]] = {}
        for chunk_start in range(0, len(alternating_ids), max(1, config.snapshot_commit_interval_sentences)):
            chunk = alternating_ids[
                chunk_start:chunk_start + max(1, config.snapshot_commit_interval_sentences)
            ]
            chunk_results = self._run_alternate_chunk(
                chunk,
                effective_sentences,
                caches,
                model,
                alternate_quantized,
                alternate_config,
            )
            alternate_results.update(chunk_results)
            for sid, result in chunk_results.items():
                if result.get("status") == "OK" and result.get("outcome") is not None:
                    caches.setdefault(sid, {})[alternate_mode] = result["outcome"].cache
        alternate_elapsed = (time.perf_counter() - alternate_start) * 1000.0

        arc_start = time.perf_counter()
        arcs: list[dict[str, Any]] = []
        for sid in sorted(products):
            arcs.extend(products[sid]["arcs"])
        arcs.sort(key=lambda arc: (arc["sentence_id"], arc["head"], arc["dependent"]))
        arc_elapsed = (time.perf_counter() - arc_start) * 1000.0

        tree_start = time.perf_counter()
        trees: list[dict[str, Any]] = []
        for sid in sorted(products):
            trees.append(
                {
                    "sentence_id": sid,
                    "flat": products[sid]["flat_tree"],
                    "nested": products[sid]["nested_tree"],
                    "tree_key": products[sid]["tree_key"],
                }
            )
        trees.sort(key=lambda tree: tree["sentence_id"])
        tree_elapsed = (time.perf_counter() - tree_start) * 1000.0

        sentence_reports: list[dict[str, Any]] = []
        for sid in sorted(active_sentences):
            sentence = active_sentences[sid]
            if sid in main_results:
                result = main_results[sid]
                if result["status"] == "OK":
                    sentence_reports.append(
                        self._ok_sentence_report(
                            result["sentence"],
                            result["outcome"],
                            products[sid],
                            config,
                            model,
                            result.get("injection_updates", []),
                        )
                    )
                else:
                    sentence_reports.append(
                        self._failed_sentence_report(
                            result["sentence"],
                            result["failure"],
                            config,
                            result.get("injection_updates", []),
                        )
                    )
            elif sid in remaining_ids:
                sentence_reports.append(
                    {
                        "sentence_id": sid,
                        "status": "PENDING",
                        "words": list(sentence.words),
                        "pos_tags": list(sentence.pos_tags),
                        "error_code": "E_PARSE_BUDGET_PENDING",
                        "action_sequence": [],
                        "action_names": [],
                        "action_sequence_length": 0,
                        "steps": [],
                        "arcs": [],
                        "tree": None,
                        "reference_metrics": {"available": sentence.gold_heads is not None},
                        "cache_ledger": None,
                        "task_key": stable_task_key(sid, 0, "parse"),
                    }
                )
            elif sid in load_closed_ids:
                failure = next(
                    (item for item in load_failures if item.sentence_id == sid),
                    Failure(
                        sentence_id=sid,
                        code="E_SENT_INVALID",
                        message="sentence closed during load validation",
                        line_no=sentence.line_no,
                        stage="sentence_load",
                    ),
                )
                sentence_reports.append(
                    self._failed_sentence_report(
                        sentence,
                        failure,
                        config,
                        [
                            {
                                "injection_id": failure.injection_id,
                                "business_code": failure.code,
                                "detail": failure.message,
                            }
                        ]
                        if failure.injection_id
                        else [],
                    )
                )

        for sid in sorted(main_results):
            result = main_results[sid]
            if result["status"] == "FAILED":
                failures.append(result["failure"])

        sentence_reports.sort(key=lambda item: item["sentence_id"])
        failures.sort(key=lambda failure: failure.sort_key)
        mode_differences = self._mode_differences(main_results, alternate_results, config.mode, alternate_mode)
        boundary_ties = self._boundary_ties(sentence_reports)

        snapshot_start = time.perf_counter()
        final_manifest: dict[str, Any] | None = None
        if self.snapshot is not None and commit_snapshots:
            final_manifest = self._commit_snapshot(
                caches=caches,
                products=products,
                active_sentences=active_sentences,
                load_failures=load_failures,
                processed_ids=processed_ids,
                config=config,
                mode=config.mode,
                stage_status="complete" if not pending_ids else "budget_pending",
            )
        snapshot_elapsed = (time.perf_counter() - snapshot_start) * 1000.0

        injection_items = sorted(
            injection_report.values(),
            key=lambda item: (item["sentence_id"], item["injection_id"]),
        )
        report: dict[str, Any] = {
            "schema_version": 1,
            "mode": config.mode,
            "model_hash": model.fingerprint,
            "config_hash": config.fingerprint,
            "corpus_hash": self.project.corpus_fingerprint,
            "quantization_fingerprint": config.quantization_fingerprint,
            "config": config.to_dict(),
            "run_profile": {
                "max_threads": config.max_threads,
                "batch_parse_point_budget": config.batch_parse_point_budget,
                "snapshot_commit_interval_sentences": config.snapshot_commit_interval_sentences,
                "point_unit": "words",
            },
            "sentences": sentence_reports,
            "failures": [failure.to_dict() for failure in failures],
            "pending": pending,
            "budget": {
                "batch_budget": config.batch_parse_point_budget,
                "point_unit": "words",
                "batches": budgets,
                "scheduled_transition_steps": total_transition_steps,
            },
            "boundary_ties": boundary_ties,
            "mode_differences": mode_differences,
            "arcs": arcs,
            "trees": trees,
            "injections": injection_items,
            "snapshot_stale": stale_event,
            "snapshot": self._snapshot_summary(final_manifest),
            "stage_timings_ms": {
                "load_validation": round(load_elapsed, 3),
                "scheduled_parse": round(schedule_elapsed, 3),
                "alternate_mode": round(alternate_elapsed, 3),
                "arc_aggregation": round(arc_elapsed, 3),
                "tree_output": round(tree_elapsed, 3),
                "snapshot_commit": round(snapshot_elapsed, 3),
                "total": round((time.perf_counter() - total_start) * 1000.0, 3),
            },
        }
        report["semantic_digest"] = self._semantic_digest(report)
        return report

    def _resume_mismatch(
        self,
        manifest: dict[str, Any],
        model: Model,
        config: Config,
        expected_version: int | None,
    ) -> str | None:
        if expected_version is not None and int(manifest.get("global_version", -1)) != expected_version:
            return (
                f"expected global_version {expected_version}, snapshot has "
                f"{manifest.get('global_version')}"
            )
        if manifest.get("model_hash") != model.fingerprint:
            return "model hash differs from snapshot"
        if manifest.get("config_hash") != config.fingerprint:
            return "config hash differs from snapshot"
        if manifest.get("corpus_hash") != self.project.corpus_fingerprint:
            return "corpus hash differs from snapshot"
        return None

    def _config_with_mode(self, config: Config, mode_override: str | None) -> Config:
        mode = mode_override or config.mode
        if mode == config.mode and config.raw.get("mode") == mode:
            return config
        raw = copy.deepcopy(config.raw)
        raw["mode"] = mode
        return config_from_dict(raw)

    def _injections_for(self, sentence_id: str, stage: str) -> list[Injection]:
        return sorted(
            [
                injection
                for injection in self.project.injections
                if injection.sentence_id == sentence_id and injection.stage == stage
            ],
            key=lambda injection: injection.injection_id,
        )

    @staticmethod
    def _parse_point_cost(sentence: Sentence) -> int:
        return max(1, len(sentence.words))

    def _run_parse_chunk(
        self,
        chunk: list[str],
        sentences: dict[str, Sentence],
        caches: dict[str, dict[str, dict[str, Any]]],
        model: Model,
        quantized: Any,
        config: Config,
    ) -> dict[str, dict[str, Any]]:
        workers = max(1, min(config.max_threads, len(chunk)))
        results: dict[str, dict[str, Any]] = {}
        with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="dep-parse") as executor:
            future_by_sid = {
                sid: executor.submit(
                    self._parse_job,
                    sentences[sid],
                    caches.get(sid, {}).get(config.mode),
                    model,
                    quantized,
                    config,
                )
                for sid in sorted(chunk)
            }
            for sid in sorted(future_by_sid):
                future = future_by_sid[sid]
                try:
                    results[sid] = future.result()
                except Exception as exc:  # isolated by sentence, never reordered
                    results[sid] = {
                        "status": "FAILED",
                        "sentence": sentences[sid],
                        "outcome": None,
                        "failure": Failure(
                            sentence_id=sid,
                            code="E_PARSE_EXCEPTION",
                            message=f"unexpected parse failure: {type(exc).__name__}: {exc}",
                            line_no=sentences[sid].line_no,
                            stage="parse",
                        ),
                        "injection_updates": [],
                    }
        return results

    def _parse_job(
        self,
        sentence: Sentence,
        old_cache: dict[str, Any] | None,
        model: Model,
        quantized: Any,
        config: Config,
    ) -> dict[str, Any]:
        effective = sentence
        updates: list[dict[str, Any]] = []
        terminal_failure: Failure | None = None
        for injection in self._injections_for(sentence.sentence_id, "parse"):
            if injection.kind == "empty_words":
                effective = effective.with_words((), ())
                terminal_failure = Failure(
                    sentence_id=sentence.sentence_id,
                    code="E_SENT_INVALID",
                    message="injected empty word sequence",
                    line_no=sentence.line_no,
                    injection_id=injection.injection_id,
                    stage="parse",
                )
                updates.append(self._injection_update(injection, "E_SENT_INVALID", "empty_words"))
                break
            if injection.kind == "unknown_pos":
                if not effective.words:
                    code = "E_INJECTION_TARGET_INVALID"
                    message = "cannot inject POS into an empty sentence"
                elif injection.token_index >= len(effective.words):
                    code = "E_INJECTION_TARGET_INVALID"
                    message = f"token index {injection.token_index} is out of range"
                else:
                    effective = effective.with_pos(injection.token_index, injection.pos_tag)
                    code = "E_FEATURE_UNDECLARED"
                    message = f"injected undeclared POS {injection.pos_tag!r}"
                updates.append(self._injection_update(injection, code, message))
                continue
            if injection.kind == "nonfinite_weight":
                bad_raw = copy_model_with_nonfinite(
                    model.raw,
                    injection.weight_action,
                    injection.weight_index,
                )
                try:
                    bad_model = model_from_dict(
                        bad_raw,
                        vocabulary=list(model.vocabulary),
                        pos_tags=list(model.pos_tags),
                    )
                    quantize_model(bad_model, config)
                    code = "E_MODEL_INVALID"
                    message = "injected non-finite action weight was not rejected"
                except ValidationError as exc:
                    code = exc.code
                    message = exc.message
                terminal_failure = Failure(
                    sentence_id=sentence.sentence_id,
                    code=code,
                    message=message,
                    line_no=sentence.line_no,
                    injection_id=injection.injection_id,
                    stage="parse",
                )
                updates.append(self._injection_update(injection, code, message))
                break
            if injection.kind == "zero_max_steps":
                bad_raw = copy.deepcopy(config.raw)
                bad_raw["max_transition_steps"] = 0
                try:
                    config_from_dict(bad_raw)
                    code = "E_CONFIG_INVALID"
                    message = "injected zero max_transition_steps was not rejected"
                except ValidationError as exc:
                    code = exc.code
                    message = exc.message
                terminal_failure = Failure(
                    sentence_id=sentence.sentence_id,
                    code=code,
                    message=message,
                    line_no=sentence.line_no,
                    injection_id=injection.injection_id,
                    stage="parse",
                )
                updates.append(self._injection_update(injection, code, message))
                break
        if terminal_failure is not None:
            return {
                "status": "FAILED",
                "sentence": effective,
                "outcome": None,
                "failure": terminal_failure,
                "injection_updates": updates,
            }
        try:
            outcome = parse_sentence(effective, model, quantized, config, old_cache)
        except ValidationError as exc:
            failure = Failure(
                sentence_id=sentence.sentence_id,
                code=exc.code,
                message=exc.message,
                line_no=sentence.line_no,
                stage="parse",
                injection_id=updates[0]["injection_id"] if updates else None,
            )
            for update in updates:
                if update["business_code"] == "E_FEATURE_UNDECLARED":
                    update["business_code"] = exc.code
                    update["detail"] = exc.message
            return {
                "status": "FAILED",
                "sentence": effective,
                "outcome": None,
                "failure": failure,
                "injection_updates": updates,
            }
        if not outcome.completed:
            failure = Failure(
                sentence_id=sentence.sentence_id,
                code=outcome.error_code or "E_PARSE_DEAD_END",
                message=outcome.error_message or "parse did not complete",
                line_no=sentence.line_no,
                stage="parse",
                injection_id=updates[0]["injection_id"] if updates else None,
            )
            return {
                "status": "FAILED",
                "sentence": effective,
                "outcome": outcome,
                "failure": failure,
                "injection_updates": updates,
            }
        for update in updates:
            if update["business_code"] == "E_FEATURE_UNDECLARED":
                update["detail"] = "parse completed despite injected feature"
        return {
            "status": "OK",
            "sentence": effective,
            "outcome": outcome,
            "failure": None,
            "injection_updates": updates,
        }

    def _run_alternate_chunk(
        self,
        chunk: list[str],
        sentences: dict[str, Sentence],
        caches: dict[str, dict[str, dict[str, Any]]],
        model: Model,
        quantized: Any,
        config: Config,
    ) -> dict[str, dict[str, Any]]:
        if not chunk:
            return {}
        workers = max(1, min(config.max_threads, len(chunk)))
        results: dict[str, dict[str, Any]] = {}
        with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="dep-alt") as executor:
            futures = {
                sid: executor.submit(
                    self._parse_alternate_job,
                    sentences[sid],
                    caches.get(sid, {}).get(config.mode),
                    model,
                    quantized,
                    config,
                )
                for sid in sorted(chunk)
            }
            for sid in sorted(futures):
                try:
                    results[sid] = futures[sid].result()
                except Exception as exc:
                    results[sid] = {
                        "status": "FAILED",
                        "outcome": None,
                        "error_code": "E_PARSE_EXCEPTION",
                        "error_message": str(exc),
                    }
        return results

    def _parse_alternate_job(
        self,
        sentence: Sentence,
        old_cache: dict[str, Any] | None,
        model: Model,
        quantized: Any,
        config: Config,
    ) -> dict[str, Any]:
        try:
            outcome = parse_sentence(sentence, model, quantized, config, old_cache)
        except ValidationError as exc:
            return {
                "status": "FAILED",
                "outcome": None,
                "error_code": exc.code,
                "error_message": exc.message,
            }
        if not outcome.completed:
            return {
                "status": "FAILED",
                "outcome": outcome,
                "error_code": outcome.error_code,
                "error_message": outcome.error_message,
            }
        return {"status": "OK", "outcome": outcome, "error_code": None, "error_message": None}

    @staticmethod
    def _injection_update(injection: Injection, code: str, detail: str) -> dict[str, Any]:
        return {
            "injection_id": injection.injection_id,
            "business_code": code,
            "detail": detail,
            "hit": True,
        }

    @staticmethod
    def _apply_injection_updates(
        report: dict[str, dict[str, Any]], updates: list[dict[str, Any]]
    ) -> None:
        for update in updates:
            item = report.get(update["injection_id"])
            if item is not None:
                item["hit"] = True
                item["business_code"] = update["business_code"]
                item["detail"] = update["detail"]

    def _ok_sentence_report(
        self,
        sentence: Sentence,
        outcome: ParseOutcome,
        product: dict[str, Any],
        config: Config,
        model: Model,
        injection_updates: list[dict[str, Any]],
    ) -> dict[str, Any]:
        steps = copy.deepcopy(outcome.steps)
        return {
            "sentence_id": sentence.sentence_id,
            "status": "OK",
            "mode": outcome.mode,
            "words": list(sentence.words),
            "pos_tags": list(sentence.pos_tags),
            "sentence_fingerprint": sentence.fingerprint,
            "sentence_load_key": sentence_load_key(sentence, model, config),
            "task_key": stable_task_key(sentence.sentence_id, 0, "parse"),
            "action_sequence": list(outcome.final_state.action_codes),
            "action_names": list(outcome.final_state.actions),
            "action_sequence_length": len(outcome.final_state.actions),
            "final_score": outcome.final_state.total_score,
            "steps": steps,
            "arcs": product["arcs"],
            "arc_count": len(product["arcs"]),
            "tree": {
                "flat": product["flat_tree"],
                "nested": product["nested_tree"],
            },
            "tree_structure": product["nested_tree"],
            "arc_key": product["arc_key"],
            "tree_key": product["tree_key"],
            "reference_metrics": product["reference_metrics"],
            "cache_ledger": outcome.ledger,
            "pruned_paths": outcome.pruned_paths or [],
            "pruned_count_by_reason": outcome.pruned_count_by_reason or {},
            "pruned_omitted_count": outcome.pruned_omitted_count,
            "search_stats": outcome.search_stats or {},
            "injection_ids": [item["injection_id"] for item in injection_updates],
        }

    def _failed_sentence_report(
        self,
        sentence: Sentence,
        failure: Failure,
        config: Config,
        injection_updates: list[dict[str, Any]],
    ) -> dict[str, Any]:
        return {
            "sentence_id": sentence.sentence_id,
            "status": "FAILED",
            "mode": config.mode,
            "words": list(sentence.words),
            "pos_tags": list(sentence.pos_tags),
            "sentence_fingerprint": sentence.fingerprint,
            "task_key": stable_task_key(sentence.sentence_id, 0, "parse"),
            "error_code": failure.code,
            "error_message": failure.message,
            "action_sequence": [],
            "action_names": [],
            "action_sequence_length": 0,
            "steps": [],
            "arcs": [],
            "arc_count": 0,
            "tree": None,
            "reference_metrics": {"available": sentence.gold_heads is not None},
            "cache_ledger": None,
            "pruned_paths": [],
            "injection_ids": [item["injection_id"] for item in injection_updates],
        }

    def _mode_differences(
        self,
        main_results: dict[str, dict[str, Any]],
        alternate_results: dict[str, dict[str, Any]],
        main_mode: str,
        alternate_mode: str,
    ) -> dict[str, Any]:
        differences: list[dict[str, Any]] = []
        same_count = 0
        different_count = 0
        evaluated_difference_count = 0
        failed_count = 0
        for sid in sorted(main_results):
            main = main_results[sid]
            if main["status"] != "OK":
                continue
            alternate = alternate_results.get(sid)
            main_outcome = main["outcome"]
            if alternate is None or alternate.get("status") != "OK":
                failed_count += 1
                differences.append(
                    {
                        "sentence_id": sid,
                        "same": False,
                        "comparison_status": "FAILED",
                        "error_code": alternate.get("error_code") if alternate else "E_MODE_COMPARISON_MISSING",
                    }
                )
                continue
            alt_outcome = alternate["outcome"]
            main_actions = list(main_outcome.final_state.action_codes)
            alt_actions = list(alt_outcome.final_state.action_codes)
            first_difference = None
            for index, (left, right) in enumerate(zip(main_actions, alt_actions)):
                if left != right:
                    first_difference = index
                    break
            if first_difference is None and len(main_actions) != len(alt_actions):
                first_difference = min(len(main_actions), len(alt_actions))
            same = main_actions == alt_actions
            eval_main = (
                len(main_actions)
                if main_mode == "greedy"
                else int(main_outcome.search_stats.get("nodes", len(main_actions)))
            )
            eval_alt = (
                len(alt_actions)
                if alternate_mode == "greedy"
                else int(alt_outcome.search_stats.get("nodes", len(alt_actions)))
            )
            greedy_eval = eval_main if main_mode == "greedy" else eval_alt
            exhaustive_eval = eval_main if main_mode == "exhaustive" else eval_alt
            if greedy_eval != exhaustive_eval:
                evaluated_difference_count += 1
            if same:
                same_count += 1
            else:
                different_count += 1
            differences.append(
                {
                    "sentence_id": sid,
                    "same": same,
                    "comparison_status": "OK",
                    "greedy": {
                        "actions": main_actions if main_mode == "greedy" else alt_actions,
                        "step_count": len(main_actions) if main_mode == "greedy" else len(alt_actions),
                        "total_score": main_outcome.final_state.total_score
                        if main_mode == "greedy"
                        else alt_outcome.final_state.total_score,
                        "evaluated_transition_steps": greedy_eval,
                    },
                    "exhaustive": {
                        "actions": main_actions if main_mode == "exhaustive" else alt_actions,
                        "step_count": len(main_actions) if main_mode == "exhaustive" else len(alt_actions),
                        "total_score": main_outcome.final_state.total_score
                        if main_mode == "exhaustive"
                        else alt_outcome.final_state.total_score,
                        "evaluated_transition_steps": exhaustive_eval,
                    },
                    "first_difference_step": first_difference,
                }
            )
        return {
            "main_mode": main_mode,
            "comparison_mode": alternate_mode,
            "same_sentence_count": same_count,
            "different_sentence_count": different_count,
            "different_evaluated_transition_count": evaluated_difference_count,
            "comparison_failed_count": failed_count,
            "selected_step_count_note": (
                "arc-standard complete derivations always contain 2*n transitions; "
                "mode work differs in evaluated transition states and chosen actions"
            ),
            "differences": differences,
        }

    @staticmethod
    def _boundary_ties(sentence_reports: list[dict[str, Any]]) -> list[dict[str, Any]]:
        ties: list[dict[str, Any]] = []
        for sentence in sentence_reports:
            for step in sentence.get("steps", []):
                tied = step.get("tied_best_actions", [])
                if len(tied) > 1:
                    ties.append(
                        {
                            "sentence_id": sentence["sentence_id"],
                            "step_index": step["step_index"],
                            "tied_actions": tied,
                            "scores": {
                                item["action"]: item["score"]
                                for item in step.get("feasible_actions", [])
                                if item["action"] in tied
                            },
                            "selected_action": step["action"],
                        }
                    )
        return ties

    def _commit_snapshot(
        self,
        *,
        caches: dict[str, dict[str, dict[str, Any]]],
        products: dict[str, dict[str, Any]],
        active_sentences: dict[str, Sentence],
        load_failures: list[Failure],
        processed_ids: set[str],
        config: Config,
        mode: str,
        stage_status: str,
    ) -> dict[str, Any] | None:
        if self.snapshot is None:
            return None
        active_cache_ids = set(active_sentences)
        filtered_caches = {
            sid: modes for sid, modes in caches.items() if sid in active_cache_ids
        }
        arcs: list[dict[str, Any]] = []
        trees: list[dict[str, Any]] = []
        for sid in sorted(products):
            arcs.extend(products[sid]["arcs"])
            trees.append(
                {
                    "sentence_id": sid,
                    "flat": products[sid]["flat_tree"],
                    "nested": products[sid]["nested_tree"],
                    "tree_key": products[sid]["tree_key"],
                }
            )
        arcs.sort(key=lambda arc: (arc["sentence_id"], arc["head"], arc["dependent"]))
        trees.sort(key=lambda tree: tree["sentence_id"])
        sentence_cursors: dict[str, Any] = {}
        for sid in sorted(filtered_caches):
            mode_cache = filtered_caches[sid].get(mode)
            if mode_cache is None:
                continue
            sentence_cursors[sid] = {
                "steps": len(mode_cache.get("steps", [])),
                "completed": bool(mode_cache.get("completed", False)),
                "last_step_key": (
                    mode_cache["steps"][-1]["cache_key"] if mode_cache.get("steps") else None
                ),
            }
        stages = {
            "load": {
                "sentences": [
                    active_sentences[sid].to_dict() for sid in sorted(active_sentences)
                ],
                "failures": [failure.to_dict() for failure in sorted(load_failures, key=lambda item: item.sort_key)],
            },
            "parse": {
                "caches": filtered_caches,
                "mode": mode,
            },
            "arcs": {"arcs": arcs},
            "trees": {"trees": trees},
        }
        return self.snapshot.commit(
            stages=stages,
            model_hash=self.project.model.fingerprint,
            config_hash=config.fingerprint,
            corpus_hash=self.project.corpus_fingerprint,
            mode=mode,
            completed_sentences=sorted(processed_ids),
            sentence_cursors=sentence_cursors,
            extra_manifest={"stage_status": stage_status},
        )

    @staticmethod
    def _snapshot_summary(manifest: dict[str, Any] | None) -> dict[str, Any] | None:
        if manifest is None:
            return None
        return {
            "global_version": manifest.get("global_version"),
            "snapshot_name": manifest.get("snapshot_name"),
            "model_hash": manifest.get("model_hash"),
            "config_hash": manifest.get("config_hash"),
            "corpus_hash": manifest.get("corpus_hash"),
            "completed_sentences": manifest.get("completed_sentences", []),
            "sentence_cursors": manifest.get("sentence_cursors", {}),
            "stage_versions": manifest.get("stage_versions", {}),
            "stage_status": manifest.get("stage_status"),
        }

    @staticmethod
    def _semantic_digest(report: dict[str, Any]) -> str:
        payload = {
            "mode": report["mode"],
            "sentences": [semantic_sentence_payload(item) for item in report["sentences"]],
            "failures": report["failures"],
            "pending": report["pending"],
            "budget": report["budget"],
            "boundary_ties": report["boundary_ties"],
            "mode_differences": report["mode_differences"],
            "injections": report["injections"],
        }
        return sha256_json(payload)


class Workbench:
    """Convenience facade for programmatic and interactive incremental runs."""

    def __init__(
        self,
        sample_dir: str | Path,
        *,
        snapshot_dir: str | Path | None = None,
        sentence_file: str | Path | None = None,
        model_file: str | Path | None = None,
        config_file: str | Path | None = None,
        vocabulary_file: str | Path | None = None,
        injection_file: str | Path | None = None,
        report_path: str | Path | None = None,
    ):
        self.sample_dir = Path(sample_dir).resolve()
        self.snapshot_dir = snapshot_dir
        self.sentence_file = sentence_file
        self.model_file = model_file
        self.config_file = config_file
        self.vocabulary_file = vocabulary_file
        self.injection_file = injection_file
        self.report_path = Path(report_path) if report_path is not None else None

    def load_project(self) -> Project:
        return load_project(
            self.sample_dir,
            sentence_file=self.sentence_file,
            model_file=self.model_file,
            config_file=self.config_file,
            vocabulary_file=self.vocabulary_file,
            injection_file=self.injection_file,
        )

    def run(
        self,
        *,
        mode_override: str | None = None,
        resume: bool = False,
        fresh: bool = False,
        expected_version: int | None = None,
        crash_after_sentences: int | None = None,
        crash_partial_steps: int | None = None,
        commit_snapshots: bool = True,
        write_report: bool = True,
    ) -> dict[str, Any]:
        project = self.load_project()
        engine = Engine(project, self.snapshot_dir)
        report = engine.run(
            mode_override=mode_override,
            resume=resume,
            fresh=fresh,
            expected_version=expected_version,
            crash_after_sentences=crash_after_sentences,
            crash_partial_steps=crash_partial_steps,
            commit_snapshots=commit_snapshots,
        )
        if write_report and self.report_path is not None:
            write_json_atomic(self.report_path, report)
        return report

    def replace_sentence(
        self,
        sentence_id: str,
        words: str | list[str],
        pos_tags: str | list[str],
        *,
        gold_heads: list[int] | None = None,
        gold_relations: list[str] | None = None,
    ) -> None:
        rows = self._read_sentence_rows()
        word_list = words.split() if isinstance(words, str) else list(words)
        pos_list = pos_tags.split() if isinstance(pos_tags, str) else list(pos_tags)
        found = False
        for row in rows:
            sid = row.get("sentence_id", row.get("id", row.get("sid")))
            if sid == sentence_id:
                row["words"] = " ".join(word_list)
                row["pos_tags"] = " ".join(pos_list)
                if gold_heads is None:
                    row.pop("gold_heads", None)
                else:
                    row["gold_heads"] = list(gold_heads)
                if gold_relations is None:
                    row.pop("gold_relations", None)
                else:
                    row["gold_relations"] = list(gold_relations)
                found = True
                break
        if not found:
            raise KeyError(f"sentence id not found: {sentence_id}")
        write_jsonl_atomic(self._sentence_path(), rows)

    def add_sentence(
        self,
        sentence_id: str,
        words: str | list[str],
        pos_tags: str | list[str],
        *,
        gold_heads: list[int] | None = None,
        gold_relations: list[str] | None = None,
    ) -> None:
        rows = self._read_sentence_rows()
        existing = {
            row.get("sentence_id", row.get("id", row.get("sid"))) for row in rows
        }
        if sentence_id in existing:
            raise KeyError(f"sentence id already exists: {sentence_id}")
        word_list = words.split() if isinstance(words, str) else list(words)
        pos_list = pos_tags.split() if isinstance(pos_tags, str) else list(pos_tags)
        row: dict[str, Any] = {
            "sentence_id": sentence_id,
            "words": " ".join(word_list),
            "pos_tags": " ".join(pos_list),
        }
        if gold_heads is not None:
            row["gold_heads"] = list(gold_heads)
        if gold_relations is not None:
            row["gold_relations"] = list(gold_relations)
        rows.append(row)
        write_jsonl_atomic(self._sentence_path(), rows)

    def delete_sentence(self, sentence_id: str) -> None:
        rows = self._read_sentence_rows()
        kept = [
            row
            for row in rows
            if row.get("sentence_id", row.get("id", row.get("sid"))) != sentence_id
        ]
        if len(kept) == len(rows):
            raise KeyError(f"sentence id not found: {sentence_id}")
        write_jsonl_atomic(self._sentence_path(), kept)

    def set_action_weight(self, action: str, index: int, value: float) -> None:
        path = self._model_path()
        model = self._read_json(path)
        weights = model.get("action_weights", model.get("weights"))
        if not isinstance(weights, dict):
            raise KeyError("model has no action_weights object")
        from .models import normalize_action_name

        canonical = normalize_action_name(action)
        target_key = None
        for key in weights:
            if normalize_action_name(key) == canonical:
                target_key = key
                break
        if target_key is None:
            raise KeyError(f"action not found: {action}")
        values = weights[target_key]
        if isinstance(values, dict):
            values = values.get("values", values.get("weights"))
            weights[target_key] = values
        if not isinstance(values, list) or index < 0 or index >= len(values):
            raise IndexError(f"weight index out of range: {index}")
        values[index] = value
        write_json_atomic(path, model)

    def set_constraint(self, name: str, value: Any) -> None:
        path = self._config_path()
        config = self._read_json(path)
        constraints = config.setdefault("constraints", {})
        if not isinstance(constraints, dict):
            raise TypeError("config constraints is not an object")
        constraints[name] = value
        write_json_atomic(path, config)

    def set_config(self, name: str, value: Any) -> None:
        path = self._config_path()
        config = self._read_json(path)
        config[name] = value
        write_json_atomic(path, config)

    def set_mode(self, mode: str) -> None:
        self.set_config("mode", mode)

    def _read_sentence_rows(self) -> list[dict[str, Any]]:
        import json

        path = self._sentence_path()
        rows: list[dict[str, Any]] = []
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    value = json.loads(line)
                    if not isinstance(value, dict):
                        raise ValueError("sentence row is not an object")
                    rows.append(value)
        return rows

    def _sentence_path(self) -> Path:
        return self._resolve_edit_path(self.sentence_file, "sentences.jsonl")

    def _model_path(self) -> Path:
        return self._resolve_edit_path(self.model_file, "model.json")

    def _config_path(self) -> Path:
        return self._resolve_edit_path(self.config_file, "config.json")

    def _resolve_edit_path(self, value: str | Path | None, default_name: str) -> Path:
        path = Path(value) if value is not None else Path(default_name)
        return path if path.is_absolute() else self.sample_dir / path

    @staticmethod
    def _read_json(path: Path) -> Any:
        import json

        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
