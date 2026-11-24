"""Deterministic incremental/full routing engine."""

from __future__ import annotations

import copy
import json
import math
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from fractions import Fraction
from pathlib import Path
from typing import Any, Callable, Iterable

from .cache import CacheEntry, STAGES, StageCache
from .commands import CommandResult, apply_command, ensure_config_shape
from .config import ConfigValidationError, parse_config
from .errors import BusinessError, ErrorCode, Failure, SimulatedCrash
from .fingerprints import (
    recalibration_fingerprint,
    stage_cache_key,
    trigger_table_fingerprint,
)
from .io_layer import (
    dataset_fingerprint,
    load_commands,
    load_config,
    load_json,
    load_utterances,
    load_yaml,
    parse_utterance,
    validate_utterances,
)
from .models import IntentSpec, RoutingConfig, SlotPatternSpec, Utterance
from .routing import (
    route_domain,
    route_intent,
    routing_point_cost,
    stable_task_key,
    trigger_hit_count,
)
from .slots import fill_slots, recalibrate_confidence, SlotResult
from .snapshot import SnapshotManager, decide_resume
from .util import (
    canonical_json,
    deep_merge,
    fingerprint,
    format_fixed,
    jsonable,
    quantize,
    sha256_text,
)


@dataclass(frozen=True)
class RuntimeInjection:
    injection_id: str
    utterance_id: str
    stage: str
    message: str = "injected runtime exception"
    code: str = ErrorCode.ROUTE_INJECTED_RUNTIME.value


@dataclass
class WorkItem:
    utterance_id: str
    command_index: int | None = None


@dataclass
class CachePlan:
    complete: bool
    keys: dict[str, str] = field(default_factory=dict)
    entries: dict[str, CacheEntry] = field(default_factory=dict)
    missing_stage: str | None = None
    injected: bool = False


@dataclass
class TaskOutcome:
    utterance_id: str
    result: dict[str, Any] | None = None
    failures: list[Failure] = field(default_factory=list)
    entries: dict[str, CacheEntry] = field(default_factory=dict)
    statuses: dict[str, str] = field(default_factory=dict)
    pending_stages: set[str] = field(default_factory=set)
    injection_hits: list[str] = field(default_factory=list)


def _failure_sort_key(failure: Failure) -> tuple[Any, ...]:
    return (
        failure.utterance_id,
        STAGES.index(failure.stage) if failure.stage in STAGES else len(STAGES),
        failure.stage,
        failure.business_code,
        failure.field or "",
        failure.message,
        failure.command_index if failure.command_index is not None else -1,
        failure.injection_id or "",
    )


def _failure_dict_key(failure: dict[str, Any]) -> str:
    return canonical_json(failure)


class Workbench:
    def __init__(
        self,
        *,
        samples_dir: str | Path | None = None,
        utterances_path: str | Path | None = None,
        config_path: str | Path | None = None,
        commands_path: str | Path | None = None,
        vocabulary_path: str | Path | None = None,
        state_dir: str | Path | None = None,
        full: bool = False,
        resume: bool = True,
        expected_global_version: int | None = None,
        batch_budget: int | None = None,
        workers: int | None = None,
        crash_after_commands: int | None = None,
        runtime_injections: Iterable[RuntimeInjection | dict[str, Any]] | None = None,
        keep_history: bool = True,
    ) -> None:
        from .io_layer import resolve_inputs

        if samples_dir is None and (utterances_path is None or config_path is None):
            raise ValueError("provide samples_dir or both utterances_path and config_path")
        paths = resolve_inputs(samples_dir, utterances_path, config_path, commands_path, vocabulary_path)
        if paths["utterances"] is None or paths["config"] is None:
            raise ValueError("cannot locate utterances JSONL and routing configuration")
        self.utterances_path = Path(paths["utterances"])
        self.config_path = Path(paths["config"])
        self.commands_path = Path(paths["commands"]) if paths["commands"] is not None else None
        self.vocabulary_path = Path(paths["vocabulary"]) if paths["vocabulary"] is not None else None
        self.state_dir = Path(state_dir) if state_dir is not None else None
        self.full = bool(full)
        self.resume = bool(resume)
        self.expected_global_version = expected_global_version
        self.batch_budget_override = batch_budget
        self.workers_override = workers
        self.crash_after_commands = crash_after_commands
        self.keep_history = keep_history
        self.runtime_injections: list[RuntimeInjection] = []
        for item in runtime_injections or []:
            if isinstance(item, RuntimeInjection):
                self.runtime_injections.append(item)
            else:
                self.runtime_injections.append(
                    RuntimeInjection(
                        injection_id=str(item.get("id", item.get("injection_id", "INJ-RUNTIME"))),
                        utterance_id=str(item.get("utterance_id", item.get("target", ""))),
                        stage=str(item.get("stage", "intent")),
                        message=str(item.get("message", "injected runtime exception")),
                        code=str(item.get("code", ErrorCode.ROUTE_INJECTED_RUNTIME.value)),
                    )
                )

    def run(self) -> dict[str, Any]:
        base_config_raw, config_load_failures = self._load_base_config()
        base_utterances, utterance_load_failures = load_utterances(self.utterances_path)
        commands, command_load_failures = (
            load_commands(self.commands_path) if self.commands_path is not None else ([], [])
        )
        base_config_raw = ensure_config_shape(base_config_raw) if isinstance(base_config_raw, dict) else base_config_raw
        base_raw_map = self._raw_map(base_utterances)
        base_input_fingerprint = fingerprint(
            {
                "config": base_config_raw,
                "utterances": [base_raw_map[key] for key in sorted(base_raw_map)],
                "vocabulary_path": str(self.vocabulary_path) if self.vocabulary_path else None,
            }
        )
        command_stream_fingerprint = fingerprint(commands)
        expected_config_hash = self._expected_final_config_hash(base_config_raw, base_raw_map, commands)

        diagnostics: list[dict[str, Any]] = []
        initial_failures: list[Failure] = [*config_load_failures, *utterance_load_failures, *command_load_failures]
        snapshots: SnapshotManager | None = SnapshotManager(self.state_dir) if self.state_dir is not None else None
        resumed = False
        stale = False
        snapshot_manifest: dict[str, Any] | None = None
        prior_cache = StageCache()
        state_data: dict[str, Any] | None = None

        if snapshots is not None and not self.full and self.resume:
            loaded = snapshots.load()
            cursor_hint = int(loaded.manifest.get("completed_command_cursor", 0)) if loaded.manifest else 0
            if cursor_hint < 0 or cursor_hint > len(commands):
                expected_at_cursor = ""
            else:
                expected_at_cursor = self._expected_final_config_hash(
                    base_config_raw,
                    base_raw_map,
                    commands[:cursor_hint],
                )
            should_resume, stale_code, stale_message = decide_resume(
                loaded,
                expected_config_hash=expected_at_cursor,
                base_input_fingerprint=base_input_fingerprint,
                command_stream_fingerprint=fingerprint(commands[:cursor_hint]),
                expected_global_version=self.expected_global_version,
            )
            if should_resume and loaded.data is not None and loaded.cache is not None:
                resumed = True
                state_data = loaded.data
                prior_cache = loaded.cache
                snapshot_manifest = loaded.manifest
            elif loaded.manifest is not None:
                stale = True
                diagnostic = {
                    "business_code": stale_code or ErrorCode.SNAPSHOT_STALE.value,
                    "stage": "snapshot",
                    "message": stale_message,
                    "actual": loaded.manifest.get("global_version"),
                    "expected": self.expected_global_version,
                }
                diagnostics.append(diagnostic)

        if state_data is not None:
            config_raw = copy.deepcopy(state_data.get("config", base_config_raw))
            raw_map = {
                str(item.get("utterance_id", item.get("id"))): copy.deepcopy(item)
                for item in state_data.get("utterances", [])
                if isinstance(item, dict)
            }
            cursor = int(state_data.get("cursor", 0))
            pending_items = [
                WorkItem(str(item["utterance_id"]), item.get("command_index"))
                for item in state_data.get("pending", [])
                if isinstance(item, dict) and item.get("utterance_id")
            ]
            results = copy.deepcopy(state_data.get("results", {}))
            ledger = copy.deepcopy(state_data.get("ledger", []))
            next_ledger_sequence = int(state_data.get("next_ledger_sequence", len(ledger) + 1))
            consumed_budget = int(state_data.get("consumed_budget", sum(int(entry.get("cost", 0)) for entry in ledger)))
            persisted_failures = self._failures_from_dicts(state_data.get("persisted_failures", []))
        else:
            config_raw = copy.deepcopy(base_config_raw)
            raw_map = copy.deepcopy(base_raw_map)
            cursor = 0
            pending_items = []
            results = {}
            ledger = []
            next_ledger_sequence = 1
            consumed_budget = 0
            persisted_failures = list(initial_failures)

        config, config_errors = self._parse_config_or_failure(config_raw)
        persisted_failures.extend(config_errors)
        effective_workers = self._effective_workers(config)
        batch_limit = self.batch_budget_override if self.batch_budget_override is not None else (config.batch_point_budget if config else 1)
        if batch_limit is None or batch_limit <= 0:
            batch_limit = 1
        remaining_budget = int(batch_limit)
        snapshot_interval = config.snapshot_interval_commands if config else 1

        # On a fresh run every valid utterance is a work item; on resume only unfinished work remains.
        if not resumed:
            parsed_for_seed, _ = self._parse_and_validate(raw_map, config)
            for utterance_id in sorted(parsed_for_seed):
                pending_items.append(WorkItem(utterance_id, 0))

        pending_items = self._dedupe_work(pending_items)
        command_failures: list[Failure] = list(persisted_failures)
        runtime_injection_hits: set[str] = set()
        command_results: list[dict[str, Any]] = []

        commands_to_apply = commands[cursor:]
        command_start_cursor = cursor
        # Existing pending work has priority and can be drained before new commands.
        pending_items, remaining_budget, outcomes, ledger, next_ledger_sequence, consumed_budget, runtime_injection_hits = self._process_work(
            pending_items,
            raw_map,
            config,
            prior_cache,
            results,
            ledger,
            next_ledger_sequence,
            consumed_budget,
            remaining_budget,
            batch_limit,
            effective_workers,
            runtime_injection_hits,
        )

        for offset, command in enumerate(commands_to_apply):
            command_index = command_start_cursor + offset
            command["command_index"] = command_index
            config_before = copy.deepcopy(config_raw)
            raw_before = copy.deepcopy(raw_map)
            result = apply_command(command, config_raw, raw_map, set(raw_map))
            command_results.append(result.to_dict())
            if result.failures:
                command_failures.extend(result.failures)
            if result.status == "failed":
                config_raw = config_before
                raw_map = raw_before
            elif result.status == "applied":
                new_config, config_problem = self._parse_config_or_failure(config_raw)
                if config_problem:
                    config_raw = config_before
                    raw_map = raw_before
                    failure = Failure(
                        ErrorCode.ROUTE_INVALID.value,
                        result.command_id,
                        "commands",
                        "command made configuration invalid: " + config_problem[0].message,
                        field="config",
                        command_index=command_index,
                    )
                    command_failures.append(failure)
                    result.status = "failed"
                else:
                    config = new_config
                    snapshot_interval = config.snapshot_interval_commands
                    dirty_remaining = set(result.dirty_ids)
                    reordered: list[WorkItem] = []
                    for item in pending_items:
                        if item.utterance_id in result.dirty_ids:
                            reordered.append(WorkItem(item.utterance_id, command_index))
                            dirty_remaining.discard(item.utterance_id)
                        else:
                            reordered.append(item)
                    for utterance_id in sorted(dirty_remaining):
                        reordered.append(WorkItem(utterance_id, command_index))
                    pending_items = reordered
                    for utterance_id in result.dirty_ids:
                        results.pop(utterance_id, None)
            cursor = command_index + 1
            pending_items = self._dedupe_work(pending_items)
            pending_items, remaining_budget, batch_outcomes, ledger, next_ledger_sequence, consumed_budget, runtime_injection_hits = self._process_work(
                pending_items,
                raw_map,
                config,
                prior_cache,
                results,
                ledger,
                next_ledger_sequence,
                consumed_budget,
                remaining_budget,
                batch_limit,
                effective_workers,
                runtime_injection_hits,
            )
            outcomes.extend(batch_outcomes)
            if snapshots is not None and ((command_index + 1) % max(1, snapshot_interval) == 0 or command_index + 1 == len(commands)):
                snapshot_manifest = self._commit_snapshot(
                    snapshots,
                    prior_cache,
                    config_raw,
                    raw_map,
                    cursor,
                    pending_items,
                    results,
                    ledger,
                    next_ledger_sequence,
                    consumed_budget,
                    command_failures,
                    base_input_fingerprint,
                    fingerprint(commands[:cursor]),
                    config,
                )
            if self.crash_after_commands is not None and cursor >= self.crash_after_commands:
                raise SimulatedCrash(f"simulated termination after command {cursor}")

        # Materialize fully cached, unaffected utterances so reports consistently show reuse decisions.
        parsed_final, validation_failures = self._parse_and_validate(raw_map, config)
        materialize = [
            WorkItem(utterance_id, None)
            for utterance_id in sorted(parsed_final)
            if not any(item.utterance_id == utterance_id for item in pending_items)
        ]
        pending_items = self._dedupe_work([*pending_items, *materialize])
        pending_items, remaining_budget, batch_outcomes, ledger, next_ledger_sequence, consumed_budget, runtime_injection_hits = self._process_work(
            pending_items,
            raw_map,
            config,
            prior_cache,
            results,
            ledger,
            next_ledger_sequence,
            consumed_budget,
            remaining_budget,
            batch_limit,
            effective_workers,
            runtime_injection_hits,
        )
        outcomes.extend(batch_outcomes)

        parsed_final, validation_failures = self._parse_and_validate(raw_map, config)
        if config is None:
            pending_items = []
        # Commands and validation run may leave stale results for now-invalid/deleted records.
        valid_ids = set(parsed_final)
        results = {utterance_id: value for utterance_id, value in results.items() if utterance_id in valid_ids}
        pending_items = [item for item in pending_items if item.utterance_id in valid_ids]

        all_failures = self._merge_failures(command_failures, validation_failures)
        for outcome in outcomes:
            all_failures.extend(outcome.failures)
        for item in pending_items:
            utterance = parsed_final.get(item.utterance_id)
            if utterance is None:
                continue
            all_failures.append(
                Failure(
                    ErrorCode.ROUTE_BUDGET_PENDING.value,
                    item.utterance_id,
                    "scheduler",
                    "routing point budget exhausted; item retained in pending queue",
                    field="batch_point_budget",
                    command_index=item.command_index,
                )
            )
        all_failures = self._merge_failures(all_failures, [])

        if snapshot_manifest is None and snapshots is not None:
            snapshot_manifest = self._commit_snapshot(
                snapshots,
                prior_cache,
                config_raw,
                raw_map,
                cursor,
                pending_items,
                results,
                ledger,
                next_ledger_sequence,
                consumed_budget,
                command_failures,
                base_input_fingerprint,
                fingerprint(commands[:cursor]),
                config,
            )
        elif snapshots is not None:
            snapshot_manifest = self._commit_snapshot(
                snapshots,
                prior_cache,
                config_raw,
                raw_map,
                cursor,
                pending_items,
                results,
                ledger,
                next_ledger_sequence,
                consumed_budget,
                command_failures,
                base_input_fingerprint,
                fingerprint(commands[:cursor]),
                config,
            )

        report = self._build_report(
            mode="full" if self.full else ("incremental" if resumed else "fresh"),
            config=config,
            config_raw=config_raw,
            parsed=parsed_final,
            results=results,
            failures=all_failures,
            pending_items=pending_items,
            cache=prior_cache,
            ledger=ledger,
            consumed_budget=consumed_budget,
            batch_limit=batch_limit,
            remaining_budget=remaining_budget,
            diagnostics=diagnostics,
            resumed=resumed,
            stale=stale,
            snapshot_manifest=snapshot_manifest,
            command_results=command_results,
            runtime_injection_hits=runtime_injection_hits,
            workers=effective_workers,
            base_input_fingerprint=base_input_fingerprint,
            command_stream_fingerprint=command_stream_fingerprint,
            raw_record_count=len(raw_map),
        )
        return report

    def _load_base_config(self) -> tuple[Any, list[Failure]]:
        failures: list[Failure] = []
        fallback_vocabulary: tuple[str, ...] = ()
        if self.vocabulary_path is not None:
            try:
                from .config import parse_vocabulary_file

                fallback_vocabulary = parse_vocabulary_file(self.vocabulary_path)
            except OSError as exc:
                failures.append(
                    Failure(ErrorCode.ROUTE_INVALID.value, "__config__", "validation", f"cannot read vocabulary: {exc}", field="vocabulary")
                )
        try:
            raw = load_yaml(self.config_path)
        except Exception as exc:
            failures.append(
                Failure(ErrorCode.ROUTE_INVALID.value, "__config__", "validation", f"cannot parse routing configuration: {exc}", field="config")
            )
            return {}, failures
        if not isinstance(raw, dict):
            failures.append(
                Failure(ErrorCode.ROUTE_INVALID.value, "__config__", "validation", "routing configuration must be a mapping", field="config")
            )
            return {}, failures
        if fallback_vocabulary:
            existing = raw.get("vocabulary", raw.get("vocab", []))
            combined = list(fallback_vocabulary) + [str(item) for item in existing] if isinstance(existing, list) else list(fallback_vocabulary)
            raw["vocabulary"] = list(dict.fromkeys(combined))
        return raw, failures

    @staticmethod
    def _raw_map(utterances: Iterable[Utterance]) -> dict[str, dict[str, Any]]:
        return {
            utterance.utterance_id: copy.deepcopy(utterance.raw)
            for utterance in sorted(utterances, key=lambda item: item.utterance_id)
        }

    def _parse_config_or_failure(self, raw: Any) -> tuple[RoutingConfig | None, list[Failure]]:
        try:
            return parse_config(raw), []
        except ConfigValidationError as exc:
            return None, [exc.as_failure()]

    def _parse_and_validate(
        self,
        raw_map: dict[str, dict[str, Any]],
        config: RoutingConfig | None,
    ) -> tuple[dict[str, Utterance], list[Failure]]:
        parsed: list[Utterance] = []
        failures: list[Failure] = []
        for utterance_id in sorted(raw_map):
            raw = raw_map[utterance_id]
            try:
                parsed.append(parse_utterance(raw, 0))
            except ValueError as exc:
                failures.append(
                    Failure(
                        ErrorCode.ROUTE_INVALID.value,
                        utterance_id,
                        "validation",
                        str(exc),
                        field="utterance",
                    )
                )
        valid, validation_failures = validate_utterances(parsed, config)
        return valid, self._merge_failures(failures, validation_failures)

    def _expected_final_config_hash(
        self,
        base_config: Any,
        base_raw_map: dict[str, dict[str, Any]],
        commands: list[dict[str, Any]],
    ) -> str:
        if not isinstance(base_config, dict):
            return sha256_text(canonical_json(base_config))
        config = copy.deepcopy(base_config)
        raw_map = copy.deepcopy(base_raw_map)
        parsed, _ = self._parse_config_or_failure(config)
        for index, command in enumerate(commands):
            command = dict(command)
            command["command_index"] = index
            before = copy.deepcopy(config)
            before_raw = copy.deepcopy(raw_map)
            result = apply_command(command, config, raw_map, set(raw_map))
            if result.status == "failed":
                config = before
                raw_map = before_raw
                continue
            parsed_new, failures = self._parse_config_or_failure(config)
            if failures:
                config = before
                raw_map = before_raw
            else:
                parsed = parsed_new
        if parsed is not None:
            return parsed.config_hash
        return fingerprint(config)

    def _effective_workers(self, config: RoutingConfig | None) -> int:
        if config is None:
            return 1
        workers = self.workers_override if self.workers_override is not None else config.thread_pool_size
        if workers <= 0:
            workers = 1
        if workers > config.max_thread_pool_size:
            workers = config.max_thread_pool_size
        return max(1, min(workers, config.max_thread_pool_size))

    @staticmethod
    def _dedupe_work(items: Iterable[WorkItem]) -> list[WorkItem]:
        result: "OrderedDict[str, WorkItem]" = OrderedDict()
        for item in items:
            if item.utterance_id not in result:
                result[item.utterance_id] = WorkItem(item.utterance_id, item.command_index)
            elif item.command_index is not None:
                old = result[item.utterance_id]
                result[item.utterance_id] = WorkItem(item.utterance_id, item.command_index if old.command_index is None else old.command_index)
        return list(result.values())

    def _cache_plan(
        self,
        utterance: Utterance,
        config: RoutingConfig,
        cache: StageCache,
    ) -> CachePlan:
        keys: dict[str, str] = {}
        entries: dict[str, CacheEntry] = {}
        validation_key = stage_cache_key("validation", utterance, config)
        keys["validation"] = validation_key
        validation_entry = cache.get("validation", validation_key)
        if validation_entry is None:
            return CachePlan(False, keys, entries, "validation", self._injection_for(utterance.utterance_id, "validation") is not None)
        entries["validation"] = validation_entry
        domain_key = stage_cache_key("domain", utterance, config)
        keys["domain"] = domain_key
        domain_entry = cache.get("domain", domain_key)
        if domain_entry is None:
            return CachePlan(False, keys, entries, "domain", self._injection_for(utterance.utterance_id, "domain") is not None)
        entries["domain"] = domain_entry
        selected_domain = domain_entry.product.get("selected") if isinstance(domain_entry.product, dict) else None
        intent_key = stage_cache_key("intent", utterance, config, selected_domain=selected_domain)
        keys["intent"] = intent_key
        intent_entry = cache.get("intent", intent_key)
        if intent_entry is None:
            return CachePlan(False, keys, entries, "intent", self._injection_for(utterance.utterance_id, "intent") is not None)
        entries["intent"] = intent_entry
        selected_intent = intent_entry.product.get("selected") if isinstance(intent_entry.product, dict) else None
        slots_key = stage_cache_key("slots", utterance, config, selected_domain=selected_domain, selected_intent=selected_intent)
        keys["slots"] = slots_key
        slots_entry = cache.get("slots", slots_key)
        if slots_entry is None:
            return CachePlan(False, keys, entries, "slots", self._injection_for(utterance.utterance_id, "slots") is not None)
        entries["slots"] = slots_entry
        score_int = (
            int(intent_entry.product.get("score_int", 0))
            if isinstance(intent_entry.product, dict) and selected_intent is not None
            else 0
        )
        recal_key = stage_cache_key(
            "recalibration",
            utterance,
            config,
            selected_domain=selected_domain,
            selected_intent=selected_intent,
            slots_digest=slots_entry.digest,
            base_confidence_int=score_int,
        )
        keys["recalibration"] = recal_key
        recal_entry = cache.get("recalibration", recal_key)
        if recal_entry is None:
            return CachePlan(False, keys, entries, "recalibration", self._injection_for(utterance.utterance_id, "recalibration") is not None)
        entries["recalibration"] = recal_entry
        return CachePlan(True, keys, entries)

    def _injection_for(self, utterance_id: str, stage: str) -> RuntimeInjection | None:
        for injection in self.runtime_injections:
            if injection.utterance_id == utterance_id and injection.stage == stage:
                return injection
        return None

    def _process_work(
        self,
        work_items: list[WorkItem],
        raw_map: dict[str, dict[str, Any]],
        config: RoutingConfig | None,
        cache: StageCache,
        results: dict[str, dict[str, Any]],
        ledger: list[dict[str, Any]],
        next_ledger_sequence: int,
        consumed_budget: int,
        remaining_budget: int,
        batch_limit: int,
        workers: int,
        injection_hits: set[str],
    ) -> tuple[list[WorkItem], int, list[TaskOutcome], list[dict[str, Any]], int, int, set[str]]:
        if config is None:
            return work_items, remaining_budget, [], ledger, next_ledger_sequence, consumed_budget, injection_hits
        parsed, _ = self._parse_and_validate(raw_map, config)
        selected_items: list[WorkItem] = []
        deferred: list[WorkItem] = []
        budget_blocked = False
        for item in work_items:
            if item.utterance_id not in parsed:
                results.pop(item.utterance_id, None)
                continue
            utterance = parsed[item.utterance_id]
            plan = self._cache_plan(utterance, config, cache)
            if budget_blocked:
                if plan.complete and not plan.injected:
                    selected_items.append(item)
                else:
                    for stage in STAGES:
                        cache.mark_pending(stage, item.utterance_id)
                    deferred.append(item)
                continue
            if plan.complete and not plan.injected:
                selected_items.append(item)
                continue
            cost = routing_point_cost(utterance, config)
            if cost > remaining_budget:
                budget_blocked = True
                for stage in STAGES:
                    cache.mark_pending(stage, item.utterance_id)
                deferred.append(item)
                continue
            before_consumed = consumed_budget
            consumed_budget += cost
            remaining_budget -= cost
            hit_count = trigger_hit_count(utterance, config)
            ledger.append(
                {
                    "sequence": next_ledger_sequence,
                    "command_index": item.command_index,
                    "utterance_id": item.utterance_id,
                    "token_count": len(utterance.tokens),
                    "trigger_hit_count": hit_count,
                    "cost": cost,
                    "consumed_before": before_consumed,
                    "consumed_after": consumed_budget,
                    "remaining": remaining_budget,
                    "batch_limit": batch_limit,
                }
            )
            next_ledger_sequence += 1
            selected_items.append(item)

        if not selected_items:
            return deferred, remaining_budget, [], ledger, next_ledger_sequence, consumed_budget, injection_hits

        def execute(item: WorkItem) -> TaskOutcome:
            return self._route_one(parsed[item.utterance_id], config, cache)

        outcomes: list[TaskOutcome] = []
        if workers > 1 and len(selected_items) > 1:
            with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="route") as executor:
                outcomes = list(executor.map(execute, selected_items))
        else:
            outcomes = [execute(item) for item in selected_items]

        for outcome in outcomes:
            if outcome.result is not None:
                results[outcome.utterance_id] = outcome.result
            for stage, entry in outcome.entries.items():
                cache.entries.setdefault(stage, {})[entry.key] = entry
            for stage, status in outcome.statuses.items():
                if status == "reuse":
                    cache.mark_reuse(stage, outcome.utterance_id)
                elif status == "recompute":
                    cache.mark_recompute(stage, outcome.utterance_id)
            for stage in outcome.pending_stages:
                cache.mark_pending(stage, outcome.utterance_id)
            injection_hits.update(outcome.injection_hits)
        return deferred, remaining_budget, outcomes, ledger, next_ledger_sequence, consumed_budget, injection_hits

    def _route_one(self, utterance: Utterance, config: RoutingConfig, prior_cache: StageCache) -> TaskOutcome:
        outcome = TaskOutcome(utterance.utterance_id)
        keys: dict[str, str] = {}
        products: dict[str, Any] = {}
        entries: dict[str, CacheEntry] = {}
        stage_order = list(STAGES)

        def fail(stage: str, failure: Failure) -> TaskOutcome:
            outcome.failures.append(failure)
            outcome.entries = entries
            outcome.statuses = {name: status for name, status in outcome.statuses.items()}
            outcome.pending_stages = set(stage_order[stage_order.index(stage) + 1 :])
            return outcome

        def injection_failure(stage: str) -> TaskOutcome | None:
            injection = self._injection_for(utterance.utterance_id, stage)
            if injection is None:
                return None
            outcome.injection_hits.append(injection.injection_id)
            outcome.statuses[stage] = "recompute"
            failure = Failure(
                injection.code,
                utterance.utterance_id,
                stage,
                injection.message,
                injection_id=injection.injection_id,
            )
            # Mark the stage itself as recomputed and downstream stages pending.
            outcome.pending_stages = set(stage_order[stage_order.index(stage) + 1 :])
            outcome.failures.append(failure)
            outcome.entries = entries
            return outcome

        try:
            injected = injection_failure("validation")
            if injected is not None:
                return injected
            validation_key = stage_cache_key("validation", utterance, config)
            keys["validation"] = validation_key
            validation_entry = prior_cache.get("validation", validation_key)
            if validation_entry is not None:
                outcome.statuses["validation"] = "reuse"
                products["validation"] = validation_entry.product
                entries["validation"] = validation_entry
            else:
                outcome.statuses["validation"] = "recompute"
                product = {
                    "utterance_id": utterance.utterance_id,
                    "valid": True,
                    "session_id": utterance.session_id,
                    "turn": utterance.turn,
                    "content_fingerprint": fingerprint(utterance.canonical_content()),
                }
                products["validation"] = product
                entries["validation"] = CacheEntry("validation", validation_key, product, {"utterance_ids": [utterance.utterance_id]}, [utterance.utterance_id]).finalize()

            injected = injection_failure("domain")
            if injected is not None:
                return injected
            domain_key = stage_cache_key("domain", utterance, config)
            keys["domain"] = domain_key
            domain_entry = prior_cache.get("domain", domain_key)
            if domain_entry is not None:
                outcome.statuses["domain"] = "reuse"
                domain_product = domain_entry.product
                entries["domain"] = domain_entry
            else:
                outcome.statuses["domain"] = "recompute"
                tier = route_domain(utterance, config)
                domain_product = tier.to_dict(config.quantization_scale, config.fixed_decimals)
                references = {"utterance_ids": [utterance.utterance_id], "trigger_entry_ids": sorted({item["entry_id"] for item in tier.matched + tier.suppressed})}
                domain_entry = CacheEntry("domain", domain_key, domain_product, references, [utterance.utterance_id]).finalize()
                entries["domain"] = domain_entry
            products["domain"] = domain_product
            selected_domain = domain_product.get("selected")

            injected = injection_failure("intent")
            if injected is not None:
                return injected
            intent_key = stage_cache_key("intent", utterance, config, selected_domain=selected_domain)
            keys["intent"] = intent_key
            intent_entry = prior_cache.get("intent", intent_key)
            if intent_entry is not None:
                outcome.statuses["intent"] = "reuse"
                intent_product = intent_entry.product
                entries["intent"] = intent_entry
            else:
                outcome.statuses["intent"] = "recompute"
                tier = route_intent(utterance, selected_domain, config)
                intent_product = tier.to_dict(config.quantization_scale, config.fixed_decimals)
                references = {"utterance_ids": [utterance.utterance_id], "trigger_entry_ids": sorted({item["entry_id"] for item in tier.matched + tier.suppressed})}
                intent_entry = CacheEntry("intent", intent_key, intent_product, references, [utterance.utterance_id]).finalize()
                entries["intent"] = intent_entry
            products["intent"] = intent_product
            selected_intent = intent_product.get("selected")
            intent_spec = config.selected_intent_spec(selected_domain, selected_intent)

            injected = injection_failure("slots")
            if injected is not None:
                return injected
            slots_key = stage_cache_key("slots", utterance, config, selected_domain=selected_domain, selected_intent=selected_intent)
            keys["slots"] = slots_key
            slots_entry = prior_cache.get("slots", slots_key)
            if slots_entry is not None:
                outcome.statuses["slots"] = "reuse"
                slots_product = slots_entry.product
                entries["slots"] = slots_entry
            else:
                outcome.statuses["slots"] = "recompute"
                slot_result = fill_slots(utterance, intent_spec, config)
                slots_product = slot_result.to_dict(config)
                references = {
                    "utterance_ids": [utterance.utterance_id],
                    "slot_patterns": sorted({item.get("pattern", "") for item in slot_result.candidates + slot_result.suppression if item.get("pattern")}),
                }
                slots_entry = CacheEntry("slots", slots_key, slots_product, references, [utterance.utterance_id]).finalize()
                entries["slots"] = slots_entry
            products["slots"] = slots_product

            injected = injection_failure("recalibration")
            if injected is not None:
                return injected
            score_int = int(intent_product.get("score_int", 0)) if selected_intent is not None else 0
            recal_key = stage_cache_key(
                "recalibration",
                utterance,
                config,
                selected_domain=selected_domain,
                selected_intent=selected_intent,
                slots_digest=entries["slots"].digest,
                base_confidence_int=score_int,
            )
            keys["recalibration"] = recal_key
            recal_entry = prior_cache.get("recalibration", recal_key)
            if recal_entry is not None:
                outcome.statuses["recalibration"] = "reuse"
                recal_product = recal_entry.product
                entries["recalibration"] = recal_entry
            else:
                outcome.statuses["recalibration"] = "recompute"
                if intent_spec is None:
                    slot_result = SlotResult({}, (), (), Fraction(0))
                else:
                    slot_result = SlotResult(
                        slots={str(key): str(value) for key, value in slots_product.get("values", {}).items()},
                        required_slots=tuple(slots_product.get("required_slots", [])),
                        missing_slots=tuple(slots_product.get("missing_slots", [])),
                        completeness=Fraction(int(slots_product.get("completeness_int", 0)), config.quantization_scale),
                        candidates=list(slots_product.get("candidates", [])),
                        suppression=list(slots_product.get("suppression", [])),
                    )
                base_score = Fraction(score_int, config.quantization_scale)
                recal = recalibrate_confidence(base_score, slot_result, config)
                recal_product = recal.to_dict()
                references = {
                    "utterance_ids": [utterance.utterance_id],
                    "config_fingerprint": recalibration_fingerprint(config),
                }
                recal_entry = CacheEntry("recalibration", recal_key, recal_product, references, [utterance.utterance_id]).finalize()
                entries["recalibration"] = recal_entry
            products["recalibration"] = recal_product

            result = self._compose_result(utterance, config, products, entries, keys)
            outcome.result = result
            outcome.entries = entries
            return outcome
        except BusinessError as exc:
            stage = str(exc.details.get("stage", "domain"))
            failure = Failure(exc.code, utterance.utterance_id, stage, exc.message)
            return fail(stage, failure)
        except Exception as exc:  # A single record must never stop the rest of the batch.
            last_stage = next((stage for stage in reversed(STAGES) if stage in outcome.statuses), "domain")
            failure = Failure(
                ErrorCode.ROUTE_INVALID.value,
                utterance.utterance_id,
                last_stage,
                f"stage exception: {type(exc).__name__}: {exc}",
            )
            return fail(last_stage, failure)

    def _compose_result(
        self,
        utterance: Utterance,
        config: RoutingConfig,
        products: dict[str, Any],
        entries: dict[str, CacheEntry],
        keys: dict[str, str],
    ) -> dict[str, Any]:
        domain_product = products["domain"]
        intent_product = products["intent"]
        slots_product = products["slots"]
        recal_product = products["recalibration"]
        trigger_suppression = [
            *domain_product.get("suppressed", []),
            *intent_product.get("suppressed", []),
        ]
        trigger_suppression.sort(key=lambda item: (item.get("domain", ""), item.get("intent", ""), item.get("start", 0), item.get("term", ""), item.get("entry_id", "")))
        slot_suppression = list(slots_product.get("suppression", []))
        boundary_reasons: list[str] = []
        if domain_product.get("boundary"):
            boundary_reasons.extend(f"domain:{reason}" for reason in domain_product.get("boundary_reasons", []))
        if intent_product.get("boundary"):
            boundary_reasons.extend(f"intent:{reason}" for reason in intent_product.get("boundary_reasons", []))
        trigger_entry_ids = sorted(
            {
                item.get("entry_id", "")
                for item in domain_product.get("matched", []) + intent_product.get("matched", []) + trigger_suppression
                if item.get("entry_id")
            }
        )
        slot_patterns = sorted(
            {
                item.get("pattern", "")
                for item in slots_product.get("candidates", []) + slot_suppression
                if item.get("pattern")
            }
        )
        result = {
            "utterance_id": utterance.utterance_id,
            "session_id": utterance.session_id,
            "turn": utterance.turn,
            "tokens": list(utterance.tokens),
            "domain": domain_product.get("selected"),
            "intent": intent_product.get("selected"),
            "domain_score": domain_product.get("score"),
            "domain_score_int": domain_product.get("score_int"),
            "domain_threshold": domain_product.get("threshold"),
            "domain_threshold_int": domain_product.get("threshold_int"),
            "intent_score": intent_product.get("score"),
            "intent_score_int": intent_product.get("score_int"),
            "intent_threshold": intent_product.get("threshold"),
            "intent_threshold_int": intent_product.get("threshold_int"),
            "score_scale": config.quantization_scale,
            "score_decimals": config.fixed_decimals,
            "domain_candidates": domain_product.get("candidates", []),
            "intent_candidates": intent_product.get("candidates", []),
            "slots": slots_product.get("values", {}),
            "slot_candidates": slots_product.get("candidates", []),
            "slot_suppression": slot_suppression,
            "trigger_suppression": trigger_suppression,
            "confidence_before_int": recal_product.get("before_int"),
            "confidence_after_int": recal_product.get("after_int"),
            "confidence_before": recal_product.get("before"),
            "confidence_after": recal_product.get("after"),
            "confidence_scale": recal_product.get("scale"),
            "confidence_decimals": recal_product.get("decimals"),
            "recalibration": recal_product,
            "boundary": bool(boundary_reasons),
            "boundary_reasons": boundary_reasons,
            "domain_tied_names": domain_product.get("tied_names", []),
            "intent_tied_names": intent_product.get("tied_names", []),
            "contributions": {
                "domain": {
                    "formula": "sum(unique matched trigger weights) / sum(configured domain trigger weights)",
                    "matched": domain_product.get("matched", []),
                    "suppressed": domain_product.get("suppressed", []),
                    "candidates": domain_product.get("candidates", []),
                },
                "intent": {
                    "formula": "sum(unique effective intent trigger weights after negation) / sum(configured intent trigger weights)",
                    "matched": intent_product.get("matched", []),
                    "suppressed": intent_product.get("suppressed", []),
                    "candidates": intent_product.get("candidates", []),
                },
                "slots": {
                    "formula": "candidate rank: match length desc, token start asc, value lexicographic",
                    "completeness": recal_product.get("completeness_int"),
                    "required_slots": slots_product.get("required_slots", []),
                    "missing_slots": slots_product.get("missing_slots", []),
                    "suppressed_count": len(slot_suppression),
                },
                "recalibration": recal_product,
            },
            "stage_cache_keys": {stage: keys[stage] for stage in STAGES if stage in keys},
            "references": {
                "utterance_ids": [utterance.utterance_id],
                "trigger_entry_ids": trigger_entry_ids,
                "slot_patterns": slot_patterns,
            },
            "truth": {
                "domain": utterance.domain_truth,
                "intent": utterance.intent_truth,
                "slots": dict(sorted(utterance.slots_truth.items())),
            },
        }
        return result

    def _merge_failures(self, left: Iterable[Failure], right: Iterable[Failure]) -> list[Failure]:
        mapping: dict[str, Failure] = {}
        for failure in [*left, *right]:
            mapping[_failure_dict_key(failure.to_dict())] = failure
        return sorted(mapping.values(), key=_failure_sort_key)

    @staticmethod
    def _failures_from_dicts(items: Iterable[dict[str, Any]]) -> list[Failure]:
        failures: list[Failure] = []
        for item in items:
            try:
                failures.append(
                    Failure(
                        business_code=str(item["business_code"]),
                        utterance_id=str(item["utterance_id"]),
                        stage=str(item["stage"]),
                        message=str(item["message"]),
                        field=item.get("field"),
                        command_index=item.get("command_index"),
                        injection_id=item.get("injection_id"),
                        details=item.get("details", {}),
                    )
                )
            except (KeyError, TypeError):
                continue
        return failures

    def _commit_snapshot(
        self,
        snapshots: SnapshotManager,
        cache: StageCache,
        config_raw: Any,
        raw_map: dict[str, dict[str, Any]],
        cursor: int,
        pending_items: list[WorkItem],
        results: dict[str, dict[str, Any]],
        ledger: list[dict[str, Any]],
        next_ledger_sequence: int,
        consumed_budget: int,
        persisted_failures: list[Failure],
        base_input_fingerprint: str,
        command_stream_fingerprint: str,
        config: RoutingConfig | None,
    ) -> dict[str, Any]:
        data = {
            "config": config_raw,
            "utterances": [raw_map[key] for key in sorted(raw_map)],
            "cursor": cursor,
            "pending": [
                {"utterance_id": item.utterance_id, "command_index": item.command_index}
                for item in pending_items
            ],
            "results": results,
            "ledger": ledger,
            "next_ledger_sequence": next_ledger_sequence,
            "consumed_budget": consumed_budget,
            "persisted_failures": [failure.to_dict() for failure in self._merge_failures(persisted_failures, [])],
        }
        current_hash = config.config_hash if config is not None else fingerprint(config_raw)
        trigger_fp = trigger_table_fingerprint(config) if config is not None else ""
        manifest_updates = {
            "config_hash": current_hash,
            "trigger_fingerprint": trigger_fp,
            "completed_command_cursor": cursor,
            "consumed_budget": consumed_budget,
            "pending_queue": [item.utterance_id for item in pending_items],
            "dataset_fingerprint": fingerprint({"utterances": [raw_map[key] for key in sorted(raw_map)]}),
            "base_input_fingerprint": base_input_fingerprint,
            "command_stream_fingerprint": command_stream_fingerprint,
        }
        return snapshots.save(data=jsonable(data), cache=cache, manifest_updates=manifest_updates)

    def _build_report(
        self,
        *,
        mode: str,
        config: RoutingConfig | None,
        config_raw: Any,
        parsed: dict[str, Utterance],
        results: dict[str, dict[str, Any]],
        failures: list[Failure],
        pending_items: list[WorkItem],
        cache: StageCache,
        ledger: list[dict[str, Any]],
        consumed_budget: int,
        batch_limit: int,
        remaining_budget: int,
        diagnostics: list[dict[str, Any]],
        resumed: bool,
        stale: bool,
        snapshot_manifest: dict[str, Any] | None,
        command_results: list[dict[str, Any]],
        runtime_injection_hits: set[str],
        workers: int,
        base_input_fingerprint: str,
        command_stream_fingerprint: str,
        raw_record_count: int,
    ) -> dict[str, Any]:
        failure_dicts = [failure.to_dict() for failure in self._merge_failures(failures, [])]
        result_list = [results[utterance_id] for utterance_id in sorted(results) if utterance_id in parsed]
        boundary_samples: list[dict[str, Any]] = []
        for result in result_list:
            if not result.get("boundary"):
                continue
            for reason in result.get("boundary_reasons", []):
                stage, _, boundary_reason = reason.partition(":")
                boundary_samples.append(
                    {
                        "utterance_id": result["utterance_id"],
                        "stage": stage,
                        "reason": boundary_reason,
                        "score_int": result.get(f"{stage}_score_int"),
                        "threshold_int": result.get(f"{stage}_threshold_int"),
                        "tied_names": result.get(f"{stage}_tied_names", []),
                    }
                )
        boundary_samples.sort(key=lambda item: (item["utterance_id"], STAGES.index(item["stage"]), item["reason"]))
        pending_queue = [{"utterance_id": item.utterance_id, "command_index": item.command_index} for item in pending_items]
        business_payload = {
            "results": result_list,
            "failures": failure_dicts,
            "boundary_samples": boundary_samples,
            "pending_queue": pending_queue,
            "runtime_injection_hits": sorted(runtime_injection_hits),
        }
        business_digest = fingerprint(business_payload)
        cache_rows = cache.reuse_recompute()
        for row in cache_rows["rows"]:
            row["task_key"] = stable_task_key(row["utterance_id"], row["stage"])
        cache_rows["rows"].sort(key=lambda item: (item["utterance_id"], STAGES.index(item["stage"]), item["status"]))
        summary = {
            "utterance_count": raw_record_count,
            "valid_count": len(parsed),
            "invalid_count": len({
                failure["utterance_id"]
                for failure in failure_dicts
                if failure["business_code"] == ErrorCode.ROUTE_INVALID.value
                and failure["stage"] == "validation"
                and failure["utterance_id"] not in {"__config__", "__commands__"}
                and not failure["utterance_id"].startswith("__")
            }),
            "result_count": len(result_list),
            "failure_count": len(failure_dicts),
            "pending_count": len(pending_items),
            "boundary_count": len({item["utterance_id"] for item in boundary_samples}),
            "reused_count": sum(1 for row in cache_rows["rows"] if row["status"] == "reuse"),
            "recomputed_count": sum(1 for row in cache_rows["rows"] if row["status"] == "recompute"),
            "business_digest": business_digest,
        }
        summary["failed_count"] = summary["invalid_count"]
        recovery = {
            "resumed": resumed,
            "stale": stale,
            "completed_command_cursor": snapshot_manifest.get("completed_command_cursor") if snapshot_manifest else None,
            "global_version": snapshot_manifest.get("global_version") if snapshot_manifest else None,
            "manifest": snapshot_manifest,
            "stage_versions": snapshot_manifest.get("stage_versions", {}) if snapshot_manifest else {},
        }
        report = {
            "schema_version": 1,
            "mode": mode,
            "config_hash": config.config_hash if config is not None else fingerprint(config_raw),
            "base_input_fingerprint": base_input_fingerprint,
            "command_stream_fingerprint": command_stream_fingerprint,
            "summary": summary,
            "business_digest": business_digest,
            "results": result_list,
            "failures": failure_dicts,
            "boundary_samples": boundary_samples,
            "pending_queue": pending_queue,
            "cache": {
                "stages": cache_rows["summary"],
                "reuse_recompute": cache_rows["rows"],
                "workers": workers,
                "parallel": workers > 1,
            },
            "budget": {
                "batch_limit": batch_limit,
                "remaining": remaining_budget,
                "consumed_in_snapshot": consumed_budget,
                "sequence": ledger,
                "pending_queue": [item.utterance_id for item in pending_items],
            },
            "commands": command_results,
            "diagnostics": diagnostics,
            "snapshot": recovery,
            "injections": [
                {"id": injection_id, "fault": "runtime_exception", "hit": injection_id in runtime_injection_hits}
                for injection_id in sorted({injection.injection_id for injection in self.runtime_injections})
            ],
        }
        return jsonable(report)


def run_full(**kwargs: Any) -> dict[str, Any]:
    kwargs["full"] = True
    return Workbench(**kwargs).run()


def run_incremental(**kwargs: Any) -> dict[str, Any]:
    kwargs["full"] = False
    return Workbench(**kwargs).run()
