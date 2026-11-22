"""Stage-cached, deterministic query-intent pipeline."""

from __future__ import annotations

import copy
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

from .canonical import digest
from .entities import (
    build_entity_patterns,
    find_candidates,
    resolve_candidates,
    slot_cache_key,
    slot_output_fingerprint,
)
from .errors import E_QUERY_EXEC, issue, sort_issues
from .intent import (
    classify_intent,
    find_feature_matches,
    intent_cache_key,
    intent_output_fingerprint,
    intent_parameter_fingerprint,
)
from .normalization import (
    normalization_cache_key,
    normalize_tokens,
    output_fingerprint as normalization_output_fingerprint,
)
from .rewrite import (
    build_rewrite,
    rewrite_cache_key,
    rewrite_output_fingerprint,
    select_rewrite_rule,
)


STAGES = ("normalization", "slots", "intent", "rewrite")


def query_content_fingerprint(query: dict[str, Any]) -> str:
    return digest(
        {
            "query_id": query["query_id"],
            "terms": query["terms"],
            "session_id": query["session_id"],
            "turn": query["turn"],
            "intent_label": query["intent_label"],
        }
    )


def config_fingerprint(config: dict[str, Any]) -> str:
    return digest(config)


def rules_fingerprint(rules: dict[str, Any]) -> str:
    return digest(rules)


def dictionary_fingerprint(config: dict[str, Any]) -> str:
    return digest(config.get("entities", []))


def semantic_projection(run: dict[str, Any]) -> dict[str, Any]:
    """Project completed state while excluding scheduler/cache hit metadata."""

    queries: list[dict[str, Any]] = []
    for record in run.get("queries", []):
        projected = {
            "query_id": record["query_id"],
            "query_sort_key": record["query_sort_key"],
            "content_fingerprint": record["content_fingerprint"],
            "input": record["input"],
            "normalization": record["normalization"],
            "slots": record["slots"],
            "intent": record["intent"],
            "rewrite": record["rewrite"],
            "stage_keys": {
                stage: {
                    "cache_key": record["stages"][stage]["cache_key"],
                    "output_fingerprint": record["stages"][stage]["output_fingerprint"],
                    "sort_key": record["stages"][stage]["sort_key"],
                }
                for stage in STAGES
            },
        }
        queries.append(projected)
    return {"queries": queries, "failures": run.get("failures", [])}


class Workbench:
    """Runs the four query stages with exact, per-query stage reuse."""

    def __init__(
        self,
        config: dict[str, Any],
        rules: dict[str, Any],
        *,
        max_workers: int | None = None,
        previous_stage_cache: dict[tuple[str, str], dict[str, Any]] | None = None,
    ) -> None:
        self.config = config
        self.rules = rules
        configured_workers = int(config.get("max_workers", 1))
        requested_workers = configured_workers if max_workers is None else int(max_workers)
        self.requested_workers = requested_workers
        self.max_workers = max(1, min(configured_workers, requested_workers))
        self.previous_stage_cache = previous_stage_cache or {}
        self._active = 0
        self._peak = 0
        self._lock = threading.Lock()
        self.entity_patterns = build_entity_patterns(config, rules)
        self.config_hash = config_fingerprint(config)
        self.rule_fingerprint = rules_fingerprint(rules)
        self.dictionary_hash = dictionary_fingerprint(config)

    def _enter_task(self) -> None:
        with self._lock:
            self._active += 1
            self._peak = max(self._peak, self._active)

    def _exit_task(self) -> None:
        with self._lock:
            self._active -= 1

    def _reuse(
        self,
        query_id: str,
        stage: str,
        cache_key: str,
    ) -> tuple[dict[str, Any] | None, str | None]:
        previous = self.previous_stage_cache.get((query_id, stage))
        if previous is None:
            return None, None
        if previous.get("cache_key") == cache_key:
            return previous, "reused"
        return None, "invalidated"

    @staticmethod
    def _stage_record(
        *,
        cache_key: str,
        output_fingerprint: str,
        sort_key: str,
        reuse_status: str | None,
        data: dict[str, Any],
        references: dict[str, Any],
    ) -> dict[str, Any]:
        if reuse_status == "reused":
            status = "reused"
        elif reuse_status == "invalidated":
            status = "invalidated"
        else:
            status = "new"
        return {
            "cache_key": cache_key,
            "output_fingerprint": output_fingerprint,
            "sort_key": sort_key,
            "cache_status": status,
            "reused": status == "reused",
            "references": references,
            "data": data,
        }

    def _run_query(self, query: dict[str, Any]) -> dict[str, Any]:
        query_id = query["query_id"]
        content_fingerprint = query_content_fingerprint(query)
        stage_records: dict[str, dict[str, Any]] = {}

        norm_key, rule_references = normalization_cache_key(
            content_fingerprint, query["terms"], self.rules, self.config
        )
        previous, reuse_status = self._reuse(query_id, "normalization", norm_key)
        if previous is not None:
            norm_data = copy.deepcopy(previous["data"])
            norm_output_fingerprint = previous["output_fingerprint"]
        else:
            tokens, token_records, normalized_terms = normalize_tokens(
                query["terms"], self.rules, self.config
            )
            norm_data = {
                "tokens": tokens,
                "token_records": token_records,
                "normalized_terms": normalized_terms,
                "normalized_text": " ".join(tokens),
            }
            norm_output_fingerprint = normalization_output_fingerprint(
                tokens, normalized_terms, rule_references
            )
        stage_records["normalization"] = self._stage_record(
            cache_key=norm_key,
            output_fingerprint=norm_output_fingerprint,
            sort_key=f"{query_id}::normalization",
            reuse_status=reuse_status,
            data=norm_data,
            references={
                "rules": rule_references,
                "terms": norm_data["tokens"],
                "entity_entries": [],
                "intents": [],
                "rewrite_rules": [],
            },
        )

        tokens = norm_data["tokens"]
        candidates = find_candidates(tokens, self.entity_patterns)
        slot_key, dictionary_references = slot_cache_key(
            content_fingerprint,
            norm_key,
            candidates,
            self.rules,
            self.config,
        )
        previous, reuse_status = self._reuse(query_id, "slots", slot_key)
        if previous is not None:
            slot_data = copy.deepcopy(previous["data"])
            slots_output_fingerprint = previous["output_fingerprint"]
        else:
            slots, suppressed = resolve_candidates(candidates)
            slot_data = {
                "slots": slots,
                "suppressed": suppressed,
                "candidate_count": len(candidates),
                "accepted_count": len(slots),
                "suppressed_count": len(suppressed),
            }
            slots_output_fingerprint = slot_output_fingerprint(slots, suppressed)
        stage_records["slots"] = self._stage_record(
            cache_key=slot_key,
            output_fingerprint=slots_output_fingerprint,
            sort_key=f"{query_id}::slots",
            reuse_status=reuse_status,
            data=slot_data,
            references={
                "rules": [],
                "terms": tokens,
                "entity_entries": dictionary_references,
                "intents": [],
                "rewrite_rules": [],
            },
        )

        feature_matches = find_feature_matches(tokens, self.config, self.rules)
        parameter_fingerprint, parameter_references = intent_parameter_fingerprint(
            feature_matches, self.config, self.rules
        )
        intent_key = intent_cache_key(
            content_fingerprint,
            norm_key,
            parameter_fingerprint,
            self.rules,
            slot_key,
        )
        previous, reuse_status = self._reuse(query_id, "intent", intent_key)
        if previous is not None:
            intent_data = copy.deepcopy(previous["data"])
            intent_result_fingerprint = previous["output_fingerprint"]
        else:
            intent_data = classify_intent(feature_matches, self.config)
            intent_result_fingerprint = intent_output_fingerprint(intent_data)
        stage_records["intent"] = self._stage_record(
            cache_key=intent_key,
            output_fingerprint=intent_result_fingerprint,
            sort_key=f"{query_id}::intent",
            reuse_status=reuse_status,
            data=intent_data,
            references={
                "rules": [],
                "terms": tokens,
                "entity_entries": dictionary_references,
                "intents": parameter_references,
                "rewrite_rules": [],
            },
        )

        primary_intent = intent_data["primary_intent"]
        rewrite_rule = select_rewrite_rule(primary_intent, self.rules)
        rewrite_key = rewrite_cache_key(
            content_fingerprint,
            norm_key,
            intent_key,
            slot_key,
            rewrite_rule,
            self.rules,
        )
        previous, reuse_status = self._reuse(query_id, "rewrite", rewrite_key)
        if previous is not None:
            rewrite_data = copy.deepcopy(previous["data"])
            rewrite_result_fingerprint = previous["output_fingerprint"]
        else:
            rewrite_data = build_rewrite(
                norm_data["normalized_text"],
                primary_intent,
                slot_data["slots"],
                self.rules,
            )
            rewrite_result_fingerprint = rewrite_output_fingerprint(rewrite_data)
        stage_records["rewrite"] = self._stage_record(
            cache_key=rewrite_key,
            output_fingerprint=rewrite_result_fingerprint,
            sort_key=f"{query_id}::rewrite",
            reuse_status=reuse_status,
            data=rewrite_data,
            references={
                "rules": [],
                "terms": tokens,
                "entity_entries": dictionary_references,
                "intents": parameter_references,
                "rewrite_rules": (
                    [rewrite_rule["id"]] if rewrite_rule is not None else []
                ),
            },
        )

        return {
            "query_id": query_id,
            "query_sort_key": query_id,
            "content_fingerprint": content_fingerprint,
            "input": copy.deepcopy(query),
            "normalization": stage_records["normalization"]["data"],
            "slots": stage_records["slots"]["data"],
            "intent": stage_records["intent"]["data"],
            "rewrite": stage_records["rewrite"]["data"],
            "stages": stage_records,
        }

    def _task_with_failure(
        self, query: dict[str, Any]
    ) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
        self._enter_task()
        try:
            # Yield briefly so the diagnostic peak actually observes overlapping
            # tasks even when individual stage work is sub-millisecond.
            time.sleep(0.001)
            return self._run_query(query), None
        except Exception as exc:  # single-query isolation boundary
            failure = issue(
                E_QUERY_EXEC,
                f"{type(exc).__name__}: {exc}",
                query_id=query.get("query_id", "unknown"),
                stage="pipeline",
            )
            return None, failure
        finally:
            self._exit_task()

    def run(
        self,
        queries: list[dict[str, Any]],
        *,
        input_failures: list[dict[str, Any]] | None = None,
        global_version: int = 0,
        command_index: int = 0,
    ) -> dict[str, Any]:
        ordered_queries = sorted(queries, key=lambda item: item["query_id"])
        records: list[dict[str, Any]] = []
        runtime_failures: list[dict[str, Any]] = []
        with ThreadPoolExecutor(
            max_workers=self.max_workers,
            thread_name_prefix="intent-stage",
        ) as executor:
            futures = {
                executor.submit(self._task_with_failure, query): query["query_id"]
                for query in ordered_queries
            }
            for future in as_completed(futures):
                record, failure = future.result()
                if record is not None:
                    records.append(record)
                if failure is not None:
                    runtime_failures.append(failure)
        records.sort(key=lambda item: item["query_id"])
        runtime_failures.sort(key=lambda item: item["query_id"])

        reuse_report: dict[str, dict[str, list[str]]] = {}
        for stage in STAGES:
            reused: list[str] = []
            invalidated: list[str] = []
            new: list[str] = []
            for record in records:
                status = record["stages"][stage]["cache_status"]
                if status == "reused":
                    reused.append(record["query_id"])
                elif status == "invalidated":
                    invalidated.append(record["query_id"])
                else:
                    new.append(record["query_id"])
            reuse_report[stage] = {
                "reused": reused,
                "invalidated": invalidated,
                "new": new,
                "recomputed": sorted(invalidated + new),
            }

        failures = sort_issues(list(input_failures or []) + runtime_failures)
        return {
            "global_version": global_version,
            "command_index": command_index,
            "queries": records,
            "failures": failures,
            "reuse_report": reuse_report,
            "metadata": {
                "config_hash": self.config_hash,
                "rule_fingerprint": self.rule_fingerprint,
                "dictionary_hash": self.dictionary_hash,
                "query_count": len(records),
                "failure_count": len(failures),
                "configured_max_workers": int(self.config.get("max_workers", 1)),
                "requested_max_workers": self.requested_workers,
                "effective_max_workers": self.max_workers,
                "peak_active_tasks": self._peak,
                "stage_order": list(STAGES),
                "task_schedule": [
                    f"{query['query_id']}::{stage}"
                    for query in ordered_queries
                    for stage in STAGES
                ],
            },
        }

    @staticmethod
    def stage_cache_from_run(run: dict[str, Any]) -> dict[tuple[str, str], dict[str, Any]]:
        cache: dict[tuple[str, str], dict[str, Any]] = {}
        for record in run.get("queries", []):
            for stage in STAGES:
                stage_record = record["stages"][stage]
                cache[(record["query_id"], stage)] = {
                    "cache_key": stage_record["cache_key"],
                    "output_fingerprint": stage_record["output_fingerprint"],
                    "data": copy.deepcopy(stage_record["data"]),
                }
        return cache
