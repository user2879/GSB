"""End-to-end deterministic workbench orchestration."""

from __future__ import annotations

import shutil
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from decimal import Decimal
from functools import partial
from pathlib import Path
from typing import Any, Callable

from .canonical import (
    decimal_text,
    fixed_str,
    payload_fingerprint_is_valid,
    sha256_obj,
    sha256_text,
    to_fixed,
)
from .codes import (
    BusinessFault,
    E_EVIDENCE_BUDGET_PENDING,
    E_INTERNAL,
    E_INPUT_IO,
    E_RELATION_UNKNOWN_TYPE,
    E_SNAPSHOT_CONFIG_MISMATCH,
    E_SNAPSHOT_CORRUPT,
    E_SNAPSHOT_IO,
    E_SNAPSHOT_VERSION_MISMATCH,
    E_WORKORDER_BUDGET_EXHAUSTED,
    failure_sort_key,
    sorted_failures,
)
from .loading import (
    group_evidence,
    load_config_file,
    load_evidence_file,
    load_injections as _load_injections,
    load_input_version,
    load_rules_file,
    validate_workorder_evidence,
)
from .models import ArbitrationConfig, Evidence, Injection, Relation, RulePackage, WorkOrderContext
from .scoring import (
    AGGREGATION_ALGORITHM,
    DECAY_ALGORITHM,
    RELATION_ALGORITHM,
    applicable_relations,
    compute_aggregation_stage,
    compute_decay_stage,
    compute_relation_stage,
    relation_strength_q,
)
from .snapshot import SnapshotStore


INJECTION_CODE_MAP = {
    "unknown_evidence_type": "E_UNKNOWN_EVIDENCE_TYPE",
    "confidence_out_of_range": "E_CONFIDENCE_RANGE",
    "negative_timestamp": "E_EVENT_TIME_INVALID",
    "zero_half_life": "E_HALF_LIFE_ZERO",
    "relation_unknown_type": "E_RELATION_UNKNOWN_TYPE",
}


def _failure_from_fault(fault: BusinessFault) -> dict[str, Any]:
    failure = fault.as_failure()
    if not failure.get("workorder_id"):
        failure["workorder_id"] = "<GLOBAL>"
    return failure


def _empty_report() -> dict[str, Any]:
    return {
        "report_version": 1,
        "fatal": False,
        "run": {},
        "summary": {
            "workorder_count": 0,
            "successful_count": 0,
            "failure_count": 0,
            "pending_count": 0,
            "boundary_count": 0,
            "boundary_workorders": [],
            "boundary_by_priority": {},
            "priority_distribution": {},
            "failure_code_distribution": {},
        },
        "workorders": [],
        "failures": [],
        "ledger": {
            "reused_workorders": [],
            "recomputed_workorders": [],
            "removed_workorders": [],
            "pending_workorders": [],
            "scheduled_decay_order": [],
            "budget": {
                "batch_limit": 0,
                "consumed_this_run": 0,
                "remaining_this_run": 0,
                "consumed_total": 0,
            },
            "stage_reuse_counts": {},
        },
        "snapshot": {},
        "injections": {"configured_count": 0, "hits": [], "misses": []},
    }


def _fatal_report(
    fault: BusinessFault,
    *,
    mode: str,
    config: ArbitrationConfig | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    report = _empty_report()
    report["fatal"] = True
    report["run"] = {
        "mode": mode,
        "config_hash": config.config_hash if config else None,
        "deterministic": True,
    }
    if extra:
        report.update(extra)
    report["failures"] = sorted_failures([_failure_from_fault(fault)])
    report["summary"]["failure_count"] = 1
    report["summary"]["failure_code_distribution"] = {fault.code: 1}
    return report


def _validate_mode(mode: str) -> str:
    if mode not in {"serial", "parallel"}:
        raise BusinessFault("E_CONFIG_SCHEMA", "mode must be serial or parallel", "global")
    return mode


def _evidence_fingerprint(evidence: list[Evidence]) -> str:
    return sha256_obj([record.canonical() for record in sorted(evidence, key=lambda item: item.evidence_seq)])


def _decay_cache_key(
    evidence: list[Evidence], rules: RulePackage, config: ArbitrationConfig, half_life_ms: int
) -> str:
    evidence_types = sorted({record.evidence_type for record in evidence})
    source_roles = sorted({record.source_role for record in evidence})
    return sha256_obj(
        {
            "algorithm": DECAY_ALGORITHM,
            "fixed_decimal_width": config.fixed_decimal_width,
            "quantization_step": config.quantization_step,
            "half_life_ms": half_life_ms,
            "evidence": [record.canonical() for record in sorted(evidence, key=lambda item: item.evidence_seq)],
            "weights": {key: decimal_text(rules.weights[key]) for key in evidence_types},
            "role_credibility": {key: decimal_text(rules.role_credibility[key]) for key in source_roles},
        }
    )


def _relation_cache_key(
    decay_payload: dict[str, Any],
    relations: list[Relation],
    config: ArbitrationConfig,
    injection_ids_by_relation: dict[str, str],
) -> str:
    relation_inputs = []
    for relation in sorted(relations, key=lambda item: item.relation_id):
        relation_inputs.append(
            {
                "relation": relation.canonical(),
                "resolved_strength_q": relation_strength_q(relation, config),
                "injection_id": injection_ids_by_relation.get(relation.relation_id),
            }
        )
    return sha256_obj(
        {
            "algorithm": RELATION_ALGORITHM,
            "upstream_content_fingerprint": decay_payload["content_fingerprint"],
            "fixed_decimal_width": config.fixed_decimal_width,
            "quantization_step": config.quantization_step,
            "exclusion_strength": decimal_text(config.exclusion_strength),
            "support_strength": decimal_text(config.support_strength),
            "zero_lower_bound": decimal_text(config.zero_lower_bound),
            "relations": relation_inputs,
        }
    )


def _aggregation_cache_key(
    relation_payload: dict[str, Any], rules: RulePackage, config: ArbitrationConfig
) -> str:
    return sha256_obj(
        {
            "algorithm": AGGREGATION_ALGORITHM,
            "upstream_content_fingerprint": relation_payload["content_fingerprint"],
            "fixed_decimal_width": config.fixed_decimal_width,
            "quantization_step": config.quantization_step,
            "fusion_mode": config.fusion_mode,
            "boundary_policy": "INCLUSIVE_UPPER",
            "thresholds": [threshold.canonical() for threshold in rules.thresholds],
        }
    )


def _reusable_payload(record: dict[str, Any] | None, key_field: str, key: str, payload_field: str) -> dict[str, Any] | None:
    if not record or record.get(key_field) != key:
        return None
    payload = record.get(payload_field)
    if not payload_fingerprint_is_valid(payload):
        raise BusinessFault(
            E_SNAPSHOT_CORRUPT,
            f"persisted {payload_field} failed content fingerprint validation",
            "snapshot",
            details={"snapshot_workorder_id": record.get("workorder_id")},
        )
    return payload


def _apply_load_injections(
    records: list[Evidence], injections: list[Injection]
) -> tuple[list[Evidence], dict[int, str], dict[str, dict[str, Any]], set[str]]:
    records = list(records)
    injection_by_seq: dict[int, str] = {}
    hits: dict[str, dict[str, Any]] = {}
    applied: set[str] = set()
    for injection in injections:
        if injection.stage != "load":
            continue
        target_index = next(
            (index for index, record in enumerate(records) if record.evidence_seq == injection.evidence_seq),
            None,
        )
        if target_index is None:
            continue
        original = records[target_index]
        if injection.fault == "unknown_evidence_type":
            mutated = replace(original, evidence_type="INJECTED_UNKNOWN_EVIDENCE_TYPE")
        elif injection.fault == "confidence_out_of_range":
            mutated = replace(original, confidence=Decimal("1.25"), confidence_text="1.25")
        elif injection.fault == "negative_timestamp":
            mutated = replace(original, event_ms=-1)
        else:
            continue
        records[target_index] = mutated
        injection_by_seq[original.evidence_seq] = injection.injection_id
        applied.add(injection.injection_id)
        hits[injection.injection_id] = {
            "injection_id": injection.injection_id,
            "workorder_id": injection.workorder_id,
            "stage": injection.stage,
            "fault": injection.fault,
            "mapped_code": INJECTION_CODE_MAP[injection.fault],
            "evidence_seq": original.evidence_seq,
            "hit": True,
        }
    return records, injection_by_seq, hits, applied


def _execute_stage_tasks(
    tasks: list[dict[str, Any]], mode: str, concurrency: int
) -> dict[str, dict[str, Any]]:
    if not tasks:
        return {}

    def run_one(task: dict[str, Any]) -> dict[str, Any]:
        try:
            payload = task["runner"]()
            return {"workorder_id": task["workorder_id"], "stage": task["stage"], "ok": True, "payload": payload}
        except BusinessFault as fault:
            if not fault.workorder_id:
                fault.workorder_id = task["workorder_id"]
            if fault.stage != task["stage"]:
                fault.stage = task["stage"]
            return {
                "workorder_id": task["workorder_id"],
                "stage": task["stage"],
                "ok": False,
                "failure": _failure_from_fault(fault),
            }
        except Exception as exc:  # noqa: BLE001 - isolated into the report by design
            return {
                "workorder_id": task["workorder_id"],
                "stage": task["stage"],
                "ok": False,
                "failure": {
                    "workorder_id": task["workorder_id"],
                    "stage": task["stage"],
                    "code": E_INTERNAL,
                    "message": "unexpected exception in work order stage",
                    "exception_type": exc.__class__.__name__,
                },
            }

    outcomes: dict[str, dict[str, Any]] = {}
    if mode == "serial" or concurrency == 1:
        for task in tasks:
            outcomes[task["workorder_id"]] = run_one(task)
        return outcomes

    workers = max(1, min(concurrency, len(tasks)))
    with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="arbitration-stage") as executor:
        futures = [executor.submit(run_one, task) for task in tasks]
        for task, future in zip(tasks, futures):
            outcomes[task["workorder_id"]] = future.result()
    return outcomes


def _task_key(workorder_id: str, stage: str) -> str:
    return sha256_text(f"{workorder_id}:{stage}:stage-task-v1")


def _record_from_context(context: WorkOrderContext) -> dict[str, Any] | None:
    if context.failure is not None:
        if context.failure.get("code") == E_EVIDENCE_BUDGET_PENDING:
            return None
        return {
            "workorder_id": context.workorder_id,
            "evidence_fingerprint": context.evidence_fingerprint,
            "failure": context.failure,
        }
    if not all(
        (
            context.decay_key,
            context.decay_payload,
            context.relation_key,
            context.relation_payload,
            context.aggregate_key,
            context.aggregate_payload,
        )
    ):
        return None
    references = context.references or {
        "evidence_seqs": sorted(record.evidence_seq for record in context.evidence),
        "relation_ids": list(context.relation_payload.get("referenced_relation_ids", [])),
    }
    return {
        "workorder_id": context.workorder_id,
        "evidence_fingerprint": context.evidence_fingerprint,
        "decay_key": context.decay_key,
        "decay_payload": context.decay_payload,
        "relation_key": context.relation_key,
        "relation_payload": context.relation_payload,
        "aggregate_key": context.aggregate_key,
        "aggregate_payload": context.aggregate_payload,
        "references": references,
        "failure": None,
    }


def _workorder_view(context: WorkOrderContext) -> dict[str, Any]:
    assert context.decay_payload is not None
    assert context.relation_payload is not None
    assert context.aggregate_payload is not None
    aggregate = context.aggregate_payload
    return {
        "workorder_id": context.workorder_id,
        "evidence_count": len(context.evidence),
        "evidence_seqs": sorted(record.evidence_seq for record in context.evidence),
        "decision_score_q": aggregate["decision_score_q"],
        "decision_score": aggregate["decision_score"],
        "upgrade_priority": aggregate["upgrade_priority"],
        "boundary": aggregate["boundary"],
        "boundary_threshold_priorities": aggregate["boundary_threshold_priorities"],
        "matched_threshold_priority": aggregate["matched_threshold_priority"],
        "references": context.references,
        "cache_keys": {
            "decay": context.decay_key,
            "relations": context.relation_key,
            "aggregation": context.aggregate_key,
        },
        "decay": context.decay_payload,
        "relations": context.relation_payload,
        "aggregation": context.aggregate_payload,
    }


def _run_workbench(
    data_dir: Path,
    state_dir: Path,
    *,
    mode: str,
    injections_path: Path | None,
    rebuild_state: bool,
) -> dict[str, Any]:
    mode = _validate_mode(mode)
    data_dir = data_dir.resolve()
    state_dir = state_dir.resolve()
    if state_dir == data_dir:
        raise BusinessFault("E_CONFIG_SCHEMA", "state directory must differ from data directory", "global")

    config = load_config_file(data_dir / "config.json")
    if rebuild_state and state_dir.exists():
        shutil.rmtree(state_dir)
    state_dir.mkdir(parents=True, exist_ok=True)
    store = SnapshotStore(state_dir)
    previous = store.load_current()
    input_version = load_input_version(data_dir)
    previous_records: dict[str, dict[str, Any]] = {}
    previous_pending: list[str] = []
    previous_total_budget = 0
    if previous is not None:
        previous_manifest, previous_records = previous
        if previous_manifest.get("config_hash") != config.config_hash:
            fault = BusinessFault(
                E_SNAPSHOT_CONFIG_MISMATCH,
                "snapshot config_hash differs from current config; full rebuild is required",
                "snapshot",
                details={
                    "snapshot_config_hash": previous_manifest.get("config_hash"),
                    "current_config_hash": config.config_hash,
                },
            )
            return _fatal_report(
                fault,
                mode=mode,
                config=config,
                extra={"snapshot": {"rejected_manifest": previous_manifest}},
            )
        snapshot_input_version = previous_manifest.get("input_version", 0)
        if not isinstance(snapshot_input_version, int) or isinstance(snapshot_input_version, bool):
            raise BusinessFault(E_SNAPSHOT_CORRUPT, "snapshot input_version is invalid", "snapshot")
        if input_version < snapshot_input_version:
            fault = BusinessFault(
                E_SNAPSHOT_VERSION_MISMATCH,
                "input version is older than the snapshot version; full rebuild is required",
                "snapshot",
                details={
                    "snapshot_input_version": snapshot_input_version,
                    "current_input_version": input_version,
                },
            )
            return _fatal_report(
                fault,
                mode=mode,
                config=config,
                extra={"snapshot": {"rejected_manifest": previous_manifest}},
            )
        previous_pending = sorted(
            item for item in previous_manifest.get("pending_workorders", []) if isinstance(item, str)
        )
        raw_total_budget = previous_manifest.get("consumed_budget_total", 0)
        if not isinstance(raw_total_budget, int) or isinstance(raw_total_budget, bool) or raw_total_budget < 0:
            raise BusinessFault(E_SNAPSHOT_CORRUPT, "snapshot consumed_budget_total is invalid", "snapshot")
        previous_total_budget = raw_total_budget

    rules = load_rules_file(data_dir / "rules.json")
    raw_evidence, evidence_failures = load_evidence_file(data_dir / "evidence.jsonl")
    injections = _load_injections(injections_path)
    evidence_groups = group_evidence(raw_evidence)

    failures: list[dict[str, Any]] = []
    structural_by_workorder: dict[str, list[dict[str, Any]]] = {}
    for failure in evidence_failures:
        workorder_id = str(failure.get("workorder_id", "<GLOBAL>"))
        if workorder_id == "<GLOBAL>":
            failures.append(failure)
        else:
            structural_by_workorder.setdefault(workorder_id, []).append(failure)

    injections_by_workorder: dict[str, list[Injection]] = {}
    for injection in injections:
        injections_by_workorder.setdefault(injection.workorder_id, []).append(injection)
    injection_status: dict[str, dict[str, Any] | None] = {injection.injection_id: None for injection in injections}

    current_ids = sorted(set(evidence_groups) | set(structural_by_workorder))
    contexts: dict[str, WorkOrderContext] = {}
    stage_injections: dict[str, list[Injection]] = {}
    relation_injection_ids: dict[str, dict[str, str]] = {}

    for workorder_id in current_ids:
        base_records = list(evidence_groups.get(workorder_id, []))
        workorder_injections = sorted(
            injections_by_workorder.get(workorder_id, []), key=lambda item: item.injection_id
        )
        load_injections = [item for item in workorder_injections if item.stage == "load"]
        mutated_records, injection_by_seq, load_hits, _ = _apply_load_injections(base_records, load_injections)
        for injection in load_injections:
            if injection.injection_id not in load_hits:
                injection_status[injection.injection_id] = {
                    "injection_id": injection.injection_id,
                    "workorder_id": injection.workorder_id,
                    "stage": injection.stage,
                    "fault": injection.fault,
                    "mapped_code": INJECTION_CODE_MAP[injection.fault],
                    "evidence_seq": injection.evidence_seq,
                    "hit": False,
                    "reason": "target evidence_seq not found",
                }
            else:
                injection_status[injection.injection_id] = load_hits[injection.injection_id]

        semantic_failures = validate_workorder_evidence(
            mutated_records, rules, injection_by_seq=injection_by_seq
        )
        load_failures = [*structural_by_workorder.get(workorder_id, []), *semantic_failures]
        load_failures = sorted(load_failures, key=failure_sort_key)
        failures.extend(load_failures)

        effective_half_life = rules.half_life_ms
        extra_relations: list[Relation] = []
        relation_ids_for_workorder: dict[str, str] = {}
        for injection in workorder_injections:
            if injection.stage == "decay" and injection.fault == "zero_half_life":
                if base_records:
                    effective_half_life = 0
                else:
                    injection_status[injection.injection_id] = {
                        "injection_id": injection.injection_id,
                        "workorder_id": injection.workorder_id,
                        "stage": injection.stage,
                        "fault": injection.fault,
                        "mapped_code": INJECTION_CODE_MAP[injection.fault],
                        "evidence_seq": injection.evidence_seq,
                        "hit": False,
                        "reason": "work order has no evidence",
                    }
            elif injection.stage == "relations" and injection.fault == "relation_unknown_type":
                if base_records:
                    target_type = base_records[0].evidence_type
                    relation_id = f"__injected__{injection.injection_id}"
                    extra_relations.append(
                        Relation(
                            relation_id=relation_id,
                            kind="mutual_exclusion",
                            left_type="INJECTED_UNKNOWN_RELATION_TYPE",
                            right_type=target_type,
                        )
                    )
                    relation_ids_for_workorder[relation_id] = injection.injection_id
                else:
                    injection_status[injection.injection_id] = {
                        "injection_id": injection.injection_id,
                        "workorder_id": injection.workorder_id,
                        "stage": injection.stage,
                        "fault": injection.fault,
                        "mapped_code": INJECTION_CODE_MAP[injection.fault],
                        "evidence_seq": injection.evidence_seq,
                        "hit": False,
                        "reason": "work order has no evidence",
                    }

        context = WorkOrderContext(
            workorder_id=workorder_id,
            evidence=mutated_records,
            load_failures=load_failures,
            injection_hits=[],
            extra_relations=extra_relations,
            effective_half_life_ms=effective_half_life,
            evidence_fingerprint=_evidence_fingerprint(mutated_records),
            previous_record=previous_records.get(workorder_id),
        )
        if load_failures:
            context.failure = load_failures[0]
        contexts[workorder_id] = context
        stage_injections[workorder_id] = workorder_injections
        relation_injection_ids[workorder_id] = relation_ids_for_workorder

    current_ids = sorted(contexts)
    removed_workorders = sorted(set(previous_records) - set(current_ids))
    stage_reuse_counts = {"decay": 0, "relations": 0, "aggregation": 0}
    scheduled_decay_order: list[str] = []
    pending_workorders: list[str] = []
    budget_remaining = config.batch_evidence_budget
    budget_consumed = 0

    for workorder_id in current_ids:
        context = contexts[workorder_id]
        if context.failure is not None:
            continue
        context.decay_key = _decay_cache_key(
            context.evidence, rules, config, context.effective_half_life_ms
        )
        reusable = _reusable_payload(context.previous_record, "decay_key", context.decay_key, "decay_payload")
        if reusable is not None:
            context.decay_payload = reusable
            stage_reuse_counts["decay"] += 1

    needs_decay = [
        context for context in contexts.values() if context.failure is None and context.decay_payload is None
    ]
    pending_priority = set(previous_pending)
    needs_decay.sort(key=lambda context: (0 if context.workorder_id in pending_priority else 1, context.workorder_id))
    scheduled_decay_order = [context.workorder_id for context in needs_decay]
    decay_task_contexts: list[WorkOrderContext] = []
    for context in needs_decay:
        required_points = len(context.evidence)
        if required_points <= 0:
            context.failure = {
                "workorder_id": context.workorder_id,
                "stage": "load",
                "code": "E_EVIDENCE_EMPTY",
                "message": "work order has no evidence",
                "exception_type": "BusinessFault",
            }
            failures.append(context.failure)
            continue
        if required_points > config.workorder_evidence_quota:
            charged = min(config.workorder_evidence_quota, budget_remaining)
            budget_remaining -= charged
            budget_consumed += charged
            context.failure = {
                "workorder_id": context.workorder_id,
                "stage": "scheduler",
                "code": E_WORKORDER_BUDGET_EXHAUSTED,
                "message": "work order evidence points exceed the per-work-order quota",
                "exception_type": "BusinessFault",
                "details": {
                    "required_points": required_points,
                    "quota": config.workorder_evidence_quota,
                    "charged_points": charged,
                },
            }
            failures.append(context.failure)
            continue
        if required_points > budget_remaining:
            pending_workorders.append(context.workorder_id)
            context.failure = {
                "workorder_id": context.workorder_id,
                "stage": "scheduler",
                "code": E_EVIDENCE_BUDGET_PENDING,
                "message": "global evidence budget exhausted; work order retained for the next batch",
                "exception_type": "BusinessFault",
                "details": {
                    "required_points": required_points,
                    "remaining_points": budget_remaining,
                },
            }
            failures.append(context.failure)
            continue
        budget_remaining -= required_points
        budget_consumed += required_points
        decay_task_contexts.append(context)

    decay_tasks: list[dict[str, Any]] = []
    for context in decay_task_contexts:
        zero_injection = next(
            (
                injection
                for injection in stage_injections.get(context.workorder_id, [])
                if injection.stage == "decay" and injection.fault == "zero_half_life"
            ),
            None,
        )
        decay_tasks.append(
            {
                "workorder_id": context.workorder_id,
                "stage": "decay",
                "stable_key": _task_key(context.workorder_id, "decay"),
                "runner": partial(
                    compute_decay_stage,
                    context.workorder_id,
                    context.evidence,
                    rules,
                    config,
                    effective_half_life_ms=context.effective_half_life_ms,
                    injection_id=zero_injection.injection_id if zero_injection else None,
                ),
            }
        )
    decay_outcomes = _execute_stage_tasks(decay_tasks, mode, config.concurrency)
    for context in decay_task_contexts:
        outcome = decay_outcomes[context.workorder_id]
        if outcome["ok"]:
            context.decay_payload = outcome["payload"]
            context.stage_computed["decay"] = True
        else:
            context.failure = outcome["failure"]
            failures.append(context.failure)
            injection_id = context.failure.get("injection_id")
            if injection_id:
                injection_status[injection_id] = {
                    "injection_id": injection_id,
                    "workorder_id": context.workorder_id,
                    "stage": "decay",
                    "fault": "zero_half_life",
                    "mapped_code": context.failure.get("code"),
                    "evidence_seq": None,
                    "hit": True,
                }

    relation_tasks: list[dict[str, Any]] = []
    for workorder_id in current_ids:
        context = contexts[workorder_id]
        if context.failure is not None or context.decay_payload is None:
            continue
        relations = applicable_relations(context.evidence, rules, context.extra_relations)
        relation_ids_for_workorder = relation_injection_ids.get(workorder_id, {})
        context.relation_key = _relation_cache_key(
            context.decay_payload, relations, config, relation_ids_for_workorder
        )
        reusable = _reusable_payload(
            context.previous_record, "relation_key", context.relation_key, "relation_payload"
        )
        if reusable is not None:
            context.relation_payload = reusable
            stage_reuse_counts["relations"] += 1
            continue
        relation_tasks.append(
            {
                "workorder_id": workorder_id,
                "stage": "relations",
                "stable_key": _task_key(workorder_id, "relations"),
                "runner": partial(
                    compute_relation_stage,
                    workorder_id,
                    context.evidence,
                    context.decay_payload,
                    relations,
                    rules,
                    config,
                    injection_ids_by_relation=relation_ids_for_workorder,
                ),
            }
        )
    relation_outcomes = _execute_stage_tasks(relation_tasks, mode, config.concurrency)
    for workorder_id, outcome in sorted(relation_outcomes.items()):
        context = contexts[workorder_id]
        if outcome["ok"]:
            context.relation_payload = outcome["payload"]
            context.stage_computed["relations"] = True
        else:
            context.failure = outcome["failure"]
            failures.append(context.failure)
            injection_id = context.failure.get("injection_id")
            if injection_id:
                injection_status[injection_id] = {
                    "injection_id": injection_id,
                    "workorder_id": workorder_id,
                    "stage": "relations",
                    "fault": "relation_unknown_type",
                    "mapped_code": context.failure.get("code"),
                    "evidence_seq": None,
                    "hit": True,
                }

    aggregation_tasks: list[dict[str, Any]] = []
    for workorder_id in current_ids:
        context = contexts[workorder_id]
        if context.failure is not None or context.relation_payload is None:
            continue
        context.aggregate_key = _aggregation_cache_key(context.relation_payload, rules, config)
        reusable = _reusable_payload(
            context.previous_record, "aggregate_key", context.aggregate_key, "aggregate_payload"
        )
        if reusable is not None:
            context.aggregate_payload = reusable
            stage_reuse_counts["aggregation"] += 1
            continue
        aggregation_tasks.append(
            {
                "workorder_id": workorder_id,
                "stage": "aggregation",
                "stable_key": _task_key(workorder_id, "aggregation"),
                "runner": partial(
                    compute_aggregation_stage,
                    workorder_id,
                    context.relation_payload,
                    rules,
                    config,
                ),
            }
        )
    aggregation_outcomes = _execute_stage_tasks(aggregation_tasks, mode, config.concurrency)
    for workorder_id, outcome in sorted(aggregation_outcomes.items()):
        context = contexts[workorder_id]
        if outcome["ok"]:
            context.aggregate_payload = outcome["payload"]
            context.stage_computed["aggregation"] = True
        else:
            context.failure = outcome["failure"]
            failures.append(context.failure)

    workorders: list[dict[str, Any]] = []
    for workorder_id in current_ids:
        context = contexts[workorder_id]
        if context.failure is None and context.decay_payload and context.relation_payload and context.aggregate_payload:
            context.references = {
                "evidence_seqs": sorted(record.evidence_seq for record in context.evidence),
                "relation_ids": list(context.relation_payload.get("referenced_relation_ids", [])),
            }
            workorders.append(_workorder_view(context))

    reused_workorders: list[str] = []
    recomputed_workorders: list[str] = []
    for workorder_id in current_ids:
        context = contexts[workorder_id]
        if context.failure is not None and context.failure.get("code") == E_EVIDENCE_BUDGET_PENDING:
            continue
        if context.failure is None and context.decay_payload and context.relation_payload and context.aggregate_payload:
            computed = context.stage_computed or {}
            if any(computed.values()):
                recomputed_workorders.append(workorder_id)
            else:
                reused_workorders.append(workorder_id)
        else:
            recomputed_workorders.append(workorder_id)

    for injection in injections:
        if injection_status.get(injection.injection_id) is not None:
            continue
        context = contexts.get(injection.workorder_id)
        if context is None:
            reason = "workorder not found"
        elif context.failure is not None and context.failure.get("code") == E_EVIDENCE_BUDGET_PENDING:
            reason = "workorder pending on global budget"
        elif context.failure is not None:
            reason = "workorder failed before target stage"
        else:
            reason = "target stage was not reached"
        injection_status[injection.injection_id] = {
            "injection_id": injection.injection_id,
            "workorder_id": injection.workorder_id,
            "stage": injection.stage,
            "fault": injection.fault,
            "mapped_code": INJECTION_CODE_MAP[injection.fault],
            "evidence_seq": injection.evidence_seq,
            "hit": False,
            "reason": reason,
        }

    injection_rows = [row for row in injection_status.values() if row is not None]
    injection_hits = sorted(
        (row for row in injection_rows if row.get("hit")), key=lambda row: row["injection_id"]
    )
    injection_misses = sorted(
        (row for row in injection_rows if not row.get("hit")), key=lambda row: row["injection_id"]
    )

    failures = sorted_failures(failures)
    priority_distribution = dict(sorted(Counter(row["upgrade_priority"] for row in workorders).items()))
    boundary_workorders = sorted(row["workorder_id"] for row in workorders if row["boundary"])
    boundary_by_priority = dict(
        sorted(Counter(row["upgrade_priority"] for row in workorders if row["boundary"]).items())
    )
    failure_code_distribution = dict(sorted(Counter(row["code"] for row in failures).items()))

    snapshot_records = {
        workorder_id: record for workorder_id, record in previous_records.items() if workorder_id in contexts
    }
    completed_workorders: list[str] = []
    snapshot_commits: list[dict[str, Any]] = []
    consumed_budget_total = previous_total_budget + budget_consumed
    last_committed_completed = -1
    total_deterministic = sum(
        1
        for context in contexts.values()
        if context.failure is None
        or context.failure.get("code") != E_EVIDENCE_BUDGET_PENDING
    )

    def commit_snapshot(reason: str, completed_now: list[str]) -> None:
        manifest = store.commit(
            snapshot_records,
            config_hash=config.config_hash,
            rules_fingerprint=rules.fingerprint,
            input_version=input_version,
            completed_workorders=completed_now,
            pending_workorders=pending_workorders,
            consumed_budget_this_run=budget_consumed,
            consumed_budget_total=consumed_budget_total,
            stage_reuse_counts=stage_reuse_counts,
            commit_reason=reason,
        )
        snapshot_commits.append(
            {
                "version": manifest["version"],
                "snapshot": manifest.get("snapshot_name"),
                "completed_count": len(completed_now),
                "reason": reason,
            }
        )

    try:
        for workorder_id in current_ids:
            context = contexts[workorder_id]
            record = _record_from_context(context)
            if record is not None:
                snapshot_records[workorder_id] = record
                if context.failure is None or context.failure.get("code") != E_EVIDENCE_BUDGET_PENDING:
                    completed_workorders.append(workorder_id)
                    if len(completed_workorders) % config.snapshot_commit_interval == 0:
                        commit_snapshot("interval", list(completed_workorders))
                        last_committed_completed = len(completed_workorders)
        if last_committed_completed != len(completed_workorders) or not snapshot_commits:
            commit_snapshot("final", list(completed_workorders))
            last_committed_completed = len(completed_workorders)
    except BusinessFault as fault:
        failures.append(_failure_from_fault(fault))
        failures = sorted_failures(failures)
        failure_code_distribution = dict(sorted(Counter(row["code"] for row in failures).items()))

    final_manifest: dict[str, Any] = {}
    if snapshot_commits:
        final_manifest = {
            "version": snapshot_commits[-1]["version"],
            "snapshot": snapshot_commits[-1]["snapshot"],
            "commits": snapshot_commits,
            "completed_workorders": sorted(completed_workorders),
            "pending_workorders": sorted(pending_workorders),
            "consumed_budget_this_run": budget_consumed,
            "consumed_budget_total": consumed_budget_total,
            "config_hash": config.config_hash,
            "rules_fingerprint": rules.fingerprint,
            "input_version": input_version,
            "stage_files": ["load.jsonl", "decay.jsonl", "relations.jsonl", "aggregation.jsonl"],
            "atomic_commit": True,
        }

    report = _empty_report()
    report["fatal"] = False
    report["run"] = {
        "mode": mode,
        "requested_concurrency": config.concurrency,
        "executor_workers": 1 if mode == "serial" else config.concurrency,
        "config_hash": config.config_hash,
        "rules_fingerprint": rules.fingerprint,
        "input_version": input_version,
        "fixed_decimal_width": config.fixed_decimal_width,
        "quantization_step": config.quantization_step,
        "fusion_mode": config.fusion_mode,
        "deterministic": True,
        "threshold_boundary_policy": "INCLUSIVE_UPPER",
        "decay_algorithm": DECAY_ALGORITHM,
        "relation_algorithm": RELATION_ALGORITHM,
        "aggregation_algorithm": AGGREGATION_ALGORITHM,
    }
    report["summary"] = {
        "workorder_count": len(current_ids),
        "successful_count": len(workorders),
        "failure_count": len(failures),
        "pending_count": len(pending_workorders),
        "boundary_count": len(boundary_workorders),
        "boundary_workorders": boundary_workorders,
        "boundary_by_priority": boundary_by_priority,
        "priority_distribution": priority_distribution,
        "failure_code_distribution": failure_code_distribution,
    }
    report["workorders"] = workorders
    report["failures"] = failures
    report["ledger"] = {
        "reused_workorders": sorted(reused_workorders),
        "recomputed_workorders": sorted(recomputed_workorders),
        "removed_workorders": removed_workorders,
        "pending_workorders": sorted(pending_workorders),
        "scheduled_decay_order": scheduled_decay_order,
        "budget": {
            "batch_limit": config.batch_evidence_budget,
            "consumed_this_run": budget_consumed,
            "remaining_this_run": budget_remaining,
            "consumed_total": consumed_budget_total,
        },
        "stage_reuse_counts": dict(sorted(stage_reuse_counts.items())),
    }
    report["snapshot"] = final_manifest
    report["injections"] = {
        "configured_count": len(injections),
        "hits": injection_hits,
        "misses": injection_misses,
    }
    return report


def run_workbench(
    data_dir: str | Path,
    state_dir: str | Path,
    *,
    mode: str = "parallel",
    injections_path: str | Path | None = None,
    rebuild_state: bool = False,
) -> dict[str, Any]:
    """Run one deterministic batch and return a JSON-serializable report."""

    try:
        return _run_workbench(
            Path(data_dir),
            Path(state_dir),
            mode=mode,
            injections_path=Path(injections_path) if injections_path is not None else None,
            rebuild_state=rebuild_state,
        )
    except BusinessFault as fault:
        return _fatal_report(fault, mode=mode)
    except OSError as exc:
        return _fatal_report(
            BusinessFault(E_INPUT_IO, f"I/O failure: {exc.__class__.__name__}", "global"),
            mode=mode,
        )
    except Exception as exc:  # noqa: BLE001 - the CLI must always emit a diagnosis
        return _fatal_report(
            BusinessFault(E_INTERNAL, "unexpected workbench failure", "global", exception_type=exc.__class__.__name__),
            mode=mode,
        )
