from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from decimal import Decimal
from typing import Any

from .cache import ContentCache
from .constants import (
    DIMENSION_ORDER,
    E_CONFIG_INVALID,
    E_GENERATION_BUDGET_EXHAUSTED,
    E_SCORE_BUDGET_EXHAUSTED,
    E_TASK_BUDGET_EXHAUSTED,
    SCHEMA_VERSION,
)
from .dedupe import dedupe_cache_key, dedupe_task
from .errors import Failure, FailureCollector
from .loader import LoadedInputs
from .plan import ConstructionState
from .sampling import (
    candidate_id_for,
    materialize_sampling,
    prepare_sampling,
    stable_sort_key,
)
from .scoring import (
    Quantizer,
    score_candidate,
    score_cost,
    scoring_cache_key,
)
from .tokenizer import SubwordTokenizer
from .util import as_decimal, decimal_text, fingerprint_object, stable_key_hash


@dataclass
class PipelineResult:
    report: dict[str, Any]
    phase_versions: dict[str, str]
    stage_artifacts: dict[str, Any]
    invalidation: list[dict[str, Any]]
    failures: list[Failure]


def _planned_candidates(state: ConstructionState) -> list[dict[str, Any]]:
    planned: list[dict[str, Any]] = []
    global_seq = 0
    for task_type in sorted(state.tasks):
        target = int(state.tasks[task_type]["target_count"])
        for sample_seq in range(1, target + 1):
            global_seq += 1
            planned.append(
                {
                    "candidate_id": candidate_id_for(task_type, sample_seq),
                    "task_type": task_type,
                    "sample_seq": sample_seq,
                    "global_seq": global_seq,
                }
            )
    return planned


def _event(
    stage: str,
    object_id: str,
    cache_key: str,
    reused: bool,
    upstream_keys: list[str],
) -> dict[str, Any]:
    return {
        "stage": stage,
        "object_id": object_id,
        "cache_key": cache_key,
        "action": "reuse" if reused else "recompute",
        "upstream_keys": upstream_keys,
    }


def _upstream_artifacts(state: ConstructionState) -> dict[str, Any]:
    templates: list[dict[str, Any]] = []
    for task_type in sorted(state.tasks):
        active = {
            variant_id
            for variant_id, weight in state.tasks[task_type]["template_weights"].items()
            if as_decimal(weight) > 0
        }
        for variant_id in sorted(active):
            template = state.templates[task_type][variant_id]
            content = {
                "variant_id": variant_id,
                "instruction_template": template["instruction_template"],
                "slots": template["slots"],
                "reference_answer": template["reference_answer"],
            }
            templates.append(
                {
                    "task_type": task_type,
                    "variant_id": variant_id,
                    "content_fingerprint": fingerprint_object(content),
                    "cache_key": stable_key_hash("template-v1", task_type, variant_id, fingerprint_object(content)),
                }
            )
    sources: list[dict[str, Any]] = []
    for task_type in sorted(state.tasks):
        slots = sorted(
            {
                slot
                for variant_id, weight in state.tasks[task_type]["template_weights"].items()
                if as_decimal(weight) > 0
                for slot in state.templates[task_type][variant_id]["slots"]
            }
        )
        active_sources = sorted(
            source
            for source, weight in state.tasks[task_type]["slot_source_weights"].items()
            if as_decimal(weight) > 0
        )
        for slot in slots:
            for source_name in active_sources:
                values = state.sources[source_name]
                content = {"name": source_name, "values": values}
                source_fp = fingerprint_object(content)
                sources.append(
                    {
                        "task_type": task_type,
                        "slot": slot,
                        "source": source_name,
                        "content_fingerprint": source_fp,
                        "cache_key": stable_key_hash("source-v1", task_type, slot, source_name, source_fp),
                    }
                )
    return {"templates": templates, "slot_sources": sources}


def _candidate_object(
    *,
    planned: dict[str, Any],
    sampling: dict[str, Any],
    scoring: dict[str, Any],
    retained_ids: set[str],
    group_by_candidate: dict[str, dict[str, Any]],
    removed_by_candidate: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    candidate_id = planned["candidate_id"]
    return {
        "candidate_id": candidate_id,
        "task_type": planned["task_type"],
        "sample_seq": planned["sample_seq"],
        "global_seq": planned["global_seq"],
        "template_variant_id": sampling["template_variant_id"],
        "source_name": sampling["source_name"],
        "slot_values": sampling["slot_values"],
        "instruction": sampling["instruction"],
        "reference_answer": sampling["reference_answer"],
        "dimensions": scoring["dimensions"],
        "raw_total": scoring["raw_total"],
        "total_fixed": scoring["total_fixed"],
        "total_display": scoring["total_display"],
        "quantization_residual": scoring["quantization_residual"],
        "quality_pass": scoring["quality_pass"],
        "boundary": scoring["boundary"],
        "boundary_decision": scoring["boundary_decision"],
        "retained": candidate_id in retained_ids,
        "dedupe_group_id": group_by_candidate.get(candidate_id, {}).get("group_id"),
        "representative_candidate_id": group_by_candidate.get(candidate_id, {}).get("representative_candidate_id"),
        "removed_as_duplicate": candidate_id in removed_by_candidate,
        "sampling_cache_key": sampling["cache_key"],
        "scoring_cache_key": scoring["cache_key"],
        "sampling_references": {
            "template": sampling["template_reference"],
            "slot_sources": sampling["slot_source_references"],
        },
    }


def _distribution(values: list[int], quantizer: Quantizer) -> dict[str, Any]:
    if not values:
        return {
            "count": 0,
            "min": None,
            "max": None,
            "mean": None,
            "median": None,
            "buckets": {f"{index * 10}-{(index + 1) * 10}": 0 for index in range(10)},
        }
    ordered = sorted(values)
    scale = Decimal(10) ** quantizer.places
    buckets = {f"{index * 10}-{(index + 1) * 10}": 0 for index in range(10)}
    for value in ordered:
        score = (Decimal(value) / scale).quantize(Decimal("0.000001"))
        index = min(9, max(0, int(score // 10)))
        buckets[f"{index * 10}-{(index + 1) * 10}"] += 1
    middle = len(ordered) // 2
    median = ordered[middle] if len(ordered) % 2 else (ordered[middle - 1] + ordered[middle]) / 2
    represented_mean = Decimal(sum(ordered)) / (Decimal(len(ordered)) * scale)
    represented_median = Decimal(str(median)) / scale
    return {
        "count": len(ordered),
        "min": ordered[0],
        "max": ordered[-1],
        "mean": decimal_text(represented_mean, 6),
        "median": decimal_text(represented_median, 6),
        "buckets": buckets,
    }


def _task_statistics(
    *,
    state: ConstructionState,
    planned_by_task: dict[str, list[dict[str, Any]]],
    sampling_by_id: dict[str, dict[str, Any]],
    scoring_by_id: dict[str, dict[str, Any]],
    quality_rejections: list[dict[str, Any]],
    dedupe_by_task: dict[str, dict[str, Any]],
    quantizer: Quantizer,
) -> dict[str, Any]:
    rejected_by_task: dict[str, int] = {}
    for item in quality_rejections:
        rejected_by_task[item["task_type"]] = rejected_by_task.get(item["task_type"], 0) + 1
    stats: dict[str, Any] = {}
    for task_type in sorted(state.tasks):
        planned = planned_by_task.get(task_type, [])
        sampled = [item for item in sampling_by_id.values() if item["task_type"] == task_type]
        scored = [item for item in scoring_by_id.values() if item["task_type"] == task_type]
        dedupe = dedupe_by_task.get(task_type, {"groups": [], "retained_candidate_ids": [], "removed_candidates": []})
        distribution: dict[str, Any] = {}
        for dimension in DIMENSION_ORDER:
            distribution[dimension] = _distribution(
                [item["dimensions"][dimension]["fixed"] for item in scored],
                quantizer,
            )
        distribution["total"] = _distribution([item["total_fixed"] for item in scored], quantizer)
        stats[task_type] = {
            "planned_candidate_count": len(planned),
            "generated_candidate_count": len(sampled),
            "scored_candidate_count": len(scored),
            "quality_pass_count": sum(1 for item in scored if item["quality_pass"]),
            "quality_rejected_count": rejected_by_task.get(task_type, 0),
            "dedupe_before_count": sum(1 for item in scored if item["quality_pass"]),
            "dedupe_after_count": len(dedupe["retained_candidate_ids"]),
            "removed_duplicate_count": len(dedupe["removed_candidates"]),
            "near_duplicate_group_count": len(dedupe["groups"]),
            "dimension_distribution": distribution,
        }
    return stats


def run_pipeline(
    *,
    state: ConstructionState,
    inputs: LoadedInputs,
    cache: ContentCache,
    base_failures: list[Failure] | None = None,
) -> PipelineResult:
    base_failures = list(base_failures or [])
    failures = FailureCollector(items=list(base_failures))
    config = state.config
    quantizer = Quantizer(
        places=int(config["fixed_decimal_places"]),
        step=as_decimal(config["quantization_step"]),
    )
    tokenizer = SubwordTokenizer(state.lexicon["subword"])
    threshold = as_decimal(config["near_duplicate_threshold"])
    planned = _planned_candidates(state)
    planned_by_task: dict[str, list[dict[str, Any]]] = {}
    for item in planned:
        planned_by_task.setdefault(item["task_type"], []).append(item)

    remaining_global = int(config["global_score_budget"])
    initial_global = remaining_global
    remaining_task_quota = {task: int(state.quotas[task]) for task in sorted(state.tasks)}
    initial_task_quota = dict(remaining_task_quota)
    remaining_batch_points = int(config["per_batch_generation_points"])
    initial_batch_points = remaining_batch_points

    sampling_by_id: dict[str, dict[str, Any]] = {}
    scoring_by_id: dict[str, dict[str, Any]] = {}
    sampling_events: list[dict[str, Any]] = []
    scoring_events: list[dict[str, Any]] = []
    dedupe_events: list[dict[str, Any]] = []
    ledger: list[dict[str, Any]] = []
    pending_queue: list[dict[str, Any]] = []
    quality_rejections: list[dict[str, Any]] = []
    futures: list[Future[Any]] = []
    future_candidates: list[dict[str, Any]] = []
    global_exhausted = False
    generation_exhausted = False

    def add_budget_failure(code: str, planned_item: dict[str, Any], message: str) -> None:
        failures.add(
            code,
            message,
            stage="scheduler",
            candidate_id=planned_item["candidate_id"],
            task_type=planned_item["task_type"],
            field="budget",
        )

    with ThreadPoolExecutor(max_workers=int(config["max_concurrency"]), thread_name_prefix="score") as executor:
        for planned_item in planned:
            task_type = planned_item["task_type"]
            candidate_id = planned_item["candidate_id"]
            if global_exhausted:
                add_budget_failure(E_SCORE_BUDGET_EXHAUSTED, planned_item, "global score budget exhausted before candidate started")
                pending_queue.append(
                    {
                        "candidate_id": candidate_id,
                        "task_type": task_type,
                        "sample_seq": planned_item["sample_seq"],
                        "code": E_SCORE_BUDGET_EXHAUSTED,
                        "stage": "scoring",
                    }
                )
                continue
            if generation_exhausted:
                add_budget_failure(E_GENERATION_BUDGET_EXHAUSTED, planned_item, "per-batch candidate generation point budget exhausted")
                pending_queue.append(
                    {
                        "candidate_id": candidate_id,
                        "task_type": task_type,
                        "sample_seq": planned_item["sample_seq"],
                        "code": E_GENERATION_BUDGET_EXHAUSTED,
                        "stage": "sampling",
                    }
                )
                continue
            if remaining_task_quota.get(task_type, 0) <= 0:
                add_budget_failure(E_TASK_BUDGET_EXHAUSTED, planned_item, f"task quota for {task_type!r} exhausted")
                continue
            if remaining_global <= 0:
                global_exhausted = True
                add_budget_failure(E_SCORE_BUDGET_EXHAUSTED, planned_item, "global score budget exhausted before candidate started")
                pending_queue.append(
                    {
                        "candidate_id": candidate_id,
                        "task_type": task_type,
                        "sample_seq": planned_item["sample_seq"],
                        "code": E_SCORE_BUDGET_EXHAUSTED,
                        "stage": "scoring",
                    }
                )
                continue
            if remaining_batch_points <= 0:
                generation_exhausted = True
                add_budget_failure(E_GENERATION_BUDGET_EXHAUSTED, planned_item, "per-batch candidate generation point budget exhausted")
                pending_queue.append(
                    {
                        "candidate_id": candidate_id,
                        "task_type": task_type,
                        "sample_seq": planned_item["sample_seq"],
                        "code": E_GENERATION_BUDGET_EXHAUSTED,
                        "stage": "sampling",
                    }
                )
                continue

            selection = prepare_sampling(
                state=state,
                task_type=task_type,
                sample_seq=planned_item["sample_seq"],
                seed_file_fingerprint=inputs.fingerprints["seed_file"],
            )
            template = state.templates[task_type][selection["template_variant_id"]]
            sampling_lookup = cache.get_or_compute(
                "sampling",
                selection["cache_key"],
                lambda selection=selection, template=template: materialize_sampling(selection, template),
            )
            sampling = sampling_lookup.value
            sampling_by_id[candidate_id] = sampling
            sampling_events.append(
                _event(
                    "sampling",
                    candidate_id,
                    selection["cache_key"],
                    sampling_lookup.reused,
                    [
                        sampling["template_reference"]["cache_key"],
                        *[reference["cache_key"] for reference in sampling["slot_source_references"]],
                    ],
                )
            )

            batch_before = remaining_batch_points
            remaining_batch_points -= 1
            cost = score_cost(sampling)
            if remaining_global < cost:
                global_exhausted = True
                ledger.append(
                    {
                        "entry_index": len(ledger) + 1,
                        "candidate_id": candidate_id,
                        "task_type": task_type,
                        "sample_seq": planned_item["sample_seq"],
                        "status": "generated_score_pending",
                        "batch_generation_points": {"before": batch_before, "cost": 1, "after": remaining_batch_points},
                        "global_score": {"before": remaining_global, "cost": 0, "after": remaining_global},
                        "task_quota": {"before": remaining_task_quota[task_type], "cost": 0, "after": remaining_task_quota[task_type]},
                        "score_cost_required": cost,
                    }
                )
                add_budget_failure(E_SCORE_BUDGET_EXHAUSTED, planned_item, "global score budget cannot cover the next candidate")
                pending_queue.append(
                    {
                        "candidate_id": candidate_id,
                        "task_type": task_type,
                        "sample_seq": planned_item["sample_seq"],
                        "code": E_SCORE_BUDGET_EXHAUSTED,
                        "stage": "scoring",
                        "required_cost": cost,
                        "remaining_budget": remaining_global,
                    }
                )
                continue

            score_key, _, _ = scoring_cache_key(
                sampling=sampling,
                scoring=state.scoring,
                quantizer=quantizer,
                tokenizer=tokenizer,
            )
            score_before = remaining_global
            quota_before = remaining_task_quota[task_type]
            remaining_global -= cost
            remaining_task_quota[task_type] -= 1
            ledger.append(
                {
                    "entry_index": len(ledger) + 1,
                    "candidate_id": candidate_id,
                    "task_type": task_type,
                    "sample_seq": planned_item["sample_seq"],
                    "status": "scoring_scheduled",
                    "batch_generation_points": {"before": batch_before, "cost": 1, "after": remaining_batch_points},
                    "global_score": {"before": score_before, "cost": cost, "after": remaining_global},
                    "task_quota": {"before": quota_before, "cost": 1, "after": remaining_task_quota[task_type]},
                    "score_cost_required": cost,
                }
            )

            def score_factory(sampling_value: dict[str, Any] = sampling) -> dict[str, Any]:
                return score_candidate(
                    sampling=sampling_value,
                    scoring=state.scoring,
                    quantizer=quantizer,
                    tokenizer=tokenizer,
                )

            future = executor.submit(cache.get_or_compute, "scoring", score_key, score_factory)
            futures.append(future)
            future_candidates.append(planned_item)

        scoring_lookups: list[tuple[dict[str, Any], Any, dict[str, Any]]] = []
        for planned_item, future in zip(future_candidates, futures):
            lookup = future.result()
            sampling = sampling_by_id[planned_item["candidate_id"]]
            scoring_lookups.append((planned_item, lookup, sampling))

    for planned_item, lookup, sampling in scoring_lookups:
        scoring = lookup.value
        scoring_by_id[planned_item["candidate_id"]] = scoring
        scoring_events.append(
            _event(
                "scoring",
                planned_item["candidate_id"],
                scoring["cache_key"],
                lookup.reused,
                [sampling["cache_key"]],
            )
        )
        if not scoring["quality_pass"]:
            quality_rejections.append(
                {
                    "candidate_id": planned_item["candidate_id"],
                    "task_type": planned_item["task_type"],
                    "sample_seq": planned_item["sample_seq"],
                    "total_fixed": scoring["total_fixed"],
                    "threshold_fixed": scoring["threshold_fixed"],
                    "reason": "below_quality_threshold",
                }
            )

    dedupe_by_task: dict[str, dict[str, Any]] = {}
    for task_type in sorted(state.tasks):
        passed = [
            scoring_by_id[item["candidate_id"]]
            for item in planned_by_task.get(task_type, [])
            if item["candidate_id"] in scoring_by_id and scoring_by_id[item["candidate_id"]]["quality_pass"]
        ]
        score_keys = [item["cache_key"] for item in passed]
        key = dedupe_cache_key(
            task_type=task_type,
            score_keys=score_keys,
            candidate_ids=[item["candidate_id"] for item in passed],
            threshold=threshold,
            tokenizer=tokenizer,
        )
        lookup = cache.get_or_compute(
            "dedupe",
            key,
            lambda task_type=task_type, passed=passed: dedupe_task(
                task_type=task_type,
                scored_candidates=passed,
                sampling_by_id=sampling_by_id,
                threshold=threshold,
                tokenizer=tokenizer,
            ),
        )
        dedupe_by_task[task_type] = lookup.value
        dedupe_events.append(
            _event(
                "dedupe",
                task_type,
                key,
                lookup.reused,
                score_keys,
            )
        )

    retained_ids: set[str] = set()
    removed_by_candidate: dict[str, dict[str, Any]] = {}
    group_by_candidate: dict[str, dict[str, Any]] = {}
    near_duplicate_groups: list[dict[str, Any]] = []
    removed_duplicates: list[dict[str, Any]] = []
    for task_type in sorted(state.tasks):
        artifact = dedupe_by_task[task_type]
        retained_ids.update(artifact["retained_candidate_ids"])
        for group in artifact["groups"]:
            near_duplicate_groups.append(group)
            for member in group["member_candidates"]:
                group_by_candidate[member["candidate_id"]] = {
                    "group_id": group["group_id"],
                    "representative_candidate_id": group["representative_candidate_id"],
                }
        for removed in artifact["removed_candidates"]:
            removed_by_candidate[removed["candidate_id"]] = removed
            removed_duplicates.append(removed)

    candidates: list[dict[str, Any]] = []
    for planned_item in planned:
        candidate_id = planned_item["candidate_id"]
        if candidate_id not in scoring_by_id:
            continue
        candidates.append(
            _candidate_object(
                planned=planned_item,
                sampling=sampling_by_id[candidate_id],
                scoring=scoring_by_id[candidate_id],
                retained_ids=retained_ids,
                group_by_candidate=group_by_candidate,
                removed_by_candidate=removed_by_candidate,
            )
        )

    boundary_samples = [
        {
            "candidate_id": candidate["candidate_id"],
            "task_type": candidate["task_type"],
            "total_fixed": candidate["total_fixed"],
            "threshold_fixed": scoring_by_id[candidate["candidate_id"]]["threshold_fixed"],
        }
        for candidate in candidates
        if candidate["boundary"]
    ]
    boundary_samples.sort(key=lambda item: item["candidate_id"])
    quality_rejections.sort(key=lambda item: (item["task_type"], item["sample_seq"], item["candidate_id"]))

    task_stats = _task_statistics(
        state=state,
        planned_by_task=planned_by_task,
        sampling_by_id=sampling_by_id,
        scoring_by_id=scoring_by_id,
        quality_rejections=quality_rejections,
        dedupe_by_task=dedupe_by_task,
        quantizer=quantizer,
    )
    stage_artifacts = {
        "sampling": [sampling_by_id[item["candidate_id"]] for item in planned if item["candidate_id"] in sampling_by_id],
        "scoring": [scoring_by_id[item["candidate_id"]] for item in planned if item["candidate_id"] in scoring_by_id],
        "dedupe": [dedupe_by_task[task_type] for task_type in sorted(state.tasks)],
    }
    invalidation = sampling_events + scoring_events + dedupe_events
    upstream_artifacts = _upstream_artifacts(state)
    phase_versions = {
        "sampling": fingerprint_object([item["cache_key"] for item in stage_artifacts["sampling"]]),
        "scoring": fingerprint_object([item["cache_key"] for item in stage_artifacts["scoring"]]),
        "dedupe": fingerprint_object([item["cache_key"] for item in stage_artifacts["dedupe"]]),
    }
    report: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "pipeline": "deterministic-canonical-v1",
        "effective_config": state.config,
        "scoring_parameters": state.scoring,
        "configuration_hash": state.fingerprint(),
        "seed_file_hash": inputs.fingerprints["seed_file"],
        "lexicon_hash": inputs.fingerprints["lexicon"],
        "quantization": {
            "fixed_decimal_places": quantizer.places,
            "quantization_step": decimal_text(quantizer.step),
            "quantum_units": quantizer.quantum_units,
        },
        "task_statistics": task_stats,
        "planned_candidates": planned,
        "upstream_artifacts": upstream_artifacts,
        "candidates": candidates,
        "retained_candidate_ids": sorted(
            retained_ids,
            key=lambda candidate_id: next(
                (item["global_seq"] for item in planned if item["candidate_id"] == candidate_id),
                0,
            ),
        ),
        "quality_rejections": quality_rejections,
        "boundary_samples": boundary_samples,
        "near_duplicate_groups": near_duplicate_groups,
        "removed_duplicates": removed_duplicates,
        "scheduler": {
            "max_concurrency": int(config["max_concurrency"]),
            "thread_pool_stage": "scoring",
            "initial_budget": {
                "global_score": initial_global,
                "task_quota": initial_task_quota,
                "per_batch_generation_points": initial_batch_points,
            },
            "remaining_budget": {
                "global_score": remaining_global,
                "task_quota": remaining_task_quota,
                "per_batch_generation_points": remaining_batch_points,
            },
            "ledger": ledger,
            "pending_queue": pending_queue,
        },
        "invalidation": invalidation,
        "failure_codes": [item.code for item in failures.items],
        "failures": failures.to_list(),
        "stage_versions": phase_versions,
    }
    aggregate_version = fingerprint_object(
        {
            "configuration_hash": report["configuration_hash"],
            "candidates": [item["scoring_cache_key"] for item in candidates],
            "retained_candidate_ids": report["retained_candidate_ids"],
            "removed_duplicates": removed_duplicates,
            "ledger": ledger,
            "pending_queue": pending_queue,
            "failure_codes": report["failure_codes"],
        }
    )
    phase_versions["aggregate"] = aggregate_version
    report["stage_versions"] = phase_versions
    return PipelineResult(
        report=report,
        phase_versions=phase_versions,
        stage_artifacts=stage_artifacts,
        invalidation=invalidation,
        failures=list(failures.items),
    )
