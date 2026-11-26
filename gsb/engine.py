from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Callable

from .canonical import fingerprint
from .cleaning import (
    cleaning_config_fingerprint,
    cleaning_cache_key,
    clean_sample,
)
from .dedup import (
    dedup_scope,
    deduplicate_shard,
    plan_shards,
    shard_cache_key,
)
from .errors import (
    E_BUDGET_PENDING,
    E_DOMAIN_QUOTA_EXCEEDED,
    E_INPUT_INVALID,
    E_SHARD_COMPARISON_LIMIT,
    PipelineError,
)
from .ledger import assign_ordinals, baseline_entry_map, explain_miss, make_entry
from .samples import SampleSet
from .sampling import sampling_cache_key, weighted_sample
from .store import SnapshotStore
from .summary import build_summary


def run_pipeline(
    config: dict[str, Any],
    sample_set: SampleSet,
    state_dir: str | Path,
    *,
    budget: int | None = None,
    workers: int | None = None,
    crash_before_commit: bool = False,
) -> dict[str, Any]:
    if budget is not None and (isinstance(budget, bool) or not isinstance(budget, int) or budget < 0):
        raise PipelineError(E_INPUT_INVALID, "budget must be a nonnegative integer", field="runtime.budget")
    if workers is not None and (isinstance(workers, bool) or not isinstance(workers, int) or workers < 1):
        raise PipelineError(E_INPUT_INVALID, "workers must be a positive integer", field="runtime.workers")

    store = SnapshotStore(state_dir)
    previous = store.load_current()
    dataset_fingerprint = sample_set.fingerprint
    config_hash = config["config_hash"]
    same_target = bool(
        previous
        and previous["manifest"].get("config_hash") == config_hash
        and previous["manifest"].get("dataset_fingerprint") == dataset_fingerprint
    )
    resume_partial = bool(same_target and previous["manifest"].get("status") == E_BUDGET_PENDING)

    if resume_partial:
        resume_state = previous
        base_version = previous["manifest"].get("change_base_global_version")
        baseline = store.load_version(base_version) if isinstance(base_version, int) else None
        if baseline is None:
            baseline = previous
    else:
        resume_state = None
        baseline = previous

    baseline_entries = baseline_entry_map(baseline)
    parent_version = previous["manifest"]["global_version"] if previous else None
    global_version = (parent_version or 0) + 1
    change_base_version = (
        previous["manifest"].get("change_base_global_version")
        if resume_partial and previous
        else parent_version
    )

    run_limit = config["budgets"]["global_units"] if budget is None else budget
    remaining_budget = run_limit
    previous_budget = (previous or {}).get("budget", {})
    cumulative_consumed = int(previous_budget.get("cumulative_consumed_units", 0))
    last_run_consumed = 0
    scheduler_stats: dict[str, Any] = {
        "cleaning_max_active": 0,
        "dedup_max_active": 0,
        "submitted_cleaning_tasks": 0,
        "submitted_dedup_tasks": 0,
        "priority_source": "stable_plan",
    }
    if resume_partial and previous and previous.get("queue", {}).get("items"):
        scheduler_stats["priority_source"] = "committed_queue"
        scheduler_stats["committed_queue_length"] = len(previous["queue"]["items"])

    quota_limits = {
        name: (config["domain_map"][name].get("quota") or config["budgets"]["domain_quota"])
        for name in config["domain_names"]
    }
    domain_seen = {name: 0 for name in config["domain_names"]}

    existing_clean = {}
    if resume_state:
        existing_clean = {
            record["sample_id"]: record
            for record in resume_state.get("cleaning", {}).get("records", [])
        }
    baseline_clean_by_key = {
        record.get("cache_key"): record
        for record in (baseline or {}).get("cleaning", {}).get("records", [])
        if record.get("cache_key")
    }

    priority_cleaning_ids: list[str] = []
    if resume_state:
        for item in resume_state.get("queue", {}).get("items", []):
            if item.get("stage") == "cleaning" and item.get("subject_id") in sample_set.samples:
                if item["subject_id"] not in priority_cleaning_ids:
                    priority_cleaning_ids.append(item["subject_id"])
    priority_cleaning_set = set(priority_cleaning_ids)
    cleaning_plan_ids = priority_cleaning_ids + [
        sample_id for sample_id in sample_set.sample_order if sample_id not in priority_cleaning_set
    ]
    if priority_cleaning_ids:
        scheduler_stats["priority_cleaning_subjects"] = priority_cleaning_ids

    cleaning_records: list[dict[str, Any]] = []
    compute_samples: list[Any] = []
    queue_items: list[dict[str, Any]] = []
    error_payload: dict[str, Any] | None = None
    overall_status = "complete"

    for sample_id in cleaning_plan_ids:
        sample = sample_set.samples[sample_id]
        domain_seen[sample["domain"]] += 1
        quota_limit = quota_limits[sample["domain"]]
        if domain_seen[sample["domain"]] > quota_limit and error_payload is None:
            error_payload = PipelineError(
                E_DOMAIN_QUOTA_EXCEEDED,
                "domain sample quota exceeded while processing batch",
                field=f"domains.{sample['domain']}.quota",
                batch_id=sample["batch_id"],
                details={
                    "domain": sample["domain"],
                    "quota": quota_limit,
                    "observed": domain_seen[sample["domain"]],
                },
            ).to_dict()
            overall_status = E_DOMAIN_QUOTA_EXCEEDED
            queue_items.append(_work_item("cleaning", sample_id, sample, 1))
            continue

        if error_payload is not None and error_payload["error_code"] == E_DOMAIN_QUOTA_EXCEEDED:
            queue_items.append(_work_item("cleaning", sample_id, sample, 1))
            continue

        cache_key = cleaning_cache_key(sample_content_fingerprint(sample), config)
        existing = existing_clean.get(sample_id)
        if existing and existing.get("cache_key") == cache_key:
            cleaning_records.append(existing)
            continue

        baseline_record = baseline_clean_by_key.get(cache_key)
        if baseline_record is not None:
            reused = dict(baseline_record)
            reused["disposition"] = "reused"
            reused["reuse_hit"] = True
            reused["invalidation_reason"] = "cache_key_match"
            reused["reused_from_global_version"] = (
                baseline["manifest"]["global_version"] if baseline else None
            )
            cleaning_records.append(reused)
            continue

        if remaining_budget < 1:
            queue_items.append(_work_item("cleaning", sample_id, sample, 1))
            overall_status = E_BUDGET_PENDING
            continue

        if overall_status == E_BUDGET_PENDING:
            queue_items.append(_work_item("cleaning", sample_id, sample, 1))
            continue

        remaining_budget -= 1
        last_run_consumed += 1
        compute_samples.append(sample)

    if compute_samples:
        worker_count = _worker_count(config, workers, "cleaning_workers")
        computed, stats = _parallel_map(
            compute_samples,
            worker_count,
            lambda item: clean_sample(item, config),
        )
        scheduler_stats["cleaning_max_active"] = stats["max_active"]
        scheduler_stats["submitted_cleaning_tasks"] = len(compute_samples)
        for record in computed:
            scope_fp = cleaning_config_fingerprint(config)
            upstream = {"sample_content_fingerprint": record["content_fingerprint"]}
            reason, changed = explain_miss(
                stage="cleaning",
                subject_id=record["sample_id"],
                cache_key=record["cache_key"],
                scope_fingerprint=scope_fp,
                upstream=upstream,
                baseline_entries=baseline_entries,
            )
            record["disposition"] = "recomputed"
            record["reuse_hit"] = False
            record["invalidation_reason"] = reason
            record["invalidation_basis"] = changed
            cleaning_records.append(record)

    cleaning_records.sort(key=lambda record: record["sample_id"])
    cleaning_complete = len(cleaning_records) == len(sample_set.sample_order) and error_payload is None

    cleaning_component = {
        "status": "complete" if len(cleaning_records) == len(sample_set.sample_order) else (
            "failed" if error_payload else "partial"
        ),
        "records": cleaning_records,
        "record_count": len(cleaning_records),
        "expected_count": len(sample_set.sample_order),
        "artifact_fingerprint": fingerprint([record["record_fingerprint"] for record in cleaning_records]),
        "config_scope_fingerprint": cleaning_config_fingerprint(config),
    }

    dedup_shards: list[dict[str, Any]] = []
    dedup_component: dict[str, Any] = {"status": "pending", "shards": [], "artifact_fingerprint": None}
    sampling_artifact: dict[str, Any] | None = None
    summary_artifact: dict[str, Any] | None = None

    if cleaning_complete:
        cleaning_by_id = {record["sample_id"]: record for record in cleaning_records}
        planned_shards = plan_shards(cleaning_records, config)
        planned_shard_by_id = {shard["shard_id"]: shard for shard in planned_shards}
        existing_dedup = {}
        if resume_state:
            existing_dedup = {
                artifact["shard_id"]: artifact
                for artifact in resume_state.get("dedup", {}).get("shards", [])
            }
        baseline_dedup_by_key = {
            artifact.get("cache_key"): artifact
            for artifact in (baseline or {}).get("dedup", {}).get("shards", [])
            if artifact.get("cache_key")
        }

        priority_dedup_ids: list[str] = []
        if resume_state:
            for item in resume_state.get("queue", {}).get("items", []):
                subject_id = item.get("subject_id")
                if item.get("stage") == "dedup" and subject_id in planned_shard_by_id:
                    if subject_id not in priority_dedup_ids:
                        priority_dedup_ids.append(subject_id)
        priority_dedup_set = set(priority_dedup_ids)
        dedup_plan = [planned_shard_by_id[shard_id] for shard_id in priority_dedup_ids] + [
            shard for shard in planned_shards if shard["shard_id"] not in priority_dedup_set
        ]
        if priority_dedup_ids:
            scheduler_stats["priority_dedup_subjects"] = priority_dedup_ids

        compute_shards: list[dict[str, Any]] = []
        for shard in dedup_plan:
            cache_key = shard_cache_key(shard, sample_set.samples, cleaning_by_id, config)
            shard["cache_key"] = cache_key
            existing = existing_dedup.get(shard["shard_id"])
            if existing and existing.get("cache_key") == cache_key:
                dedup_shards.append(existing)
                continue

            baseline_artifact = baseline_dedup_by_key.get(cache_key)
            if baseline_artifact is not None:
                reused = dict(baseline_artifact)
                reused["disposition"] = "reused"
                reused["reuse_hit"] = True
                reused["invalidation_reason"] = "cache_key_match"
                reused["reused_from_global_version"] = (
                    baseline["manifest"]["global_version"] if baseline else None
                )
                dedup_shards.append(reused)
                continue

            if shard["pair_count"] > config["budgets"]["shard_comparison_limit"]:
                err = PipelineError(
                    E_SHARD_COMPARISON_LIMIT,
                    "shard pair count exceeds comparison limit",
                    field=f"dedup.shards.{shard['shard_id']}",
                    details={
                        "pair_count": shard["pair_count"],
                        "comparison_limit": config["budgets"]["shard_comparison_limit"],
                    },
                ).to_dict()
                error_payload = err
                overall_status = E_SHARD_COMPARISON_LIMIT
                break

            if remaining_budget < shard["scheduling_cost"]:
                queue_items.append(_work_item("dedup", shard["shard_id"], None, shard["scheduling_cost"], shard=shard))
                overall_status = E_BUDGET_PENDING
                continue

            if overall_status == E_BUDGET_PENDING:
                queue_items.append(_work_item("dedup", shard["shard_id"], None, shard["scheduling_cost"], shard=shard))
                continue

            remaining_budget -= shard["scheduling_cost"]
            last_run_consumed += shard["scheduling_cost"]
            compute_shards.append(shard)

        if compute_shards and error_payload is None:
            worker_count = _worker_count(config, workers, "dedup_workers")

            def run_shard(shard: dict[str, Any]) -> dict[str, Any]:
                return deduplicate_shard(shard, sample_set.samples, cleaning_by_id, config)

            computed_shards, stats = _parallel_map(compute_shards, worker_count, run_shard)
            scheduler_stats["dedup_max_active"] = stats["max_active"]
            scheduler_stats["submitted_dedup_tasks"] = len(compute_shards)
            for artifact in computed_shards:
                scope_fp = fingerprint(dedup_scope(config, [artifact["domain"]]))
                upstream = {
                    "shard_membership_fingerprint": fingerprint(artifact["sample_ids"]),
                    "cleaning_set_fingerprint": fingerprint(
                        [cleaning_by_id[sample_id]["record_fingerprint"] for sample_id in artifact["sample_ids"]]
                    ),
                }
                reason, changed = explain_miss(
                    stage="dedup_shard",
                    subject_id=artifact["shard_id"],
                    cache_key=artifact["cache_key"],
                    scope_fingerprint=scope_fp,
                    upstream=upstream,
                    baseline_entries=baseline_entries,
                )
                artifact["disposition"] = "recomputed"
                artifact["reuse_hit"] = False
                artifact["invalidation_reason"] = reason
                artifact["invalidation_basis"] = changed
            dedup_shards.extend(computed_shards)

        known_shard_ids = {artifact["shard_id"] for artifact in dedup_shards}
        for shard in planned_shards:
            if shard["shard_id"] not in known_shard_ids and not any(
                item["subject_id"] == shard["shard_id"] for item in queue_items
            ):
                queue_items.append(_work_item("dedup", shard["shard_id"], None, shard["scheduling_cost"], shard=shard))
                if overall_status == "complete":
                    overall_status = E_BUDGET_PENDING

        dedup_shards.sort(key=lambda artifact: artifact["shard_id"])
        dedup_complete = len(dedup_shards) == len(planned_shards) and error_payload is None
        dedup_fingerprint = fingerprint([artifact["artifact_fingerprint"] for artifact in dedup_shards])
        dedup_component = {
            "status": "complete" if dedup_complete else ("failed" if error_payload else "partial"),
            "shards": dedup_shards,
            "planned_shard_count": len(planned_shards),
            "artifact_fingerprint": dedup_fingerprint if dedup_shards else None,
        }

        if dedup_complete:
            survivors = sorted(
                sample_id
                for artifact in dedup_shards
                for decision in artifact["decisions"]
                if decision["status"] == "kept"
                for sample_id in [decision["sample_id"]]
            )
            sampling_key = sampling_cache_key(survivors, dedup_fingerprint, config)
            existing_sampling = resume_state.get("sampling") if resume_state else None
            baseline_sampling = baseline.get("sampling") if baseline else None

            if existing_sampling and existing_sampling.get("cache_key") == sampling_key:
                sampling_artifact = existing_sampling
            elif baseline_sampling and baseline_sampling.get("cache_key") == sampling_key:
                sampling_artifact = dict(baseline_sampling)
                sampling_artifact["disposition"] = "reused"
                sampling_artifact["reuse_hit"] = True
                sampling_artifact["invalidation_reason"] = "cache_key_match"
            else:
                if remaining_budget >= 1 and overall_status != E_BUDGET_PENDING:
                    remaining_budget -= 1
                    last_run_consumed += 1
                    sampling_artifact = weighted_sample(
                        survivors,
                        sample_set.samples,
                        config,
                        dedup_fingerprint=dedup_fingerprint,
                    )
                    scope_fp = fingerprint(_sampling_scope_hash(config))
                    reason, changed = explain_miss(
                        stage="sampling",
                        subject_id="global",
                        cache_key=sampling_artifact["cache_key"],
                        scope_fingerprint=scope_fp,
                        upstream={"dedup_fingerprint": dedup_fingerprint, "survivor_set_fingerprint": fingerprint(survivors)},
                        baseline_entries=baseline_entries,
                    )
                    sampling_artifact["disposition"] = "recomputed"
                    sampling_artifact["reuse_hit"] = False
                    sampling_artifact["invalidation_reason"] = reason
                    sampling_artifact["invalidation_basis"] = changed
                else:
                    queue_items.append({"stage": "sampling", "subject_id": "global", "cost": 1})
                    overall_status = E_BUDGET_PENDING

            if sampling_artifact is not None and summary_artifact is None:
                summary_key_input = {
                    "sampling_fingerprint": sampling_artifact["artifact_fingerprint"],
                    "dedup_fingerprint": dedup_fingerprint,
                }
                existing_summary = resume_state.get("summary") if resume_state else None
                baseline_summary = baseline.get("summary") if baseline else None
                expected_summary_key = fingerprint(
                    {
                        "stage": "ratio-summary-v1",
                        "sampling_fingerprint": sampling_artifact["artifact_fingerprint"],
                        "dedup_fingerprint": dedup_fingerprint,
                        "targets_fp": {
                            name: config["domain_map"][name]["target_ratio_fp"]
                            for name in config["domain_names"]
                        },
                        "fixed_point": config["fixed_point"],
                    }
                )
                if existing_summary and existing_summary.get("cache_key") == expected_summary_key:
                    summary_artifact = existing_summary
                elif baseline_summary and baseline_summary.get("cache_key") == expected_summary_key:
                    summary_artifact = dict(baseline_summary)
                    summary_artifact["disposition"] = "reused"
                    summary_artifact["reuse_hit"] = True
                    summary_artifact["invalidation_reason"] = "cache_key_match"
                elif remaining_budget >= 1 and overall_status != E_BUDGET_PENDING:
                    remaining_budget -= 1
                    last_run_consumed += 1
                    summary_artifact = build_summary(
                        candidates=survivors,
                        sampling_artifact=sampling_artifact,
                        samples=sample_set.samples,
                        config=config,
                        dedup_fingerprint=dedup_fingerprint,
                    )
                    reason, changed = explain_miss(
                        stage="summary",
                        subject_id="global",
                        cache_key=summary_artifact["cache_key"],
                        scope_fingerprint=fingerprint(
                            {name: config["domain_map"][name]["target_ratio_fp"] for name in config["domain_names"]}
                        ),
                        upstream=summary_key_input,
                        baseline_entries=baseline_entries,
                    )
                    summary_artifact["disposition"] = "recomputed"
                    summary_artifact["reuse_hit"] = False
                    summary_artifact["invalidation_reason"] = reason
                    summary_artifact["invalidation_basis"] = changed
                else:
                    queue_items.append({"stage": "summary", "subject_id": "global", "cost": 1})
                    overall_status = E_BUDGET_PENDING

    if error_payload is not None:
        overall_status = error_payload["error_code"]
    elif queue_items and overall_status == "complete":
        overall_status = E_BUDGET_PENDING
    elif not queue_items and cleaning_complete and dedup_component.get("status") == "complete" and sampling_artifact and summary_artifact:
        overall_status = "complete"
    else:
        overall_status = E_BUDGET_PENDING

    ledger_entries = _build_ledger(
        config=config,
        cleaning_records=cleaning_records,
        dedup_shards=dedup_shards,
        sampling_artifact=sampling_artifact,
        summary_artifact=summary_artifact,
        baseline=baseline,
    )
    ledger_component = {"entries": ledger_entries, "ledger_fingerprint": fingerprint([e["cache_key"] for e in ledger_entries])}

    budget_component = {
        "runtime_budget_limit": run_limit,
        "remaining_budget_units": remaining_budget,
        "last_run_consumed_units": last_run_consumed,
        "cumulative_consumed_units": cumulative_consumed + last_run_consumed,
        "domain_quota_limits": quota_limits,
        "domain_seen": domain_seen,
        "shard_comparison_limit": config["budgets"]["shard_comparison_limit"],
        "scheduler_stats": scheduler_stats,
    }
    queue_component = {
        "status": overall_status if queue_items else "empty",
        "items": queue_items,
        "next_priority_subject": queue_items[0]["subject_id"] if queue_items else None,
    }

    stage_status = {
        "cleaning": cleaning_component["status"],
        "dedup": dedup_component["status"],
        "sampling": "complete" if sampling_artifact else "pending",
        "summary": "complete" if summary_artifact else "pending",
    }
    manifest = {
        "schema_version": 1,
        "global_version": global_version,
        "parent_global_version": parent_version,
        "change_base_global_version": change_base_version,
        "status": overall_status,
        "business_code": None if overall_status == "complete" else overall_status,
        "stage_status": stage_status,
        "config_version": config["config_version"],
        "config_hash": config_hash,
        "dataset_fingerprint": dataset_fingerprint,
        "input_fingerprint": fingerprint({"config_hash": config_hash, "dataset_fingerprint": dataset_fingerprint}),
        "files": {},
    }
    state = {
        "manifest": manifest,
        "cleaning": cleaning_component,
        "dedup": dedup_component,
        "sampling": sampling_artifact,
        "summary": summary_artifact,
        "ledger": ledger_component,
        "queue": queue_component,
        "budget": budget_component,
        "error": error_payload,
    }

    snapshot_dir = store.commit(state, crash_before_replace=crash_before_commit)
    state["snapshot_dir"] = str(snapshot_dir)
    state["current_snapshot"] = str(store.current_link)
    return state


def sample_content_fingerprint(sample: dict[str, Any]) -> str:
    from .samples import sample_content_fingerprint as impl

    return impl(sample)


def _sampling_scope_hash(config: dict[str, Any]) -> dict[str, Any]:
    from .sampling import sampling_scope

    return sampling_scope(config)


def _worker_count(config: dict[str, Any], override: int | None, stage_key: str) -> int:
    configured = config["concurrency"][stage_key]
    max_workers = config["concurrency"]["max_workers"]
    value = configured if override is None else min(configured, override)
    return max(1, min(value, max_workers))


def _work_item(
    stage: str,
    subject_id: str,
    sample: dict[str, Any] | None,
    cost: int,
    *,
    shard: dict[str, Any] | None = None,
) -> dict[str, Any]:
    item: dict[str, Any] = {"stage": stage, "subject_id": subject_id, "cost": cost}
    if sample is not None:
        item["domain"] = sample["domain"]
        item["batch_id"] = sample["batch_id"]
    if shard is not None:
        item["domain"] = shard["domain"]
        item["shard_index"] = shard["shard_index"]
    return item


def _parallel_map(items: list[Any], workers: int, func: Callable[[Any], Any]) -> tuple[list[Any], dict[str, int]]:
    if not items:
        return [], {"max_active": 0, "observed_active": 0}
    active = 0
    max_active = 0

    def wrapped(item: Any) -> Any:
        nonlocal active, max_active
        active += 1
        max_active = max(max_active, active)
        try:
            return func(item)
        finally:
            active -= 1

    if workers == 1:
        results = [wrapped(item) for item in items]
    else:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            results = list(executor.map(wrapped, items))
    return results, {"max_active": max_active, "observed_active": max_active}


def _build_ledger(
    *,
    config: dict[str, Any],
    cleaning_records: list[dict[str, Any]],
    dedup_shards: list[dict[str, Any]],
    sampling_artifact: dict[str, Any] | None,
    summary_artifact: dict[str, Any] | None,
    baseline: dict[str, Any] | None,
) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    baseline_versions = baseline["manifest"]["global_version"] if baseline else None

    for record in cleaning_records:
        entries.append(
            make_entry(
                stage="cleaning",
                subject_id=record["sample_id"],
                config=config,
                cache_key=record["cache_key"],
                upstream={"sample_content_fingerprint": record["content_fingerprint"]},
                artifact_fingerprint=record["record_fingerprint"],
                disposition=record.get("disposition", "recomputed"),
                invalidation_reason=record.get("invalidation_reason", "cold_build"),
                scope_fingerprint=cleaning_config_fingerprint(config),
                reused_from_version=record.get("reused_from_global_version", baseline_versions)
                if record.get("disposition") == "reused"
                else None,
            )
        )

    for artifact in dedup_shards:
        scope_fp = fingerprint(dedup_scope(config, [artifact["domain"]]))
        upstream = {
            "shard_membership_fingerprint": fingerprint(artifact["sample_ids"]),
            "cleaning_set_fingerprint": fingerprint(
                [decision["sample_id"] for decision in artifact["decisions"]]
            ),
        }
        entries.append(
            make_entry(
                stage="dedup_shard",
                subject_id=artifact["shard_id"],
                config=config,
                cache_key=artifact["cache_key"],
                upstream=upstream,
                artifact_fingerprint=artifact["artifact_fingerprint"],
                disposition=artifact.get("disposition", "recomputed"),
                invalidation_reason=artifact.get("invalidation_reason", "cold_build"),
                scope_fingerprint=scope_fp,
                reused_from_version=artifact.get("reused_from_global_version", baseline_versions)
                if artifact.get("disposition") == "reused"
                else None,
            )
        )
        for decision in artifact["decisions"]:
            decision_key = fingerprint(
                {"decision": "dedup_decision", "shard_cache_key": artifact["cache_key"], "sample_id": decision["sample_id"]}
            )
            entries.append(
                make_entry(
                    stage="dedup_decision",
                    subject_id=decision["sample_id"],
                    config=config,
                    cache_key=decision_key,
                    upstream={
                        "dedup_shard_cache_key": artifact["cache_key"],
                        "dedup_shard_fingerprint": artifact["artifact_fingerprint"],
                    },
                    artifact_fingerprint=fingerprint(decision),
                    disposition=artifact.get("disposition", "recomputed"),
                    invalidation_reason=artifact.get("invalidation_reason", "cold_build"),
                    scope_fingerprint=scope_fp,
                )
            )

    if sampling_artifact:
        entries.append(
            make_entry(
                stage="sampling",
                subject_id="global",
                config=config,
                cache_key=sampling_artifact["cache_key"],
                upstream={
                    "dedup_fingerprint": sampling_artifact["upstream_dedup_fingerprint"],
                    "global_order_fingerprint": fingerprint(sampling_artifact["global_order"]),
                },
                artifact_fingerprint=sampling_artifact["artifact_fingerprint"],
                disposition=sampling_artifact.get("disposition", "recomputed"),
                invalidation_reason=sampling_artifact.get("invalidation_reason", "cold_build"),
                scope_fingerprint=fingerprint(_sampling_scope_hash(config)),
            )
        )

    if summary_artifact:
        entries.append(
            make_entry(
                stage="summary",
                subject_id="global",
                config=config,
                cache_key=summary_artifact["cache_key"],
                upstream={
                    "sampling_fingerprint": summary_artifact["upstream_sampling_fingerprint"],
                    "dedup_fingerprint": summary_artifact["upstream_dedup_fingerprint"],
                },
                artifact_fingerprint=summary_artifact["artifact_fingerprint"],
                disposition=summary_artifact.get("disposition", "recomputed"),
                invalidation_reason=summary_artifact.get("invalidation_reason", "cold_build"),
                scope_fingerprint=fingerprint(
                    {name: config["domain_map"][name]["target_ratio_fp"] for name in config["domain_names"]}
                ),
            )
        )
        for domain_row in summary_artifact["domains"]:
            entries.append(
                make_entry(
                    stage="summary_domain",
                    subject_id=domain_row["domain"],
                    config=config,
                    cache_key=fingerprint({"summary_cache_key": summary_artifact["cache_key"], "domain": domain_row["domain"]}),
                    upstream={"summary_fingerprint": summary_artifact["artifact_fingerprint"]},
                    artifact_fingerprint=fingerprint(domain_row),
                    disposition=summary_artifact.get("disposition", "recomputed"),
                    invalidation_reason=summary_artifact.get("invalidation_reason", "cold_build"),
                )
            )

    return assign_ordinals(entries)
