from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from .attribution import build_composition_stats, build_error_attribution
from .dedup import deduplicate_shard, expected_dedup_fingerprint
from .errors import Code, fail
from .loading import LoadedInputs
from .models import Document, PipelineConfig, content_hash
from .receipts import arbitrate_receipts, load_receipts, receipt_authority_diagnostics
from .sampling import (
    build_sample_plan,
    dedup_parameter_fingerprint,
    shard_documents_from_plan,
)
from .state import StateStore


ENGINE_VERSION = "0.1.0"


@dataclass(slots=True)
class PipelineResult:
    root: str
    run_key: str
    snapshot_name: str
    manifest: Mapping[str, Any]
    artifacts: Mapping[str, Any]

    @property
    def report(self) -> Mapping[str, Any]:
        return self.artifacts["report"]

    def to_dict(self) -> dict[str, Any]:
        return {
            "root": self.root,
            "run_key": self.run_key,
            "snapshot_name": self.snapshot_name,
            "manifest": dict(self.manifest),
            "report": dict(self.report),
        }


def _receipt_run_metadata(receipts: Sequence[Any], files: Sequence[str]) -> tuple[str, dict[str, Any]]:
    payload = {
        "folders": sorted(str(path) for path in files),
        "receipts": [
            receipt.to_dict() for receipt in sorted(receipts, key=lambda item: (item.source_file, item.receipt_id))
        ],
    }
    return content_hash(payload), payload


def _dedup_index(artifact: Mapping[str, Any] | None) -> dict[str, Mapping[str, Any]]:
    if not artifact:
        return {}
    shards = artifact.get("shards", [])
    if isinstance(shards, Mapping):
        return {str(key): value for key, value in shards.items()}
    return {str(shard["shard_id"]): shard for shard in shards}


def _sampling_index(artifact: Mapping[str, Any] | None) -> dict[str, Mapping[str, Any]]:
    if not artifact:
        return {}
    return {str(shard["shard_id"]): shard for shard in artifact.get("shards", [])}


def _domain_component_changes(previous: Mapping[str, Any], current: Mapping[str, Any]) -> list[str]:
    if not previous:
        return ["no_previous_domain_fingerprint"]
    changes: list[str] = []
    if previous.get("declaration") != current.get("declaration"):
        changes.append("domain_declaration_changed")
    if previous.get("weight") != current.get("weight"):
        changes.append("domain_weight_changed")
    if previous.get("documents") != current.get("documents"):
        changes.append("domain_documents_changed")
    if not changes and previous.get("combined") != current.get("combined"):
        changes.append("domain_fingerprint_changed")
    return changes


def _sampling_invalidation_reasons(
    *,
    previous_sampling: Mapping[str, Any] | None,
    current_shard: Mapping[str, Any],
    previous_domain_components: Mapping[str, Any],
    current_domain_components: Mapping[str, Any],
    previous_plan: Mapping[str, Any] | None,
    current_plan: Mapping[str, Any],
) -> list[str]:
    if previous_sampling is None:
        return ["new_or_unpublished_shard"]
    reasons = _domain_component_changes(
        previous_domain_components.get(current_shard["domain"], {}),
        current_domain_components.get(current_shard["domain"], {}),
    )
    if previous_plan is not None:
        previous_allocation = previous_plan.get("allocations", {}).get(current_shard["domain"])
        current_allocation = current_plan.get("allocations", {}).get(current_shard["domain"])
        if previous_allocation != current_allocation:
            reasons.append("sampling_allocation_changed")
        previous_context = previous_sampling.get("domain_context_fingerprint")
        if previous_context != current_shard.get("domain_context_fingerprint"):
            reasons.append("sampling_order_or_boundary_changed")
    if not reasons:
        reasons.append("sampling_fingerprint_mismatch")
    return sorted(set(reasons))


def _dedup_invalidation_reasons(
    *,
    previous_result: Mapping[str, Any] | None,
    current_shard: Mapping[str, Any],
    previous_sampling: Mapping[str, Any] | None,
    current_parameter_fingerprint: str,
) -> list[str]:
    reasons: list[str] = []
    if previous_result is None:
        reasons.append("no_previous_dedup_artifact")
    else:
        if previous_result.get("sampling_fingerprint") != current_shard["sampling_fingerprint"]:
            reasons.append("upstream_sampling_changed")
        if previous_result.get("parameter_fingerprint") != current_parameter_fingerprint:
            reasons.append("dedup_parameters_changed")
        if previous_sampling is None:
            reasons.append("sampling_artifact_missing")
    if not reasons:
        reasons.append("dedup_fingerprint_mismatch")
    return sorted(set(reasons))


def _sort_failures(failures: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    return [
        dict(item)
        for item in sorted(
            failures,
            key=lambda item: (
                int(item.get("code", 0)),
                str(item.get("domain", "")),
                str(item.get("shard_id", "")),
                str(item.get("doc_id", "")),
                str(item.get("message", "")),
            ),
        )
    ]


def _make_event(event: str, **fields: Any) -> dict[str, Any]:
    payload = {"event": event, **fields}
    payload["event_fingerprint"] = content_hash(payload)
    return payload


def _load_result_from_snapshot(root: Path, snapshot: Mapping[str, Any]) -> PipelineResult:
    return PipelineResult(
        root=str(root),
        run_key=str(snapshot["manifest"]["run_key"]),
        snapshot_name=str(snapshot["name"]),
        manifest=snapshot["manifest"],
        artifacts={
            "sampling": snapshot.get("sampling"),
            "dedup": snapshot.get("dedup"),
            "composition": snapshot.get("composition"),
            "attribution": snapshot.get("attribution"),
            "failures": snapshot.get("failures"),
            "receipts": snapshot.get("receipts"),
            "report": snapshot.get("report"),
        },
    )


def _reference_config(config: PipelineConfig) -> PipelineConfig:
    return config.with_precision(config.precision.higher_precision())


def _run_reference_deduplication(
    config: PipelineConfig,
    plan: Mapping[str, Any],
    documents: Mapping[str, Sequence[Document]],
    workers: int,
) -> list[dict[str, Any]]:
    shards = []
    for shard in sorted(plan["shards"], key=lambda item: item["shard_id"]):
        shard_docs = shard_documents_from_plan(shard, documents)
        shards.append((shard, shard_docs))
    if workers <= 1 or len(shards) <= 1:
        return [deduplicate_shard(config, shard, docs) for shard, docs in shards]
    with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="reference-dedup") as executor:
        futures = [executor.submit(deduplicate_shard, config, shard, docs) for shard, docs in shards]
        results = [future.result() for future in futures]
    return sorted(results, key=lambda item: item["shard_id"])


def run_pipeline(
    root: str | Path,
    inputs: LoadedInputs,
    *,
    receipts_source: str | Path | Sequence[str | Path] | None = None,
    workers: int = 4,
    force: bool = False,
    kill_after_shards: int | None = None,
) -> PipelineResult:
    if not isinstance(inputs, LoadedInputs):
        fail(Code.INVALID_SCHEMA, "run_pipeline requires LoadedInputs")
    if workers < 1:
        fail(Code.INVALID_SCHEMA, "workers must be a positive integer", workers=workers)
    root_path = Path(root)
    store = StateStore(root_path)
    store.initialize()

    receipts, receipt_files = load_receipts(receipts_source, inputs.config)
    receipts_fingerprint, receipts_payload = _receipt_run_metadata(receipts, receipt_files)
    receipt_authority = arbitrate_receipts(receipts)
    run_key = content_hash(
        {
            "engine_version": ENGINE_VERSION,
            "config_hash": inputs.config_hash,
            "input_hash": inputs.input_hash,
            "receipts_fingerprint": receipts_fingerprint,
        }
    )
    kill_limit = kill_after_shards
    if kill_limit is None:
        environment_value = os.environ.get("CORPUS_PIPELINE_KILL_AFTER_SHARDS")
        if environment_value:
            try:
                kill_limit = int(environment_value)
            except ValueError:
                fail(Code.INVALID_SCHEMA, "kill-after-shards environment value must be an integer")

    with store.exclusive_run_lock():
        previous = store.load_current_snapshot()
        previous_manifest = previous["manifest"] if previous else None
        recovery_reasons: list[str] = []
        if previous_manifest:
            if previous_manifest.get("global_version") != inputs.effective_version():
                recovery_reasons.append("global_version_changed")
            if previous_manifest.get("config_hash") != inputs.config_hash:
                recovery_reasons.append("config_hash_changed")
            if previous_manifest.get("input_hash") != inputs.input_hash:
                recovery_reasons.append("input_hash_changed")
            if previous_manifest.get("receipts_fingerprint") != receipts_fingerprint:
                recovery_reasons.append("receipts_changed")

        if (
            previous is not None
            and previous_manifest is not None
            and previous_manifest.get("run_key") == run_key
            and previous_manifest.get("status") == "complete"
            and not force
        ):
            event = _make_event(
                "snapshot_reuse",
                run_key=run_key,
                snapshot=previous["name"],
                global_version=inputs.effective_version(),
                config_hash=inputs.config_hash,
                input_hash=inputs.input_hash,
            )
            store.append_ledger([event])
            return _load_result_from_snapshot(root_path, previous)

        work_manifest = store.load_work_manifest(run_key)
        if work_manifest and work_manifest.get("status") != "complete":
            recovery_decision = "resume"
            recovery_reasons = ["matching_incomplete_work_checkpoint", "resume_completed_shards_only"]
        elif previous_manifest is not None:
            recovery_decision = "forced_rebuild" if force else "rebuild"
            if not recovery_reasons:
                recovery_reasons = ["new_run_key"]
        else:
            recovery_decision = "fresh"
            if force:
                recovery_decision = "forced_rebuild"

        if work_manifest is None:
            work_manifest = {
                "schema_version": 1,
                "engine_version": ENGINE_VERSION,
                "run_key": run_key,
                "global_version": inputs.effective_version(),
                "config_hash": inputs.config_hash,
                "input_hash": inputs.input_hash,
                "receipts_fingerprint": receipts_fingerprint,
                "status": "running",
                "completed_stages": [],
                "completed_shards": [],
                "budget_consumed": 0,
                "created_by": "run_pipeline",
                "recovery_decision": recovery_decision,
            }
            store.save_work_manifest(run_key, work_manifest)
        elif work_manifest.get("status") == "complete":
            # A completed work manifest without a visible snapshot can happen
            # after an interrupted pointer swap.  The stage artifacts are still
            # authoritative and deterministic.
            recovery_decision = "resume"

        base_events: list[dict[str, Any]] = [
            _make_event(
                "run_recovery",
                run_key=run_key,
                global_version=inputs.effective_version(),
                config_hash=inputs.config_hash,
                input_hash=inputs.input_hash,
                decision=recovery_decision,
                reasons=sorted(recovery_reasons),
                previous_snapshot=previous["name"] if previous else None,
            )
        ]

        sampling_stage = store.load_work_stage(run_key, "sampling")
        reference_stage = store.load_work_stage(run_key, "reference_sampling")
        if sampling_stage is not None:
            current_plan = build_sample_plan(inputs.config, inputs.documents)
            if sampling_stage.get("plan_fingerprint") == current_plan["plan_fingerprint"]:
                stored_plan = sampling_stage
                base_events.append(
                    _make_event(
                        "sampling_plan_reuse",
                        run_key=run_key,
                        action="reused",
                        basis="checkpoint_plan_fingerprint_match",
                        plan_fingerprint=stored_plan["plan_fingerprint"],
                    )
                )
            else:
                stored_plan = current_plan
                store.save_work_stage(run_key, "sampling", stored_plan)
                base_events.append(
                    _make_event(
                        "sampling_plan_decision",
                        run_key=run_key,
                        action="recomputed",
                        basis="checkpoint_plan_fingerprint_mismatch",
                        plan_fingerprint=stored_plan["plan_fingerprint"],
                    )
                )
            current_plan = stored_plan
        else:
            current_plan = build_sample_plan(inputs.config, inputs.documents)
            store.save_work_stage(run_key, "sampling", current_plan)
            base_events.append(
                _make_event(
                    "sampling_plan_decision",
                    run_key=run_key,
                    action="recomputed",
                    basis="no_checkpoint",
                    plan_fingerprint=current_plan["plan_fingerprint"],
                )
            )

        if reference_stage is not None:
            reference_plan = reference_stage
        else:
            reference_plan = build_sample_plan(_reference_config(inputs.config), inputs.documents)
            store.save_work_stage(run_key, "reference_sampling", reference_plan)

        work_manifest["budget_consumed"] = current_plan["budget_consumed"]
        work_manifest["completed_stages"] = sorted(
            set(work_manifest.get("completed_stages", [])) | {"sampling"}
        )
        store.save_work_manifest(run_key, work_manifest)
        store.append_ledger(base_events)

        previous_sampling = previous.get("sampling") if previous else None
        previous_dedup = previous.get("dedup") if previous else None
        previous_plan = previous_sampling if previous_sampling else None
        previous_sampling_index = _sampling_index(previous_sampling)
        previous_dedup_index = _dedup_index(previous_dedup)
        previous_components = (
            previous_sampling.get("domain_components", {}) if previous_sampling else {}
        )
        current_components = {
            domain: inputs.domain_component_fingerprints(domain) for domain in inputs.config.domain_ids
        }

        current_shards = sorted(current_plan["shards"], key=lambda item: item["shard_id"])
        shard_entries: dict[str, tuple[Mapping[str, Any], list[Document], str]] = {}
        for shard in current_shards:
            docs = shard_documents_from_plan(shard, inputs.documents)
            expected = expected_dedup_fingerprint(inputs.config, shard, docs)
            shard_entries[shard["shard_id"]] = (shard, docs, expected)

        sample_reuse_count = 0
        snapshot_dedup_reuse_count = 0
        resume_checkpoint_reuse_count = 0
        recompute_count = 0
        results: dict[str, Mapping[str, Any]] = {}
        events: list[dict[str, Any]] = []
        completed_shards = set(work_manifest.get("completed_shards", []))

        def sync_commit(shard_id: str, result: Mapping[str, Any], action_basis: str, reasons: list[str]) -> None:
            results[shard_id] = result
            completed_shards.add(shard_id)
            store.save_shard_checkpoint(run_key, shard_id, result)

        for shard_id, (shard, docs, expected_fingerprint) in shard_entries.items():
            previous_sample = previous_sampling_index.get(shard_id)
            previous_result = previous_dedup_index.get(shard_id)
            if (
                previous_sample is not None
                and previous_sample.get("sampling_fingerprint") == shard["sampling_fingerprint"]
            ):
                sample_reuse_count += 1
                sampling_action = "reused"
                sampling_basis = "snapshot_sampling_fingerprint_match"
                sampling_reasons: list[str] = []
            else:
                sampling_action = "recomputed"
                sampling_basis = "fingerprint_mismatch"
                sampling_reasons = _sampling_invalidation_reasons(
                    previous_sampling=previous_sample,
                    current_shard=shard,
                    previous_domain_components=previous_components,
                    current_domain_components=current_components,
                    previous_plan=previous_plan,
                    current_plan=current_plan,
                )
            events.append(
                _make_event(
                    "shard_reuse_decision",
                    run_key=run_key,
                    stage="sampling",
                    shard_id=shard_id,
                    domain=shard["domain"],
                    action=sampling_action,
                    basis=sampling_basis,
                    reasons=sampling_reasons,
                    sampling_fingerprint=shard["sampling_fingerprint"],
                )
            )

            checkpoint = store.load_shard_checkpoint(run_key, shard_id)
            if checkpoint is not None and checkpoint.get("dedup_fingerprint") == expected_fingerprint:
                sync_commit(shard_id, checkpoint, "resume_checkpoint_fingerprint_match", ["resume_exact_fingerprint_match"])
                resume_checkpoint_reuse_count += 1
                events.append(
                    _make_event(
                        "shard_reuse_decision",
                        run_key=run_key,
                        stage="dedup",
                        shard_id=shard_id,
                        domain=shard["domain"],
                        action="reused",
                        basis="resume_checkpoint_fingerprint_match",
                        reasons=["resume_exact_fingerprint_match"],
                        dedup_fingerprint=expected_fingerprint,
                    )
                )
            elif previous_result is not None and previous_result.get("dedup_fingerprint") == expected_fingerprint:
                sync_commit(shard_id, previous_result, "snapshot_dedup_fingerprint_match", ["snapshot_exact_fingerprint_match"])
                snapshot_dedup_reuse_count += 1
                events.append(
                    _make_event(
                        "shard_reuse_decision",
                        run_key=run_key,
                        stage="dedup",
                        shard_id=shard_id,
                        domain=shard["domain"],
                        action="reused",
                        basis="snapshot_dedup_fingerprint_match",
                        reasons=["snapshot_exact_fingerprint_match"],
                        dedup_fingerprint=expected_fingerprint,
                    )
                )
            # Remaining shards are computed in the thread pool below.

        compute_entries = [
            (shard_id, shard_entries[shard_id][0], shard_entries[shard_id][1], shard_entries[shard_id][2])
            for shard_id in sorted(shard_entries)
            if shard_id not in results
        ]
        current_parameter_fingerprint = dedup_parameter_fingerprint(inputs.config)
        if compute_entries:
            with ThreadPoolExecutor(max_workers=min(workers, len(compute_entries)), thread_name_prefix="dedup") as executor:
                future_map = {
                    executor.submit(deduplicate_shard, inputs.config, shard, docs): (shard_id, shard, expected)
                    for shard_id, shard, docs, expected in compute_entries
                }
                for future in as_completed(future_map):
                    shard_id, shard, expected = future_map[future]
                    result = future.result()
                    if result.get("dedup_fingerprint") != expected:
                        fail(
                            Code.STATE_CORRUPT,
                            "computed dedup fingerprint does not match the precomputed dependency fingerprint",
                            shard_id=shard_id,
                        )
                    previous_sample = previous_sampling_index.get(shard_id)
                    previous_result = previous_dedup_index.get(shard_id)
                    reasons = _dedup_invalidation_reasons(
                        previous_result=previous_result,
                        current_shard=shard,
                        previous_sampling=previous_sample,
                        current_parameter_fingerprint=current_parameter_fingerprint,
                    )
                    sync_commit(shard_id, result, "recomputed", reasons)
                    recompute_count += 1
                    completed_shards.add(shard_id)
                    work_manifest["completed_shards"] = sorted(completed_shards)
                    work_manifest["status"] = "running"
                    store.save_work_manifest(run_key, work_manifest)
                    events.append(
                        _make_event(
                            "shard_reuse_decision",
                            run_key=run_key,
                            stage="dedup",
                            shard_id=shard_id,
                            domain=shard["domain"],
                            action="recomputed",
                            basis="content_fingerprint_changed",
                            reasons=reasons,
                            dedup_fingerprint=expected,
                        )
                    )
                    if kill_limit is not None and len(completed_shards) >= kill_limit:
                        work_manifest["status"] = "interrupted"
                        work_manifest["kill_after_shards"] = kill_limit
                        work_manifest["completed_shards"] = sorted(completed_shards)
                        work_manifest["completed_stages"] = sorted(
                            set(work_manifest.get("completed_stages", [])) | {"sampling", "dedup_partial"}
                        )
                        store.save_work_manifest(run_key, work_manifest)
                        store.append_ledger(events)
                        # The work manifest, per-shard checkpoints and ledger are
                        # fsynced before the forced process termination.
                        os._exit(137)

        work_manifest["completed_shards"] = sorted(completed_shards)
        work_manifest["completed_stages"] = sorted(
            set(work_manifest.get("completed_stages", [])) | {"sampling", "dedup"}
        )
        work_manifest["status"] = "running"
        store.save_work_manifest(run_key, work_manifest)
        store.append_ledger(events)
        events = []

        primary_results = [results[shard["shard_id"]] for shard in current_shards]
        dedup_artifact = {
            "parameter_fingerprint": current_parameter_fingerprint,
            "shards": sorted(primary_results, key=lambda item: item["shard_id"]),
            "new_shard_failures": _sort_failures(
                failure for result in primary_results for failure in result.get("failures", [])
            ),
        }
        store.save_work_stage(run_key, "dedup", dedup_artifact)

        composition = build_composition_stats(inputs.config, current_plan, primary_results, inputs.documents)
        composition_fingerprint = content_hash(
            {
                "config_stats": {
                    "precision": inputs.config.precision.to_dict(),
                    "sampling": inputs.config.sampling.to_dict(),
                },
                "shards": [
                    [result["shard_id"], result["dedup_fingerprint"]]
                    for result in sorted(primary_results, key=lambda item: item["shard_id"])
                ],
                "values": composition,
            }
        )
        composition_artifact = {**composition, "composition_fingerprint": composition_fingerprint}

        ref_config = _reference_config(inputs.config)
        ref_results = _run_reference_deduplication(ref_config, reference_plan, inputs.documents, workers)
        reference_stats = build_composition_stats(ref_config, reference_plan, ref_results, inputs.documents)
        attribution = build_error_attribution(
            inputs.config,
            current_plan,
            primary_results,
            composition_artifact,
            ref_config,
            reference_plan,
            ref_results,
            reference_stats,
        )

        sampling_failures = current_plan.get("failures", [])
        dedup_failures = dedup_artifact["new_shard_failures"]
        all_failures = _sort_failures(list(sampling_failures) + list(dedup_failures))
        failures_artifact = {
            "sampling": _sort_failures(sampling_failures),
            "dedup": _sort_failures(dedup_failures),
            "all": all_failures,
            "sampling_failure_count": len(sampling_failures),
            "dedup_failure_count": len(dedup_failures),
            "total_failure_count": len(all_failures),
        }

        receipt_diagnostics = receipt_authority_diagnostics(receipt_authority, current_plan)
        receipts_artifact = {
            "authority": receipt_authority,
            "diagnostics": receipt_diagnostics,
            "source_files": sorted(str(path) for path in receipt_files),
            "receipts_fingerprint": receipts_fingerprint,
            "normalized_inputs": receipts_payload,
        }

        current_shard_count = len(current_shards)
        dedup_reuse_total = snapshot_dedup_reuse_count + resume_checkpoint_reuse_count
        total_reuse_observations = sample_reuse_count + dedup_reuse_total
        observation_total = max(1, current_shard_count * 2)
        reuse_rate = round(total_reuse_observations / observation_total, 6)
        shard_reuse_rate = round(dedup_reuse_total / max(1, current_shard_count), 6)

        summary_reasons: list[str] = []
        if any(
            previous_dedup_index.get(shard["shard_id"], {}).get("dedup_fingerprint")
            != results[shard["shard_id"]]["dedup_fingerprint"]
            for shard in current_shards
        ):
            summary_reasons.append("dedup_fingerprint_changed")
        if previous is None:
            summary_reasons.append("no_previous_composition")
        if not summary_reasons:
            summary_reasons.append("composition_rebuilt_for_consistent_reporting")
        events.append(
            _make_event(
                "aggregate_reuse_decision",
                run_key=run_key,
                stage="composition_summary",
                action="recomputed",
                basis="derived_from_dedup_fingerprints",
                reasons=sorted(set(summary_reasons)),
                composition_fingerprint=composition_fingerprint,
            )
        )
        for group in receipt_authority.get("groups", []):
            events.append(
                _make_event(
                    "receipt_authority",
                    run_key=run_key,
                    domain=group["domain"],
                    shard_id=group["shard_id"],
                    authority_receipt_id=group["authority_receipt_id"],
                    decision_basis=group["decision_basis"],
                    covered_refs=group["covered_refs"],
                    superseded_refs=group["superseded_refs"],
                    rejected_overrides=group["rejected_overrides"],
                )
            )

        remaining_budget = max(0, inputs.config.sampling.budget - current_plan["budget_consumed"])
        report = {
            "status": "complete",
            "run_key": run_key,
            "global_version": inputs.effective_version(),
            "config_hash": inputs.config_hash,
            "input_hash": inputs.input_hash,
            "receipts_fingerprint": receipts_fingerprint,
            "recovery": {
                "decision": recovery_decision,
                "reasons": sorted(recovery_reasons),
                "previous_snapshot": previous["name"] if previous else None,
            },
            "parallelism": {"workers": workers},
            "budget": {
                "requested_samples": inputs.config.sampling.budget,
                "consumed_samples": current_plan["budget_consumed"],
                "remaining_samples": remaining_budget,
                "per_domain_allocations": current_plan["allocations"],
                "per_domain_quota": inputs.config.sampling.per_domain_quota,
                "shard_size": inputs.config.sampling.shard_size,
            },
            "shards": {
                "total": current_shard_count,
                "sampling_reused": sample_reuse_count,
                "dedup_reused": dedup_reuse_total,
                "snapshot_dedup_reused": snapshot_dedup_reuse_count,
                "resume_checkpoint_reused": resume_checkpoint_reuse_count,
                "recomputed": recompute_count,
                "completed": len(results),
            },
            "reuse": {
                "overall_reuse_rate": reuse_rate,
                "dedup_shard_reuse_rate": shard_reuse_rate,
                "sampling_reuse_rate": round(sample_reuse_count / max(1, current_shard_count), 6),
                "observation_count": total_reuse_observations,
            },
            "failures": {
                "total": failures_artifact["total_failure_count"],
                "sampling": failures_artifact["sampling_failure_count"],
                "dedup": failures_artifact["dedup_failure_count"],
            },
            "dedup": {
                "sampled_documents": sum(result["sampled_count"] for result in primary_results),
                "retained_documents": sum(result["retained_count"] for result in primary_results),
                "clusters": sum(len(result["clusters"]) for result in primary_results),
                "boundary_pairs": sum(len(result["boundary_pairs"]) for result in primary_results),
                "comparison_limit_exhausted_shards": sum(
                    1 for result in primary_results if result["limit_exhausted"]
                ),
            },
            "composition": composition_artifact["totals"],
            "attribution": {
                "first_exceeded_stage": attribution["first_exceeded_stage"],
                "conclusion": attribution["conclusion"],
            },
            "receipts": {
                "authority_groups": receipt_authority["group_count"],
                "rejected_overrides": receipt_authority["rejected_override_count"],
                "sample_count_mismatches": len(receipt_diagnostics["mismatches"]),
            },
            "determinism": {
                "ordinal_model": current_plan["ordinal_model"],
                "budget_preallocated": True,
                "pair_scan_order": "doc_id_lexicographic_prefix",
                "completion_order_independent": True,
            },
        }

        sampling_artifact = {**current_plan, "domain_components": current_components}
        artifacts = {
            "sampling": sampling_artifact,
            "dedup": dedup_artifact,
            "composition": composition_artifact,
            "attribution": attribution,
            "failures": failures_artifact,
            "receipts": receipts_artifact,
            "report": report,
        }

        events.append(
            _make_event(
                "summary_commit",
                run_key=run_key,
                global_version=inputs.effective_version(),
                config_hash=inputs.config_hash,
                budget_consumed=current_plan["budget_consumed"],
                completed_shards=sorted(results),
            )
        )
        store.append_ledger(events)
        events = []

        work_manifest["status"] = "complete"
        work_manifest["completed_stages"] = ["sampling", "dedup", "composition", "attribution", "receipts"]
        work_manifest["completed_shards"] = sorted(results)
        work_manifest["budget_consumed"] = current_plan["budget_consumed"]
        work_manifest["report_fingerprint"] = content_hash(report)
        store.save_work_manifest(run_key, work_manifest)

        manifest = {
            "schema_version": 1,
            "engine_version": ENGINE_VERSION,
            "run_key": run_key,
            "status": "complete",
            "global_version": inputs.effective_version(),
            "config_hash": inputs.config_hash,
            "input_hash": inputs.input_hash,
            "documents_fingerprint": inputs.documents_fingerprint,
            "domains_fingerprint": inputs.domains_fingerprint,
            "receipts_fingerprint": receipts_fingerprint,
            "completed_stages": ["sampling", "dedup", "composition", "attribution", "receipts"],
            "completed_shards": sorted(results),
            "budget_requested": inputs.config.sampling.budget,
            "budget_consumed": current_plan["budget_consumed"],
            "budget_remaining": remaining_budget,
            "workers": workers,
            "recovery_decision": recovery_decision,
            "reuse": report["reuse"],
            "failure_count": failures_artifact["total_failure_count"],
            "domain_components": current_components,
            "report_fingerprint": content_hash(report),
        }
        snapshot_name = store.publish_snapshot(run_key, artifacts, manifest)
        commit_event = _make_event(
            "snapshot_commit",
            run_key=run_key,
            snapshot=snapshot_name,
            global_version=inputs.effective_version(),
            config_hash=inputs.config_hash,
            input_hash=inputs.input_hash,
            completed_shards=sorted(results),
            budget_consumed=current_plan["budget_consumed"],
        )
        store.append_ledger([commit_event])

        return PipelineResult(
            root=str(root_path),
            run_key=run_key,
            snapshot_name=snapshot_name,
            manifest=manifest,
            artifacts=artifacts,
        )
