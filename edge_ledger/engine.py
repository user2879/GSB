from __future__ import annotations

import copy
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

from .canonical import digest, fixed_string
from .config import PipelineConfig
from .dedupe import DedupeInput, build_dedupe, subtoken_set
from .errors import Failure, SimulatedCrash, sorted_failure_dicts
from .loader import InjectionRecord, LoadedWorkspace, ValidatedSample, load_workspace
from .models import QualityBreakdown
from .scheduler import (
    ConcurrencyProbe,
    RenderTask,
    ScheduleResult,
    run_thread_pool,
    schedule_samples,
)
from .scoring import ScoreFailure, compute_quality
from .snapshot import LoadSnapshotResult, SnapshotStore, SnapshotView
from .templates import render_template


RENDER_SCHEMA_VERSION = 1
SCORE_SCHEMA_VERSION = 1
DEDUPE_SCHEMA_VERSION = 1
STATS_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class RenderProduct:
    sample_id: str
    template_id: str
    template_revision: int
    answer_template_id: str
    answer_template_revision: int
    chain_fingerprint: str
    answer_chain_fingerprint: str
    sample_fingerprint: str
    render_cache_key: str
    render_digest: str
    instruction_text: str
    answer_text: str
    fill_order: tuple[str, ...]
    resolved_slots: tuple[dict[str, Any], ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "sample_id": self.sample_id,
            "template_id": self.template_id,
            "template_revision": self.template_revision,
            "answer_template_id": self.answer_template_id,
            "answer_template_revision": self.answer_template_revision,
            "chain_fingerprint": self.chain_fingerprint,
            "answer_chain_fingerprint": self.answer_chain_fingerprint,
            "sample_fingerprint": self.sample_fingerprint,
            "render_cache_key": self.render_cache_key,
            "render_digest": self.render_digest,
            "instruction_text": self.instruction_text,
            "answer_text": self.answer_text,
            "fill_order": list(self.fill_order),
            "resolved_slots": [dict(item) for item in self.resolved_slots],
        }


@dataclass(frozen=True)
class ScoreProduct:
    sample_id: str
    render_digest: str
    score_cache_key: str
    quality: QualityBreakdown

    def as_dict(self) -> dict[str, Any]:
        return {
            "sample_id": self.sample_id,
            "render_digest": self.render_digest,
            "score_cache_key": self.score_cache_key,
            "quality": self.quality.as_dict(),
        }


@dataclass
class WorkOutcome:
    task: RenderTask
    sample_result: dict[str, Any]
    render_product: RenderProduct | None = None
    score_product: ScoreProduct | None = None
    failure: Failure | None = None
    render_action: str = "recompute"
    score_action: str = "recompute"
    render_basis: str = ""
    score_basis: str = ""


@dataclass
class BuildResult:
    root: Path
    result: dict[str, Any]
    canonical: dict[str, Any]
    canonical_digest: str
    runtime_seconds: float
    snapshot_generation: int | None
    recovery: dict[str, Any]
    probe_max_concurrency: int

    def as_report(self) -> dict[str, Any]:
        report = copy.deepcopy(self.result)
        report["report"] = {
            "runtime_ms": round(self.runtime_seconds * 1000, 3),
            "snapshot_generation": self.snapshot_generation,
            "recovery": self.recovery,
            "max_observed_concurrency": self.probe_max_concurrency,
            "canonical_digest": self.canonical_digest,
        }
        return report


def _cache_entry(cache_key: str, payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "cache_key": cache_key,
        "payload_hash": digest(payload),
        "payload": payload,
    }


def _cache_lookup(cache: Mapping[str, Any], key: str) -> dict[str, Any] | None:
    entry = cache.get(key)
    if not isinstance(entry, dict):
        return None
    payload = entry.get("payload")
    if not isinstance(payload, dict):
        return None
    if entry.get("payload_hash") != digest(payload):
        return None
    return payload


def _resolved_values_document(sample: ValidatedSample) -> list[dict[str, Any]]:
    return [
        {
            "name": slot.name,
            "value": slot.value,
            "priority": slot.priority,
            "source_index": slot.source_index,
            "suppressed": list(slot.suppressed),
        }
        for slot in sample.resolved_slots
    ]


def render_key_for(sample: ValidatedSample) -> str:
    return digest(
        {
            "kind": "render-cache-key",
            "schema_version": RENDER_SCHEMA_VERSION,
            "chain_fingerprint": sample.instruction.chain_fingerprint,
            "answer_chain_fingerprint": sample.answer.chain_fingerprint,
            "sample_fingerprint": sample.sample_fingerprint,
            "slot_values": [
                {"name": slot.name, "value": slot.value}
                for slot in sample.resolved_slots
            ],
        }
    )


def score_key_for(render_digest: str, config: PipelineConfig) -> str:
    return digest(
        {
            "kind": "score-cache-key",
            "schema_version": SCORE_SCHEMA_VERSION,
            "render_digest": render_digest,
            "quality_fingerprint": config.quality_fingerprint,
            "quantization": config.quantization,
            "quantum_step": config.quantum_step,
            "decimal_places": config.decimal_places,
        }
    )


def template_cache_key(effective: Any) -> str:
    return digest(
        {
            "kind": "template-effective-cache-key",
            "template_id": effective.template_id,
            "revision": effective.revision,
            "chain_fingerprint": effective.chain_fingerprint,
            "effective_fingerprint": effective.effective_fingerprint,
        }
    )


def _render_sample(sample: ValidatedSample) -> RenderProduct:
    values = sample.slot_values()
    if tuple(sorted(values)) != sample.fill_order:
        raise AssertionError("slot fill order is not stable")
    instruction_text = render_template(sample.instruction.instruction_template, values)
    answer_text = render_template(sample.answer.answer_template, values)
    render_fp = _render_digest(sample, instruction_text, answer_text)
    return RenderProduct(
        sample_id=sample.sample.sample_id,
        template_id=sample.sample.template_id,
        template_revision=sample.sample.template_revision,
        answer_template_id=sample.sample.answer_template_id,
        answer_template_revision=sample.sample.answer_template_revision,
        chain_fingerprint=sample.instruction.chain_fingerprint,
        answer_chain_fingerprint=sample.answer.chain_fingerprint,
        sample_fingerprint=sample.sample_fingerprint,
        render_cache_key=render_key_for(sample),
        render_digest=render_fp,
        instruction_text=instruction_text,
        answer_text=answer_text,
        fill_order=sample.fill_order,
        resolved_slots=tuple(slot.as_dict() for slot in sample.resolved_slots),
    )


def _render_digest(sample: ValidatedSample, instruction_text: str, answer_text: str) -> str:
    return digest(
        {
            "kind": "render-digest",
            "schema_version": RENDER_SCHEMA_VERSION,
            "chain_fingerprint": sample.instruction.chain_fingerprint,
            "answer_chain_fingerprint": sample.answer.chain_fingerprint,
            "sample_fingerprint": sample.sample_fingerprint,
            "slot_values": [
                {"name": slot.name, "value": slot.value}
                for slot in sample.resolved_slots
            ],
            "instruction_text": instruction_text,
            "answer_text": answer_text,
        }
    )


def _render_product_from_payload(
    payload: Mapping[str, Any], expected_key: str, sample: ValidatedSample
) -> RenderProduct | None:
    try:
        product = RenderProduct(
            sample_id=str(payload["sample_id"]),
            template_id=str(payload["template_id"]),
            template_revision=int(payload["template_revision"]),
            answer_template_id=str(payload["answer_template_id"]),
            answer_template_revision=int(payload["answer_template_revision"]),
            chain_fingerprint=str(payload["chain_fingerprint"]),
            answer_chain_fingerprint=str(payload["answer_chain_fingerprint"]),
            sample_fingerprint=str(payload["sample_fingerprint"]),
            render_cache_key=str(payload["render_cache_key"]),
            render_digest=str(payload["render_digest"]),
            instruction_text=str(payload["instruction_text"]),
            answer_text=str(payload["answer_text"]),
            fill_order=tuple(str(value) for value in payload["fill_order"]),
            resolved_slots=tuple(dict(value) for value in payload["resolved_slots"]),
        )
        expected_digest = _render_digest(
            sample, product.instruction_text, product.answer_text
        )
    except (KeyError, TypeError, ValueError):
        return None
    if (
        product.render_cache_key != expected_key
        or product.sample_id != sample.sample.sample_id
        or product.sample_fingerprint != sample.sample_fingerprint
        or product.chain_fingerprint != sample.instruction.chain_fingerprint
        or product.answer_chain_fingerprint != sample.answer.chain_fingerprint
        or product.render_digest != expected_digest
        or product.fill_order != sample.fill_order
    ):
        return None
    return product


def _sample_base_result(sample: ValidatedSample, status: str) -> dict[str, Any]:
    return {
        "sample_id": sample.sample.sample_id,
        "source_shard": sample.sample.source_shard,
        "batch_id": sample.sample.batch_id,
        "template_id": sample.sample.template_id,
        "template_revision": sample.sample.template_revision,
        "answer_template_id": sample.sample.answer_template_id,
        "answer_template_revision": sample.sample.answer_template_revision,
        "template_chain_fingerprint": sample.instruction.chain_fingerprint,
        "answer_chain_fingerprint": sample.answer.chain_fingerprint,
        "sample_fingerprint": sample.sample_fingerprint,
        "status": status,
        "render_cache_key": render_key_for(sample),
        "resolved_slots": [slot.as_dict() for slot in sample.resolved_slots],
        "fill_order": list(sample.fill_order),
        "injection_ids": list(sample.sample.injection_ids),
    }


def _quality_document(quality: QualityBreakdown) -> dict[str, Any]:
    result = quality.as_dict()
    result["score_fixed"] = fixed_string(quality.quantized_score, quality.decimal_places)
    result["residual_fixed"] = fixed_string(quality.residual, quality.decimal_places)
    return result


def _invalidation_plan(
    loaded: LoadedWorkspace,
    previous: SnapshotView | None,
    recovery: dict[str, Any],
) -> dict[str, Any]:
    prior_inputs = previous.input_fingerprints if previous is not None else {}
    prior_templates = prior_inputs.get("templates", {}) if isinstance(prior_inputs, dict) else {}
    prior_samples = prior_inputs.get("samples", {}) if isinstance(prior_inputs, dict) else {}
    current_templates = loaded.revision_fingerprints
    current_samples = loaded.sample_fingerprints
    changed_templates = sorted(
        key
        for key in set(prior_templates) | set(current_templates)
        if prior_templates.get(key) != current_templates.get(key)
    )
    changed_samples = sorted(
        key
        for key in set(prior_samples) | set(current_samples)
        if prior_samples.get(key) != current_samples.get(key)
    )
    affected: dict[str, list[str]] = {}
    for sample in loaded.valid_samples:
        revision_keys = {
            f"{sample.sample.template_id}@{rev}" for rev in sample.instruction.revision_path
        } | {
            f"{sample.sample.answer_template_id}@{rev}" for rev in sample.answer.revision_path
        }
        matched = sorted(revision_keys & set(changed_templates))
        if matched:
            affected[sample.sample.sample_id] = matched
    prior_config = previous.manifest.get("config_hash") if previous is not None else None
    return {
        "mode": recovery["action"],
        "changed_template_revisions": changed_templates,
        "changed_sample_ids": changed_samples,
        "config_changed": prior_config != loaded.config_hash,
        "affected_samples_by_template_revision": affected,
        "previous_global_version": (
            previous.manifest.get("global_version") if previous is not None else None
        ),
        "current_global_version": loaded.global_version,
    }


def _sample_failure_map(failures: list[Failure], all_sample_ids: set[str]) -> dict[str, list[Failure]]:
    result: dict[str, list[Failure]] = {}
    for failure in failures:
        if failure.target_kind == "sample" and failure.target_id in all_sample_ids:
            result.setdefault(failure.target_id, []).append(failure)
    return result


def _build_stats(
    loaded: LoadedWorkspace,
    sample_results: list[dict[str, Any]],
    dedupe_payload: dict[str, Any],
    failures: list[Failure],
    consumed_points: int,
) -> dict[str, Any]:
    all_ids = {sample.sample_id for sample in loaded.all_samples}
    failure_map = _sample_failure_map(failures, all_ids)
    template_counts: dict[str, dict[str, int]] = {}

    def ensure(template_id: str) -> dict[str, int]:
        return template_counts.setdefault(
            template_id,
            {
                "input_samples": 0,
                "rendered_samples": 0,
                "kept_samples": 0,
                "removed_duplicates": 0,
                "failed_samples": 0,
                "pending_samples": 0,
                "quota_stopped_samples": 0,
            },
        )

    for template_id in loaded.templates.template_ids:
        ensure(template_id)
    all_sample_lookup = {sample.sample_id: sample for sample in loaded.all_samples}
    for sample in loaded.all_samples:
        ensure(sample.template_id)["input_samples"] += 1
    result_lookup = {item["sample_id"]: item for item in sample_results}
    for sample_id, sample in sorted(all_sample_lookup.items()):
        counts = ensure(sample.template_id)
        item = result_lookup.get(sample_id)
        if item is not None:
            if item["status"] == "ok":
                counts["rendered_samples"] += 1
                if item.get("dedupe_status") == "kept":
                    counts["kept_samples"] += 1
                elif item.get("dedupe_status") == "removed":
                    counts["removed_duplicates"] += 1
            elif item["status"] == "failed":
                if "instruction_text" in item:
                    counts["rendered_samples"] += 1
                counts["failed_samples"] += 1
            elif item["status"] == "pending":
                counts["pending_samples"] += 1
            elif item["status"] == "quota_stopped":
                counts["quota_stopped_samples"] += 1
        elif sample_id in failure_map:
            counts["failed_samples"] += 1
    shard_counts: dict[str, dict[str, int]] = {}
    for sample in loaded.all_samples:
        counts = shard_counts.setdefault(
            sample.source_shard,
            {"input_samples": 0, "rendered_samples": 0, "removed_duplicates": 0},
        )
        counts["input_samples"] += 1
        item = result_lookup.get(sample.sample_id)
        if item is not None and item["status"] == "ok":
            counts["rendered_samples"] += 1
            if item.get("dedupe_status") == "removed":
                counts["removed_duplicates"] += 1

    code_counts: dict[str, int] = {}
    for failure in failures:
        code_counts[failure.code] = code_counts.get(failure.code, 0) + 1
    sample_references = [
        {
            "sample_id": item["sample_id"],
            "status": item["status"],
            "render_cache_key": item.get("render_cache_key"),
            "render_digest": item.get("render_digest"),
            "score_cache_key": item.get("score_cache_key"),
            "quality_score": item.get("quality_score"),
            "dedupe_status": item.get("dedupe_status"),
            "cluster_id": item.get("cluster_id"),
        }
        for item in sorted(sample_results, key=lambda value: value["sample_id"])
    ]
    input_sample_references = [
        {
            "sample_id": sample.sample_id,
            "template_id": sample.template_id,
            "template_revision": sample.template_revision,
            "answer_template_id": sample.answer_template_id,
            "answer_template_revision": sample.answer_template_revision,
            "source_shard": sample.source_shard,
            "batch_id": sample.batch_id,
        }
        for sample in sorted(loaded.all_samples, key=lambda value: value.sample_id)
    ]
    return {
        "per_template": {key: template_counts[key] for key in sorted(template_counts)},
        "per_shard": {key: shard_counts[key] for key in sorted(shard_counts)},
        "total": {
            "input_samples": len(loaded.all_samples),
            "rendered_samples": sum(
                1 for item in sample_results if item["status"] in {"ok", "failed"} and "instruction_text" in item
            ),
            "kept_samples": sum(
                1 for item in sample_results if item.get("dedupe_status") == "kept"
            ),
            "removed_duplicates": dedupe_payload.get("removed_count", 0),
            "failed_samples": len(
                [item for item in sample_results if item["status"] == "failed"]
            )
            + len(
                [
                    failure
                    for failure in failures
                    if failure.target_kind == "sample"
                    and failure.target_id in all_ids
                    and failure.target_id not in result_lookup
                ]
            ),
            "pending_samples": sum(
                1 for item in sample_results if item["status"] == "pending"
            ),
            "quota_stopped_samples": sum(
                1 for item in sample_results if item["status"] == "quota_stopped"
            ),
            "render_points_consumed": consumed_points,
            "clusters": len(dedupe_payload.get("clusters", [])),
            "boundary_pairs": len(dedupe_payload.get("boundary_pairs", [])),
        },
        "code_counts": {key: code_counts[key] for key in sorted(code_counts)},
        "sample_references": sample_references,
        "input_sample_references": input_sample_references,
        "sample_set": sorted(all_ids),
        "rendered_sample_set": sorted(
            item["sample_id"]
            for item in sample_results
            if item["status"] in {"ok", "failed"} and "instruction_text" in item
        ),
    }


class BuildEngine:
    def __init__(
        self,
        root: Path,
        *,
        state_dir: Path | None = None,
        full: bool = False,
        crash_after_stage: str | None = None,
        crash_mode: str = "raise",
    ) -> None:
        self.root = Path(root)
        self.state_dir = Path(state_dir) if state_dir is not None else self.root / "state"
        self.full = full
        self.crash_after_stage = crash_after_stage
        self.crash_mode = crash_mode

    def build(self) -> BuildResult:
        started = time.perf_counter()
        loaded = load_workspace(self.root)
        store = SnapshotStore(self.state_dir)
        snapshot_load = store.load_current(
            expected_global_version=loaded.global_version,
            expected_config_hash=loaded.config_hash,
        )
        previous = None if self.full else snapshot_load.snapshot
        recovery = snapshot_load.decision.as_dict()
        if self.full:
            recovery = {
                "action": "full_rebuild",
                "reason": "full build requested; persisted caches intentionally ignored",
                "previous_generation": None
                if snapshot_load.snapshot is None
                else snapshot_load.snapshot.generation,
                "previous_global_version": None
                if snapshot_load.snapshot is None
                else snapshot_load.snapshot.manifest.get("global_version"),
            }
        execution: list[dict[str, Any]] = []
        invalidation = _invalidation_plan(loaded, previous, recovery)
        render_cache: dict[str, Any] = (
            {} if previous is None else copy.deepcopy(previous.render_cache)
        )
        score_cache: dict[str, Any] = (
            {} if previous is None else copy.deepcopy(previous.score_cache)
        )
        template_cache: dict[str, Any] = (
            {} if previous is None else copy.deepcopy(previous.template_cache)
        )
        prior_pending = [] if previous is None else previous.pending_queue

        template_products: list[dict[str, Any]] = []
        for effective in loaded.templates.effective_templates():
            key = template_cache_key(effective)
            cached = _cache_lookup(template_cache, key)
            cache_valid = (
                cached is not None
                and cached.get("effective_fingerprint") == effective.effective_fingerprint
                and cached.get("chain_fingerprint") == effective.chain_fingerprint
            )
            action = "reuse" if cache_valid else "recompute"
            basis = (
                f"persisted:{previous.generation}:template:{key}"
                if cache_valid and previous is not None
                else "content_key_miss"
            )
            if not cache_valid:
                template_cache[key] = _cache_entry(key, effective.as_dict())
            execution.append(
                {
                    "stage": "template_resolution",
                    "entity_id": f"{effective.template_id}@{effective.revision}",
                    "cache_key": key,
                    "action": action,
                    "basis": basis,
                }
            )
            template_products.append(
                {
                    "template_id": effective.template_id,
                    "revision": effective.revision,
                    "parent_revision": (
                        effective.revision_path[-2]
                        if len(effective.revision_path) > 1
                        else None
                    ),
                    "revision_path": list(effective.revision_path),
                    "instruction_template": effective.instruction_template,
                    "answer_template": effective.answer_template,
                    "cache_key": key,
                    "effective_fingerprint": effective.effective_fingerprint,
                    "chain_fingerprint": effective.chain_fingerprint,
                    "slots": [slot.as_dict() for slot in effective.slots],
                }
            )

        sample_lookup = {sample.sample.sample_id: sample for sample in loaded.valid_samples}
        schedule = schedule_samples(
            [(sample.sample.source_shard, sample.sample.sample_id) for sample in loaded.valid_samples],
            loaded.config,
            prior_pending,
        )
        probe = ConcurrencyProbe()

        def worker(task: RenderTask) -> WorkOutcome:
            sample = sample_lookup[task.sample_id]
            base = _sample_base_result(sample, "ok")
            render_key = base["render_cache_key"]
            cached_render = _cache_lookup(render_cache, render_key)
            render_product = (
                None
                if cached_render is None
                else _render_product_from_payload(cached_render, render_key, sample)
            )
            if render_product is not None:
                render_action = "reuse"
                render_basis = (
                    f"persisted:{previous.generation}:render:{render_key}"
                    if previous is not None
                    else "cache_present"
                )
            else:
                render_product = _render_sample(sample)
                render_action = "recompute"
                render_basis = f"render_key_miss:{render_key}"
            base.update(render_product.as_dict())

            score_key = score_key_for(render_product.render_digest, loaded.config)
            base["score_cache_key"] = score_key
            if sample.sample.quality_override == "NaN":
                failure = ScoreFailure(
                    "quality computation produced a non-finite value",
                    injection_id=(
                        sample.sample.injection_ids[0]
                        if sample.sample.injection_ids
                        else None
                    ),
                ).as_failure(sample.sample.sample_id)
                base["status"] = "failed"
                base["failure_codes"] = [failure.code]
                return WorkOutcome(
                    task=task,
                    sample_result=base,
                    render_product=render_product,
                    failure=failure,
                    render_action=render_action,
                    score_action="recompute",
                    render_basis=render_basis,
                    score_basis="nonfinite_injection",
                )
            cached_score = _cache_lookup(score_cache, score_key)
            if cached_score is not None:
                try:
                    if (
                        cached_score.get("sample_id") != sample.sample.sample_id
                        or cached_score.get("render_digest") != render_product.render_digest
                        or cached_score.get("score_cache_key") != score_key
                    ):
                        raise ValueError("cached score provenance mismatch")
                    quality = QualityBreakdown(
                        raw_score=int(cached_score["quality"]["raw_score"]),
                        quantized_score=int(cached_score["quality"]["quantized_score"]),
                        quantum_step=int(cached_score["quality"]["quantum_step"]),
                        residual=int(cached_score["quality"]["residual"]),
                        components={
                            str(key): int(value)
                            for key, value in cached_score["quality"]["components"].items()
                        },
                        decimal_places=int(cached_score["quality"]["decimal_places"]),
                        quantization=str(cached_score["quality"]["quantization"]),
                    )
                    if (
                        quality.quantum_step != loaded.config.quantum_step
                        or quality.decimal_places != loaded.config.decimal_places
                        or quality.quantization != loaded.config.quantization
                        or quality.residual != quality.raw_score - quality.quantized_score
                    ):
                        raise ValueError("cached score quality mismatch")
                    score_product = ScoreProduct(
                        sample_id=sample.sample.sample_id,
                        render_digest=render_product.render_digest,
                        score_cache_key=score_key,
                        quality=quality,
                    )
                    score_action = "reuse"
                    score_basis = (
                        f"persisted:{previous.generation}:score:{score_key}"
                        if previous is not None
                        else "cache_present"
                    )
                except (KeyError, TypeError, ValueError):
                    cached_score = None
            if cached_score is None:
                try:
                    quality = compute_quality(sample, render_product.answer_text, loaded.config)
                except ScoreFailure as exc:
                    failure = exc.as_failure(sample.sample.sample_id)
                    base["status"] = "failed"
                    base["failure_codes"] = [failure.code]
                    return WorkOutcome(
                        task=task,
                        sample_result=base,
                        render_product=render_product,
                        failure=failure,
                        render_action=render_action,
                        score_action="recompute",
                        render_basis=render_basis,
                        score_basis="score_failure",
                    )
                score_product = ScoreProduct(
                    sample_id=sample.sample.sample_id,
                    render_digest=render_product.render_digest,
                    score_cache_key=score_key,
                    quality=quality,
                )
                score_action = "recompute"
                score_basis = f"score_key_miss:{score_key}"
            base["quality"] = _quality_document(score_product.quality)
            base["quality_score"] = score_product.quality.quantized_score
            base["quality_residual"] = score_product.quality.residual
            base["quantum_step"] = score_product.quality.quantum_step
            base["quantization"] = score_product.quality.quantization
            return WorkOutcome(
                task=task,
                sample_result=base,
                render_product=render_product,
                score_product=score_product,
                render_action=render_action,
                score_action=score_action,
                render_basis=render_basis,
                score_basis=score_basis,
            )

        outcomes = run_thread_pool(
            schedule.scheduled,
            worker,
            loaded.config.max_workers,
            probe,
        )
        outcomes.sort(key=lambda item: (item.task.source_shard, item.task.sample_id))
        sample_results: dict[str, dict[str, Any]] = {}
        runtime_failures: list[Failure] = []
        physical_points = 0
        for outcome in outcomes:
            sample_results[outcome.task.sample_id] = outcome.sample_result
            execution.append(
                {
                    "stage": "render",
                    "entity_id": outcome.task.sample_id,
                    "cache_key": outcome.sample_result.get("render_cache_key"),
                    "action": outcome.render_action,
                    "basis": outcome.render_basis,
                    "sort_key": outcome.task.sort_key,
                }
            )
            execution.append(
                {
                    "stage": "score",
                    "entity_id": outcome.task.sample_id,
                    "cache_key": outcome.sample_result.get("score_cache_key"),
                    "action": outcome.score_action,
                    "basis": outcome.score_basis,
                    "sort_key": outcome.task.sort_key,
                }
            )
            if outcome.render_product is not None:
                render_cache.setdefault(
                    outcome.render_product.render_cache_key,
                    _cache_entry(
                        outcome.render_product.render_cache_key,
                        outcome.render_product.as_dict(),
                    ),
                )
                if outcome.render_action == "recompute":
                    physical_points += outcome.task.points
            if outcome.score_product is not None:
                score_cache.setdefault(
                    outcome.score_product.score_cache_key,
                    _cache_entry(
                        outcome.score_product.score_cache_key,
                        outcome.score_product.as_dict(),
                    ),
                )
            if outcome.failure is not None:
                runtime_failures.append(outcome.failure)

        for failure in schedule.quota_failures:
            sample = sample_lookup[failure.target_id]
            item = _sample_base_result(sample, "quota_stopped")
            item["failure_codes"] = [failure.code]
            sample_results[failure.target_id] = item
            runtime_failures.append(failure)
        for pending in schedule.pending:
            sample = sample_lookup[pending.sample_id]
            item = _sample_base_result(sample, "pending")
            item["pending_code"] = pending.code
            item["task_sort_key"] = pending.sort_key
            sample_results[pending.sample_id] = item

        completed = [
            sample_results[sample.sample.sample_id]
            for sample in loaded.valid_samples
            if sample.sample.sample_id in sample_results
            and sample_results[sample.sample.sample_id]["status"] == "ok"
        ]
        dedupe_items: list[DedupeInput] = []
        for item in sorted(completed, key=lambda value: value["sample_id"]):
            tokens = subtoken_set(
                item["instruction_text"] + "\n" + item["answer_text"],
                loaded.config.subword_size,
            )
            dedupe_items.append(
                DedupeInput(
                    sample_id=item["sample_id"],
                    quality_score=int(item["quality_score"]),
                    tokens=tokens,
                )
            )
        dedupe_refs = [
            {
                "sample_id": item.sample_id,
                "quality_score": item.quality_score,
                "subtoken_fingerprint": item.as_ref()["subtoken_fingerprint"],
                "render_digest": next(
                    value["render_digest"]
                    for value in completed
                    if value["sample_id"] == item.sample_id
                ),
            }
            for item in dedupe_items
        ]
        dedupe_key = digest(
            {
                "kind": "dedupe-cache-key",
                "schema_version": DEDUPE_SCHEMA_VERSION,
                "threshold_permille": loaded.config.dedupe_threshold_permille,
                "subword_size": loaded.config.subword_size,
                "items": dedupe_refs,
            }
        )
        prior_dedupe = previous.stages.get("dedupe") if previous is not None else None
        if (
            isinstance(prior_dedupe, dict)
            and prior_dedupe.get("cache_key") == dedupe_key
            and isinstance(prior_dedupe.get("product"), dict)
        ):
            dedupe_payload = copy.deepcopy(prior_dedupe["product"])
            dedupe_action = "reuse"
            dedupe_basis = f"persisted:{previous.generation}:dedupe:{dedupe_key}"
        else:
            dedupe_payload = build_dedupe(
                dedupe_items, loaded.config.dedupe_threshold_permille
            ).as_dict()
            dedupe_action = "recompute"
            dedupe_basis = f"dedupe_key_miss:{dedupe_key}"
        execution.append(
            {
                "stage": "dedupe",
                "entity_id": "global",
                "cache_key": dedupe_key,
                "action": dedupe_action,
                "basis": dedupe_basis,
            }
        )
        for sample_id in dedupe_payload.get("sample_clusters", {}):
            if sample_id not in sample_results:
                continue
            item = sample_results[sample_id]
            item["cluster_id"] = dedupe_payload["sample_clusters"][sample_id]
            item["keeper_id"] = dedupe_payload["sample_keepers"].get(sample_id)
            item["dedupe_status"] = dedupe_payload["sample_status"].get(sample_id)
            item["similarity_to_keeper_permille"] = dedupe_payload[
                "similarity_to_keeper"
            ].get(sample_id)

        all_failures = sorted(
            [*loaded.failures, *runtime_failures], key=Failure.sort_key
        )
        ordered_sample_results = [
            sample_results[key]
            for key in sorted(
                sample_results,
                key=lambda sample_id: (
                    sample_results[sample_id]["source_shard"],
                    sample_id,
                ),
            )
        ]
        stats = _build_stats(
            loaded,
            ordered_sample_results,
            dedupe_payload,
            all_failures,
            schedule.consumed_points,
        )
        stats_refs = stats["sample_references"]
        stats_key = digest(
            {
                "kind": "stats-cache-key",
                "schema_version": STATS_SCHEMA_VERSION,
                "dedupe_cache_key": dedupe_key,
                "config_hash": loaded.config_hash,
                "sample_references": stats_refs,
                "input_sample_references": stats.get("input_sample_references", []),
                "failure_refs": sorted_failure_dicts(all_failures),
                "consumed_points": schedule.consumed_points,
            }
        )
        prior_stats = previous.stages.get("stats") if previous is not None else None
        if (
            isinstance(prior_stats, dict)
            and prior_stats.get("cache_key") == stats_key
            and isinstance(prior_stats.get("product"), dict)
        ):
            stats = copy.deepcopy(prior_stats["product"])
            stats_action = "reuse"
            stats_basis = f"persisted:{previous.generation}:stats:{stats_key}"
        else:
            stats_action = "recompute"
            stats_basis = f"stats_key_miss:{stats_key}"
        execution.append(
            {
                "stage": "stats",
                "entity_id": "global",
                "cache_key": stats_key,
                "action": stats_action,
                "basis": stats_basis,
            }
        )

        injection_hits = self._injection_hits(loaded.injection_records, all_failures)
        failures_payload = sorted_failure_dicts(all_failures)
        pending_payload = [item.as_dict() for item in schedule.pending]
        stage_payloads = {
            "load_validation": {
                "templates_fingerprint": loaded.validation_products.templates_fingerprint,
                "samples_fingerprint": loaded.validation_products.samples_fingerprint,
                "config_hash": loaded.validation_products.config_hash,
                "global_version": loaded.validation_products.global_version,
                "valid_sample_count": len(loaded.valid_samples),
                "all_sample_count": len(loaded.all_samples),
                "failures": failures_payload,
                "injection_hits": injection_hits,
            },
            "template_resolution": {
                "template_chain_fingerprint": loaded.template_chain_fingerprint,
                "revision_fingerprints": loaded.revision_fingerprints,
                "templates": template_products,
            },
            "schedule": schedule.as_dict(),
            "render": {
                "render_cache_keys": {
                    item["sample_id"]: item.get("render_cache_key")
                    for item in ordered_sample_results
                },
                "physical_render_points_recomputed": physical_points,
            },
            "score": {
                "quality_score_scale": 1_000_000,
                "quantization": loaded.config.quantization,
                "quantum_step": loaded.config.quantum_step,
                "decimal_places": loaded.config.decimal_places,
                "scores": {
                    item["sample_id"]: {
                        "quality_score": item.get("quality_score"),
                        "quality_residual": item.get("quality_residual"),
                        "score_cache_key": item.get("score_cache_key"),
                    }
                    for item in ordered_sample_results
                },
            },
            "dedupe": {
                "cache_key": dedupe_key,
                "product": dedupe_payload,
            },
            "stats": {
                "cache_key": stats_key,
                "product": stats,
            },
        }
        quick_view = {
            "sample_count": stats["total"]["input_samples"],
            "rendered_count": stats["total"]["rendered_samples"],
            "removed_duplicates": stats["total"]["removed_duplicates"],
            "render_points_consumed": stats["total"]["render_points_consumed"],
            "clusters": stats["total"]["clusters"],
            "code_counts": stats["code_counts"],
        }
        completed_template_cursor = {
            "templates": {
                template_id: sorted(
                    revision
                    for (candidate_id, revision) in loaded.templates.valid_revisions
                    if candidate_id == template_id
                )
                for template_id in loaded.templates.template_ids
            },
            "template_chain_fingerprint": loaded.template_chain_fingerprint,
        }
        result: dict[str, Any] = {
            "schema_version": 1,
            "global_version": loaded.global_version,
            "config_hash": loaded.config_hash,
            "template_chain_fingerprint": loaded.template_chain_fingerprint,
            "templates": template_products,
            "samples": ordered_sample_results,
            "clusters": dedupe_payload.get("clusters", []),
            "dedupe": dedupe_payload,
            "failures": failures_payload,
            "pending_queue": pending_payload,
            "injection_hits": injection_hits,
            "stats": stats,
            "stats_cache_key": stats_key,
            "dedupe_cache_key": dedupe_key,
            "quick_view": quick_view,
            "execution": {
                "recovery": recovery,
                "invalidation": invalidation,
                "cache_ledger": execution,
                "prior_pending_count": len(prior_pending),
                "scheduled_count": len(schedule.scheduled),
                "quota_failure_count": len(schedule.quota_failures),
                "budget_remaining": schedule.budget_remaining,
                "physical_render_points_recomputed": physical_points,
            },
        }
        canonical = canonical_view(result)
        canonical_digest = digest(canonical)
        result["canonical_digest"] = canonical_digest
        generation = store.commit(
            global_version=loaded.global_version,
            config_hash=loaded.config_hash,
            template_chain_fingerprint=loaded.template_chain_fingerprint,
            stages=stage_payloads,
            render_cache=render_cache,
            score_cache=score_cache,
            template_cache=template_cache,
            pending_queue=pending_payload,
            completed_template_cursor=completed_template_cursor,
            consumed_budget=schedule.consumed_points,
            result=result,
            input_fingerprints={
                "templates": loaded.revision_fingerprints,
                "samples": loaded.sample_fingerprints,
                "global_version": loaded.global_version,
                "config_hash": loaded.config_hash,
            },
            previous_generation=(
                None if snapshot_load.snapshot is None else snapshot_load.snapshot.generation
            ),
            crash_after_stage=self.crash_after_stage,
            crash_mode=self.crash_mode,
        )
        result["snapshot_generation"] = generation
        return BuildResult(
            root=self.root,
            result=result,
            canonical=canonical,
            canonical_digest=canonical_digest,
            runtime_seconds=time.perf_counter() - started,
            snapshot_generation=generation,
            recovery=recovery,
            probe_max_concurrency=probe.max_active,
        )

    @staticmethod
    def _injection_hits(
        records: list[InjectionRecord], failures: list[Failure]
    ) -> list[dict[str, Any]]:
        result: list[dict[str, Any]] = []
        for record in records:
            matching = [
                failure for failure in failures if failure.injection_id == record.injection_id
            ]
            result.append(
                {
                    "injection_id": record.injection_id,
                    "type": record.injection_type,
                    "target_kind": record.target_kind,
                    "target_id": record.target_id,
                    "expected_stage": record.expected_stage,
                    "applied": record.applied,
                    "hit": bool(matching),
                    "observed_stages": sorted({failure.stage for failure in matching}),
                    "observed_codes": sorted({failure.code for failure in matching}),
                    "failure_messages": [failure.message for failure in matching],
                }
            )
        return sorted(result, key=lambda item: (item["injection_id"], item["target_id"]))


def canonical_view(result: Mapping[str, Any]) -> dict[str, Any]:
    """Return the deterministic projection used for incremental/full equality.

    Execution provenance (cache hit bases, recovery action, wall-clock duration and
    snapshot generation) is intentionally excluded. All construction conclusions,
    cache keys, output ordering, failures and statistics remain included.
    """

    excluded = {
        "execution",
        "canonical_digest",
        "snapshot_generation",
        "report",
    }
    return {
        key: copy.deepcopy(value)
        for key, value in sorted(result.items())
        if key not in excluded
    }


def build_workspace(
    root: Path,
    *,
    state_dir: Path | None = None,
    full: bool = False,
    crash_after_stage: str | None = None,
    crash_mode: str = "raise",
) -> BuildResult:
    return BuildEngine(
        root,
        state_dir=state_dir,
        full=full,
        crash_after_stage=crash_after_stage,
        crash_mode=crash_mode,
    ).build()
