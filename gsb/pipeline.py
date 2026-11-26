"""End-to-end deterministic pipeline and invalidation ledger."""

from __future__ import annotations

import io
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from PIL import Image

from gsb.config import RunConfig, load_config
from gsb.constants import (
    E_POINT_BUDGET_PENDING,
    E_SAMPLE_FORMAT,
    E_TEMPLATE_VARIANT_DIMENSION,
    INVALIDATION_COMPUTED,
    INVALIDATION_INVALIDATED,
    INVALIDATION_REUSED,
)
from gsb.errors import ValidationIssue, sort_issues
from gsb.fonts import TemplateLibrary, load_templates
from gsb.keypoints import KeypointResult, regress_keypoints
from gsb.localize import Candidate, Component, connected_components, sliding_candidates
from gsb.models import DatasetState, SampleRecord
from gsb.reading import ReadingEntry, ReadingResult, assemble_reading_order
from gsb.recognizer import (
    RecognitionPlan,
    RecognitionResult,
    run_budgeted_recognition,
)
from gsb.samples import load_dataset
from gsb.util import sha256_obj


@dataclass
class SampleCacheMeta:
    image_fingerprint: str = ""
    candidate_set_key: str = ""
    component_set_key: str = ""
    config_hash: str = ""
    template_fingerprint: str = ""
    quantization: str = ""
    annotation_version: str | None = None
    annotation_fingerprint: str | None = None
    keypoint_ids: tuple[str, ...] = ()
    keypoint_cache_by_id: dict[str, str] = field(default_factory=dict)
    reading_order_cache_key: str = ""


@dataclass
class PipelineCache:
    candidate_sets: dict[str, tuple[Candidate, ...]] = field(default_factory=dict)
    component_sets: dict[str, tuple[Component, ...]] = field(default_factory=dict)
    keypoints: dict[str, KeypointResult] = field(default_factory=dict)
    recognitions: dict[str, RecognitionResult] = field(default_factory=dict)
    reading_entries: dict[str, ReadingEntry] = field(default_factory=dict)
    sample_meta: dict[str, SampleCacheMeta] = field(default_factory=dict)


@dataclass(frozen=True)
class PipelineResult:
    report: dict[str, Any]
    cache: PipelineCache
    recognition_results: dict[str, RecognitionResult]
    pending_ids: tuple[str, ...]


def _candidate_set_key(
    record: SampleRecord,
    *,
    config: RunConfig,
    template: TemplateLibrary,
) -> str:
    return sha256_obj(
        {
            "stage": "candidate_set",
            "sample": record.sample_id,
            "image": record.image_fingerprint,
            "parameters": config.stage_parameter_fingerprint("candidate"),
            "template_fingerprint": template.content_fingerprint,
            "template_version": template.template_version,
            "quantization": config.quantization.level,
            "window": config.to_dict()["window"],
        }
    )


def _component_set_key(
    record: SampleRecord,
    *,
    config: RunConfig,
    template: TemplateLibrary,
    candidate_set_key: str,
) -> str:
    return sha256_obj(
        {
            "stage": "component_set",
            "sample": record.sample_id,
            "image": record.image_fingerprint,
            "parameters": config.stage_parameter_fingerprint("component"),
            "template_fingerprint": template.content_fingerprint,
            "template_version": template.template_version,
            "quantization": config.quantization.level,
            "candidates": candidate_set_key,
        }
    )


def _invalidation_reason(
    previous: SampleCacheMeta | None,
    current: SampleCacheMeta,
) -> str:
    if previous is None:
        return "no_prior_stage_state"
    if previous.image_fingerprint != current.image_fingerprint:
        return "image_content_changed"
    if previous.template_fingerprint != current.template_fingerprint:
        return "template_content_changed"
    if previous.quantization != current.quantization:
        return "quantization_changed"
    if previous.config_hash != current.config_hash:
        return "parameter_fingerprint_changed"
    return "dependency_cache_key_changed"


def _cache_record(
    *,
    artifact_type: str,
    artifact_id: str,
    cache_key: str,
    status: str,
    reason: str,
) -> dict[str, Any]:
    return {
        "artifact_type": artifact_type,
        "artifact_id": artifact_id,
        "cache_key": cache_key,
        "status": status,
        "reason": reason,
    }


def _sample_image(record: SampleRecord) -> np.ndarray:
    image = Image.open(io.BytesIO(record.image_bytes))
    image.load()
    return np.asarray(image.convert("L"), dtype=np.uint8)


def _build_sample_result(
    *,
    record: SampleRecord,
    config: RunConfig,
    template: TemplateLibrary,
    cache: PipelineCache,
    invalidation: list[dict[str, Any]],
    recognition_by_keypoint: dict[str, RecognitionResult],
    pending_by_keypoint: dict[str, dict[str, Any]],
) -> tuple[dict[str, Any], list[RecognitionPlan], list[KeypointResult]]:
    image = _sample_image(record)
    file_name = Path(record.manifest.image_path).name
    previous = cache.sample_meta.get(record.sample_id)
    current_meta = SampleCacheMeta(
        image_fingerprint=record.image_fingerprint,
        config_hash=config.config_hash,
        template_fingerprint=template.content_fingerprint,
        quantization=config.quantization.level,
        annotation_version=record.annotation.version if record.annotation is not None else None,
        annotation_fingerprint=record.annotation.content_fingerprint if record.annotation is not None else None,
    )
    annotation_cache_key = sha256_obj(
        {
            "stage": "annotation_version",
            "sample": record.sample_id,
            "version": current_meta.annotation_version,
            "content_fingerprint": current_meta.annotation_fingerprint,
        }
    )
    annotation_reused = (
        previous is not None
        and previous.annotation_version == current_meta.annotation_version
        and previous.annotation_fingerprint == current_meta.annotation_fingerprint
    )
    invalidation.append(
        _cache_record(
            artifact_type="annotation_version",
            artifact_id=record.sample_id,
            cache_key=annotation_cache_key,
            status=INVALIDATION_REUSED if annotation_reused else (
                INVALIDATION_INVALIDATED if previous is not None else INVALIDATION_COMPUTED
            ),
            reason="annotation_version_unchanged" if annotation_reused else (
                "annotation_version_changed" if previous is not None else "initial_annotation_version"
            ),
        )
    )
    candidate_key = _candidate_set_key(record, config=config, template=template)
    current_meta.candidate_set_key = candidate_key
    candidate_max_observed = 0
    if candidate_key in cache.candidate_sets:
        candidates = list(cache.candidate_sets[candidate_key])
        candidate_status = INVALIDATION_REUSED
        candidate_reason = "candidate_set_cache_hit"
    else:
        candidate_execution = sliding_candidates(
            image,
            config=config,
            image_fingerprint=record.image_fingerprint,
            parameter_fingerprint=config.stage_parameter_fingerprint("candidate"),
            template_fingerprint=template.content_fingerprint,
            file_name=file_name,
        )
        candidates = list(candidate_execution.results)
        candidate_max_observed = candidate_execution.max_observed
        cache.candidate_sets[candidate_key] = tuple(candidates)
        candidate_status = (
            INVALIDATION_INVALIDATED if previous is not None else INVALIDATION_COMPUTED
        )
        candidate_reason = _invalidation_reason(previous, current_meta)
    for candidate in candidates:
        invalidation.append(
            _cache_record(
                artifact_type="candidate",
                artifact_id=candidate.candidate_id,
                cache_key=candidate.fingerprint,
                status=candidate_status,
                reason=candidate_reason,
            )
        )

    component_key = _component_set_key(
        record,
        config=config,
        template=template,
        candidate_set_key=candidate_key,
    )
    current_meta.component_set_key = component_key
    if component_key in cache.component_sets:
        components = list(cache.component_sets[component_key])
        component_status = INVALIDATION_REUSED
        component_reason = "component_set_cache_hit"
    else:
        components = connected_components(
            image,
            candidates=candidates,
            config=config,
            image_fingerprint=record.image_fingerprint,
            parameter_fingerprint=config.stage_parameter_fingerprint("component"),
            file_name=file_name,
        )
        cache.component_sets[component_key] = tuple(components)
        component_status = (
            INVALIDATION_INVALIDATED if previous is not None else INVALIDATION_COMPUTED
        )
        component_reason = _invalidation_reason(previous, current_meta)
    for component in components:
        invalidation.append(
            _cache_record(
                artifact_type="component",
                artifact_id=component.component_id,
                cache_key=component.fingerprint,
                status=component_status,
                reason=component_reason,
            )
        )

    keypoints = regress_keypoints(
        image,
        components=components,
        manifest=record.manifest,
        annotation=record.annotation,
        config=config,
        image_fingerprint=record.image_fingerprint,
        parameter_fingerprint=config.stage_parameter_fingerprint("keypoint"),
        template_fingerprint=template.content_fingerprint,
        file_name=file_name,
        prior_by_cache=cache.keypoints,
    )
    current_meta.keypoint_ids = tuple(item.keypoint_id for item in keypoints)
    current_meta.keypoint_cache_by_id = {
        item.keypoint_id: item.cache_key for item in keypoints
    }
    for keypoint in keypoints:
        cache.keypoints[keypoint.cache_key] = keypoint
        if keypoint.reused:
            status = INVALIDATION_REUSED
            reason = "keypoint_cache_hit"
        else:
            status = (
                INVALIDATION_INVALIDATED if previous is not None else INVALIDATION_COMPUTED
            )
            if previous is not None and previous.image_fingerprint == record.image_fingerprint:
                old_key = previous.keypoint_cache_by_id.get(keypoint.keypoint_id)
                if old_key is not None and old_key != keypoint.cache_key:
                    reason = "keypoint_dependency_or_annotation_coordinate_changed"
                else:
                    reason = _invalidation_reason(previous, current_meta)
            else:
                reason = _invalidation_reason(previous, current_meta)
        invalidation.append(
            _cache_record(
                artifact_type="keypoint",
                artifact_id=keypoint.keypoint_id,
                cache_key=keypoint.cache_key,
                status=status,
                reason=reason,
            )
        )

    reading = assemble_reading_order(
        keypoints,
        columns=record.manifest.columns,
        config=config,
        image_fingerprint=record.image_fingerprint,
        parameter_fingerprint=config.stage_parameter_fingerprint("reading"),
        template_fingerprint=template.content_fingerprint,
        file_name=file_name,
    )
    order_reused = (
        previous is not None
        and previous.reading_order_cache_key == reading.order_cache_key
    )
    invalidation.append(
        _cache_record(
            artifact_type="reading_order",
            artifact_id=record.sample_id,
            cache_key=reading.order_cache_key,
            status=INVALIDATION_REUSED if order_reused else (
                INVALIDATION_INVALIDATED if previous is not None else INVALIDATION_COMPUTED
            ),
            reason="reading_order_cache_hit" if order_reused else (
                "reading_order_dependency_changed" if previous is not None else "initial_reading_order"
            ),
        )
    )
    prior_reading_ids = {
        entry.keypoint_id: entry for entry in cache.reading_entries.values()
        if entry.keypoint_id.startswith(record.sample_id)
    }
    for entry in reading.entries:
        prior_entry = prior_reading_ids.get(entry.keypoint_id)
        reused = prior_entry is not None and prior_entry.cache_key == entry.cache_key
        cache.reading_entries[entry.cache_key] = entry
        invalidation.append(
            _cache_record(
                artifact_type="reading",
                artifact_id=f"{record.sample_id}:{entry.position}:{entry.keypoint_id}",
                cache_key=entry.cache_key,
                status=INVALIDATION_REUSED if reused else (
                    INVALIDATION_INVALIDATED if previous is not None else INVALIDATION_COMPUTED
                ),
                reason="reading_entry_cache_hit" if reused else _invalidation_reason(previous, current_meta),
            )
        )
    cache.sample_meta[record.sample_id] = current_meta
    current_meta.reading_order_cache_key = reading.order_cache_key

    plans: list[RecognitionPlan] = []
    recognition_rows: list[dict[str, Any]] = []
    for keypoint in keypoints:
        plan = RecognitionPlan(
            sample_id=record.sample_id,
            file_name=file_name,
            image=image,
            keypoint=keypoint,
            template_library=template,
        )
        plans.append(plan)
        recognition = recognition_by_keypoint.get(keypoint.keypoint_id)
        if recognition is not None:
            recognition_rows.append(recognition.to_dict())
        elif keypoint.keypoint_id in pending_by_keypoint:
            recognition_rows.append(
                {
                    "sample_id": record.sample_id,
                    "keypoint_id": keypoint.keypoint_id,
                    "char": None,
                    "score": None,
                    "score_ppm": None,
                    "confidence": None,
                    "cache_key": plan.cache_key,
                    "status": E_POINT_BUDGET_PENDING,
                    "point_cost": plan.point_cost,
                    "sort_key": plan.sort_key,
                }
            )
        else:
            recognition_rows.append(
                {
                    "sample_id": record.sample_id,
                    "keypoint_id": keypoint.keypoint_id,
                    "char": None,
                    "score": None,
                    "score_ppm": None,
                    "confidence": None,
                    "cache_key": plan.cache_key,
                    "status": "E_RECOGNITION_NOT_RUN",
                    "point_cost": plan.point_cost,
                    "sort_key": plan.sort_key,
                }
            )
    recognition_rows.sort(key=lambda item: item["sort_key"])
    sample_result = {
        "sample_id": record.sample_id,
        "status": "ok",
        "image_fingerprint": record.image_fingerprint,
        "annotation_version": record.annotation.version if record.annotation is not None else None,
        "candidate_set_key": candidate_key,
        "component_set_key": component_key,
        "candidates": [candidate.to_dict() for candidate in candidates],
        "components": [component.to_dict() for component in components],
        "keypoints": [keypoint.to_dict() for keypoint in keypoints],
        "recognitions": recognition_rows,
        "reading_order": [entry.to_dict() for entry in reading.entries],
        "reading_order_cache_key": reading.order_cache_key,
        "conflicts": [conflict.to_dict() for conflict in reading.conflicts],
        "invalidation": invalidation,
        "concurrency": {
            "candidate_max_observed": candidate_max_observed,
            "candidate_worker_limit": config.recognition.max_workers,
        },
    }
    return sample_result, plans, keypoints


def run_pipeline(
    dataset: DatasetState,
    *,
    config: RunConfig,
    templates: TemplateLibrary,
    initial_issues: Iterable[ValidationIssue] = (),
    cache: PipelineCache | None = None,
    prior_recognition_results: dict[str, RecognitionResult] | None = None,
    pending_ids: Iterable[str] = (),
    prior_pending: Iterable[dict[str, Any]] = (),
) -> PipelineResult:
    cache = cache or PipelineCache()
    pending_id_set = set(pending_ids)
    sample_results: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    all_plans: list[RecognitionPlan] = []
    all_keypoints: list[KeypointResult] = []
    invalidation: list[dict[str, Any]] = []
    valid_sample_ids: set[str] = set()
    for sample_id in dataset.sample_order:
        record = dataset.samples[sample_id]
        sample_invalidation: list[dict[str, Any]] = []
        sample_recognition: dict[str, RecognitionResult] = {}
        if prior_recognition_results:
            for keypoint in cache.keypoints.values():
                if keypoint.keypoint_id.startswith(sample_id + "-") and keypoint.keypoint_id in prior_recognition_results:
                    candidate_result = prior_recognition_results[keypoint.keypoint_id]
                    plan_key = sha256_obj(
                        {
                            "stage": "recognition",
                            "sample": sample_id,
                            "keypoint_cache": keypoint.cache_key,
                            "template_fingerprint": templates.content_fingerprint,
                            "template_version": templates.template_version,
                        }
                    )
                    if candidate_result.cache_key == plan_key:
                        sample_recognition[keypoint.keypoint_id] = candidate_result
        sample_result, plans, keypoints = _build_sample_result(
            record=record,
            config=config,
            template=templates,
            cache=cache,
            invalidation=sample_invalidation,
            recognition_by_keypoint=sample_recognition,
            pending_by_keypoint={},
        )
        sample_result["invalidation"] = sample_invalidation
        sample_results.append(sample_result)
        all_plans.extend(plans)
        all_keypoints.extend(keypoints)
        valid_sample_ids.add(sample_id)
        invalidation.extend(sample_invalidation)

    prior_by_keypoint: dict[str, RecognitionResult] = {}
    if prior_recognition_results:
        for plan in all_plans:
            result = prior_recognition_results.get(plan.keypoint.keypoint_id)
            if result is not None and result.cache_key == plan.cache_key:
                prior_by_keypoint[plan.keypoint.keypoint_id] = result
    recognition_batch = run_budgeted_recognition(
        all_plans,
        config=config,
        prior_results=prior_by_keypoint,
        pending_ids=pending_id_set,
    )
    recognition_by_keypoint = {
        result.keypoint_id: result for result in recognition_batch.results
    }
    pending_by_keypoint = {
        item["keypoint_id"]: item for item in recognition_batch.budget.pending
    }
    for result in recognition_batch.results:
        cache.recognitions[result.cache_key] = result
    sample_id_by_keypoint = {
        keypoint.keypoint_id: sample_id
        for sample_id in valid_sample_ids
        for keypoint in cache.keypoints.values()
        if keypoint.keypoint_id.startswith(sample_id + "-")
    }
    for sample_result in sample_results:
        sample_id = sample_result["sample_id"]
        recognition_rows: list[dict[str, Any]] = []
        for keypoint in sample_result["keypoints"]:
            keypoint_id = keypoint["keypoint_id"]
            result = recognition_by_keypoint.get(keypoint_id)
            if result is not None:
                recognition_rows.append(result.to_dict())
                prior_entry = next(
                    (
                        item for item in sample_result["invalidation"]
                        if item["artifact_type"] == "keypoint"
                        and item["artifact_id"] == keypoint_id
                    ),
                    None,
                )
                if not result.reused:
                    prior_recognition = prior_by_keypoint.get(keypoint_id)
                    status = (
                        INVALIDATION_COMPUTED
                        if prior_entry is not None
                        and prior_entry["status"] == INVALIDATION_COMPUTED
                        and prior_recognition is None
                        else INVALIDATION_INVALIDATED
                    )
                else:
                    status = INVALIDATION_REUSED
                invalidation.append(
                    _cache_record(
                        artifact_type="recognition",
                        artifact_id=result.char_id,
                        cache_key=result.cache_key,
                        status=status,
                        reason="recognition_cache_hit" if result.reused else "keypoint_or_template_dependency_changed",
                    )
                )
            else:
                pending = pending_by_keypoint.get(keypoint_id)
                plan_key = next(
                    (
                        plan.cache_key for plan in all_plans
                        if plan.keypoint.keypoint_id == keypoint_id
                    ),
                    "",
                )
                recognition_rows.append(
                    {
                        "sample_id": sample_id,
                        "keypoint_id": keypoint_id,
                        "char": None,
                        "score": None,
                        "score_ppm": None,
                        "confidence": None,
                        "cache_key": plan_key,
                        "status": E_POINT_BUDGET_PENDING,
                        "point_cost": pending["point_cost"] if pending else 0,
                        "sort_key": pending["sort_key"] if pending else "",
                    }
                )
        recognition_rows.sort(key=lambda item: item["sort_key"])
        sample_result["recognitions"] = recognition_rows

    template_failures = []
    issue_dicts = [issue.to_dict() for issue in sort_issues(list(initial_issues))]
    report = {
        "schema_version": 1,
        "config_hash": config.config_hash,
        "template_version": templates.template_version,
        "template_fingerprint": templates.content_fingerprint,
        "quantization": config.quantization.level,
        "fixed_bits": config.fixed_bits,
        "worker_limit": config.recognition.max_workers,
        "samples": sample_results,
        "failures": issue_dicts + failures,
        "template_failures": template_failures,
        "pending": list(recognition_batch.budget.pending),
        "budget": recognition_batch.budget.to_dict(),
        "invalidation": invalidation,
        "incremental": {
            "cache_hits": sum(1 for item in invalidation if item["status"] == INVALIDATION_REUSED),
            "invalidations": sum(1 for item in invalidation if item["status"] == INVALIDATION_INVALIDATED),
            "recognition_reuses": recognition_batch.budget.reused_count,
        },
    }
    return PipelineResult(
        report=report,
        cache=cache,
        recognition_results=recognition_by_keypoint,
        pending_ids=tuple(item["keypoint_id"] for item in recognition_batch.budget.pending),
    )


def load_templates_for_config(config: RunConfig, path: str | Path) -> tuple[TemplateLibrary | None, list[ValidationIssue]]:
    return load_templates(
        path,
        expected_version=config.recognition.template_version,
        expected_width=config.recognition.template_width,
        expected_height=config.recognition.template_height,
        expected_weights=config.recognition.variant_weights,
    )


def run_from_paths(
    *,
    config_path: str | Path,
    dataset_path: str | Path,
    templates_path: str | Path,
    cache: PipelineCache | None = None,
    prior_recognition_results: dict[str, RecognitionResult] | None = None,
    pending_ids: Iterable[str] = (),
) -> PipelineResult:
    config = load_config(config_path)
    templates, template_issues = load_templates_for_config(config, templates_path)
    dataset, dataset_issues = load_dataset(dataset_path, config=config)
    if templates is None:
        return PipelineResult(
            report={
                "schema_version": 1,
                "config_hash": config.config_hash,
                "samples": [],
                "failures": [issue.to_dict() for issue in template_issues],
                "template_failures": [issue.to_dict() for issue in template_issues],
                "pending": [],
                "invalidation": [],
            },
            cache=cache or PipelineCache(),
            recognition_results={},
            pending_ids=(),
        )
    return run_pipeline(
        dataset,
        config=config,
        templates=templates,
        initial_issues=dataset_issues,
        cache=cache,
        prior_recognition_results=prior_recognition_results,
        pending_ids=pending_ids,
    )
