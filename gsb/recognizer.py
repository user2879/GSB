"""Multi-template fixed-point recognition, confidence and point budget."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from PIL import Image

from gsb.concurrency import OrderedTask, StageExecution, execute_ordered
from gsb.config import RunConfig
from gsb.constants import E_POINT_BUDGET_PENDING
from gsb.fonts import GlyphTemplate, TemplateLibrary
from gsb.keypoints import KeypointResult
from gsb.util import ratio_fixed, round_half_up, sha256_obj, task_sort_key


@dataclass(frozen=True)
class RecognitionPlan:
    sample_id: str
    file_name: str
    image: np.ndarray
    keypoint: KeypointResult
    template_library: TemplateLibrary
    priority: int = 1

    @property
    def cache_key(self) -> str:
        return sha256_obj(
            {
                "stage": "recognition",
                "sample": self.sample_id,
                "keypoint_cache": self.keypoint.cache_key,
                "template_fingerprint": self.template_library.content_fingerprint,
                "template_version": self.template_library.template_version,
            }
        )

    @property
    def point_cost(self) -> int:
        return self.keypoint.bbox.area * len(self.template_library.weights)

    @property
    def sort_key(self) -> str:
        return task_sort_key(
            Path(self.file_name).name,
            "recognition",
            self.keypoint.column_truth,
            self.keypoint.reading_index_truth,
            self.keypoint.bbox.top * 10000 + self.keypoint.bbox.left,
        )


@dataclass(frozen=True)
class RecognitionResult:
    char_id: str
    sample_id: str
    keypoint_id: str
    candidate_id: str
    component_id: str
    char: str
    code_point: int
    template_id: str
    score: int
    score_ppm: int
    confidence: int
    template_scores: tuple[dict[str, Any], ...]
    block_bbox: tuple[int, int, int, int]
    point_cost: int
    quantization_level: str
    kp_residual_ppm: int
    cache_key: str
    sort_key: str
    reused: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "char_id": self.char_id,
            "sample_id": self.sample_id,
            "keypoint_id": self.keypoint_id,
            "candidate_id": self.candidate_id,
            "component_id": self.component_id,
            "char": self.char,
            "code_point": self.code_point,
            "template_id": self.template_id,
            "score": self.score,
            "score_ppm": self.score_ppm,
            "confidence": self.confidence,
            "template_scores": list(self.template_scores),
            "block_bbox": list(self.block_bbox),
            "point_cost": self.point_cost,
            "quantization_level": self.quantization_level,
            "kp_residual_ppm": self.kp_residual_ppm,
            "cache_key": self.cache_key,
            "sort_key": self.sort_key,
            "reused": self.reused,
        }


@dataclass(frozen=True)
class BudgetReport:
    limit: int
    consumed: int
    pending: tuple[dict[str, Any], ...]
    completed_count: int
    reused_count: int
    max_observed_workers: int
    worker_limit: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "limit": self.limit,
            "consumed": self.consumed,
            "pending": list(self.pending),
            "completed_count": self.completed_count,
            "reused_count": self.reused_count,
            "max_observed_workers": self.max_observed_workers,
            "worker_limit": self.worker_limit,
        }


@dataclass(frozen=True)
class RecognitionBatch:
    results: tuple[RecognitionResult, ...]
    budget: BudgetReport
    reuse_records: tuple[dict[str, Any], ...] = field(default_factory=tuple)


def _crop_block(
    image: np.ndarray,
    keypoint: KeypointResult,
    *,
    target_width: int,
    target_height: int,
) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    half_width = max(4, target_width)
    half_height = max(4, target_height)
    center_x = round_half_up(
        6 * keypoint.x + keypoint.bbox.left + keypoint.bbox.right,
        8,
    )
    center_y = round_half_up(
        6 * keypoint.y + keypoint.bbox.top + keypoint.bbox.bottom,
        8,
    )
    left = center_x - half_width
    top = center_y - half_height
    right = center_x + half_width
    bottom = center_y + half_height
    padded = np.pad(image, ((half_height + 2, half_height + 2), (half_width + 2, half_width + 2)), mode="constant", constant_values=242)
    offset_x = half_width + 2
    offset_y = half_height + 2
    region = padded[
        top + offset_y : bottom + offset_y,
        left + offset_x : right + offset_x,
    ]
    pil = Image.fromarray(region, mode="L").resize(
        (target_width, target_height),
        resample=Image.Resampling.NEAREST,
    )
    return np.asarray(pil, dtype=np.uint8), (left, top, right, bottom)


def _match_glyph(
    target_bits: np.ndarray,
    glyph: GlyphTemplate,
    *,
    weights: tuple[int, ...],
    score_scale: int,
) -> tuple[int, tuple[dict[str, Any], ...]]:
    score_sum = 0
    variant_records: list[dict[str, Any]] = []
    target_pixels = int(target_bits.sum())
    for weight, variant in zip(weights, glyph.variants):
        pixel_count = variant.width * variant.height
        template_pixels = variant.ink_count()
        template_bits = np.asarray(variant.bits, dtype=np.uint8).reshape(variant.height, variant.width)
        best_score = -1
        best_intersection = 0
        best_shift = (0, 0)
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                aligned = np.roll(target_bits, (dy, dx), axis=(0, 1))
                intersection = int(np.logical_and(aligned, template_bits).sum())
                union = target_pixels + template_pixels - intersection
                if union <= 0:
                    candidate_score = score_scale
                else:
                    candidate_score = round_half_up(intersection * score_scale, union)
                if candidate_score > best_score:
                    best_score = candidate_score
                    best_intersection = intersection
                    best_shift = (dx, dy)
        variant_score = best_score
        score_sum += weight * variant_score
        variant_records.append(
            {
                "name": variant.name,
                "weight": weight,
                "intersection": best_intersection,
                "target_ink_pixels": target_pixels,
                "template_ink_pixels": template_pixels,
                "pixels": pixel_count,
                "score": variant_score,
                "alignment": {"dx": best_shift[0], "dy": best_shift[1]},
            }
        )
    score = round_half_up(score_sum, sum(weights))
    return score, tuple(variant_records)


def recognize_one(plan: RecognitionPlan, *, config: RunConfig) -> RecognitionResult:
    keypoint = plan.keypoint
    target_gray, block_bbox = _crop_block(
        plan.image,
        keypoint,
        target_width=plan.template_library.width,
        target_height=plan.template_library.height,
    )
    target_bits = target_gray <= config.thresholds.ink
    score_scale = 1 << (config.fixed_bits + 10)
    glyph_scores: list[
        tuple[int, int, GlyphTemplate, tuple[dict[str, Any], ...]]
    ] = []
    for glyph in plan.template_library.glyphs:
        score, variants = _match_glyph(
            target_bits,
            glyph,
            weights=plan.template_library.weights,
            score_scale=score_scale,
        )
        glyph_scores.append((score, glyph.code_point, glyph, variants))
    glyph_scores.sort(key=lambda item: (-item[0], item[1]))
    best_score, best_code_point, best_glyph, _ = glyph_scores[0]
    score_ppm = ratio_fixed(best_score, score_scale, 20)
    quality_ppm = max(0, 1_000_000 - keypoint.residual_ppm)
    score_weight = config.thresholds.confidence_score_weight
    residual_weight = config.thresholds.confidence_residual_weight
    confidence = round_half_up(
        score_ppm * score_weight + quality_ppm * residual_weight,
        score_weight + residual_weight,
    )
    template_scores = tuple(
        {
            "char": glyph.char,
            "code_point": code_point,
            "template_id": glyph.template_id,
            "score": score,
            "score_ppm": ratio_fixed(score, score_scale, 20),
            "variants": list(variants),
        }
        for score, code_point, glyph, variants in sorted(
            glyph_scores,
            key=lambda item: item[1],
        )
    )
    sort_key = plan.sort_key
    return RecognitionResult(
        char_id=f"char:{keypoint.keypoint_id}",
        sample_id=plan.sample_id,
        keypoint_id=keypoint.keypoint_id,
        candidate_id=keypoint.candidate_id,
        component_id=keypoint.component_id,
        char=best_glyph.char,
        code_point=best_code_point,
        template_id=best_glyph.template_id,
        score=best_score,
        score_ppm=score_ppm,
        confidence=confidence,
        template_scores=template_scores,
        block_bbox=block_bbox,
        point_cost=plan.point_cost,
        quantization_level=config.quantization.level,
        kp_residual_ppm=keypoint.residual_ppm,
        cache_key=plan.cache_key,
        sort_key=sort_key,
    )


def _pending_record(plan: RecognitionPlan) -> dict[str, Any]:
    return {
        "code": E_POINT_BUDGET_PENDING,
        "sample_id": plan.sample_id,
        "keypoint_id": plan.keypoint.keypoint_id,
        "sort_key": plan.sort_key,
        "point_cost": plan.point_cost,
        "priority": 0 if plan.priority == 0 else 1,
    }


def run_budgeted_recognition(
    plans: Iterable[RecognitionPlan],
    *,
    config: RunConfig,
    prior_results: dict[str, RecognitionResult] | None = None,
    pending_ids: set[str] | None = None,
) -> RecognitionBatch:
    prior_results = prior_results or {}
    pending_ids = pending_ids or set()
    plan_list = list(plans)
    plan_list.sort(
        key=lambda plan: (
            0 if plan.keypoint.keypoint_id in pending_ids else 1,
            plan.sort_key,
        )
    )
    consumed = 0
    selected: list[RecognitionPlan] = []
    pending: list[dict[str, Any]] = []
    reused: dict[str, RecognitionResult] = {}
    reuse_records: list[dict[str, Any]] = []
    for plan in plan_list:
        keypoint_id = plan.keypoint.keypoint_id
        known = prior_results.get(keypoint_id)
        if known is not None and keypoint_id not in pending_ids:
            reused_result = RecognitionResult(**{**known.__dict__, "reused": True})
            reused[keypoint_id] = reused_result
            reuse_records.append(
                {
                    "artifact_type": "recognition",
                    "artifact_id": reused_result.char_id,
                    "cache_key": reused_result.cache_key,
                    "status": "reused",
                    "reason": "prior_recognition_cache_hit",
                }
            )
            continue
        if consumed + plan.point_cost <= config.budget.max_template_points:
            selected.append(plan)
            consumed += plan.point_cost
        else:
            pending.append(_pending_record(plan))

    if selected:
        tasks = [OrderedTask(sort_key=plan.sort_key, payload=plan) for plan in selected]
        execution: StageExecution[RecognitionResult] = execute_ordered(
            tasks,
            lambda plan: recognize_one(plan, config=config),
            max_workers=config.recognition.max_workers,
            stage="recognition",
        )
        computed = list(execution.results)
    else:
        execution = StageExecution((), (), 0, config.recognition.max_workers)
        computed = []
    all_results = list(reused.values()) + computed
    all_results.sort(key=lambda item: item.sort_key)
    pending.sort(key=lambda item: item["sort_key"])
    return RecognitionBatch(
        results=tuple(all_results),
        budget=BudgetReport(
            limit=config.budget.max_template_points,
            consumed=consumed,
            pending=tuple(pending),
            completed_count=len(computed),
            reused_count=len(reused),
            max_observed_workers=execution.max_observed,
            worker_limit=execution.worker_limit,
        ),
        reuse_records=tuple(reuse_records),
    )
