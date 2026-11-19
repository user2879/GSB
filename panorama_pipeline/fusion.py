from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np

from .config import PipelineConfig
from .errors import E_BOUNDARY_SENSITIVE
from .io_utils import canonical_json_bytes, sha256_bytes
from .models import InstanceMask, instance_numeric_sort_key
from .scheduler import OrderedTask, ordered_parallel_map


_FAST_DECISION_DECIMALS = 6


def fast_quantize(value: float) -> float:
    return round(float(value), _FAST_DECISION_DECIMALS)


def fusion_dependency_fingerprint(
    instances: list[InstanceMask],
    *,
    iou_threshold: float,
    retention_mode: str,
    min_coverage: float,
) -> str:
    participants = [
        [item.key, item.cache_key]
        for item in sorted(instances, key=lambda value: value.key)
    ]
    return sha256_bytes(
        canonical_json_bytes(
            {
                "participants": participants,
                "iou_match_threshold": float(iou_threshold),
                "retention_mode": retention_mode,
                "min_coverage": float(min_coverage),
            }
        )
    )


@dataclass(frozen=True)
class PairMetrics:
    left_key: str
    right_key: str
    class_id: int
    left_area: int
    right_area: int
    intersection: int
    union: int
    reference_iou: float
    actual_iou: float
    reference_coverage: float
    actual_coverage: float
    left_confidence: float
    right_confidence: float

    @property
    def pair_key(self) -> str:
        return f"{self.left_key}|{self.right_key}"

    @property
    def area_weighted_confidence(self) -> float:
        total = self.left_area + self.right_area
        if total == 0:
            return (self.left_confidence + self.right_confidence) / 2.0
        return (
            self.left_confidence * self.left_area
            + self.right_confidence * self.right_area
        ) / total


@dataclass
class FusedInstance:
    identifier: str
    decision_id: str
    instance_keys: tuple[str, ...]
    class_id: int
    mask: np.ndarray
    confidence: float
    crowd: bool
    palette: tuple[int, int, int, int]
    iou: float
    coverage: float
    matched: bool
    mask_area: int = 0
    weighted_pixel_support: float = 0.0
    assigned_pixels: int = 0

    def summary(self) -> dict[str, Any]:
        return {
            "identifier": self.identifier,
            "decision_id": self.decision_id,
            "instance_keys": list(self.instance_keys),
            "class_id": self.class_id,
            "confidence": self.confidence,
            "crowd": self.crowd,
            "palette": list(self.palette),
            "iou": self.iou,
            "coverage": self.coverage,
            "matched": self.matched,
            "mask_area": self.mask_area,
            "weighted_pixel_support": self.weighted_pixel_support,
            "assigned_pixels": self.assigned_pixels,
        }


@dataclass
class FusionResult:
    category_grid: np.ndarray
    fused_instances: list[FusedInstance]
    matches: list[dict[str, Any]]
    boundary_diagnostics: list[dict[str, Any]]
    class_distribution: dict[str, int]
    fusion_result_version: str
    conflict_pixels: int
    failure_codes: list[str] = field(default_factory=list)

    @property
    def boundary_sensitive(self) -> list[dict[str, Any]]:
        return [
            item
            for item in self.boundary_diagnostics
            if item["decision_mismatch"]
        ]

    def summary(self) -> dict[str, Any]:
        return {
            "fusion_result_version": self.fusion_result_version,
            "class_distribution": self.class_distribution,
            "matched_pair_count": len(self.matches),
            "fused_instance_count": len(self.fused_instances),
            "conflict_pixels": self.conflict_pixels,
            "failure_codes": self.failure_codes,
            "boundary_diagnostic_count": len(self.boundary_diagnostics),
            "boundary_sensitive_count": len(self.boundary_sensitive),
            "category_grid_sha256": sha256_bytes(
                np.ascontiguousarray(self.category_grid, dtype=np.int32).tobytes()
            ),
        }


def _pair_metrics(
    left: InstanceMask,
    right: InstanceMask,
) -> PairMetrics:
    intersection = int(np.count_nonzero(left.mask & right.mask))
    union = int(np.count_nonzero(left.mask | right.mask))
    left_area = left.area
    right_area = right.area
    reference_iou = float(intersection / union) if union else 0.0
    actual_iou = fast_quantize(reference_iou)
    denominator = min(left_area, right_area)
    reference_coverage = float(intersection / denominator) if denominator else 0.0
    actual_coverage = fast_quantize(reference_coverage)
    return PairMetrics(
        left_key=left.key,
        right_key=right.key,
        class_id=left.class_id,
        left_area=left_area,
        right_area=right_area,
        intersection=intersection,
        union=union,
        reference_iou=reference_iou,
        actual_iou=actual_iou,
        reference_coverage=reference_coverage,
        actual_coverage=actual_coverage,
        left_confidence=left.confidence,
        right_confidence=right.confidence,
    )


def _diagnostic_entry(
    metric: PairMetrics,
    *,
    iou_threshold: float,
    coverage_floor: float,
    epsilon: float,
) -> dict[str, Any] | None:
    near_iou = (
        abs(metric.reference_iou - iou_threshold) <= epsilon
        or abs(metric.actual_iou - iou_threshold) <= epsilon
    )
    reference_iou_decision = metric.reference_iou >= iou_threshold
    actual_iou_decision = metric.actual_iou >= iou_threshold
    reference_coverage_decision = metric.reference_coverage >= coverage_floor
    actual_coverage_decision = metric.actual_coverage >= coverage_floor
    near_coverage = (
        abs(metric.reference_coverage - coverage_floor) <= epsilon
        or abs(metric.actual_coverage - coverage_floor) <= epsilon
    )
    if not near_iou and not near_coverage:
        return None
    iou_mismatch = reference_iou_decision != actual_iou_decision
    coverage_mismatch = reference_coverage_decision != actual_coverage_decision
    mismatch = iou_mismatch or coverage_mismatch
    sources: list[str] = []
    if iou_mismatch:
        sources.append("iou_fast_path_quantization")
    if coverage_mismatch:
        sources.append("coverage_fast_path_quantization")
    return {
        "pair_key": metric.pair_key,
        "instance_keys": [metric.left_key, metric.right_key],
        "class_id": metric.class_id,
        "metric": "iou_and_coverage",
        "iou_boundary_value": iou_threshold,
        "coverage_boundary_value": coverage_floor,
        "epsilon": epsilon,
        "iou": {
            "reference_double": metric.reference_iou,
            "actual_fast": metric.actual_iou,
            "difference": metric.reference_iou - metric.actual_iou,
            "reference_decision": reference_iou_decision,
            "actual_decision": actual_iou_decision,
        },
        "coverage": {
            "reference_double": metric.reference_coverage,
            "actual_fast": metric.actual_coverage,
            "difference": metric.reference_coverage - metric.actual_coverage,
            "reference_decision": reference_coverage_decision,
            "actual_decision": actual_coverage_decision,
        },
        "decision_mismatch": mismatch,
        "deviation_sources": sources,
        "business_code": E_BOUNDARY_SENSITIVE if mismatch else None,
    }


def _fuse_pair(
    left: InstanceMask,
    right: InstanceMask,
    metric: PairMetrics,
    *,
    retention_mode: str,
) -> FusedInstance:
    primary, secondary = sorted(
        (left, right),
        key=lambda item: (-item.confidence, item.key),
    )
    if retention_mode == "drop_crowd":
        crowd = left.crowd or right.crowd
    else:
        crowd = left.crowd and right.crowd
    mask = left.mask | right.mask
    weighted_pixel_support = float(
        np.count_nonzero(left.mask) * left.confidence
        + np.count_nonzero(right.mask) * right.confidence
    )
    identifier = f"fused:{metric.left_key}+{metric.right_key}"
    decision_id = min(
        (left.lexical_id, left.source_format),
        (right.lexical_id, right.source_format),
    )[0]
    return FusedInstance(
        identifier=identifier,
        decision_id=decision_id,
        instance_keys=tuple(sorted((left.key, right.key))),
        class_id=left.class_id,
        mask=mask,
        confidence=metric.area_weighted_confidence,
        crowd=crowd,
        palette=primary.palette,
        iou=metric.actual_iou,
        coverage=metric.actual_coverage,
        matched=True,
        mask_area=int(np.count_nonzero(mask)),
        weighted_pixel_support=weighted_pixel_support,
    )


def _unmatched_instance(item: InstanceMask) -> FusedInstance:
    return FusedInstance(
        identifier=item.key,
        decision_id=item.lexical_id,
        instance_keys=(item.key,),
        class_id=item.class_id,
        mask=item.mask,
        confidence=item.confidence,
        crowd=item.crowd,
        palette=item.palette,
        iou=0.0,
        coverage=1.0 if item.area else 0.0,
        matched=False,
        mask_area=item.area,
        weighted_pixel_support=float(item.area * item.confidence),
    )


def _raster_priority(item: FusedInstance, retention_mode: str) -> tuple:
    if retention_mode == "prefer_non_crowd":
        crowd_tier = 1 if item.crowd else 0
    else:
        crowd_tier = 0
    return (
        crowd_tier,
        -item.iou,
        -item.confidence,
        item.decision_id,
        item.identifier,
    )


def fuse_instances(
    instances: list[InstanceMask],
    *,
    config: PipelineConfig,
    max_workers: int | None = None,
    progress_callback: Callable[[float, int], None] | None = None,
) -> FusionResult:
    workers = config.max_workers if max_workers is None else max_workers
    ordered_instances = sorted(instances, key=instance_numeric_sort_key)
    if progress_callback:
        progress_callback(0.05, 0)
    by_format: dict[str, list[InstanceMask]] = {"v1": [], "v2": []}
    for item in ordered_instances:
        by_format[item.source_format].append(item)
    candidate_pairs = [
        (left, right)
        for left in by_format["v1"]
        for right in by_format["v2"]
        if left.class_id == right.class_id
    ]
    pair_tasks = [
        OrderedTask(
            instance_id=left.instance_id,
            stage=f"iou_match_against_v2_{right.instance_id}",
            payload=(left, right),
            source_format="v1",
        )
        for left, right in candidate_pairs
    ]
    metrics = ordered_parallel_map(
        pair_tasks,
        lambda pair: _pair_metrics(pair[0], pair[1]),
        workers,
    )
    if progress_callback:
        progress_callback(0.35, len(candidate_pairs))

    diagnostics: list[dict[str, Any]] = []
    for metric in sorted(metrics, key=lambda value: value.pair_key):
        entry = _diagnostic_entry(
            metric,
            iou_threshold=config.iou_match_threshold,
            coverage_floor=config.pixel_retention.min_coverage,
            epsilon=config.boundary_epsilon,
        )
        if entry is not None:
            diagnostics.append(entry)

    eligible = [
        metric
        for metric in metrics
        if metric.actual_iou >= config.iou_match_threshold
        and metric.actual_coverage >= config.pixel_retention.min_coverage
    ]
    eligible.sort(
        key=lambda value: (
            -value.actual_iou,
            -value.area_weighted_confidence,
            value.pair_key,
        )
    )
    instance_lookup = {item.key: item for item in ordered_instances}
    used_keys: set[str] = set()
    accepted_matches: list[PairMetrics] = []
    match_decisions: list[dict[str, Any]] = []
    for metric in eligible:
        if metric.left_key in used_keys or metric.right_key in used_keys:
            match_decisions.append(
                {
                    "pair_key": metric.pair_key,
                    "accepted": False,
                    "reason": "one_to_one_conflict",
                    "iou": metric.actual_iou,
                    "coverage": metric.actual_coverage,
                    "confidence": metric.area_weighted_confidence,
                }
            )
            continue
        used_keys.add(metric.left_key)
        used_keys.add(metric.right_key)
        accepted_matches.append(metric)
        match_decisions.append(
            {
                "pair_key": metric.pair_key,
                "accepted": True,
                "reason": "matched",
                "iou": metric.actual_iou,
                "coverage": metric.actual_coverage,
                "confidence": metric.area_weighted_confidence,
            }
        )
    if progress_callback:
        progress_callback(0.55, len(accepted_matches))

    fused_instances: list[FusedInstance] = []
    for metric in accepted_matches:
        fused_instances.append(
            _fuse_pair(
                instance_lookup[metric.left_key],
                instance_lookup[metric.right_key],
                metric,
                retention_mode=config.pixel_retention.mode,
            )
        )
    for item in ordered_instances:
        if item.key not in used_keys:
            fused_instances.append(_unmatched_instance(item))

    if config.pixel_retention.mode == "drop_crowd":
        raster_candidates = [item for item in fused_instances if not item.crowd]
    else:
        raster_candidates = list(fused_instances)
    raster_candidates.sort(
        key=lambda item: _raster_priority(item, config.pixel_retention.mode)
    )
    max_class = max(
        [item.class_id for item in fused_instances] + [0]
    )
    category_grid = np.zeros_like(fused_instances[0].mask, dtype=np.int32) if fused_instances else np.zeros((1, 1), dtype=np.int32)
    conflict_pixels = 0
    assigned_by_identifier: dict[str, int] = {}
    for item in raster_candidates:
        target = category_grid == 0
        assignment = item.mask & target
        conflict_pixels += int(np.count_nonzero(item.mask & ~target))
        assigned = int(np.count_nonzero(assignment))
        assigned_by_identifier[item.identifier] = assigned
        category_grid[assignment] = item.class_id
    for item in fused_instances:
        item.assigned_pixels = assigned_by_identifier.get(item.identifier, 0)
    if progress_callback:
        progress_callback(0.8, len(fused_instances))

    counts = np.bincount(category_grid.reshape(-1), minlength=max_class + 1)
    class_distribution = {
        str(class_id): int(counts[class_id])
        for class_id in range(len(counts))
        if counts[class_id] or class_id == 0
    }
    dependencies = fusion_dependency_fingerprint(
        ordered_instances,
        iou_threshold=config.iou_match_threshold,
        retention_mode=config.pixel_retention.mode,
        min_coverage=config.pixel_retention.min_coverage,
    )
    failure_codes = sorted(
        {E_BOUNDARY_SENSITIVE}
        if any(entry["decision_mismatch"] for entry in diagnostics)
        else set()
    )
    if progress_callback:
        progress_callback(1.0, len(fused_instances))
    return FusionResult(
        category_grid=category_grid,
        fused_instances=sorted(
            fused_instances,
            key=lambda item: (item.decision_id, item.identifier),
        ),
        matches=match_decisions,
        boundary_diagnostics=sorted(diagnostics, key=lambda item: item["pair_key"]),
        class_distribution=class_distribution,
        fusion_result_version=dependencies,
        conflict_pixels=conflict_pixels,
        failure_codes=failure_codes,
    )
