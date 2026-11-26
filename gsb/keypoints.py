"""Fixed-point keypoint regression with annotation-version overrides."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from gsb.config import RunConfig
from gsb.fixed_image import regress_center
from gsb.localize import Component
from gsb.models import AnnotationVersion, BBox, SampleManifest, TruthKeypoint
from gsb.util import residual_fixed, sha256_obj, short_fingerprint, task_sort_key


@dataclass(frozen=True)
class KeypointResult:
    keypoint_id: str
    cache_key: str
    sort_key: str
    x: int
    y: int
    bbox: BBox
    column_truth: int
    reading_index_truth: int
    text_truth: str
    template_id: str
    candidate_id: str
    component_id: str
    component_fingerprint: str
    raw_fixed: tuple[int, int]
    quantized_fixed: tuple[int, int]
    regression_rounding: dict[str, str]
    offset_rounding: dict[str, str]
    residual_fixed: int
    residual_ppm: int
    source: str
    annotation_version: str | None
    annotation_coordinate_fingerprint: str | None
    reused: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "keypoint_id": self.keypoint_id,
            "cache_key": self.cache_key,
            "sort_key": self.sort_key,
            "x": self.x,
            "y": self.y,
            "bbox": self.bbox.to_list(),
            "column_truth": self.column_truth,
            "reading_index_truth": self.reading_index_truth,
            "text_truth": self.text_truth,
            "template_id": self.template_id,
            "candidate_id": self.candidate_id,
            "component_id": self.component_id,
            "component_fingerprint": self.component_fingerprint,
            "raw_fixed": list(self.raw_fixed),
            "quantized_fixed": list(self.quantized_fixed),
            "rounding": {
                "regression": dict(self.regression_rounding),
                "annotation_offset": dict(self.offset_rounding),
            },
            "residual_fixed": self.residual_fixed,
            "residual_ppm": self.residual_ppm,
            "source": self.source,
            "annotation_version": self.annotation_version,
            "annotation_coordinate_fingerprint": self.annotation_coordinate_fingerprint,
            "reused": self.reused,
        }


def _bbox_iou(left: BBox, right: BBox) -> int:
    intersection = left.intersection(right)
    if intersection is None:
        return 0
    union_area = left.area + right.area - intersection.area
    if union_area <= 0:
        return 0
    return intersection.area * 1_000_000 // union_area


def match_components_to_truth(
    components: list[Component],
    manifest: SampleManifest,
) -> dict[int, TruthKeypoint]:
    truth = list(manifest.keypoints)
    assignments: dict[int, TruthKeypoint] = {}
    used: set[str] = set()
    for component_index, component in enumerate(components):
        candidates = [item for item in truth if item.keypoint_id not in used]
        if not candidates:
            break
        best = max(
            candidates,
            key=lambda item: (
                _bbox_iou(component.bbox, item.bbox),
                -abs(component.bbox.left - item.bbox.left)
                - abs(component.bbox.top - item.bbox.top),
                -item.reading_index,
                item.keypoint_id,
            ),
        )
        if _bbox_iou(component.bbox, best.bbox) > 0:
            assignments[component_index] = best
            used.add(best.keypoint_id)
    return assignments


def _annotation_fingerprint(
    annotation: AnnotationVersion | None,
    keypoint_id: str,
) -> tuple[str | None, tuple[int, int] | None]:
    if annotation is None:
        return None, None
    coordinate = annotation.coordinate_for(keypoint_id)
    if coordinate is None:
        return None, None
    fingerprint = sha256_obj(
        {
            "sample_id": annotation.sample_id,
            "keypoint_id": keypoint_id,
            "coordinate": list(coordinate),
        }
    )
    return fingerprint, coordinate


def _offset_direction(reference: int, observed: int) -> str:
    if observed > reference:
        return "UP"
    if observed < reference:
        return "DOWN"
    return "EXACT"


def regress_keypoints(
    image: np.ndarray,
    *,
    components: list[Component],
    manifest: SampleManifest,
    annotation: AnnotationVersion | None,
    config: RunConfig,
    image_fingerprint: str,
    parameter_fingerprint: str,
    template_fingerprint: str,
    file_name: str,
    prior_by_cache: dict[str, KeypointResult] | None = None,
) -> list[KeypointResult]:
    prior_by_cache = prior_by_cache or {}
    assignments = match_components_to_truth(components, manifest)
    results: list[KeypointResult] = []
    for component_index, component in enumerate(components):
        truth = assignments.get(component_index)
        if truth is None:
            truth = TruthKeypoint(
                keypoint_id=f"auto-{component.component_id}",
                x=(component.bbox.left + component.bbox.right) // 2,
                y=(component.bbox.top + component.bbox.bottom) // 2,
                bbox=component.bbox,
                column=-1,
                reading_index=len(manifest.keypoints) + component_index,
                text="",
                template_id="",
            )
        annotation_fingerprint, annotation_coordinate = _annotation_fingerprint(
            annotation,
            truth.keypoint_id,
        )
        cache_payload = {
            "stage": "keypoint",
            "image": image_fingerprint,
            "parameters": parameter_fingerprint,
            "template_fingerprint": template_fingerprint,
            "template_version": config.recognition.template_version,
            "quantization": config.quantization.level,
            "component": component.fingerprint,
            "annotation": annotation_fingerprint,
            "truth_id": truth.keypoint_id,
        }
        cache_key = sha256_obj(cache_payload)
        known = prior_by_cache.get(cache_key)
        if known is not None:
            results.append(
                KeypointResult(
                    **{
                        **known.__dict__,
                        "reused": True,
                    }
                )
            )
            continue
        regression = regress_center(
            image,
            threshold=config.thresholds.ink,
            bits=config.fixed_bits,
            bbox=component.bbox,
        )
        if regression is None:
            continue
        raw_x, raw_y = regression["raw_fixed"]
        regressed_x = int(regression["x"])
        regressed_y = int(regression["y"])
        if annotation_coordinate is None:
            x, y = regressed_x, regressed_y
            source = "regression"
            offset_rounding = {"x": "EXACT", "y": "EXACT"}
        else:
            x, y = annotation_coordinate
            source = "annotation"
            offset_rounding = {
                "x": _offset_direction(regressed_x, x),
                "y": _offset_direction(regressed_y, y),
            }
        residual = residual_fixed(raw_x, regression["quantized_fixed"][0]) + residual_fixed(
            raw_y, regression["quantized_fixed"][1]
        )
        residual_ppm = min(
            1_000_000,
            residual * 1_000_000 // (1 << config.fixed_bits),
        )
        sort_key = task_sort_key(
            Path(file_name).name,
            "keypoint",
            truth.column,
            truth.reading_index,
            component.bbox.top * 10000 + component.bbox.left,
        )
        results.append(
            KeypointResult(
                keypoint_id=truth.keypoint_id,
                cache_key=cache_key,
                sort_key=sort_key,
                x=x,
                y=y,
                bbox=component.bbox,
                column_truth=truth.column,
                reading_index_truth=truth.reading_index,
                text_truth=truth.text,
                template_id=truth.template_id,
                candidate_id=component.candidate_id,
                component_id=component.component_id,
                component_fingerprint=component.fingerprint,
                raw_fixed=(raw_x, raw_y),
                quantized_fixed=tuple(regression["quantized_fixed"]),
                regression_rounding=dict(regression["rounding"]),
                offset_rounding=offset_rounding,
                residual_fixed=residual,
                residual_ppm=residual_ppm,
                source=source,
                annotation_version=annotation.version if annotation is not None else None,
                annotation_coordinate_fingerprint=annotation_fingerprint,
                reused=False,
            )
        )
    results.sort(key=lambda item: (item.sort_key, item.keypoint_id))
    return results
