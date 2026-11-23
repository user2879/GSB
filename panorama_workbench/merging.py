from __future__ import annotations

from dataclasses import dataclass, field
from fractions import Fraction
from typing import Any, Iterable

import numpy as np

from .config import WorkbenchConfig
from .inference import Fragment, TileMask, extract_fragments


@dataclass
class MergedInstance:
    instance_id: int
    category_code: int
    pixels: tuple[int, ...]
    area: int
    min_row: int
    min_col: int
    max_row: int
    max_col: int
    fragment_ids: tuple[str, ...]
    tile_keys: tuple[str, ...]
    representative_fragment_id: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "area": self.area,
            "bbox": [self.min_col, self.min_row, self.max_col - self.min_col + 1, self.max_row - self.min_row + 1],
            "category_code": self.category_code,
            "fragment_ids": list(self.fragment_ids),
            "instance_id": self.instance_id,
            "representative_fragment_id": self.representative_fragment_id,
            "tile_keys": list(self.tile_keys),
        }


@dataclass
class MergeResult:
    instances: tuple[MergedInstance, ...]
    semantic_raster: np.ndarray
    instance_raster: np.ndarray
    fragments: tuple[Fragment, ...]
    dropped_instances: tuple[dict[str, Any], ...]
    truncated_instances: tuple[dict[str, Any], ...]
    merge_events: tuple[dict[str, Any], ...]
    processed_mask_pixels: int
    conflict_pixels: int


@dataclass
class _Group:
    order: int
    category_code: int
    pixels: set[int]
    min_row: int
    min_col: int
    max_row: int
    max_col: int
    fragments: list[Fragment] = field(default_factory=list)

    @property
    def area(self) -> int:
        return len(self.pixels)


def merge_masks(
    width: int,
    height: int,
    masks: Iterable[TileMask],
    config: WorkbenchConfig,
) -> MergeResult:
    mask_list = sorted(masks, key=lambda mask: (mask.geometry.row, mask.geometry.col))
    fragments: list[Fragment] = []
    for mask in mask_list:
        fragments.extend(extract_fragments(width, height, mask))
    fragments.sort(
        key=lambda fragment: (
            fragment.tile_row,
            fragment.tile_col,
            fragment.fragment_index,
            fragment.category_code,
            fragment.fragment_id,
        )
    )
    threshold = Fraction(str(config.merge_iou))
    groups: list[_Group] = []
    groups_by_category: dict[int, list[_Group]] = {}
    merge_events: list[dict[str, Any]] = []

    for fragment in fragments:
        fragment_pixels = set(fragment.pixels)
        candidate: _Group | None = None
        candidate_iou: Fraction | None = None
        for group in groups_by_category.get(fragment.category_code, []):
            intersection = len(fragment_pixels.intersection(group.pixels))
            if intersection == 0:
                continue
            union = len(fragment_pixels) + len(group.pixels) - intersection
            iou = Fraction(intersection, union)
            if iou < threshold:
                continue
            if _candidate_better(candidate, candidate_iou, group, iou):
                candidate = group
                candidate_iou = iou
        if candidate is None:
            min_row = min(pixel // width for pixel in fragment_pixels)
            min_col = min(pixel % width for pixel in fragment_pixels)
            max_row = max(pixel // width for pixel in fragment_pixels)
            max_col = max(pixel % width for pixel in fragment_pixels)
            group = _Group(
                order=len(groups),
                category_code=fragment.category_code,
                pixels=fragment_pixels,
                min_row=min_row,
                min_col=min_col,
                max_row=max_row,
                max_col=max_col,
                fragments=[fragment],
            )
            groups.append(group)
            groups_by_category.setdefault(fragment.category_code, []).append(group)
            continue

        before_area = candidate.area
        candidate.pixels.update(fragment_pixels)
        candidate.fragments.append(fragment)
        candidate.min_row = min(candidate.min_row, fragment.min_row)
        candidate.min_col = min(candidate.min_col, fragment.min_col)
        candidate.max_row = max(candidate.max_row, fragment.min_row + 1)
        candidate.max_col = max(candidate.max_col, fragment.min_col + 1)
        event = {
            "fragment_area": fragment.area,
            "fragment_id": fragment.fragment_id,
            "group_area_before": before_area,
            "iou": _fraction_float(candidate_iou),
            "target_order": candidate.order,
            "threshold_tie": candidate_iou == threshold,
        }
        if candidate_iou == threshold:
            event["tie_break"] = (
                "smaller_fragment_into_larger"
                if fragment.area <= before_area
                else "larger_fragment_absorbs_smaller_group"
            )
        merge_events.append(event)

    claim_order = sorted(
        groups,
        key=lambda group: (
            -group.area,
            group.category_code,
            group.min_row,
            group.min_col,
            group.fragments[0].fragment_id,
            group.order,
        ),
    )
    claimed = np.zeros(height * width, dtype=bool)
    resolved: list[_Group] = []
    conflict_pixels = 0
    for group in claim_order:
        retained = {pixel for pixel in group.pixels if not claimed[pixel]}
        conflict_pixels += len(group.pixels) - len(retained)
        if not retained:
            continue
        group.pixels = retained
        group.min_row = min(pixel // width for pixel in retained)
        group.min_col = min(pixel % width for pixel in retained)
        group.max_row = max(pixel // width for pixel in retained)
        group.max_col = max(pixel % width for pixel in retained)
        for pixel in retained:
            claimed[pixel] = True
        resolved.append(group)

    dropped: list[dict[str, Any]] = []
    retained_groups: list[_Group] = []
    for group in resolved:
        if group.area < config.min_instance_area:
            dropped.append(
                {
                    "area": group.area,
                    "category_code": group.category_code,
                    "fragment_ids": sorted(fragment.fragment_id for fragment in group.fragments),
                    "reason_code": "E_INSTANCE_BELOW_MIN_AREA",
                    "threshold": config.min_instance_area,
                }
            )
        else:
            retained_groups.append(group)

    retained_groups.sort(
        key=lambda group: (
            -group.area,
            group.category_code,
            group.min_row,
            group.min_col,
            group.fragments[0].fragment_id,
            group.order,
        )
    )
    instances: list[MergedInstance] = []
    truncated: list[dict[str, Any]] = []
    for rank, group in enumerate(retained_groups):
        fragment_ids = tuple(sorted(fragment.fragment_id for fragment in group.fragments))
        representative = min(fragment_ids)
        payload = {
            "area": group.area,
            "category_code": group.category_code,
            "fragment_ids": list(fragment_ids),
            "representative_fragment_id": representative,
        }
        if rank >= config.max_instances_per_image:
            payload["reason_code"] = "E_INSTANCE_LIMIT_TRUNCATED"
            payload["rank"] = rank + 1
            truncated.append(payload)
            continue
        tile_keys = tuple(
            sorted({fragment.tile_key for fragment in group.fragments})
        )
        instances.append(
            MergedInstance(
                instance_id=rank + 1,
                category_code=group.category_code,
                pixels=tuple(sorted(group.pixels)),
                area=group.area,
                min_row=group.min_row,
                min_col=group.min_col,
                max_row=group.max_row,
                max_col=group.max_col,
                fragment_ids=fragment_ids,
                tile_keys=tile_keys,
                representative_fragment_id=representative,
            )
        )

    semantic_raster = np.zeros((height, width), dtype=np.int32)
    instance_raster = np.zeros((height, width), dtype=np.int32)
    for instance in instances:
        rows = np.fromiter((pixel // width for pixel in instance.pixels), dtype=np.int32)
        cols = np.fromiter((pixel % width for pixel in instance.pixels), dtype=np.int32)
        semantic_raster[rows, cols] = instance.category_code
        instance_raster[rows, cols] = instance.instance_id

    return MergeResult(
        instances=tuple(instances),
        semantic_raster=semantic_raster,
        instance_raster=instance_raster,
        fragments=tuple(fragments),
        dropped_instances=tuple(dropped),
        truncated_instances=tuple(truncated),
        merge_events=tuple(merge_events),
        processed_mask_pixels=sum(fragment.area for fragment in fragments),
        conflict_pixels=conflict_pixels,
    )


def _candidate_better(
    current: _Group | None,
    current_iou: Fraction | None,
    candidate: _Group,
    candidate_iou: Fraction,
) -> bool:
    if current is None or current_iou is None:
        return True
    if candidate_iou != current_iou:
        return candidate_iou > current_iou
    if candidate.area != current.area:
        return candidate.area > current.area
    return candidate.order < current.order


def _fraction_float(value: Fraction | None) -> float:
    if value is None:
        return 0.0
    return float(value)
