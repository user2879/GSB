"""Sliding-window coarse localization and connected-component refinement."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from gsb.concurrency import OrderedTask, StageExecution, execute_ordered
from gsb.config import RunConfig
from gsb.models import BBox
from gsb.util import (
    ratio_fixed,
    sha256_bytes,
    sha256_obj,
    short_fingerprint,
    task_sort_key,
)


@dataclass(frozen=True)
class Candidate:
    candidate_id: str
    fingerprint: str
    sort_key: str
    x: int
    y: int
    width: int
    height: int
    ink_count: int
    density_ppm: int
    vertical_valley_ppm: int
    texture_gradient_ppm: int
    projection: tuple[int, ...]
    projection_weights: tuple[int, int, int]
    score: int
    selected: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "fingerprint": self.fingerprint,
            "sort_key": self.sort_key,
            "window": {
                "x": self.x,
                "y": self.y,
                "width": self.width,
                "height": self.height,
            },
            "metrics": {
                "ink_count": self.ink_count,
                "density_ppm": self.density_ppm,
                "vertical_valley_ppm": self.vertical_valley_ppm,
                "texture_gradient_ppm": self.texture_gradient_ppm,
                "vertical_projection": list(self.projection),
                "projection_weights": list(self.projection_weights),
            },
            "score": self.score,
            "selected": self.selected,
        }


@dataclass(frozen=True)
class Component:
    component_id: str
    fingerprint: str
    sort_key: str
    candidate_id: str
    candidate_fingerprint: str
    bbox: BBox
    area: int
    weight_sum: int
    centroid_x_fixed: int
    centroid_y_fixed: int
    mean_ink: int
    fragment_ids: tuple[str, ...]
    merge_basis: tuple[dict[str, Any], ...]
    split_basis: tuple[dict[str, Any], ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "component_id": self.component_id,
            "fingerprint": self.fingerprint,
            "sort_key": self.sort_key,
            "candidate_id": self.candidate_id,
            "candidate_fingerprint": self.candidate_fingerprint,
            "bbox": self.bbox.to_list(),
            "area": self.area,
            "weight_sum": self.weight_sum,
            "centroid_fixed": [self.centroid_x_fixed, self.centroid_y_fixed],
            "mean_ink": self.mean_ink,
            "fragment_ids": list(self.fragment_ids),
            "merge_basis": list(self.merge_basis),
            "split_basis": list(self.split_basis),
        }


@dataclass(frozen=True)
class Fragment:
    fragment_id: str
    bbox: BBox
    area: int
    pixel_sum: int
    pixels: tuple[tuple[int, int], ...]
    source_fragment_ids: tuple[str, ...]

    @property
    def mean_ink(self) -> int:
        return self.pixel_sum // self.area if self.area else 0


@dataclass(frozen=True)
class LocalizationResult:
    candidates: tuple[Candidate, ...]
    components: tuple[Component, ...]
    candidate_execution: StageExecution[Candidate]


def _candidate_for_window(
    image: np.ndarray,
    *,
    x: int,
    y: int,
    config: RunConfig,
    image_fingerprint: str,
    parameter_fingerprint: str,
    template_fingerprint: str,
    file_name: str,
) -> Candidate:
    height, width = config.window.height, config.window.width
    region = image[y : y + height, x : x + width]
    mask = region <= config.thresholds.ink
    ink_count = int(mask.sum())
    density_ppm = ratio_fixed(ink_count, mask.size, 20)
    projection = tuple(int(value) for value in mask.sum(axis=0))
    max_projection = max(projection) if projection else 0
    deepest_zero_run = 0
    current_zero_run = 0
    for value in projection:
        if value == 0:
            current_zero_run += 1
            deepest_zero_run = max(deepest_zero_run, current_zero_run)
        else:
            current_zero_run = 0
    if max_projection == 0:
        valley_ppm = 1_000_000
    else:
        valley_ppm = ratio_fixed(deepest_zero_run, len(projection), 20)
    gradient_sum = 0
    if region.shape[0] > 1:
        gradient_sum += int(np.abs(np.diff(region.astype(np.int16), axis=0)).sum())
    if region.shape[1] > 1:
        gradient_sum += int(np.abs(np.diff(region.astype(np.int16), axis=1)).sum())
    gradient_denominator = max(1, int(mask.size) * 255)
    gradient_ppm = ratio_fixed(gradient_sum, gradient_denominator, 20)
    identity = {
        "image": image_fingerprint,
        "parameters": parameter_fingerprint,
        "template": template_fingerprint,
        "window": [x, y, width, height],
    }
    fingerprint = sha256_obj(identity)
    seed = sha256_bytes(f"projection|{fingerprint}".encode("ascii"))
    weights = tuple(64 + ord(seed[index * 2]) % 65 for index in range(3))
    score = (
        density_ppm * weights[0]
        + valley_ppm * weights[1]
        + gradient_ppm * weights[2]
    ) // sum(weights)
    selected = score >= config.thresholds.candidate_score
    sort_key = task_sort_key(
        Path(file_name).name,
        "sliding",
        x // config.window.stride_x,
        y // config.window.stride_y,
        ((y // config.window.stride_y) << 16) + (x // config.window.stride_x),
    )
    return Candidate(
        candidate_id=short_fingerprint(fingerprint),
        fingerprint=fingerprint,
        sort_key=sort_key,
        x=x,
        y=y,
        width=width,
        height=height,
        ink_count=ink_count,
        density_ppm=density_ppm,
        vertical_valley_ppm=valley_ppm,
        texture_gradient_ppm=gradient_ppm,
        projection=projection,
        projection_weights=weights,
        score=score,
        selected=selected,
    )


def sliding_candidates(
    image: np.ndarray,
    *,
    config: RunConfig,
    image_fingerprint: str,
    parameter_fingerprint: str,
    template_fingerprint: str,
    file_name: str,
) -> StageExecution[Candidate]:
    tasks = []
    for y in range(0, config.layout.height - config.window.height + 1, config.window.stride_y):
        for x in range(0, config.layout.width - config.window.width + 1, config.window.stride_x):
            sort_key = task_sort_key(
                Path(file_name).name,
                "sliding",
                x // config.window.stride_x,
                y // config.window.stride_y,
                ((y // config.window.stride_y) << 16) + (x // config.window.stride_x),
            )
            tasks.append(
                OrderedTask(
                    sort_key=sort_key,
                    payload=(x, y),
                )
            )

    def worker(payload: tuple[int, int]) -> Candidate:
        x, y = payload
        return _candidate_for_window(
            image,
            x=x,
            y=y,
            config=config,
            image_fingerprint=image_fingerprint,
            parameter_fingerprint=parameter_fingerprint,
            template_fingerprint=template_fingerprint,
            file_name=file_name,
        )

    return execute_ordered(
        tasks,
        worker,
        max_workers=config.recognition.max_workers,
        stage="sliding",
    )


def _fragments(image: np.ndarray, threshold: int) -> list[Fragment]:
    mask = image <= threshold
    height, width = mask.shape
    visited = np.zeros_like(mask, dtype=bool)
    fragments: list[Fragment] = []
    for start_y in range(height):
        for start_x in range(width):
            if not mask[start_y, start_x] or visited[start_y, start_x]:
                continue
            stack = [(start_x, start_y)]
            visited[start_y, start_x] = True
            pixels: list[tuple[int, int]] = []
            while stack:
                x, y = stack.pop()
                pixels.append((x, y))
                for dy in (-1, 0, 1):
                    for dx in (-1, 0, 1):
                        if dx == 0 and dy == 0:
                            continue
                        nx = x + dx
                        ny = y + dy
                        if 0 <= nx < width and 0 <= ny < height and mask[ny, nx] and not visited[ny, nx]:
                            visited[ny, nx] = True
                            stack.append((nx, ny))
            pixels.sort(key=lambda item: (item[1], item[0]))
            xs = [item[0] for item in pixels]
            ys = [item[1] for item in pixels]
            bbox = BBox(min(xs), min(ys), max(xs) + 1, max(ys) + 1)
            fragment_identity = sha256_obj(
                {
                    "bbox": bbox.to_list(),
                    "pixels_sha256": sha256_bytes(
                        b"".join(int(y * width + x).to_bytes(4, "big") for x, y in pixels)
                    ),
                }
            )
            fragments.append(
                Fragment(
                    fragment_id=short_fingerprint(fragment_identity),
                    bbox=bbox,
                    area=len(pixels),
                    pixel_sum=int(sum(int(image[y, x]) for x, y in pixels)),
                    pixels=tuple(pixels),
                    source_fragment_ids=(short_fingerprint(fragment_identity),),
                )
            )
    fragments.sort(key=lambda item: (item.bbox.top, item.bbox.left, item.fragment_id))
    return fragments


def _bbox_gap(left: BBox, right: BBox) -> int:
    horizontal = max(0, max(left.left, right.left) - min(left.right, right.right))
    vertical = max(0, max(left.top, right.top) - min(left.bottom, right.bottom))
    return max(horizontal, vertical)


def _union_bbox(left: BBox, right: BBox) -> BBox:
    return BBox(
        min(left.left, right.left),
        min(left.top, right.top),
        max(left.right, right.right),
        max(left.bottom, right.bottom),
    )


def _candidate_for_region(
    bbox: BBox,
    candidates: list[Candidate],
) -> Candidate | None:
    center_x = (bbox.left + bbox.right) // 2
    center_y = (bbox.top + bbox.bottom) // 2
    selected = [candidate for candidate in candidates if candidate.selected]
    pool = selected or candidates
    containing: list[Candidate] = []
    for candidate in pool:
        window_bbox = BBox(
            candidate.x,
            candidate.y,
            candidate.x + candidate.width,
            candidate.y + candidate.height,
        )
        if window_bbox.contains_point(center_x, center_y):
            containing.append(candidate)
    if containing:
        return max(containing, key=lambda candidate: (candidate.score, candidate.fingerprint))
    if not pool:
        return None
    return min(
        pool,
        key=lambda candidate: (
            abs(candidate.x + candidate.width // 2 - center_x)
            + abs(candidate.y + candidate.height // 2 - center_y),
            candidate.fingerprint,
        ),
    )


def _merge_fragments(
    fragments: list[Fragment],
    *,
    config: RunConfig,
) -> tuple[list[Fragment], list[list[dict[str, Any]]]]:
    parent = list(range(len(fragments)))
    basis: list[list[dict[str, Any]]] = [[] for _ in fragments]

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    for left_index in range(len(fragments)):
        for right_index in range(left_index + 1, len(fragments)):
            left = fragments[left_index]
            right = fragments[right_index]
            gap = _bbox_gap(left.bbox, right.bbox)
            if gap > config.thresholds.cc_merge_gap:
                continue
            tone_delta = abs(left.mean_ink - right.mean_ink)
            union = _union_bbox(left.bbox, right.bbox)
            if tone_delta > config.thresholds.component_tone_tolerance:
                continue
            if union.width > config.thresholds.component_max_span or union.height > config.thresholds.component_max_span:
                continue
            left_root = find(left_index)
            right_root = find(right_index)
            if left_root == right_root:
                continue
            parent[right_root] = left_root
            record = {
                "rule": "same_candidate_same_tone_nearby",
                "left_fragment": left.fragment_id,
                "right_fragment": right.fragment_id,
                "bbox_gap": gap,
                "tone_delta": tone_delta,
                "union_bbox": union.to_list(),
            }
            basis[left_root].append(record)
            basis[right_root].append(record)

    groups: dict[int, list[int]] = {}
    for index in range(len(fragments)):
        root = find(index)
        groups.setdefault(root, []).append(index)
    merged: list[Fragment] = []
    merged_basis: list[list[dict[str, Any]]] = []
    for root in sorted(groups):
        indices = sorted(groups[root])
        if len(indices) == 1:
            merged.append(fragments[indices[0]])
            merged_basis.append([])
            continue
        members = [fragments[index] for index in indices]
        pixels = tuple(sorted((pixel for member in members for pixel in member.pixels), key=lambda item: (item[1], item[0])))
        xs = [pixel[0] for pixel in pixels]
        ys = [pixel[1] for pixel in pixels]
        bbox = BBox(min(xs), min(ys), max(xs) + 1, max(ys) + 1)
        combined_id = short_fingerprint(
            sha256_obj(
                {
                    "bbox": bbox.to_list(),
                    "fragments": [member.fragment_id for member in members],
                    "pixels": len(pixels),
                }
            )
        )
        union_basis: list[dict[str, Any]] = []
        for index in indices:
            for record in basis[index]:
                if record not in union_basis:
                    union_basis.append(record)
        merged.append(
            Fragment(
                fragment_id=combined_id,
                bbox=bbox,
                area=len(pixels),
                pixel_sum=sum(member.pixel_sum for member in members),
                pixels=pixels,
                source_fragment_ids=tuple(
                    source
                    for member in members
                    for source in member.source_fragment_ids
                ),
            )
        )
        merged_basis.append(sorted(union_basis, key=lambda item: (item["left_fragment"], item["right_fragment"])))
    merged.sort(key=lambda item: (item.bbox.top, item.bbox.left, item.fragment_id))
    alignment = {fragment.fragment_id: basis for fragment, basis in zip(merged, merged_basis)}
    return merged, [alignment[fragment.fragment_id] for fragment in merged]


def _split_large_fragment(
    fragment: Fragment,
    *,
    config: RunConfig,
    image: np.ndarray,
) -> list[Fragment]:
    if (
        fragment.bbox.width <= config.thresholds.component_max_span
        and fragment.bbox.height <= config.thresholds.component_max_span
    ):
        return [fragment]
    occupancy: dict[int, int] = {}
    for x, y in fragment.pixels:
        occupancy[y] = occupancy.get(y, 0) + 1
    rows = sorted(occupancy)
    cuts: list[int] = []
    for top, bottom in zip(rows, rows[1:]):
        if bottom - top > config.thresholds.split_min_gap + 1:
            cuts.append(top + 1)
    if not cuts:
        return [fragment]
    buckets: list[list[tuple[int, int]]] = []
    current: list[tuple[int, int]] = []
    cut_index = 0
    for pixel in sorted(fragment.pixels, key=lambda item: (item[1], item[0])):
        while cut_index < len(cuts) and pixel[1] >= cuts[cut_index]:
            if current:
                buckets.append(current)
                current = []
            cut_index += 1
        current.append(pixel)
    if current:
        buckets.append(current)
    if len(buckets) <= 1:
        return [fragment]
    split: list[Fragment] = []
    for bucket in buckets:
        xs = [pixel[0] for pixel in bucket]
        ys = [pixel[1] for pixel in bucket]
        bbox = BBox(min(xs), min(ys), max(xs) + 1, max(ys) + 1)
        split.append(
            Fragment(
                fragment_id=short_fingerprint(
                    sha256_obj({"parent": fragment.fragment_id, "bbox": bbox.to_list()})
                ),
                bbox=bbox,
                area=len(bucket),
                pixel_sum=sum(int(image[y, x]) for x, y in bucket),
                pixels=tuple(bucket),
                source_fragment_ids=fragment.source_fragment_ids,
            )
        )
    return split


def _component_from_fragment(
    fragment: Fragment,
    *,
    config: RunConfig,
    image_fingerprint: str,
    parameter_fingerprint: str,
    candidate: Candidate,
    merge_basis: list[dict[str, Any]],
    split_basis: list[dict[str, Any]],
    file_name: str,
    component_index: int,
    image: np.ndarray,
) -> Component:
    total_weight = 0
    weighted_x = 0
    weighted_y = 0
    for x, y in fragment.pixels:
        value = max(1, config.thresholds.ink - int(image[y, x]))
        total_weight += value
        weighted_x += x * value
        weighted_y += y * value
    if total_weight == 0:
        centroid_x_fixed = 0
        centroid_y_fixed = 0
    else:
        centroid_x_fixed = ratio_fixed(weighted_x, total_weight, config.fixed_bits)
        centroid_y_fixed = ratio_fixed(weighted_y, total_weight, config.fixed_bits)
    fingerprint = sha256_obj(
        {
            "image": image_fingerprint,
            "parameters": parameter_fingerprint,
            "candidate": candidate.fingerprint,
            "fragment": fragment.fragment_id,
            "bbox": fragment.bbox.to_list(),
            "merge_basis": merge_basis,
            "split_basis": split_basis,
        }
    )
    sort_key = task_sort_key(
        Path(file_name).name,
        "component",
        candidate.x // max(1, config.window.stride_x),
        component_index,
        (fragment.bbox.top << 16) + fragment.bbox.left,
    )
    return Component(
        component_id=short_fingerprint(fingerprint),
        fingerprint=fingerprint,
        sort_key=sort_key,
        candidate_id=candidate.candidate_id,
        candidate_fingerprint=candidate.fingerprint,
        bbox=fragment.bbox,
        area=fragment.area,
        weight_sum=total_weight,
        centroid_x_fixed=centroid_x_fixed,
        centroid_y_fixed=centroid_y_fixed,
        mean_ink=fragment.mean_ink,
        fragment_ids=fragment.source_fragment_ids,
        merge_basis=tuple(merge_basis),
        split_basis=tuple(split_basis),
    )


def connected_components(
    image: np.ndarray,
    *,
    candidates: list[Candidate],
    config: RunConfig,
    image_fingerprint: str,
    parameter_fingerprint: str,
    file_name: str,
) -> list[Component]:
    fragments = [
        fragment
        for fragment in _fragments(image, config.thresholds.ink)
        if fragment.area >= config.thresholds.component_min_area
    ]
    merged, bases = _merge_fragments(fragments, config=config)
    components: list[Component] = []
    for index, (fragment, merge_basis) in enumerate(zip(merged, bases)):
        candidate = _candidate_for_region(fragment.bbox, candidates)
        if candidate is None:
            continue
        split_fragments = _split_large_fragment(fragment, config=config, image=image)
        for split_index, split_fragment in enumerate(split_fragments):
            split_basis = []
            if len(split_fragments) > 1:
                split_basis.append(
                    {
                        "rule": "projection_span_split",
                        "parent_fragment": fragment.fragment_id,
                        "part": split_index,
                        "parts": len(split_fragments),
                    }
                )
            components.append(
                _component_from_fragment(
                    split_fragment,
                    config=config,
                    image_fingerprint=image_fingerprint,
                    parameter_fingerprint=parameter_fingerprint,
                    candidate=candidate,
                    merge_basis=merge_basis,
                    split_basis=split_basis,
                    file_name=file_name,
                    component_index=index * 1000 + split_index,
                    image=image,
                )
            )
    components.sort(key=lambda item: (item.sort_key, item.fingerprint))
    return components


def localize(
    image: np.ndarray,
    *,
    config: RunConfig,
    image_fingerprint: str,
    parameter_fingerprint: str,
    template_fingerprint: str,
    file_name: str,
) -> LocalizationResult:
    execution = sliding_candidates(
        image,
        config=config,
        image_fingerprint=image_fingerprint,
        parameter_fingerprint=parameter_fingerprint,
        template_fingerprint=template_fingerprint,
        file_name=file_name,
    )
    candidates = list(execution.results)
    components = connected_components(
        image,
        candidates=candidates,
        config=config,
        image_fingerprint=image_fingerprint,
        parameter_fingerprint=parameter_fingerprint,
        file_name=file_name,
    )
    return LocalizationResult(
        candidates=tuple(candidates),
        components=tuple(components),
        candidate_execution=execution,
    )
