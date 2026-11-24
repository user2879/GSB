"""Deterministic layout detection, box regression, splitting and matching."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import math
from typing import Any, Iterable, Sequence

import numpy as np
from PIL import Image

from .config import Config
from .templates import TemplateSet

COLUMN_ALGORITHM_VERSION = "columns-projection-v1"
SEGMENT_ALGORITHM_VERSION = "segments-projection-v1"
REGRESSION_ALGORITHM_VERSION = "regression-centroid-dp-v1"
MATCH_ALGORITHM_VERSION = "match-fixed-v1"


def normalize_window(window: int) -> int:
    return max(1, int(window))


def _moving_sum(values: np.ndarray, window: int) -> np.ndarray:
    """Centered integer moving sum with clipped edges."""
    window = normalize_window(window)
    radius = window // 2
    prefix = np.zeros(values.size + 1, dtype=np.int64)
    np.cumsum(values.astype(np.int64), out=prefix[1:])
    result = np.empty(values.size, dtype=np.int64)
    for i in range(values.size):
        lo = max(0, i - radius)
        hi = min(values.size, i + radius + 1)
        result[i] = prefix[hi] - prefix[lo]
    return result


def moving_average(values: np.ndarray, window: int) -> np.ndarray:
    window = normalize_window(window)
    sums = _moving_sum(values, window)
    counts = np.empty(values.size, dtype=np.int64)
    radius = window // 2
    for i in range(values.size):
        counts[i] = min(values.size, i + radius + 1) - max(0, i - radius)
    return sums * 1000 // np.maximum(1, counts)


def binarize(pixels: np.ndarray, threshold: int) -> np.ndarray:
    return np.asarray(pixels, dtype=np.uint8) < int(threshold)


def binary_digest(binary: np.ndarray) -> str:
    arr = np.ascontiguousarray(binary.astype(np.uint8, copy=False))
    payload = f"{arr.shape[0]}x{arr.shape[1]}:".encode("ascii") + arr.tobytes()
    return sha256(payload).hexdigest()


def rgb_digest_bytes(payload: bytes) -> str:
    return sha256(payload).hexdigest()


@dataclass(frozen=True)
class DetectedColumn:
    reading_index: int
    x0: int
    x1: int
    raw_ink: int
    peak_density: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "reading_index": self.reading_index,
            "x0": self.x0,
            "x1": self.x1,
            "width": self.x1 - self.x0,
            "raw_ink": self.raw_ink,
            "peak_density": self.peak_density,
        }


def _runs_from_occupancy(occupied: np.ndarray) -> list[tuple[int, int]]:
    runs: list[tuple[int, int]] = []
    start: int | None = None
    for index, active in enumerate(occupied):
        if active and start is None:
            start = index
        elif not active and start is not None:
            runs.append((start, index))
            start = None
    if start is not None:
        runs.append((start, len(occupied)))
    return runs


def _merge_close_runs(runs: Sequence[tuple[int, int]], max_gap: int) -> list[tuple[int, int]]:
    if not runs:
        return []
    merged = [list(runs[0])]
    for start, end in runs[1:]:
        if start - merged[-1][1] <= max_gap:
            merged[-1][1] = end
        else:
            merged.append([start, end])
    return [(int(a), int(b)) for a, b in merged]


def detect_columns(pixels: np.ndarray, config: Config) -> list[DetectedColumn]:
    binary = binarize(pixels, config.binarization_threshold)
    height, width = binary.shape
    profile = binary.sum(axis=0, dtype=np.int64)
    smoothed = moving_average(profile, config.row_projection_window)
    minimum = max(1, int(math.ceil(height * 2 / 1000)))
    occupied = smoothed >= minimum * 1000
    runs = _merge_close_runs(
        _runs_from_occupancy(occupied), max_gap=max(1, config.row_projection_window // 2)
    )
    raw_occupied = profile > 0
    refined: list[tuple[int, int]] = []
    radius = max(1, config.row_projection_window)
    for start, end in runs:
        lo = max(0, start - radius)
        hi = min(width, end + radius)
        raw_positions = np.flatnonzero(raw_occupied[lo:hi])
        if raw_positions.size == 0:
            continue
        x0 = int(lo + raw_positions[0])
        x1 = int(lo + raw_positions[-1] + 1)
        if x1 - x0 < 2:
            continue
        if refined and x0 <= refined[-1][1]:
            refined[-1] = (refined[-1][0], max(refined[-1][1], x1))
        else:
            refined.append((x0, x1))
    refined.sort(key=lambda pair: (-pair[0], -pair[1]))
    columns: list[DetectedColumn] = []
    for index, (x0, x1) in enumerate(refined, start=1):
        raw_ink = int(profile[x0:x1].sum())
        peak = int(smoothed[x0:x1].max()) if x1 > x0 else 0
        columns.append(DetectedColumn(index, x0, x1, raw_ink, peak))
    return columns


@dataclass(frozen=True)
class CandidateSegment:
    segment_index: int
    y0: int
    y1: int
    height: int
    raw_ink: int
    height_class: str
    needs_regression: bool

    def as_dict(self) -> dict[str, Any]:
        return {
            "segment_index": self.segment_index,
            "y0": self.y0,
            "y1": self.y1,
            "height": self.height,
            "raw_ink": self.raw_ink,
            "height_class": self.height_class,
            "needs_regression": self.needs_regression,
        }


def segment_column(
    pixels: np.ndarray,
    column: DetectedColumn,
    config: Config,
) -> list[CandidateSegment]:
    binary = binarize(pixels[:, column.x0 : column.x1], config.binarization_threshold)
    profile = binary.sum(axis=1, dtype=np.int64)
    smoothed = moving_average(profile, config.row_projection_window)
    occupied = smoothed > 0
    runs = _merge_close_runs(
        _runs_from_occupancy(occupied), max_gap=max(1, config.row_projection_window // 4)
    )
    raw_rows = profile > 0
    height = pixels.shape[0]
    refined: list[tuple[int, int]] = []
    radius = max(1, config.row_projection_window // 2)
    for start, end in runs:
        lo = max(0, start - radius)
        hi = min(height, end + radius)
        positions = np.flatnonzero(raw_rows[lo:hi])
        if positions.size == 0:
            continue
        y0 = int(lo + positions[0])
        y1 = int(lo + positions[-1] + 1)
        if refined and y0 <= refined[-1][1]:
            refined[-1] = (refined[-1][0], max(refined[-1][1], y1))
        else:
            refined.append((y0, y1))
    segments: list[CandidateSegment] = []
    for index, (y0, y1) in enumerate(refined, start=1):
        h = y1 - y0
        if h < config.char_min_height:
            classification = "under_min"
            needs = True
        elif h > config.char_max_height:
            classification = "over_max"
            needs = True
        else:
            classification = "normal"
            needs = False
        segments.append(
            CandidateSegment(
                segment_index=index,
                y0=y0,
                y1=y1,
                height=h,
                raw_ink=int(profile[y0:y1].sum()),
                height_class=classification,
                needs_regression=needs,
            )
        )
    return segments


@dataclass
class SplitSolution:
    cuts: tuple[int, ...]
    costs: tuple[dict[str, int], ...]
    total_cost: int
    tied: bool
    parts: int


def _cut_cost(projection: np.ndarray, cut: int, ideal: int, penalty: int) -> dict[str, int]:
    max_projection = max(1, int(projection.max()))
    p = int(projection[cut])
    valley_depth = max(0, max_projection - p)
    balance = abs(int(cut) - int(ideal))
    # Normalize valley depth to roughly 0..1000, while the split penalty weights
    # deviation from an even division. The base balance term keeps a zero penalty
    # from choosing a deep valley that cuts through the middle of a glyph.
    normalized_projection = p * 1000 // max_projection
    # Squaring strongly favors true blank rows over soft intra-glyph valleys.
    valley_cost = normalized_projection * normalized_projection // 100
    balance_cost = balance * (4 + penalty) * 128
    return {
        "cut": int(cut),
        "ideal": int(ideal),
        "valley_depth": int(valley_depth),
        "balance_distance": int(balance),
        "valley_cost": int(valley_cost),
        "balance_cost": int(balance_cost),
        "total": int(valley_cost + balance_cost),
    }


def _solve_exact_parts(
    start: int,
    end: int,
    parts: int,
    projection: np.ndarray,
    min_height: int,
    max_height: int,
    penalty: int,
) -> tuple[int, tuple[int, ...], tuple[dict[str, int], ...], bool] | None:
    height = end - start
    memo: dict[tuple[int, int], tuple[int, tuple[int, ...], tuple[dict[str, int], ...], bool]] = {}

    def solve(pos: int, remaining: int) -> tuple[int, tuple[int, ...], tuple[dict[str, int], ...], bool]:
        state = (pos, remaining)
        if state in memo:
            return memo[state]
        if remaining == 1:
            length = end - pos
            if min_height <= length <= max_height:
                result = (0, (), (), False)
            else:
                result = (10**18, (), (), False)
            memo[state] = result
            return result
        best_cost = 10**18
        best_cuts: tuple[int, ...] = ()
        best_costs: tuple[dict[str, int], ...] = ()
        best_tied = False
        minimum_rest = (remaining - 1) * min_height
        lo = pos + min_height
        hi = min(end - minimum_rest, pos + max_height)
        boundary_index = parts - remaining + 1
        ideal = start + int(round(height * boundary_index / parts))
        for cut in range(lo, hi + 1):
            if cut <= pos or cut >= end:
                continue
            tail = solve(cut, remaining - 1)
            if tail[0] >= 10**18:
                continue
            own = _cut_cost(projection, cut, ideal, penalty)
            total = own["total"] + tail[0]
            candidate_cuts = (cut,) + tail[1]
            candidate_costs = (own,) + tail[2]
            tied = tail[3]
            if total < best_cost:
                best_cost = total
                best_cuts = candidate_cuts
                best_costs = candidate_costs
                best_tied = tied
            elif total == best_cost:
                if not best_cuts or candidate_cuts < best_cuts:
                    best_cuts = candidate_cuts
                    best_costs = candidate_costs
                best_tied = True
        result = (best_cost, best_cuts, best_costs, best_tied)
        memo[state] = result
        return result

    solved = solve(start, parts)
    if solved[0] >= 10**18:
        return None
    return solved[0], solved[1], solved[2], solved[3]


def split_oversize(
    y0: int,
    y1: int,
    projection: np.ndarray,
    config: Config,
) -> SplitSolution | None:
    height = y1 - y0
    if height <= config.char_max_height:
        return None
    minimum_parts = max(2, int(math.ceil(height / config.char_max_height)))
    maximum_parts = max(minimum_parts, height // max(1, config.char_min_height))
    maximum_parts = min(maximum_parts, 8)
    solutions: list[SplitSolution] = []
    for parts in range(minimum_parts, maximum_parts + 1):
        solved = _solve_exact_parts(
            y0,
            y1,
            parts,
            projection,
            config.char_min_height,
            config.char_max_height,
            config.split_penalty,
        )
        if solved is None:
            continue
        total, cuts, costs, tied = solved
        group_cost = (parts - 1) * config.split_penalty * 4096
        solutions.append(
            SplitSolution(
                cuts=cuts,
                costs=costs,
                total_cost=int(total + group_cost),
                tied=bool(tied),
                parts=parts,
            )
        )
    if not solutions:
        return None
    solutions.sort(key=lambda s: (s.total_cost, s.cuts))
    best = solutions[0]
    best_tied = best.tied or any(s.total_cost == best.total_cost for s in solutions[1:])
    best.tied = best_tied
    return best


def _regress_one(
    pixels: np.ndarray,
    threshold: int,
    x0: int,
    x1: int,
    y0: int,
    y1: int,
    min_height: int,
    max_height: int,
) -> dict[str, Any]:
    image_height, image_width = pixels.shape
    local = pixels[y0:y1, x0:x1]
    ink = local < threshold
    if not ink.any():
        tight_x0, tight_y0, tight_x1, tight_y1 = x0, y0, x1, y1
        centroid_y_fp = (y0 + y1) * 2
        total_ink = 0
    else:
        ys, xs = np.nonzero(ink)
        tight_y0 = y0 + int(ys.min())
        tight_y1 = y0 + int(ys.max()) + 1
        tight_x0 = x0 + int(xs.min())
        tight_x1 = x0 + int(xs.max()) + 1
        row_counts = ink.sum(axis=1, dtype=np.int64)
        weighted = int((np.arange(y0, y1, dtype=np.int64) * row_counts).sum())
        total_ink = int(row_counts.sum())
        centroid_y_fp = (2 * weighted + total_ink) * 2 // max(1, total_ink)
    tight_height = max(1, tight_y1 - tight_y0)
    tight_width = max(1, tight_x1 - tight_x0)
    tight_mid_fp = (tight_y0 + tight_y1) * 2
    target_center_fp = (3 * centroid_y_fp + 2 * tight_mid_fp) // 5
    target_height = min(max_height, max(min_height, tight_height + 2))
    box_center_fp = target_center_fp
    candidate_y0 = (box_center_fp - target_height * 2) // 4
    candidate_y1 = candidate_y0 + target_height
    if candidate_y0 > tight_y0 - 1:
        candidate_y0 = tight_y0 - 1
        candidate_y1 = candidate_y0 + target_height
    if candidate_y1 < tight_y1 + 1:
        candidate_y1 = tight_y1 + 1
        candidate_y0 = candidate_y1 - target_height
    candidate_y0 = max(y0, candidate_y0)
    candidate_y1 = min(y1, candidate_y1)
    if candidate_y1 <= candidate_y0:
        candidate_y0, candidate_y1 = y0, y1
    target_width = min(max(2, tight_width + 2), max(2, candidate_y1 - candidate_y0 + 4))
    x_center = (tight_x0 + tight_x1) // 2
    candidate_x0 = x_center - target_width // 2
    candidate_x1 = candidate_x0 + target_width
    if candidate_x0 > tight_x0 - 1:
        candidate_x0 = tight_x0 - 1
        candidate_x1 = candidate_x0 + target_width
    if candidate_x1 < tight_x1 + 1:
        candidate_x1 = tight_x1 + 1
        candidate_x0 = candidate_x1 - target_width
    candidate_x0 = max(x0, candidate_x0)
    candidate_x1 = min(x1, candidate_x1)
    if candidate_x1 <= candidate_x0:
        candidate_x0, candidate_x1 = x0, x1
    box_center_y = (candidate_y0 + candidate_y1) * 2
    center_residual_fp = abs(box_center_y - centroid_y_fp)
    height_residual = abs((candidate_y1 - candidate_y0) - tight_height)
    center_residual = (center_residual_fp + 2) // 4
    residual = int(center_residual + height_residual)
    return {
        "bbox": [int(candidate_x0), int(candidate_y0), int(candidate_x1), int(candidate_y1)],
        "tight_bbox": [int(tight_x0), int(tight_y0), int(tight_x1), int(tight_y1)],
        "centroid_y_q4": int(centroid_y_fp),
        "ink_pixels": int(total_ink),
        "residual": residual,
        "residual_detail": {
            "center_distance": int(center_residual),
            "height_difference": int(height_residual),
        },
    }


def regress_segment(
    pixels: np.ndarray,
    column: DetectedColumn,
    segment: CandidateSegment,
    config: Config,
) -> dict[str, Any]:
    binary = binarize(pixels[:, column.x0 : column.x1], config.binarization_threshold)
    projection = moving_average(binary.sum(axis=1, dtype=np.int64), config.row_projection_window)
    split = split_oversize(segment.y0, segment.y1, projection, config)
    parts: list[tuple[int, int]] = []
    split_record: dict[str, Any]
    if split is not None:
        boundaries = (segment.y0, *split.cuts, segment.y1)
        parts = [(boundaries[i], boundaries[i + 1]) for i in range(len(boundaries) - 1)]
        split_record = {
            "applied": True,
            "penalty": int(config.split_penalty),
            "split_count": int(len(split.cuts)),
            "cuts": [int(cut) for cut in split.cuts],
            "tied": bool(split.tied),
            "total_cost": int(split.total_cost),
            "parts": int(split.parts),
            "cut_costs": [dict(cost) for cost in split.costs],
            "segment_costs": [
                {
                    "y0": int(parts[i][0]),
                    "y1": int(parts[i][1]),
                    "height": int(parts[i][1] - parts[i][0]),
                    "cost": int(split.costs[i]["total"]) if i < len(split.costs) else 0,
                }
                for i in range(len(parts))
            ],
        }
    else:
        parts = [(segment.y0, segment.y1)]
        split_record = {
            "applied": False,
            "penalty": 0,
            "split_count": 0,
            "cuts": [],
            "tied": False,
            "total_cost": 0,
            "parts": 1,
            "cut_costs": [],
            "segment_costs": [
                {
                    "y0": int(segment.y0),
                    "y1": int(segment.y1),
                    "height": int(segment.height),
                    "cost": 0,
                }
            ],
        }
    regressed: list[dict[str, Any]] = []
    for part_index, (py0, py1) in enumerate(parts, start=1):
        one = _regress_one(
            pixels,
            config.binarization_threshold,
            column.x0,
            column.x1,
            py0,
            py1,
            config.char_min_height,
            config.char_max_height,
        )
        one["part_index"] = part_index
        one["split_record"] = split_record
        one["source_segment"] = segment.as_dict()
        regressed.append(one)
    return {"split": split_record, "boxes": regressed}


def quantize_block(
    box_pixels: np.ndarray,
    levels: int,
    size: int,
    *,
    threshold: int | None = None,
) -> np.ndarray:
    """Tight-crop ink, resize to a fixed dot grid and quantize ink levels."""
    arr = np.asarray(box_pixels, dtype=np.uint8)
    if arr.size == 0:
        arr = np.zeros((1, 1), dtype=np.uint8)
    if threshold is not None:
        ink = arr < int(threshold)
        ys, xs = np.nonzero(ink)
        if xs.size:
            arr = arr[int(ys.min()) : int(ys.max()) + 1, int(xs.min()) : int(xs.max()) + 1]
    image = Image.fromarray(arr, mode="L").resize((size, size), Image.Resampling.BOX)
    resized = np.asarray(image, dtype=np.uint8).astype(np.int64)
    low = int(resized.min())
    high = int(resized.max())
    if high <= low:
        normalized = np.zeros_like(resized)
    else:
        normalized = (high - resized) * 255 // (high - low)
    return ((normalized * (levels - 1) + 127) // 255).astype(np.int64)


def _variant_score(a: np.ndarray, b: np.ndarray, levels: int, scale: int, metric: str) -> int:
    n = int(a.size)
    denom = max(1, n * max(1, levels - 1))
    if metric == "l1":
        diff = int(np.abs(a - b).sum())
        score = scale - (diff * scale + denom - 1) // denom
    elif metric == "hamming":
        cutoff = max(1, levels // 2)
        equal = int(((a >= cutoff) == (b >= cutoff)).sum())
        score = equal * scale // max(1, n)
    elif metric == "iou":
        cutoff = max(1, levels // 2)
        left = a >= cutoff
        right = b >= cutoff
        union = int((left | right).sum())
        if union == 0:
            score = scale
        else:
            score = int((left & right).sum()) * scale // union
    elif metric == "cosine":
        dot = int((a * b).sum())
        aa = int((a * a).sum())
        bb = int((b * b).sum())
        denominator = math.isqrt(max(1, aa * bb))
        if denominator == 0:
            score = scale
        else:
            score = dot * scale // denominator
    else:
        raise ValueError(f"未知匹配度量: {metric}")
    return max(0, min(scale, int(score)))


def match_block(
    pixels: np.ndarray,
    bbox: Sequence[int],
    config: Config,
    templates: TemplateSet,
) -> dict[str, Any]:
    x0, y0, x1, y1 = (int(v) for v in bbox)
    block = pixels[y0:y1, x0:x1]
    levels = config.quant_levels
    scale = config.fixed_scale
    query = quantize_block(block, levels, templates.size, threshold=config.binarization_threshold)
    weight_sum = sum(templates.weights)
    best: list[dict[str, Any]] = []
    for char in templates.ordered_characters():
        variant_scores: list[int] = []
        weighted_total = 0
        for variant, weight in zip(templates.variants[char], templates.weights):
            template_q = variant.astype(np.int64) * (levels - 1)
            score = _variant_score(query, template_q, levels, scale, config.match_metric)
            variant_scores.append(int(score))
            weighted_total += int(score) * int(weight)
        fused = weighted_total // weight_sum
        best.append(
            {
                "text": char,
                "score_q": int(fused),
                "variant_scores_q": variant_scores,
            }
        )
    best.sort(key=lambda item: (-int(item["score_q"]), ord(item["text"])))
    top = best[0]
    second = best[1] if len(best) > 1 else {"text": None, "score_q": 0}
    tied = len(best) > 1 and int(second["score_q"]) == int(top["score_q"])
    tied_characters = [
        item["text"] for item in best if int(item["score_q"]) == int(top["score_q"])
    ]
    # The list is already code-point ordered, so the first entry is the tie-break winner.
    return {
        "text": top["text"],
        "score_q": int(top["score_q"]),
        "score": round(int(top["score_q"]) / scale, 8),
        "second_text": second["text"],
        "second_score_q": int(second["score_q"]),
        "margin_q": int(top["score_q"]) - int(second["score_q"]),
        "tie": bool(tied),
        "tied_characters": tied_characters,
        "variant_scores_q": top["variant_scores_q"],
        "metric": config.match_metric,
        "quant_levels": int(levels),
        "fixed_bits": int(config.fixed_bits),
    }


def confidence_details(
    match: dict[str, Any],
    regression: dict[str, Any],
    config: Config,
) -> dict[str, Any]:
    scale = config.fixed_scale
    tolerance_q = min(scale, max(0, int(round(config.confidence_tolerance * scale))))
    score_q = int(match["score_q"])
    residual = int(regression["residual"])
    residual_denominator = max(1, config.border_residual_tolerance * 4)
    raw_border_penalty = min(scale, residual * scale // residual_denominator)
    border_penalty = raw_border_penalty * 25 // 100
    split_record = regression.get("split_record", {})
    split_count = int(split_record.get("split_count", 0))
    split_total_cost = int(split_record.get("total_cost", 0))
    if split_count > 0:
        split_denominator = max(1, config.split_penalty * 50 + 1)
        raw_split_penalty = min(scale, split_total_cost * scale // split_denominator)
        split_penalty = raw_split_penalty * 20 // 100
    else:
        raw_split_penalty = 0
        split_penalty = 0
    after_border = score_q - border_penalty
    after_split = after_border - split_penalty
    low_q = max(0, min(scale, int(config.confidence_low_q)))
    high_q = scale if config.confidence_high_q is None else max(low_q, min(scale, int(config.confidence_high_q)))
    final_q = max(low_q, min(high_q, after_split))
    reasons: list[dict[str, Any]] = []
    if match.get("tie"):
        reasons.append(
            {
                "code": "TIE_BREAK",
                "detail": "".join(match.get("tied_characters", [])) or str(match.get("text")),
            }
        )
    if residual > config.border_residual_tolerance:
        reasons.append(
            {
                "code": "BORDER_RESIDUAL",
                "value": residual,
                "tolerance": config.border_residual_tolerance,
            }
        )
    if split_count > 0:
        reasons.append(
            {
                "code": "SPLIT_BLOCK",
                "split_count": split_count,
                "total_cost": split_total_cost,
            }
        )
    if score_q < tolerance_q:
        reasons.append(
            {
                "code": "LOW_MATCH_SCORE",
                "score_q": score_q,
                "tolerance_q": tolerance_q,
            }
        )
    if final_q < tolerance_q:
        reasons.append(
            {
                "code": "LOW_CONFIDENCE",
                "confidence_q": final_q,
                "tolerance_q": tolerance_q,
            }
        )
    low_confidence = bool(reasons)
    return {
        "score_q": score_q,
        "border_residual": residual,
        "raw_border_penalty_q": int(raw_border_penalty),
        "border_penalty_q": int(border_penalty),
        "split_count": int(split_count),
        "split_total_cost": int(split_total_cost),
        "raw_split_penalty_q": int(raw_split_penalty),
        "split_penalty_q": int(split_penalty),
        "after_border_q": int(after_border),
        "after_split_q": int(after_split),
        "confidence_q": int(final_q),
        "confidence": round(final_q / scale, 8),
        "tolerance_q": int(tolerance_q),
        "low_confidence": low_confidence,
        "low_reasons": reasons,
        "operation_order": [
            "score_q",
            "subtract_border_penalty",
            "subtract_split_penalty",
            "clip_confidence_interval",
        ],
    }


def block_digest(block: np.ndarray) -> str:
    arr = np.ascontiguousarray(block, dtype=np.uint8)
    return sha256(f"{arr.shape[0]}x{arr.shape[1]}:".encode("ascii") + arr.tobytes()).hexdigest()
