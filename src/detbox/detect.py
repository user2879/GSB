"""Deterministic sliding-window candidate detection on synthetic frames."""
from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

from .config import DetectionParams
from .errors import DetboxError
from .models import Candidate, FORMAT_V2


@dataclass
class EvictionEvent:
    sequence: str
    frame: int
    evicted_box: tuple[int, int, int, int]
    evicted_score: int
    reason: str
    pool_size: int

    def to_dict(self) -> dict:
        return {
            "sequence": self.sequence,
            "frame": self.frame,
            "evicted_box": list(self.evicted_box),
            "evicted_score": self.evicted_score,
            "reason": self.reason,
            "pool_size": self.pool_size,
        }


def load_frame_image(path: str | Path):
    try:
        image = Image.open(path)
        image.load()
    except Exception as exc:  # Pillow raises various subclasses on truncation
        raise DetboxError(f"cannot read PNG {path}: {exc}", code="E_IMAGE_TRUNCATED") from exc
    return image.convert("RGB")


def _window_features(array: np.ndarray, x: int, y: int, size: int) -> tuple[float, float, float]:
    patch = array[y : y + size, x : x + size, :].astype(np.float64)
    color_mean = float(patch.mean())
    gray = patch.mean(axis=2)
    if size <= 1:
        edge_density = 0.0
    else:
        dx = np.abs(np.diff(gray, axis=1))
        dy = np.abs(np.diff(gray, axis=0))
        edges = (dx > 30.0).sum() + (dy > 30.0).sum()
        edge_density = float(edges) / float(2 * size * (size - 1))
    chroma = float(np.abs(patch[:, :, 0] - patch[:, :, 1]).mean()) + float(
        np.abs(patch[:, :, 2] - patch[:, :, 1]).mean()
    )
    return color_mean, edge_density, chroma


def _pool_rank_key(candidate: Candidate):
    # Higher score first; ties broken by coordinate dictionary order ascending.
    return (-candidate.score, candidate.x1, candidate.y1, candidate.x2, candidate.y2)


def enforce_pool(
    candidates: list[Candidate],
    capacity: int,
    sequence: str,
    frame: int,
) -> tuple[list[Candidate], list[EvictionEvent]]:
    if len(candidates) <= capacity:
        return list(candidates), []
    ordered = sorted(candidates, key=_pool_rank_key)
    kept = ordered[:capacity]
    evicted = ordered[capacity:]
    events = [
        EvictionEvent(
            sequence=sequence,
            frame=frame,
            evicted_box=candidate.int_box_tuple(),
            evicted_score=candidate.int_score_value(),
            reason="pool_full_score_desc_coord_lex",
            pool_size=capacity,
        )
        for candidate in evicted
    ]
    return kept, events


def _iou(box_a, box_b) -> float:
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def non_max_suppress(candidates: list[Candidate], iou_limit: float = 0.5) -> list[Candidate]:
    ordered = sorted(candidates, key=_pool_rank_key)
    kept: list[Candidate] = []
    for candidate in ordered:
        if all(_iou(candidate.int_box_tuple(), other.int_box_tuple()) <= iou_limit for other in kept):
            kept.append(candidate)
    return kept


def _dominant_class(array: np.ndarray, x: int, y: int, size: int) -> int:
    patch = array[y : y + size, x : x + size, :].astype(np.int64)
    means = patch.reshape(-1, 3).mean(axis=0)
    r, g, b = means
    if r >= g and r >= b:
        return 0
    if g >= r and g >= b:
        return 1
    return 2


def detect_frame(
    image: Image.Image,
    params: DetectionParams,
    sequence: str,
    frame: int,
) -> tuple[list[Candidate], list[EvictionEvent]]:
    params.score_weights  # validated at load time; keep explicit dependency
    array = np.asarray(image, dtype=np.uint8)
    height, width = array.shape[:2]
    weights = params.score_weights
    w_color = float(weights.get("color", 0.0))
    w_edge = float(weights.get("edge", 0.0))
    weight_sum = w_color + w_edge
    if not math.isfinite(weight_sum) or weight_sum == 0:
        raise DetboxError(
            f"invalid weight sum {weight_sum!r}", code="E_WEIGHT_NONFINITE"
        )
    raw: list[Candidate] = []
    for size in params.window_sizes:
        for y in range(0, height - size + 1, params.step):
            for x in range(0, width - size + 1, params.step):
                color_mean, edge_density, chroma = _window_features(array, x, y, size)
                if color_mean < params.color_mean_threshold:
                    continue
                # Flat filled shapes have no internal edges: accept either a
                # strong boundary response or a saturated chromatic target.
                if edge_density < params.edge_density_threshold and chroma < 40.0:
                    continue
                color_term = min(1.0, color_mean / 255.0)
                edge_term = min(1.0, edge_density)
                score = (w_color * color_term + w_edge * edge_term) / weight_sum
                score = max(0.0, min(1.0, score))
                class_code = _dominant_class(array, x, y, size)
                raw.append(
                    Candidate(
                        x1=float(x),
                        y1=float(y),
                        x2=float(x + size),
                        y2=float(y + size),
                        score=score,
                        class_code=class_code,
                        source=FORMAT_V2,
                        precision_label="detected-float",
                        int_box=(x, y, x + size, y + size),
                        int_score=int(round(score * 1000.0)),
                    )
                )
    raw = non_max_suppress(raw)
    kept, events = enforce_pool(raw, params.pool_capacity, sequence, frame)
    kept.sort(key=lambda c: (c.x1, c.y1, c.x2, c.y2, -c.score, c.class_code))
    return kept, events
