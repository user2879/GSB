from __future__ import annotations

from typing import Any

import numpy as np

from .config import RecognitionConfig
from .font import FONT_VERSION, font_templates

CANVAS_SIZE = 24


def _resize_binary(array: np.ndarray, height: int, width: int) -> np.ndarray:
    if array.shape == (height, width):
        return array.astype(np.uint8, copy=False)
    src_h, src_w = array.shape
    if src_h == 0 or src_w == 0:
        return np.zeros((height, width), dtype=np.uint8)
    ys = (np.arange(height, dtype=np.int64) * src_h) // height
    xs = (np.arange(width, dtype=np.int64) * src_w) // width
    return array[ys[:, None], xs[None, :]].astype(np.uint8, copy=False)


def _normalize_canvas(array: np.ndarray, canvas_size: int = CANVAS_SIZE) -> np.ndarray:
    """Aspect-preserving nearest-neighbour normalization with centering."""
    src_h, src_w = array.shape
    if src_h == 0 or src_w == 0:
        return np.zeros((canvas_size, canvas_size), dtype=np.uint8)
    scale = min(canvas_size / src_h, canvas_size / src_w)
    height = max(1, min(canvas_size, int(round(src_h * scale))))
    width = max(1, min(canvas_size, int(round(src_w * scale))))
    resized = _resize_binary(array, height, width)
    result = np.zeros((canvas_size, canvas_size), dtype=np.uint8)
    y = (canvas_size - height) // 2
    x = (canvas_size - width) // 2
    result[y : y + height, x : x + width] = resized
    return result


def _metric_score(
    block: np.ndarray,
    template: np.ndarray,
    metric: str,
) -> float:
    if block.size == 0 or template.size == 0:
        return 0.0
    b = (block > 0).astype(np.float64)
    t = (template > 0).astype(np.float64)
    if metric == "l1":
        return float(1.0 - np.mean(np.abs(b - t)))
    if metric == "xor":
        mismatch = float(np.count_nonzero(b != t))
        union = float(np.count_nonzero((b > 0) | (t > 0)))
        return 1.0 if union == 0 else float(1.0 - mismatch / union)
    if metric == "cosine":
        denominator = float(np.linalg.norm(b) * np.linalg.norm(t))
        return 0.0 if denominator == 0 else float(np.dot(b.ravel(), t.ravel()) / denominator)
    if metric == "ink_balance":
        block_density = float(b.mean())
        template_density = float(t.mean())
        block_aspect = block.shape[1] / max(1, block.shape[0])
        template_aspect = template.shape[1] / max(1, template.shape[0])
        aspect_scale = max(block_aspect, template_aspect, 1.0)
        score = 1.0 - 0.78 * abs(block_density - template_density)
        score -= 0.22 * (abs(block_aspect - template_aspect) / aspect_scale)
        return float(max(0.0, score))
    raise ValueError(f"unsupported match metric: {metric}")


def match_point_cost() -> int:
    return len(font_templates())


def recognize_segment(
    gray: np.ndarray,
    bbox: list[int],
    config: RecognitionConfig,
    *,
    metric_override: str | None = None,
) -> dict[str, Any]:
    x0, y0, x1, y1 = bbox
    crop = (gray[y0:y1, x0:x1] <= config.binarization_threshold).astype(np.uint8)
    metric = metric_override or config.match_metric
    normalized_block = _normalize_canvas(crop)
    scored: list[dict[str, Any]] = []
    for char, template in font_templates().items():
        normalized_template = _normalize_canvas(template)
        score = round(_metric_score(normalized_block, normalized_template, metric), 6)
        scored.append({"char": char, "codepoint": ord(char), "score": score})
    scored.sort(key=lambda item: (-item["score"], item["codepoint"]))
    selected = scored[0]
    top_score = selected["score"]
    candidates: list[dict[str, Any]] = []
    for rank, candidate in enumerate(scored[: config.top_k_candidates]):
        if rank == 0:
            reason = "selected_highest_score"
        elif candidate["score"] == top_score:
            reason = "tie_lost_lower_codepoint"
        else:
            reason = "lower_score"
        candidates.append(
            {
                "rank": rank + 1,
                "char": candidate["char"],
                "codepoint": candidate["codepoint"],
                "score": candidate["score"],
                "eliminated_reason": reason,
            }
        )
    return {
        "stage": "recognition",
        "font_version": FONT_VERSION,
        "metric": metric,
        "point_cost": match_point_cost(),
        "bbox": [int(x0), int(y0), int(x1), int(y1)],
        "text": selected["char"],
        "score": top_score,
        "tie_break": "codepoint_ascending",
        "candidates": candidates,
        "candidate_count": len(scored),
        "eliminated_beyond_top_k": max(0, len(scored) - len(candidates)),
    }
