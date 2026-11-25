"""Deterministic sliding-window coarse candidate localization."""

from __future__ import annotations

import hashlib
from typing import Any

import numpy as np

from .config import EffectiveConfig
from .constants import (
    DEFAULT_MIN_WINDOW_DENSITY_Q,
    DEFAULT_WINDOW_CUTOFF,
    SCORE_SCALE,
)


def stable_window_starts(length: int, window: int, step: int) -> list[int]:
    if window <= 0 or step <= 0:
        return []
    starts = list(range(0, max(0, length - window) + 1, step))
    final = length - window
    if final >= 0 and final not in starts:
        starts.append(final)
    return starts


def _window_projection(image_fp: str, x: int, y: int, window: tuple[int, int], step: tuple[int, int]) -> int:
    digest = hashlib.sha256(
        f"{image_fp}|{x}|{y}|{window[0]}|{window[1]}|{step[0]}|{step[1]}".encode("ascii")
    ).digest()
    raw = int.from_bytes(digest[:4], "big")
    return raw % 49 - 24


def _iou(a: dict[str, int], b: dict[str, int]) -> float:
    x0 = max(a["x0"], b["x0"])
    y0 = max(a["y0"], b["y0"])
    x1 = min(a["x1"], b["x1"])
    y1 = min(a["y1"], b["y1"])
    if x1 <= x0 or y1 <= y0:
        return 0.0
    intersection = (x1 - x0) * (y1 - y0)
    area_a = (a["x1"] - a["x0"]) * (a["y1"] - a["y0"])
    area_b = (b["x1"] - b["x0"]) * (b["y1"] - b["y0"])
    return intersection / float(area_a + area_b - intersection)


def run_sliding_stage(
    image_name: str,
    image_fp: str,
    grayscale: np.ndarray,
    effective: EffectiveConfig,
) -> dict[str, Any]:
    height, width = grayscale.shape
    window = effective.window
    xs = stable_window_starts(width, window.width, window.step_x)
    ys = stable_window_starts(height, window.height, window.step_y)
    ink = grayscale < effective.cc.ink_threshold
    gradients = np.abs(np.diff(grayscale.astype(np.int16), axis=0)).astype(np.int32)
    gradients_x = np.abs(np.diff(grayscale.astype(np.int16), axis=1)).astype(np.int32)

    scored: list[dict[str, Any]] = []
    for y in ys:
        for x in xs:
            crop = ink[y : y + window.height, x : x + window.width]
            area = int(crop.size)
            ink_count = int(crop.sum())
            density_q = int(round(ink_count * SCORE_SCALE / area)) if area else 0
            row_profile = crop.sum(axis=1).astype(np.float64) / max(1, window.width)
            if row_profile.size >= 3:
                kernel = np.ones(3, dtype=np.float64) / 3.0
                smooth = np.convolve(row_profile, kernel, mode="same")
                local = smooth[1:-1] if smooth.size > 2 else smooth
                valley_ratio = float(local.min()) if local.size else float(row_profile.min())
            else:
                valley_ratio = float(row_profile.min()) if row_profile.size else 0.0
            valley_q = int(round(max(0.0, min(1.0, 1.0 - valley_ratio)) * SCORE_SCALE))
            gradient_crop = gradients[y : y + window.height, x : x + window.width]
            gradient_x_crop = gradients_x[y : y + window.height, x : x + window.width]
            gradient_mean = float(gradient_crop.mean() + gradient_x_crop.mean()) / 2.0 if area else 0.0
            gradient_q = int(round(min(SCORE_SCALE, gradient_mean * SCORE_SCALE / 255.0)))
            projection_q = _window_projection(image_fp, x, y, (window.width, window.height), (window.step_x, window.step_y))
            raw_score = (density_q * 4 + valley_q * 2 + gradient_q * 3) / 10_000.0
            score_q = int(round(raw_score + projection_q))
            scored.append(
                {
                    "window_id": "",
                    "x0": int(x),
                    "y0": int(y),
                    "x1": int(x + window.width),
                    "y1": int(y + window.height),
                    "ink_pixels": ink_count,
                    "density_q": density_q,
                    "valley_q": valley_q,
                    "gradient_q": gradient_q,
                    "projection_q": int(projection_q),
                    "score_q": score_q,
                }
            )

    scored.sort(key=lambda item: (item["y0"], item["x0"], item["window_id"]))
    for index, item in enumerate(scored):
        item["window_id"] = f"w{index:06d}"

    eligible = [
        item
        for item in scored
        if item["density_q"] >= DEFAULT_MIN_WINDOW_DENSITY_Q and item["score_q"] >= DEFAULT_WINDOW_CUTOFF
    ]
    ranked = sorted(eligible, key=lambda item: (-item["score_q"], item["y0"], item["x0"], item["window_id"]))
    selected: list[dict[str, Any]] = []
    for item in ranked:
        if any(_iou(item, other) >= 0.5 for other in selected):
            continue
        selected.append(item)
    selected.sort(key=lambda item: (item["y0"], item["x0"], item["window_id"]))

    candidates: list[dict[str, Any]] = []
    for index, item in enumerate(selected):
        candidates.append(
            {
                "candidate_id": f"cand{index:06d}",
                "window_id": item["window_id"],
                "x0": item["x0"],
                "y0": item["y0"],
                "x1": item["x1"],
                "y1": item["y1"],
                "score_q": item["score_q"],
                "density_q": item["density_q"],
                "valley_q": item["valley_q"],
                "gradient_q": item["gradient_q"],
                "projection_q": item["projection_q"],
            }
        )
    return {
        "image": image_name,
        "image_fp": image_fp,
        "window": window.as_dict(),
        "window_count": len(scored),
        "eligible_count": len(eligible),
        "candidate_count": len(candidates),
        "candidates": candidates,
    }


def window_count(width: int, height: int, window_size: tuple[int, int], step: tuple[int, int]) -> int:
    return len(stable_window_starts(width, window_size[0], step[0])) * len(
        stable_window_starts(height, window_size[1], step[1])
    )
