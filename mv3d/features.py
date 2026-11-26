"""Deterministic block-gradient/color feature extraction in integer space."""

from __future__ import annotations

import io
import math
from typing import Any

import numpy as np
from PIL import Image

from .constants import ENGINE_VERSION
from .fixed import bytes_hash, stable_hash
from .models import ReconstructionConfig, ValidatedView


def feature_key(
    view: ValidatedView,
    config: ReconstructionConfig,
    strategy: dict[str, Any],
) -> str:
    return stable_hash(
        {
            "engine_version": ENGINE_VERSION,
            "stage": "features",
            "input_key": view.input_key,
            "bits": config.bits,
            "quant_level": config.quant_level,
            "feature_block": strategy["feature_block"],
            "chroma_floor": strategy["descriptor_chroma_floor"],
        }
    )


def _load_rgb(view: ValidatedView) -> np.ndarray:
    with Image.open(io.BytesIO(view.image_bytes)) as image:
        image.load()
        return np.array(image.convert("RGB"), dtype=np.int64, copy=True)


def _candidate_for_block(
    pixels: np.ndarray,
    block_x: int,
    block_y: int,
    block: int,
    chroma_floor: int,
) -> dict[str, Any] | None:
    crop = pixels[block_y : block_y + block, block_x : block_x + block]
    if crop.shape[0] != block or crop.shape[1] != block:
        return None
    rgb_sum = crop.sum(axis=(0, 1), dtype=np.int64)
    count = block * block
    mean_r, mean_g, mean_b = (int(value // count) for value in rgb_sum)
    gray = (crop[:, :, 0] * 2 + crop[:, :, 1] * 3 + crop[:, :, 2] * 3) // 8
    horizontal = np.abs(gray[:, 1:] - gray[:, :-1]).sum(dtype=np.int64)
    vertical = np.abs(gray[1:, :] - gray[:-1, :]).sum(dtype=np.int64)
    gradient = int(horizontal + vertical)
    channel_max = crop.max(axis=2)
    channel_min = crop.min(axis=2)
    chroma_map = channel_max - channel_min
    maximum_chroma = int(chroma_map.max())
    if maximum_chroma < chroma_floor:
        return None
    if gradient < max(16, block * block):
        return None
    weights = np.maximum(chroma_map, 0)
    total_weight = int(weights.sum(dtype=np.int64))
    if total_weight <= 0:
        return None
    weighted_x = int((weights * np.arange(block, dtype=np.int64)[None, :]).sum(dtype=np.int64))
    weighted_y = int((weights * np.arange(block, dtype=np.int64)[:, None]).sum(dtype=np.int64))
    center_x = block_x + weighted_x // total_weight
    center_y = block_y + weighted_y // total_weight
    threshold = max(chroma_floor, maximum_chroma // 2)
    selected = crop[chroma_map >= threshold]
    selected_mean = selected.mean(axis=0).astype(np.int64) if len(selected) else rgb_sum // count
    descriptor = (
        int(selected_mean[0]),
        int(selected_mean[1]),
        int(selected_mean[2]),
        gradient // max(1, block),
        maximum_chroma,
    )
    score = gradient // max(1, block * block) + maximum_chroma * 4
    block_bytes = crop.astype(np.uint8).tobytes()
    return {
        "x": center_x,
        "y": center_y,
        "block_x": block_x,
        "block_y": block_y,
        "block": block,
        "descriptor": descriptor,
        "gradient": gradient,
        "max_chroma": maximum_chroma,
        "score": score,
        "content_fingerprint": bytes_hash(block_bytes),
        "mean_rgb": (mean_r, mean_g, mean_b),
    }


def _nms(candidates: list[dict[str, Any]], minimum_distance: int) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    squared_limit = minimum_distance * minimum_distance
    for candidate in sorted(
        candidates,
        key=lambda item: (-item["score"], item["x"], item["y"], item["block_x"], item["block_y"]),
    ):
        if all(
            (candidate["x"] - existing["x"]) ** 2 + (candidate["y"] - existing["y"]) ** 2
            >= squared_limit
            for existing in selected
        ):
            selected.append(candidate)
    return selected


def extract_features(
    view: ValidatedView,
    config: ReconstructionConfig,
    strategy: dict[str, Any],
) -> dict[str, Any]:
    pixels = _load_rgb(view)
    height, width = pixels.shape[:2]
    block = int(strategy["feature_block"])
    chroma_floor = int(strategy["descriptor_chroma_floor"])
    scan_blocks = (width // block) * (height // block)
    candidates: list[dict[str, Any]] = []
    for block_y in range(0, height - block + 1, block):
        for block_x in range(0, width - block + 1, block):
            candidate = _candidate_for_block(
                pixels, block_x, block_y, block, chroma_floor
            )
            if candidate is not None:
                candidates.append(candidate)
    minimum_distance = max(5, block * 5 // 6)
    selected = _nms(candidates, minimum_distance)
    points = []
    for candidate in selected:
        point = {
            "id": f"v:{view.view_id:04d}:{candidate['x']:04d}:{candidate['y']:04d}",
            "x": candidate["x"],
            "y": candidate["y"],
            "block_x": candidate["block_x"],
            "block_y": candidate["block_y"],
            "descriptor": list(candidate["descriptor"]),
            "gradient": candidate["gradient"],
            "max_chroma": candidate["max_chroma"],
            "score": candidate["score"],
            "content_fingerprint": candidate["content_fingerprint"],
        }
        points.append(point)
    points.sort(key=lambda item: (item["x"], item["y"], item["id"]))
    key = feature_key(view, config, strategy)
    multiply_adds = scan_blocks * block * block * 6 + len(candidates) * block * 4
    return {
        "view_id": view.view_id,
        "key": key,
        "input_key": view.input_key,
        "image_sha256": view.image_sha256,
        "strategy": {
            "name": strategy["name"],
            "feature_block": block,
            "chroma_floor": chroma_floor,
        },
        "scan_blocks": scan_blocks,
        "candidate_blocks": len(candidates),
        "points": points,
        "multiply_adds": multiply_adds,
    }


def cumulative_math(points: list[dict[str, Any]]) -> int:
    return sum(point["gradient"] + point["max_chroma"] * point["block"] for point in points)
