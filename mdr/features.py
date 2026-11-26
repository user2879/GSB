from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from .utils import fingerprint, quantize_fixed, round_div, sha256_bytes


def feature_param_fingerprint(config: dict[str, Any]) -> str:
    return fingerprint(
        {
            "extractor_version": 2,
            "patch_radius": config["feature_patch_radius"],
            "threshold": config["feature_score_threshold"],
            "nms_distance": config["feature_nms_distance"],
            "frac_bits": config["frac_bits"],
            "quant_level": config["quant_level"],
            "quant_shift": config["quant_shifts"][config["quant_level"]],
            "width": config["width"],
            "height": config["height"],
            "mode": config["mode"],
        }
    )


def feature_cache_key(view_id: str, image_sha256: str, config: dict[str, Any]) -> str:
    return fingerprint(
        {
            "stage": "features",
            "view_id": view_id,
            "image_sha256": image_sha256,
            "param_fingerprint": feature_param_fingerprint(config),
        }
    )


def _patch_descriptor(
    pixels: np.ndarray,
    x: int,
    y: int,
    radius: int,
    scale: int,
    shift: int,
) -> tuple[list[int], int, str, dict[str, Any]]:
    size = radius * 2 + 1
    patch = pixels[y - radius : y + radius + 1, x - radius : x + radius + 1, :]
    raw_bytes = bytes(int(v) for row in patch for value in row for v in value)
    content_fingerprint = sha256_bytes(raw_bytes)

    descriptor: list[int] = []
    descriptor_q: list[int] = []
    quadrants = [
        (-radius, 0, -radius, 0),
        (0, radius + 1, -radius, 0),
        (-radius, 0, 0, radius + 1),
        (0, radius + 1, 0, radius + 1),
    ]
    for x0, x1, y0, y1 in quadrants:
        block = patch[y0 + radius : y1 + radius, x0 + radius : x1 + radius, :]
        count = int(block.shape[0] * block.shape[1])
        for channel in range(3):
            total = int(sum(int(block[by, bx, channel]) for by in range(block.shape[0]) for bx in range(block.shape[1])))
            mean_q, _ = round_div(total * scale, count)
            mean_q, _ = quantize_fixed(mean_q, shift)
            descriptor_q.append(int(mean_q))
            descriptor.append(int(mean_q // max(1, scale // 32)))

    gradient_score = 0
    gradient_values: list[int] = []
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            if dx < radius:
                left = pixels[y + dy, x + dx, :]
                right = pixels[y + dy, x + dx + 1, :]
                value = sum(abs(int(right[c]) - int(left[c])) for c in range(3))
                gradient_score += value
                gradient_values.append(value)
            if dy < radius:
                top = pixels[y + dy, x + dx, :]
                bottom = pixels[y + dy + 1, x + dx, :]
                value = sum(abs(int(bottom[c]) - int(top[c])) for c in range(3))
                gradient_score += value
                gradient_values.append(value)

    gradient_q, score_dir = quantize_fixed(gradient_score * scale, shift)
    descriptor_q.append(int(gradient_q))
    descriptor.extend([min(255, int(gradient_score) // 64), len(gradient_values)])
    return descriptor, gradient_score, content_fingerprint, {
        "gradient_q": gradient_q,
        "descriptor_q": descriptor_q,
    }


def extract_features(
    view: dict[str, Any],
    image_sha256: str,
    config: dict[str, Any],
) -> dict[str, Any]:
    view_id = str(view["view_id"])
    image_path = Path(str(view["image_abs"]))
    radius = int(config["feature_patch_radius"])
    threshold = int(config["feature_score_threshold"])
    nms_distance = int(config["feature_nms_distance"])
    bits = int(config["frac_bits"])
    scale = 1 << bits
    shift = int(config["quant_shifts"][config["quant_level"]])

    with Image.open(image_path) as image:
        rgb = image.convert("RGB")
        pixels = np.asarray(rgb, dtype=np.int64)
    height, width, _ = pixels.shape

    candidates: list[dict[str, Any]] = []
    for y in range(radius, height - radius):
        for x in range(radius, width - radius):
            descriptor, score, patch_hash, extra = _patch_descriptor(
                pixels, x, y, radius, scale, shift
            )
            if score < threshold:
                continue
            score_q, score_dir = quantize_fixed(score * scale, shift)
            candidates.append(
                {
                    "x": int(x),
                    "y": int(y),
                    "score": int(score),
                    "score_q": score_q,
                    "score_round_dir": score_dir,
                    "gradient_q": extra["gradient_q"],
                    "descriptor": descriptor,
                    "descriptor_q": extra["descriptor_q"],
                    "content_fingerprint": patch_hash,
                }
            )

    ranked = sorted(candidates, key=lambda item: (-int(item["score"]), int(item["x"]), int(item["y"])))
    selected: list[dict[str, Any]] = []
    for candidate in ranked:
        if all(
            max(abs(candidate["x"] - other["x"]), abs(candidate["y"] - other["y"])) >= nms_distance
            for other in selected
        ):
            selected.append(candidate)

    selected.sort(key=lambda item: (int(item["x"]), int(item["y"])))
    features: list[dict[str, Any]] = []
    for index, item in enumerate(selected):
        feature_id = f"{view_id}:f{index:03d}"
        feature_key = fingerprint(
            {
                "stage": "feature-item",
                "view": view_id,
                "image_sha256": image_sha256,
                "xy": [item["x"], item["y"]],
                "content": item["content_fingerprint"],
                "score_q": item["score_q"],
                "quant_level": config["quant_level"],
                "frac_bits": bits,
            }
        )
        features.append(
            {
                "feature_id": feature_id,
                "feature_index": index,
                "x": item["x"],
                "y": item["y"],
                "xy_lex": [item["x"], item["y"]],
                "score": item["score"],
                "score_q": item["score_q"],
                "score_round_dir": item["score_round_dir"],
                "descriptor": item["descriptor"],
                "descriptor_q": item["descriptor_q"],
                "content_fingerprint": item["content_fingerprint"],
                "feature_cache_key": feature_key,
            }
        )

    param_fingerprint = feature_param_fingerprint(config)
    cache_key = feature_cache_key(view_id, image_sha256, config)
    return {
        "stage": "features",
        "view_id": view_id,
        "image_sha256": image_sha256,
        "param_fingerprint": param_fingerprint,
        "cache_key": cache_key,
        "feature_count": len(features),
        "features": features,
    }
