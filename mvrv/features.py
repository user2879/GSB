from __future__ import annotations

from typing import Any

import numpy as np

from .hashing import sha256_object, short_fingerprint


def feature_cache_key(view_index: int, image_sha256: str, strategy: dict[str, Any]) -> str:
    return sha256_object(
        {
            "stage": "view_features",
            "view_index": view_index,
            "image_sha256": image_sha256,
            "strategy_fingerprint": strategy["fingerprint"],
            "extractor": "block_gradient_color_v1",
        }
    )


def _luma(r: int, g: int, b: int) -> int:
    return (299 * r + 587 * g + 114 * b) // 1000


def extract_view_features(
    view: dict[str, Any], image_sha256: str, strategy: dict[str, Any]
) -> dict[str, Any]:
    pixels: np.ndarray = view["pixels"]
    height, width, _ = pixels.shape
    block_size = int(strategy["block_size"])
    features: list[dict[str, Any]] = []

    for y0 in range(0, height - block_size + 1, block_size):
        for x0 in range(0, width - block_size + 1, block_size):
            patch = pixels[y0 : y0 + block_size, x0 : x0 + block_size, :].astype(np.int64)
            rgb_sum = patch.reshape(-1, 3).sum(axis=0)
            pixel_count = block_size * block_size
            r_avg = int(rgb_sum[0]) // pixel_count
            g_avg = int(rgb_sum[1]) // pixel_count
            b_avg = int(rgb_sum[2]) // pixel_count
            luma = np.apply_along_axis(lambda px: _luma(int(px[0]), int(px[1]), int(px[2])), 2, patch)
            luma_avg = int(luma.sum()) // pixel_count

            gradient_sum = 0
            for yy in range(block_size):
                for xx in range(block_size - 1):
                    gradient_sum += abs(int(luma[yy, xx + 1]) - int(luma[yy, xx]))
            for yy in range(block_size - 1):
                for xx in range(block_size):
                    gradient_sum += abs(int(luma[yy + 1, xx]) - int(luma[yy, xx]))

            spread_sum = int(abs(patch - np.array([r_avg, g_avg, b_avg], dtype=np.int64)).sum())
            activity = (gradient_sum + spread_sum // 3) // pixel_count
            if activity < int(strategy["feature_threshold"]):
                continue

            descriptor = [
                r_avg,
                g_avg,
                b_avg,
                luma_avg,
                gradient_sum // max(1, 2 * block_size * (block_size - 1)),
                spread_sum // (3 * pixel_count),
                int(activity),
            ]
            coordinate = [x0, y0]
            fingerprint = short_fingerprint(
                {
                    "coordinate": coordinate,
                    "descriptor": descriptor,
                    "patch_sha": short_fingerprint(patch.tobytes(), 32),
                    "strategy_fingerprint": strategy["fingerprint"],
                },
                20,
            )
            features.append(
                {
                    "view_index": int(view["index"]),
                    "xy": coordinate,
                    "block_size": block_size,
                    "activity_q": activity * 1024,
                    "descriptor": descriptor,
                    "content_fingerprint": fingerprint,
                    "sort_key": [int(view["index"]), x0, y0],
                }
            )

    features.sort(key=lambda item: (item["xy"][0], item["xy"][1]))
    for order, feature in enumerate(features):
        feature["lex_order"] = order
    feature_fingerprint = feature_cache_key(int(view["index"]), image_sha256, strategy)
    return {
        "view_index": int(view["index"]),
        "feature_count": len(features),
        "features": features,
        "image_sha256": image_sha256,
        "cache_key": feature_fingerprint,
        "upstream": {
            "image_sha256": image_sha256,
            "strategy_fingerprint": strategy["fingerprint"],
        },
        "strategy_id": strategy["id"],
        "sort_key": [int(view["index"])],
    }
