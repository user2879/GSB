"""Block-gradient / colour-statistics feature extraction in integer space."""

from __future__ import annotations

from typing import Any

import numpy as np

from .fixed import digest, digest_bytes
from .pngio import PngImage


def feature_params_key(
    *,
    view_id: str,
    image_sha256: str,
    width: int,
    height: int,
    block_size: int,
    descriptor_shift: int,
    fixed_bits: int,
) -> str:
    return digest(
        {
            "stage": "features",
            "view": view_id,
            "image_sha256": image_sha256,
            "resolution": [width, height],
            "block_size": block_size,
            "descriptor_shift": descriptor_shift,
            "fixed_q": fixed_bits,
        }
    )


def _block_feature(
    rgb: np.ndarray,
    *,
    view_id: str,
    image_sha256: str,
    block_x: int,
    block_y: int,
    block_size: int,
    descriptor_shift: int,
) -> dict[str, Any] | None:
    height, width = rgb.shape[:2]
    x0 = block_x * block_size
    y0 = block_y * block_size
    x1 = min(width, x0 + block_size)
    y1 = min(height, y0 + block_size)
    if x1 - x0 < 2 or y1 - y0 < 2:
        return None
    block = rgb[y0:y1, x0:x1].astype(np.int64)
    count = int(block.shape[0] * block.shape[1])
    sums = block.sum(axis=(0, 1))
    mean = [int((int(value) + count // 2) // count) for value in sums]
    saturation = max(mean) - min(mean)
    if saturation < 12:
        return None

    horizontal = np.abs(np.diff(block, axis=1)).sum()
    vertical = np.abs(np.diff(block, axis=0)).sum()
    gradient_denominator = max(1, count * 3)
    gx = int((int(horizontal) + gradient_denominator // 2) // gradient_denominator)
    gy = int((int(vertical) + gradient_denominator // 2) // gradient_denominator)
    if gx + gy < 2:
        return None

    content_fp = digest_bytes(block.astype(np.uint8).tobytes())
    coord = [x0 + (x1 - x0) // 2, y0 + (y1 - y0) // 2]
    descriptor = {
        "rgb": [channel >> descriptor_shift for channel in mean],
        "gx": gx >> descriptor_shift,
        "gy": gy >> descriptor_shift,
    }
    feature_id = f"f{coord[0]:04d},{coord[1]:04d}"
    fingerprint = digest(
        {
            "stage": "feature",
            "view": view_id,
            "image_sha256": image_sha256,
            "block_size": block_size,
            "block": [block_x, block_y],
            "coord": coord,
            "descriptor": descriptor,
            "content_fp": content_fp,
        }
    )
    return {
        "id": feature_id,
        "coord": coord,
        "block": [block_x, block_y],
        "descriptor": descriptor,
        "mean_rgb": mean,
        "saturation": saturation,
        "gradient": [gx, gy],
        "content_fp": content_fp,
        "fingerprint": fingerprint,
    }


def extract_features(
    image: PngImage,
    *,
    view_id: str,
    block_size: int,
    descriptor_shift: int,
    fixed_bits: int,
) -> dict[str, Any]:
    height, width = image.rgb.shape[:2]
    features: list[dict[str, Any]] = []
    for block_y in range((height + block_size - 1) // block_size):
        for block_x in range((width + block_size - 1) // block_size):
            feature = _block_feature(
                image.rgb,
                view_id=view_id,
                image_sha256=image.raw_sha256,
                block_x=block_x,
                block_y=block_y,
                block_size=block_size,
                descriptor_shift=descriptor_shift,
            )
            if feature is not None:
                features.append(feature)
    features.sort(key=lambda item: (item["coord"][0], item["coord"][1], item["id"]))
    params_key = feature_params_key(
        view_id=view_id,
        image_sha256=image.raw_sha256,
        width=width,
        height=height,
        block_size=block_size,
        descriptor_shift=descriptor_shift,
        fixed_bits=fixed_bits,
    )
    digest_value = digest(
        {
            "params_key": params_key,
            "features": [
                {"id": feature["id"], "coord": feature["coord"], "fingerprint": feature["fingerprint"]}
                for feature in features
            ],
        }
    )
    return {
        "view_id": view_id,
        "image_sha256": image.raw_sha256,
        "resolution": [width, height],
        "block_size": block_size,
        "descriptor_shift": descriptor_shift,
        "params_key": params_key,
        "cache_key": digest_value,
        "feature_count": len(features),
        "features": features,
    }
