from __future__ import annotations

import struct
from collections.abc import Iterable

import numpy as np

from .io_utils import canonical_json_bytes, sha256_bytes


def mask_to_runs(mask: np.ndarray) -> list[tuple[int, int]]:
    flat = np.asarray(mask, dtype=bool).reshape(-1)
    if flat.size == 0:
        return []
    padded = np.concatenate(([False], flat, [False]))
    changes = np.flatnonzero(padded[1:] != padded[:-1])
    starts = changes[0::2]
    ends = changes[1::2]
    return [(int(start), int(end - start)) for start, end in zip(starts, ends)]


def validate_runs(
    runs: Iterable[tuple[int, int]],
    pixel_count: int,
) -> list[tuple[int, int]]:
    normalized = [(int(start), int(length)) for start, length in runs]
    previous_end = 0
    for index, (start, length) in enumerate(normalized):
        if start < 0 or length <= 0:
            raise ValueError(f"run {index} has invalid start/length")
        end = start + length
        if end > pixel_count:
            raise ValueError(f"run {index} exceeds the image bounds")
        if index and start < previous_end:
            raise ValueError(f"run {index} overlaps or is unsorted")
        previous_end = end
    return normalized


def runs_to_mask(
    runs: Iterable[tuple[int, int]],
    pixel_count: int,
) -> np.ndarray:
    validated = validate_runs(runs, pixel_count)
    flat = np.zeros(pixel_count, dtype=bool)
    for start, length in validated:
        flat[start : start + length] = True
    return flat


def run_content_fingerprint(
    runs: Iterable[tuple[int, int]],
    *,
    width: int,
    height: int,
) -> str:
    normalized = [(int(start), int(length)) for start, length in runs]
    encoded = b"".join(struct.pack("<II", start, length) for start, length in normalized)
    return sha256_bytes(
        canonical_json_bytes(
            {
                "width": int(width),
                "height": int(height),
                "run_count": len(normalized),
            }
        )
        + encoded
    )
