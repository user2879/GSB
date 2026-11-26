from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from .canonical import fingerprint
from .wavio import read_wav_samples


def normalize_clip(
    clip: dict[str, Any],
    data_dir: str | Path,
    internal_sample_rate: int,
) -> tuple[dict[str, Any], np.ndarray, list[dict[str, Any]]]:
    path = Path(data_dir) / clip["path"]
    wav_info, samples, issues = read_wav_samples(path)
    if issues:
        return {}, np.zeros(0, dtype=np.float64), issues

    source_rate = int(wav_info["sample_rate"])
    method = "none"
    factor = 1
    if source_rate == internal_sample_rate:
        normalized = samples.astype(np.float64, copy=True)
        method = "identity"
    elif source_rate % internal_sample_rate == 0:
        factor = source_rate // internal_sample_rate
        usable = samples.shape[0] - (samples.shape[0] % factor)
        trimmed = samples[:usable]
        normalized = trimmed.reshape(-1, factor).mean(axis=1)
        method = "integer_block_average"
    else:
        ratio = internal_sample_rate / source_rate
        output_count = max(1, int(np.floor((samples.shape[0] - 1) * ratio)) + 1)
        source_positions = np.arange(output_count, dtype=np.float64) * source_rate / internal_sample_rate
        left = np.floor(source_positions).astype(np.int64)
        right = np.minimum(left + 1, samples.shape[0] - 1)
        fraction = source_positions - left
        normalized = samples[left] * (1.0 - fraction) + samples[right] * fraction
        factor = 0
        method = "deterministic_linear"

    grid = np.rint(np.clip(normalized, -1.0, 127.0 / 128.0) * 128.0).astype(np.int16)
    normalized_fp = fingerprint(
        {
            "internal_sample_rate": internal_sample_rate,
            "sample_count": int(grid.shape[0]),
            "grid_sha256": __import__("hashlib").sha256(np.ascontiguousarray(grid).tobytes()).hexdigest(),
        },
        "norm_",
    )
    receipt = {
        "clip_id": clip["clip_id"],
        "track_id": clip["track_id"],
        "version": int(clip["version"]),
        "role": clip.get("role", "library"),
        "original_sample_rate": source_rate,
        "original_bit_depth": int(wav_info["bit_depth"]),
        "original_channels": int(wav_info["channels"]),
        "original_format_fingerprint": clip["format_fingerprint"],
        "internal_sample_rate": internal_sample_rate,
        "resample_method": method,
        "resample_ratio": internal_sample_rate / source_rate,
        "ratio_numerator": internal_sample_rate,
        "ratio_denominator": source_rate,
        "integer_downsample_factor": factor,
        "source_sample_count": int(samples.shape[0]),
        "normalized_sample_count": int(normalized.shape[0]),
        "normalized_content_fingerprint": normalized_fp,
    }
    return receipt, normalized, []
