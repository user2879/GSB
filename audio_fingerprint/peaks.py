from __future__ import annotations

from typing import Any

from .canonical import fingerprint
from .config import peak_rule_fingerprint
from .spectrogram import peak_product_id


PEAK_PRODUCT = "peak_set"


def frame_threshold(config: dict[str, Any], clip_id: str, frame_index: int) -> int:
    overrides = config.get("frame_threshold_overrides", {})
    clip_overrides = overrides.get(clip_id, {}) if isinstance(overrides, dict) else {}
    value = clip_overrides.get(str(frame_index))
    if value is None:
        value = clip_overrides.get(frame_index) if isinstance(clip_overrides, dict) else None
    return int(value if value is not None else config["peak_threshold"])


def peak_cache_key(
    clip: dict[str, Any],
    frame_index: int,
    threshold: int,
    neighbor_frames: list[dict[str, Any]],
    config: dict[str, Any],
) -> str:
    return fingerprint(
        {
            "product_type": PEAK_PRODUCT,
            "product_id": peak_product_id(clip["clip_id"], frame_index),
            "frame_cache_key": next(
                frame["cache_key"] for frame in neighbor_frames if frame["frame_index"] == frame_index
            ),
            "neighbor_frame_keys": [
                {"frame_index": frame["frame_index"], "cache_key": frame["cache_key"]}
                for frame in sorted(neighbor_frames, key=lambda item: item["frame_index"])
            ],
            "config_fingerprint": peak_rule_fingerprint(config, threshold),
            "threshold_q": threshold,
        },
        "pkey_",
    )


def extract_peaks(
    clip: dict[str, Any],
    frame_index: int,
    neighbor_frames: list[dict[str, Any]],
    config: dict[str, Any],
) -> dict[str, Any]:
    threshold = frame_threshold(config, clip["clip_id"], frame_index)
    current = next(frame for frame in neighbor_frames if frame["frame_index"] == frame_index)
    radius = int(config["peak_neighborhood_radius"])
    band_start = int(config["peak_band_start"])
    band_end = int(config["peak_band_end"])
    candidates: list[dict[str, Any]] = []

    for band_index in range(band_start, band_end):
        strength = int(current["bands_q"][band_index])
        if strength < threshold:
            continue
        is_local_max = True
        for time_offset in range(-radius, radius + 1):
            neighbor_index = frame_index + time_offset
            neighbor = next(
                (item for item in neighbor_frames if item["frame_index"] == neighbor_index),
                None,
            )
            if neighbor is None:
                continue
            for band_offset in range(-radius, radius + 1):
                compare_band = band_index + band_offset
                if compare_band < 0 or compare_band >= len(neighbor["bands_q"]):
                    continue
                if time_offset == 0 and band_offset == 0:
                    continue
                if int(neighbor["bands_q"][compare_band]) > strength:
                    is_local_max = False
                    break
            if not is_local_max:
                break
        if is_local_max:
            candidates.append(
                {
                    "peak_id": f"peak:{clip['clip_id']}:{frame_index:06d}:{band_index:02d}",
                    "frame_index": frame_index,
                    "band_index": band_index,
                    "coordinates_q": [frame_index, band_index],
                    "strength_q": strength,
                }
            )

    candidates.sort(key=lambda peak: (-peak["strength_q"], peak["band_index"], peak["peak_id"]))
    selected = candidates[: int(config["max_peaks_per_frame"])]
    selected.sort(key=lambda peak: (peak["frame_index"], peak["band_index"]))
    cache_key = peak_cache_key(clip, frame_index, threshold, neighbor_frames, config)
    frame_refs = [
        {"type": "spec_frame", "id": frame["product_id"], "key": frame["cache_key"]}
        for frame in sorted(neighbor_frames, key=lambda item: item["frame_index"])
    ]
    return {
        "product_type": PEAK_PRODUCT,
        "product_id": peak_product_id(clip["clip_id"], frame_index),
        "cache_key": cache_key,
        "clip_id": clip["clip_id"],
        "track_id": clip["track_id"],
        "version": int(clip["version"]),
        "frame_index": frame_index,
        "threshold_q": threshold,
        "peaks": selected,
        "candidate_count": len(candidates),
        "upstream_refs": frame_refs
        + [{"type": "peak_rule_config", "key": peak_rule_fingerprint(config, threshold)}],
    }
