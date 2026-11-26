from __future__ import annotations

from typing import Any

from .canonical import fingerprint
from .config import pair_fingerprint_config


PAIR_PRODUCT = "fingerprint_pair"


def quantize_time_delta(delta: int, gear: str, step: int) -> dict[str, Any]:
    if step <= 1:
        quantized = int(delta)
    else:
        quantized = ((int(delta) + step // 2) // step) * step
    if quantized > delta:
        direction = "up"
    elif quantized < delta:
        direction = "down"
    else:
        direction = "exact"
    return {
        "raw_delta_frames": int(delta),
        "quantized_delta_frames": int(quantized),
        "quantization_step_frames": int(step),
        "gear": gear,
        "rounding_direction": direction,
    }


def fingerprint_key_for(bank_a: int, band_b: int, quantized_delta: int, gear: str) -> str:
    return f"fp:b{int(bank_a):02d}-b{int(band_b):02d}:{gear}:d{int(quantized_delta):03d}"


def pair_product_id(clip_id: str, anchor_frame: int, band_a: int, target_frame: int, band_b: int) -> str:
    return (
        f"pair:{clip_id}:a{anchor_frame:06d}:b{band_a:02d}"
        f"-t{target_frame:06d}:b{band_b:02d}"
    )


def build_pairs_for_clip(
    clip: dict[str, Any],
    peak_sets: list[dict[str, Any]],
    config: dict[str, Any],
) -> list[dict[str, Any]]:
    sets_by_frame = {int(item["frame_index"]): item for item in peak_sets}
    max_gap = int(config["pair_lookahead_frames"])
    max_targets = int(config["max_targets_per_anchor"])
    gear = str(config["time_quant_gear"])
    step = int(config["time_quant_step"])
    pair_config_fp = pair_fingerprint_config(config)
    products: list[dict[str, Any]] = []

    for frame_index in sorted(sets_by_frame):
        anchor_set = sets_by_frame[frame_index]
        for anchor in anchor_set["peaks"]:
            candidates: list[tuple[int, int, dict[str, Any], dict[str, Any]]] = []
            for target_offset in range(1, max_gap + 1):
                target_frame_index = frame_index + target_offset
                target_set = sets_by_frame.get(target_frame_index)
                if target_set is None:
                    continue
                for target in target_set["peaks"]:
                    candidates.append((target_offset, -int(target["strength_q"]), target, target_set))
            candidates.sort(key=lambda item: (item[0], item[1], item[2]["band_index"], item[2]["peak_id"]))
            selected = candidates[:max_targets]
            for _, _, target, target_set in selected:
                delta = int(target["frame_index"]) - int(anchor["frame_index"])
                quantization = quantize_time_delta(delta, gear, step)
                fp_key = fingerprint_key_for(
                    anchor["band_index"],
                    target["band_index"],
                    quantization["quantized_delta_frames"],
                    gear,
                )
                product_id = pair_product_id(
                    clip["clip_id"],
                    int(anchor["frame_index"]),
                    int(anchor["band_index"]),
                    int(target["frame_index"]),
                    int(target["band_index"]),
                )
                strength = int(anchor["strength_q"]) + int(target["strength_q"])
                cache_key = fingerprint(
                    {
                        "product_type": PAIR_PRODUCT,
                        "product_id": product_id,
                        "anchor_peak_set_key": anchor_set["cache_key"],
                        "target_peak_set_key": target_set["cache_key"],
                        "pair_config_fingerprint": pair_config_fp,
                        "quantization": quantization,
                        "fingerprint_key": fp_key,
                        "strength_q": strength,
                    },
                    "pairkey_",
                )
                products.append(
                    {
                        "product_type": PAIR_PRODUCT,
                        "product_id": product_id,
                        "cache_key": cache_key,
                        "fingerprint_key": fp_key,
                        "clip_id": clip["clip_id"],
                        "track_id": clip["track_id"],
                        "version": int(clip["version"]),
                        "anchor_frame": int(anchor["frame_index"]),
                        "anchor_band": int(anchor["band_index"]),
                        "target_frame": int(target["frame_index"]),
                        "target_band": int(target["band_index"]),
                        "strength_q": strength,
                        "quantization": quantization,
                        "upstream_refs": [
                            {"type": "peak_set", "id": anchor_set["product_id"], "key": anchor_set["cache_key"]},
                            {"type": "peak_set", "id": target_set["product_id"], "key": target_set["cache_key"]},
                            {"type": "pair_config", "key": pair_config_fp},
                        ],
                    }
                )

    products.sort(key=lambda item: item["product_id"])
    return products
