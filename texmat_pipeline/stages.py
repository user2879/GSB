"""Pipeline compute stages: noise accumulation, fixed-point quantization,
material channel mapping and fixed-point weighted blending.

Every stage function is pure and deterministic.  The tile-level functions are
shared by the serial reference path and the thread-pool scheduler, so
concurrent and single-threaded executions are pixel-identical.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .fixedpoint import FixedSpec, quantize_float, weighted_blend
from .hashing import fingerprint_array, derive_cache_key
from .models import CHANNELS, GenerationConfig, Material, NoiseLayer
from .noise import value_noise_field


@dataclass(frozen=True)
class StageSpecs:
    unit: FixedSpec
    signed: FixedSpec

    @classmethod
    def from_config(cls, config: GenerationConfig, frac_bits: int | None = None) -> "StageSpecs":
        fp = config.fixed_point
        bits = fp.frac_bits if frac_bits is None else frac_bits
        return cls(
            FixedSpec(bits, fp.unit_levels, 0.0, 1.0),
            FixedSpec(bits, fp.signed_levels, -1.0, 1.0),
        )

    def to_meta(self) -> dict:
        return {
            "frac_bits": self.unit.frac_bits,
            "unit_levels": self.unit.levels,
            "signed_levels": self.signed.levels,
            "unit_step": self.unit.step,
            "signed_step": self.signed.step,
        }


# --------------------------------------------------------------------------- #
# Stage 1+2: deterministic layer noise accumulation and quantization
# --------------------------------------------------------------------------- #
def layer_noise_tile(layer: NoiseLayer, width: int, height: int,
                     row_start: int, row_end: int) -> np.ndarray:
    """Signed noise field tile (float) for one layer, row range [start, end)."""
    field = value_noise_field(width, height, layer.frequency, layer.fingerprint)
    return field[row_start:row_end, :] * layer.amplitude


def accumulate_layers(
    textures: dict,
    config: GenerationConfig,
    specs: StageSpecs,
) -> dict:
    """Run layers in ascending index order, row-major pixel traversal.

    Layer 0 starts from the quantized base texture; each subsequent layer adds
    its quantized noise contribution and re-quantizes to the signed domain.
    Returns per-layer fixed-point images, per-layer raw noise and metadata.
    """
    width = height = next(iter(textures.values())).width
    round_mode = config.fixed_point.round_mode
    result: dict = {"layers": [], "order": [], "boundary_pixels": 0}
    previous: np.ndarray | None = None
    base_gray_by_tex = {tid: tex.gray.astype(np.float64) / 255.0 for tid, tex in textures.items()}

    for layer in config.layers:
        base_gray = base_gray_by_tex.get(f"{layer.base_texture}_{width}")
        if base_gray is None:
            base_gray = next(iter(base_gray_by_tex.values()))
        noise = layer_noise_tile(layer, width, height, 0, height)
        if previous is None:
            # Base texture lives on [0,1]; lift it into the signed [-1,1]
            # domain before the first layer so all accumulations share one
            # fixed-point scale and quantization specification.
            current, boundary = quantize_float(base_gray + noise, specs.signed, round_mode)
        else:
            current, boundary = quantize_float(
                previous.astype(np.float64) / specs.signed.scale + noise, specs.signed, round_mode
            )
        fp = fingerprint_array(current, f"noise_layer:{layer.index}")
        key = derive_cache_key(
            "noise",
            [fp],
            {
                "layer_index": layer.index,
                "fingerprint": layer.fingerprint,
                "frequency": layer.frequency,
                "amplitude": layer.amplitude,
                "frac_bits": specs.signed.frac_bits,
                "levels": specs.signed.levels,
                "round_mode": round_mode,
                "base_texture": layer.base_texture,
            },
        )
        result["layers"].append(
            {
                "index": layer.index,
                "image": current,
                "noise": noise,
                "boundary": boundary,
                "fingerprint": fp,
                "cache_key": key,
                "base_texture": layer.base_texture,
            }
        )
        result["order"].append(layer.index)
        result["boundary_pixels"] += int(np.count_nonzero(boundary))
        previous = current
    return result


# --------------------------------------------------------------------------- #
# Stage 3: material channel mapping and per-channel weighted blending
# --------------------------------------------------------------------------- #
def map_channel_tile(channel: str, base_gray: np.ndarray, noise: np.ndarray,
                     specs: StageSpecs, round_mode: str) -> tuple[np.ndarray, np.ndarray]:
    """Map one (layer, channel) pair onto the unit fixed-point domain."""
    if channel == "diffuse":
        value = np.clip(0.82 * base_gray + 0.18 * (noise + 1.0) * 0.5, 0.0, 1.0)
    elif channel == "roughness":
        value = np.clip(0.5 + 0.45 * noise - 0.15 * base_gray, 0.0, 1.0)
    elif channel == "metallic":
        value = np.clip(0.25 * base_gray + 0.75 * np.maximum(noise, 0.0), 0.0, 1.0)
    else:  # normal perturbation
        value = np.clip(0.5 + 0.5 * noise + 0.1 * (base_gray - 0.5), 0.0, 1.0)
    return quantize_float(value, specs.unit, round_mode)


def _weight_ints(weights: dict[int, float]) -> dict[int, int]:
    # Deterministic integer weights: scale finite float weights by 1000.
    return {idx: max(1, int(round(w * 1000.0))) for idx, w in weights.items()}


def build_material_channels(
    textures: dict,
    noise_result: dict,
    materials: list[Material],
    config: GenerationConfig,
    specs: StageSpecs,
) -> dict:
    """Map every referenced layer onto four channels and blend per material."""
    width = next(iter(textures.values())).width
    round_mode = config.fixed_point.round_mode
    layer_images = {entry["index"]: entry["image"] for entry in noise_result["layers"]}
    layer_noise = {entry["index"]: entry["noise"] for entry in noise_result["layers"]}
    layer_base = {entry["index"]: entry["base_texture"] for entry in noise_result["layers"]}
    base_gray_by_tex = {tid: tex.gray.astype(np.float64) / 255.0 for tid, tex in textures.items()}

    # Cache the (layer, channel) mapped tiles: several materials share them.
    mapped: dict[tuple[int, str], tuple[np.ndarray, np.ndarray]] = {}
    for entry in noise_result["layers"]:
        idx = entry["index"]
        base_gray = base_gray_by_tex[f"{layer_base[idx]}_{width}"]
        for channel in CHANNELS:
            mapped[(idx, channel)] = map_channel_tile(
                channel, base_gray, layer_noise[idx], specs, round_mode
            )

    output: dict = {"materials": [], "boundary_pixels": 0}
    for material in materials:
        mat_entry: dict = {"material_id": material.material_id, "channels": {}, "lineage": {}}
        for channel in CHANNELS:
            weight_map = material.layer_weights[channel]
            int_weights = _weight_ints(weight_map)
            terms = []
            used_layers: list[int] = []
            for idx in sorted(weight_map):
                if idx not in layer_images:
                    continue
                terms.append((mapped[(idx, channel)][0], int_weights[idx]))
                used_layers.append(idx)
            if not terms:
                raise ValueError(f"material {material.material_id} channel {channel} has no usable layers")
            normalize = config.blend_mode == "normalized_weight"
            blended, blend_tie = weighted_blend(terms, specs.unit, round_mode, normalize)
            map_boundary = np.zeros_like(blended, dtype=bool)
            for idx in used_layers:
                map_boundary |= mapped[(idx, channel)][1]
            boundary = map_boundary | blend_tie
            upstream = [noise_result["layers"][idx]["cache_key"] for idx in used_layers]
            fp = fingerprint_array(blended, f"channel:{material.material_id}:{channel}")
            key = derive_cache_key(
                "channel",
                upstream,
                {
                    "material": material.material_id,
                    "channel": channel,
                    "weights": {str(k): int_weights[k] for k in sorted(int_weights)},
                    "blend_mode": config.blend_mode,
                    "frac_bits": specs.unit.frac_bits,
                    "levels": specs.unit.levels,
                    "round_mode": round_mode,
                },
            )
            mat_entry["channels"][channel] = {
                "image": blended,
                "boundary": boundary,
                "fingerprint": fp,
                "cache_key": key,
                "used_layers": used_layers,
            }
            mat_entry["lineage"][channel] = used_layers
            output["boundary_pixels"] += int(np.count_nonzero(boundary))
        output["materials"].append(mat_entry)
    return output


def composite_fingerprint(material_entry: dict, config: GenerationConfig, specs: StageSpecs) -> tuple[str, str]:
    """Final composite fingerprint/key derived from the four channel keys."""
    upstream = [material_entry["channels"][c]["cache_key"] for c in CHANNELS]
    fp_input = np.stack([material_entry["channels"][c]["image"] for c in CHANNELS], axis=-1)
    fp = fingerprint_array(fp_input, f"composite:{material_entry['material_id']}")
    key = derive_cache_key(
        "composite",
        upstream,
        {
            "material": material_entry["material_id"],
            "blend_mode": config.blend_mode,
            "frac_bits": specs.unit.frac_bits,
            "channels": CHANNELS,
        },
    )
    return fp, key
