"""Stable content and parameter fingerprints."""

from __future__ import annotations

import hashlib
import json
from typing import Any

from .models import ChannelSpec, LayerSpec, MaterialConfig, RenderConfig, TextureData

KEY_VERSION = "fixedtex-key-v1"


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False)


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def content_key(kind: str, payload: dict[str, Any]) -> str:
    digest = hashlib.sha256()
    digest.update(KEY_VERSION.encode("ascii"))
    digest.update(b"\0")
    digest.update(kind.encode("ascii"))
    digest.update(b"\0")
    digest.update(canonical_json(payload).encode("utf-8"))
    return digest.hexdigest()


def texture_key(texture: TextureData) -> str:
    return content_key(
        "texture",
        {
            "width": texture.width,
            "height": texture.height,
            "mode": texture.mode,
            "content_sha256": hashlib.sha256(texture.rgb.tobytes()).hexdigest(),
        },
    )


def layer_parameter_payload(
    layer: LayerSpec,
    config: RenderConfig,
    texture: TextureData,
) -> dict[str, Any]:
    return {
        "texture_content_key": texture.content_key,
        "resolution": [texture.width, texture.height],
        "global_seed": config.seed,
        "algorithm": config.algorithm,
        "layer": layer.parameter_dict(),
    }


def layer_parameter_key(
    layer: LayerSpec,
    config: RenderConfig,
    texture: TextureData,
) -> str:
    return content_key("layer-params", layer_parameter_payload(layer, config, texture))


def layer_key(
    layer: LayerSpec,
    config: RenderConfig,
    texture: TextureData,
) -> str:
    return content_key(
        "layer",
        {
            "upstream": texture.content_key,
            "parameters": layer_parameter_key(layer, config, texture),
            "fraction_bits": config.fixed.fraction_bits,
            "quant_step": config.fixed.quant_step,
            "guard_bits": config.fixed.guard_bits,
            "tie_rule": config.fixed.tie_rule,
        },
    )


def channel_key(
    channel: ChannelSpec,
    config: RenderConfig,
    material: MaterialConfig,
    texture: TextureData,
    layer_keys: dict[int, str],
) -> str:
    referenced = {str(index): layer_keys[index] for index in channel.layer_indices if index in layer_keys}
    return content_key(
        "channel",
        {
            "upstream_texture": texture.content_key,
            "upstream_layers": referenced,
            "parameters": channel.parameter_dict(),
            "material_tie_rule": material.tie_rule,
            "equal_weight_rule": material.equal_weight_rule,
            "blend_mode": config.blend_mode,
            "fraction_bits": config.fixed.fraction_bits,
            "quant_step": config.fixed.quant_step,
            "resolution": [texture.width, texture.height],
        },
    )


def composite_key(
    config: RenderConfig,
    material: MaterialConfig,
    channel_keys: dict[str, str],
) -> str:
    return content_key(
        "composite",
        {
            "channels": {name: channel_keys[name] for name in sorted(channel_keys)},
            "blend_mode": config.blend_mode,
            "tie_rule": material.tie_rule,
            "fraction_bits": config.fixed.fraction_bits,
            "quant_step": config.fixed.quant_step,
            "resolution": config.resolution,
        },
    )


def object_hash(value: Any) -> str:
    return sha256_text(canonical_json(value))
