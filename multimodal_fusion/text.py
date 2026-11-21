"""Deterministic sparse assertion encoding with explicit fill provenance."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .config import FusionConfig, QuantizationProfile
from .inputs import SampleInput
from .util import derive_int_vector, hash_json, round_div, saturate_signed


ROLE_WEIGHTS = {"entity": 2, "attribute": 3, "value": 2}
TOKEN_FANOUT = 4
TOKEN_DERIVATION_SPAN = 1024


@dataclass(frozen=True)
class TextStage:
    vector: list[int]
    raw_vector: list[int]
    residuals: list[int]
    merge_sources: list[dict[str, Any]]
    fill_positions: list[int]
    filled_fields: list[dict[str, Any]]
    quantization: dict[str, Any]
    text_cache_key: str
    saturation_count: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "vector": self.vector,
            "raw_vector": self.raw_vector,
            "residuals": self.residuals,
            "merge_sources": self.merge_sources,
            "fill_positions": self.fill_positions,
            "filled_fields": self.filled_fields,
            "quantization": self.quantization,
            "text_cache_key": self.text_cache_key,
            "saturation_count": self.saturation_count,
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "TextStage":
        return cls(
            vector=[int(item) for item in value["vector"]],
            raw_vector=[int(item) for item in value["raw_vector"]],
            residuals=[int(item) for item in value["residuals"]],
            merge_sources=value["merge_sources"],
            fill_positions=[int(item) for item in value["fill_positions"]],
            filled_fields=value["filled_fields"],
            quantization=value["quantization"],
            text_cache_key=value["text_cache_key"],
            saturation_count=int(value["saturation_count"]),
        )


def text_stage_key(sample: SampleInput, config: FusionConfig, profile: QuantizationProfile) -> str:
    return hash_json(
        {
            "schema": "text-stage-v1",
            "text_config_hash": config.text_config_hash,
            "assertions_hash": sample.assertions_hash,
            "profile": profile.as_dict(),
        },
        namespace="text-stage-key-v1",
    )


def _token_vector(namespace: str, token: str, dimension: int, one: int) -> list[int]:
    seed = f"{namespace}\0{token}"
    fanout = min(max(1, TOKEN_FANOUT), dimension)
    positions = derive_int_vector(seed + "\0positions", fanout, span=max(0, dimension - 1))
    values = derive_int_vector(
        seed + "\0values",
        fanout,
        span=TOKEN_DERIVATION_SPAN,
    )
    vector = [0] * dimension
    for position, value in zip(positions, values):
        index = int(position) % dimension
        vector[index] += round_div(int(value) * one, 2 * TOKEN_DERIVATION_SPAN)
    return vector


def _default_vector(config: FusionConfig, one: int) -> list[int]:
    namespace = (
        f"text-default-v1|{config.projection_seed}|{config.text_config_hash}|{config.text_dim}"
    )
    values = derive_int_vector(namespace, config.text_dim, span=TOKEN_DERIVATION_SPAN)
    return [round_div(int(value) * one, 2 * TOKEN_DERIVATION_SPAN) for value in values]


def _sorted_assertions(sample: SampleInput) -> list[dict[str, Any]]:
    return sorted(
        sample.assertions,
        key=lambda assertion: (
            str(assertion.get("entity", "")),
            str(assertion.get("attribute", "")),
            str(assertion.get("value", "")),
        ),
    )


def encode_text(
    sample: SampleInput,
    config: FusionConfig,
    profile: QuantizationProfile,
) -> TextStage:
    one = profile.one
    namespace = f"text-token-v1|{config.projection_seed}|{config.text_config_hash}"
    accumulation = [0] * config.text_dim
    support = [0] * config.text_dim
    merge_sources: list[dict[str, Any]] = []
    filled_fields: list[dict[str, Any]] = []
    sorted_assertions = _sorted_assertions(sample)

    grouped: dict[str, list[dict[str, Any]]] = {}
    for assertion_index, assertion in enumerate(sorted_assertions):
        entity = str(assertion.get("entity", ""))
        sources = grouped.setdefault(entity, [])
        source = {"attribute": str(assertion.get("attribute", "")), "value": str(assertion.get("value", ""))}
        sources.append(source)
        for role in ("entity", "attribute", "value"):
            token = str(assertion.get(role, ""))
            if not token:
                token = f"<missing:{role}>"
                filled_fields.append(
                    {
                        "assertion_index": assertion_index,
                        "field": role,
                        "fill_token": token,
                        "namespace": namespace,
                    }
                )
            vector = _token_vector(namespace, f"{role}:{token}", config.text_dim, one)
            role_weight = ROLE_WEIGHTS[role]
            for dimension, value in enumerate(vector):
                if value:
                    accumulation[dimension] += value * role_weight
                    support[dimension] += role_weight

    for entity in sorted(grouped):
        merge_sources.append(
            {
                "entity": entity,
                "assertion_count": len(grouped[entity]),
                "sources": sorted(
                    grouped[entity], key=lambda value: (value["attribute"], value["value"])
                ),
            }
        )

    raw_vector = [0] * config.text_dim
    fill_positions: list[int] = []
    defaults = _default_vector(config, one)
    for dimension in range(config.text_dim):
        if support[dimension] > 0:
            raw_vector[dimension] = round_div(accumulation[dimension], support[dimension])
        else:
            raw_vector[dimension] = defaults[dimension]
            fill_positions.append(dimension)

    vector: list[int] = []
    residuals: list[int] = []
    saturation_count = 0
    for value in raw_vector:
        value, saturated = saturate_signed(value, profile.word_bits)
        saturation_count += int(saturated)
        quantized = round_div(value, profile.quant_step) * profile.quant_step
        quantized, _ = saturate_signed(quantized, profile.word_bits)
        vector.append(quantized)
        residuals.append(value - quantized)

    text_cache_key = text_stage_key(sample, config, profile)
    quantization = {
        "profile": profile.as_dict(),
        "quant_step": profile.quant_step,
        "quant_step_ratio": f"{profile.quant_step}/{profile.one}",
        "support": support,
        "assertion_count": len(sorted_assertions),
    }
    return TextStage(
        vector=vector,
        raw_vector=raw_vector,
        residuals=residuals,
        merge_sources=merge_sources,
        fill_positions=fill_positions,
        filled_fields=filled_fields,
        quantization=quantization,
        text_cache_key=text_cache_key,
        saturation_count=saturation_count,
    )
