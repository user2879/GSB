"""Fusion configuration loading and strict deterministic validation."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
from typing import Any

import yaml

from .errors import ConfigError
from .util import canonical_json, decimal_to_q, hash_json, sha256_text


@dataclass(frozen=True)
class QuantizationProfile:
    tier: int
    fractional_bits: int
    word_bits: int
    quant_step: int

    @property
    def one(self) -> int:
        return 1 << self.fractional_bits

    def as_dict(self) -> dict[str, Any]:
        return {
            "tier": self.tier,
            "fractional_bits": self.fractional_bits,
            "word_bits": self.word_bits,
            "quant_step": self.quant_step,
            "quant_step_ratio": f"{self.quant_step}/{self.one}",
        }


@dataclass(frozen=True)
class FusionConfig:
    raw: dict[str, Any]
    config_hash: str
    visual_config_hash: str
    text_config_hash: str
    fusion_config_hash: str
    block_size: int
    visual_dim: int
    text_dim: int
    projection_seed: str
    tier: int
    base_step: int
    fractional_bits: int
    word_bits: int
    tolerance_q: int
    weights: dict[str, Decimal]
    temperature: Decimal
    threshold: Decimal
    tie_break: str
    global_macs: int
    batch_points: int
    cache_blocks: int
    threads: int
    snapshot_interval: int
    image_dir: str
    truth_dir: str
    text_jsonl: str
    truth_required: bool
    strict_assertions: bool
    expected_width: int | None
    expected_height: int | None

    @property
    def profile(self) -> QuantizationProfile:
        return make_profile(self.tier, self.fractional_bits, self.word_bits, self.base_step)

    @property
    def diagnostic_profile(self) -> QuantizationProfile:
        high_fractional = min(self.fractional_bits + 4, 58)
        high_word = min(self.word_bits + 8, 63)
        return make_profile(self.tier + 1, high_fractional, high_word, self.base_step)

    def weight_q(self, modality: str, profile: QuantizationProfile) -> int:
        return decimal_to_q(self.weights[modality], profile.fractional_bits)

    def temperature_q(self, profile: QuantizationProfile) -> int:
        return decimal_to_q(self.temperature, profile.fractional_bits)

    def threshold_q(self, profile: QuantizationProfile) -> int:
        return decimal_to_q(self.threshold, profile.fractional_bits)


def make_profile(tier: int, fractional_bits: int, word_bits: int, base_step: int) -> QuantizationProfile:
    step = max(1, int(base_step) >> int(tier))
    return QuantizationProfile(
        tier=int(tier),
        fractional_bits=int(fractional_bits),
        word_bits=int(word_bits),
        quant_step=step,
    )


def _mapping(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ConfigError(f"{name} must be a mapping")
    return value


def _integer(mapping: dict[str, Any], key: str, default: int | None = None) -> int:
    value = mapping.get(key, default)
    if isinstance(value, bool) or not isinstance(value, int):
        raise ConfigError(f"{key} must be an integer")
    return value


def _number(mapping: dict[str, Any], key: str) -> Decimal:
    value = mapping.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float, Decimal)):
        raise ConfigError(f"{key} must be numeric")
    return Decimal(str(value))


def _path_string(mapping: dict[str, Any], key: str) -> str:
    value = mapping.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ConfigError(f"input.{key} must be a non-empty string")
    return value


def load_config(path: str | Path) -> FusionConfig:
    config_path = Path(path)
    try:
        with config_path.open("r", encoding="utf-8") as handle:
            raw = yaml.safe_load(handle)
    except OSError as exc:
        raise ConfigError(f"cannot read config: {exc}") from exc
    except yaml.YAMLError as exc:
        raise ConfigError(f"invalid YAML: {exc}") from exc

    raw = _mapping(raw, "config")
    schema_version = raw.get("schema_version", 1)
    if schema_version != 1:
        raise ConfigError("schema_version must be 1")

    input_section = _mapping(raw.get("input", {}), "input")
    features = _mapping(raw.get("features"), "features")
    quantization = _mapping(raw.get("quantization"), "quantization")
    fusion = _mapping(raw.get("fusion"), "fusion")
    budgets = _mapping(raw.get("budgets", {}), "budgets")

    visual_dim = _integer(features, "visual_dim")
    text_dim = _integer(features, "text_dim")
    block_size = _integer(features, "block_size")
    if visual_dim <= 0 or visual_dim > 4096:
        raise ConfigError("features.visual_dim must be in [1, 4096]")
    if text_dim <= 0 or text_dim > 4096:
        raise ConfigError("features.text_dim must be in [1, 4096]")
    if block_size <= 0 or block_size > 4096:
        raise ConfigError("features.block_size must be in [1, 4096]")

    projection_seed = features.get("projection_seed", "default-v1")
    if not isinstance(projection_seed, str) or not projection_seed:
        raise ConfigError("features.projection_seed must be a non-empty string")

    tier = _integer(quantization, "tier")
    base_step = _integer(quantization, "base_step")
    fractional_bits = _integer(quantization, "fractional_bits")
    word_bits = _integer(quantization, "word_bits")
    if tier < 0 or tier > 30:
        raise ConfigError("quantization.tier must be in [0, 30]")
    if base_step <= 0:
        raise ConfigError("quantization.base_step must be positive")
    if (base_step >> tier) <= 0:
        raise ConfigError("quantization.tier is too high for base_step")
    if fractional_bits < 1 or fractional_bits > 58:
        raise ConfigError("quantization.fractional_bits must be in [1, 58]")
    if word_bits < 8 or word_bits > 63 or fractional_bits >= word_bits:
        raise ConfigError("quantization.word_bits must be in [8, 63] and exceed fractional_bits")
    tolerance = _number(quantization, "tolerance")
    if tolerance < 0:
        raise ConfigError("quantization.tolerance must be non-negative")
    tolerance_q = decimal_to_q(tolerance, fractional_bits)

    weights_raw = _mapping(fusion.get("weights"), "fusion.weights")
    if set(weights_raw) != {"text", "visual"}:
        raise ConfigError("fusion.weights must contain exactly text and visual")
    weights = {name: _number(weights_raw, name) for name in ("text", "visual")}
    if any(value < 0 for value in weights.values()) or sum(weights.values()) <= 0:
        raise ConfigError("fusion weights must be non-negative and not all zero")
    temperature = _number(fusion, "temperature")
    threshold = _number(fusion, "threshold")
    if temperature <= 0:
        raise ConfigError("fusion.temperature must be positive")
    if threshold < 0 or threshold > 1:
        raise ConfigError("fusion.threshold must be in [0, 1]")
    tie_break = fusion.get("tie_break", "positive")
    if tie_break not in {"positive", "negative"}:
        raise ConfigError("fusion.tie_break must be positive or negative")

    global_macs = _integer(budgets, "global_macs", 0)
    batch_points = _integer(budgets, "batch_points", 0)
    cache_blocks = _integer(budgets, "cache_blocks", 0)
    threads = _integer(budgets, "threads", 1)
    snapshot_interval = _integer(budgets, "snapshot_interval", 1)
    if global_macs < 0 or batch_points < 0 or cache_blocks < 0:
        raise ConfigError("budgets must be non-negative")
    if threads < 1 or threads > 256:
        raise ConfigError("budgets.threads must be in [1, 256]")
    if snapshot_interval < 1:
        raise ConfigError("budgets.snapshot_interval must be positive")

    width = input_section.get("expected_width")
    height = input_section.get("expected_height")
    if width is not None and (isinstance(width, bool) or not isinstance(width, int) or width <= 0):
        raise ConfigError("input.expected_width must be a positive integer or null")
    if height is not None and (isinstance(height, bool) or not isinstance(height, int) or height <= 0):
        raise ConfigError("input.expected_height must be a positive integer or null")

    image_dir = _path_string(input_section, "image_dir")
    truth_dir = _path_string(input_section, "truth_dir")
    text_jsonl = _path_string(input_section, "text_jsonl")
    truth_required = input_section.get("truth_required", True)
    strict_assertions = input_section.get("strict_assertions", True)
    if not isinstance(truth_required, bool):
        raise ConfigError("input.truth_required must be boolean")
    if not isinstance(strict_assertions, bool):
        raise ConfigError("input.strict_assertions must be boolean")

    config_hash = hash_json(raw, namespace="fusion-config-v1")
    visual_fingerprint = {
        "schema": "visual-v1",
        "block_size": block_size,
        "visual_dim": visual_dim,
        "projection_seed": projection_seed,
        "metrics": ["mean_r", "mean_g", "mean_b", "gradient_strength", "edge_density"],
    }
    text_fingerprint = {
        "schema": "text-v1",
        "text_dim": text_dim,
        "projection_seed": projection_seed,
        "roles": {"entity": 2, "attribute": 3, "value": 2},
    }
    fusion_fingerprint = {
        "schema": "fusion-v1",
        "weights": {key: str(weights[key]) for key in sorted(weights)},
        "temperature": str(temperature),
        "threshold": str(threshold),
        "tie_break": tie_break,
    }

    return FusionConfig(
        raw=raw,
        config_hash=config_hash,
        visual_config_hash=hash_json(visual_fingerprint, namespace="visual-config-v1"),
        text_config_hash=hash_json(text_fingerprint, namespace="text-config-v1"),
        fusion_config_hash=hash_json(fusion_fingerprint, namespace="fusion-config-v1"),
        block_size=block_size,
        visual_dim=visual_dim,
        text_dim=text_dim,
        projection_seed=projection_seed,
        tier=tier,
        base_step=base_step,
        fractional_bits=fractional_bits,
        word_bits=word_bits,
        tolerance_q=tolerance_q,
        weights=weights,
        temperature=temperature,
        threshold=threshold,
        tie_break=tie_break,
        global_macs=global_macs,
        batch_points=batch_points,
        cache_blocks=cache_blocks,
        threads=threads,
        snapshot_interval=snapshot_interval,
        image_dir=image_dir,
        truth_dir=truth_dir,
        text_jsonl=text_jsonl,
        truth_required=truth_required,
        strict_assertions=strict_assertions,
        expected_width=width,
        expected_height=height,
    )


def config_fields_for_report(config: FusionConfig) -> dict[str, Any]:
    return {
        "canonical": canonical_json(config.raw),
        "config_hash": config.config_hash,
        "visual_config_hash": config.visual_config_hash,
        "text_config_hash": config.text_config_hash,
        "fusion_config_hash": config.fusion_config_hash,
        "projection_fingerprint": sha256_text(config.projection_seed + config.visual_config_hash),
    }
