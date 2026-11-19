from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Any

from .compat import CompatTable
from .errors import ValidationError, PipelineError
from .normalization import NormalizationRules, normalize_vocabulary
from .util import canonical_json, deep_merge, fraction_from_json_number, read_json, sha256_json


DEFAULT_TIER_SPECS = {
    1: {"id": 1, "codec": "delta_varint", "low_frequency_filter": False},
    2: {"id": 2, "codec": "delta_varint_lowfreq_filter", "low_frequency_filter": True, "low_frequency_threshold": 2},
}


@dataclass(frozen=True)
class TierSpec:
    id: int
    codec: str
    low_frequency_filter: bool
    low_frequency_threshold: int

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "codec": self.codec,
            "low_frequency_filter": self.low_frequency_filter,
            "low_frequency_threshold": self.low_frequency_threshold,
        }


@dataclass(frozen=True)
class Config:
    raw: dict
    config_hash: str
    normalization: NormalizationRules
    controlled_vocabulary: tuple[str, ...]
    allowed_shards: tuple[str, ...]
    posting_block_size: int
    tiers: dict[int, TierSpec]
    active_tier: int
    keep_positions: bool
    index_format_version: str
    lexical_weight: Fraction
    cooccurrence_weight: Fraction
    fusion_temperature: Fraction
    fixed_decimal_bits: int
    fixed_tolerance_units: Fraction
    interaction_budget_ms: float
    concurrency: int
    snapshot_interval_batches: int
    top_k: int
    compat: CompatTable

    @property
    def fixed_scale(self) -> int:
        return 1 << self.fixed_decimal_bits

    def tier(self, tier_id: int | None = None) -> TierSpec:
        key = self.active_tier if tier_id is None else tier_id
        try:
            return self.tiers[key]
        except KeyError as exc:
            raise ValidationError("E_INDEX_INVALID", f"unknown compression tier: {key}") from exc

    def tier_semantic(self, version: str | None = None, tier_id: int | None = None) -> str:
        return self.compat.tier_semantic(version or self.index_format_version, tier_id or self.active_tier)

    def index_hash_payload(self) -> dict:
        return {
            "normalization": self.normalization.to_dict(),
            "controlled_vocabulary": list(self.controlled_vocabulary),
            "allowed_shards": list(self.allowed_shards),
            "posting_block_size": self.posting_block_size,
            "tiers": {str(k): v.to_dict() for k, v in sorted(self.tiers.items())},
            "active_tier": self.active_tier,
            "tier_semantic": self.tier_semantic(),
            "keep_positions": self.keep_positions,
            "index_format_version": self.index_format_version,
            "compat_semantic_hash": self.compat.semantic_hash,
        }

    def scoring_hash_payload(self) -> dict:
        return {
            "lexical_weight": f"{self.lexical_weight.numerator}/{self.lexical_weight.denominator}",
            "cooccurrence_weight": f"{self.cooccurrence_weight.numerator}/{self.cooccurrence_weight.denominator}",
            "fusion_temperature": f"{self.fusion_temperature.numerator}/{self.fusion_temperature.denominator}",
            "fixed_decimal_bits": self.fixed_decimal_bits,
            "fixed_tolerance_units": f"{self.fixed_tolerance_units.numerator}/{self.fixed_tolerance_units.denominator}",
            "top_k": self.top_k,
        }

    def index_hash(self) -> str:
        return sha256_json(self.index_hash_payload())

    def scoring_hash(self) -> str:
        return sha256_json(self.scoring_hash_payload())

    def to_dict(self) -> dict:
        return self.raw

    @classmethod
    def load(cls, path: str | Path, compat: CompatTable) -> "Config":
        try:
            raw = read_json(path)
        except (OSError, ValueError) as exc:
            raise ValidationError("E_INDEX_INVALID", f"cannot load fusion config: {exc}") from exc
        return cls.from_mapping(raw, compat)

    @classmethod
    def from_mapping(cls, raw: dict, compat: CompatTable) -> "Config":
        if not isinstance(raw, dict):
            raise ValidationError("E_INDEX_INVALID", "config must be an object")

        normalization = NormalizationRules.from_mapping(raw.get("normalization", {}))
        vocabulary_raw = raw.get("controlled_vocabulary")
        if not isinstance(vocabulary_raw, list) or not vocabulary_raw:
            raise ValidationError("E_INDEX_INVALID", "controlled_vocabulary must be a non-empty array")
        vocabulary = normalize_vocabulary(vocabulary_raw, normalization)

        shards_raw = raw.get("allowed_shards")
        if not isinstance(shards_raw, list) or not shards_raw:
            raise ValidationError("E_INDEX_INVALID", "allowed_shards must be a non-empty array")
        if any(not isinstance(shard, str) or not shard for shard in shards_raw):
            raise ValidationError("E_INDEX_INVALID", "allowed_shards entries must be non-empty strings")
        if len(shards_raw) != len(set(shards_raw)):
            raise ValidationError("E_INDEX_INVALID", "allowed_shards entries must be unique")
        allowed_shards = tuple(sorted(shards_raw))

        block_size = raw.get("posting_block_size")
        if not isinstance(block_size, int) or isinstance(block_size, bool) or block_size <= 0:
            raise ValidationError("E_INDEX_INVALID", "posting_block_size must be a positive integer")

        tiers = _parse_tiers(raw.get("compression_tiers"))
        if 1 not in tiers or 2 not in tiers:
            raise ValidationError("E_INDEX_INVALID", "compression tier 1 and tier 2 must both be declared")
        for tier in tiers.values():
            if tier.id not in (1, 2):
                raise ValidationError("E_INDEX_INVALID", f"unsupported compression tier id: {tier.id}")
            if not tier.codec:
                raise ValidationError("E_INDEX_INVALID", "compression codec must be non-empty")
            if tier.low_frequency_threshold <= 0:
                raise ValidationError("E_INDEX_INVALID", "low_frequency_threshold must be positive")

        active_tier_raw = raw.get("active_compression_tier", 1)
        if isinstance(active_tier_raw, str) and active_tier_raw.startswith("tier"):
            try:
                active_tier = int(active_tier_raw[4:])
            except ValueError as exc:
                raise ValidationError("E_INDEX_INVALID", "invalid active_compression_tier") from exc
        else:
            active_tier = active_tier_raw
        if not isinstance(active_tier, int) or isinstance(active_tier, bool) or active_tier not in tiers:
            raise ValidationError("E_INDEX_INVALID", "active_compression_tier must be 1 or 2")

        keep_positions = raw.get("keep_positions")
        if not isinstance(keep_positions, bool):
            raise ValidationError("E_INDEX_INVALID", "keep_positions must be a boolean")

        format_version = raw.get("index_format_version")
        if not isinstance(format_version, str):
            raise ValidationError("E_INDEX_INVALID", "index_format_version must be a string")
        version_spec = compat.require_version(format_version)
        for tier in tiers.values():
            if tier.id not in version_spec.supported_tiers:
                raise ValidationError("E_INDEX_INVALID", f"compression tier {tier.id} is not supported by {format_version}")
        if active_tier not in version_spec.supported_tiers:
            raise ValidationError("E_INDEX_INVALID", "active compression tier is not supported by the target format")

        weights = raw.get("channel_weights")
        if not isinstance(weights, dict):
            raise ValidationError("E_INDEX_INVALID", "channel_weights must be an object")
        try:
            lexical_weight = fraction_from_json_number(weights.get("lexical"))
            cooccurrence_weight = fraction_from_json_number(weights.get("cooccurrence"))
        except (ValueError, TypeError) as exc:
            raise ValidationError("E_INDEX_INVALID", "channel weights must be numeric") from exc
        if lexical_weight < 0 or cooccurrence_weight < 0 or lexical_weight + cooccurrence_weight <= 0:
            raise ValidationError("E_INDEX_INVALID", "channel weights must be non-negative with a positive sum")

        try:
            temperature = fraction_from_json_number(raw.get("fusion_temperature"))
        except (ValueError, TypeError) as exc:
            raise ValidationError("E_INDEX_INVALID", "fusion_temperature must be numeric") from exc
        if temperature <= 0:
            raise ValidationError("E_INDEX_INVALID", "fusion_temperature must be positive")

        bits = raw.get("fixed_decimal_bits")
        if not isinstance(bits, int) or isinstance(bits, bool) or bits < 1 or bits > 30:
            raise ValidationError("E_INDEX_INVALID", "fixed_decimal_bits must be an integer in [1, 30]")

        tolerance_raw = raw.get("fixed_tolerance_units", 0.5)
        try:
            tolerance = fraction_from_json_number(tolerance_raw)
        except (ValueError, TypeError) as exc:
            raise ValidationError("E_INDEX_INVALID", "fixed_tolerance_units must be numeric") from exc
        if tolerance <= 0:
            raise ValidationError("E_INDEX_INVALID", "fixed_tolerance_units must be positive")

        budget = raw.get("interaction_budget_ms")
        if not isinstance(budget, (int, float)) or isinstance(budget, bool) or budget < 0:
            raise ValidationError("E_INDEX_INVALID", "interaction_budget_ms must be non-negative")

        concurrency = raw.get("concurrency")
        if not isinstance(concurrency, int) or isinstance(concurrency, bool) or concurrency <= 0:
            raise ValidationError("E_INDEX_INVALID", "concurrency must be a positive integer")

        interval = raw.get("snapshot_interval_batches")
        if not isinstance(interval, int) or isinstance(interval, bool) or interval <= 0:
            raise ValidationError("E_INDEX_INVALID", "snapshot_interval_batches must be a positive integer")

        top_k = raw.get("top_k", 10)
        if not isinstance(top_k, int) or isinstance(top_k, bool) or top_k <= 0:
            raise ValidationError("E_INDEX_INVALID", "top_k must be a positive integer")

        normalized_raw = deep_merge(raw, {})
        return cls(
            raw=normalized_raw,
            config_hash=sha256_json(normalized_raw),
            normalization=normalization,
            controlled_vocabulary=vocabulary,
            allowed_shards=allowed_shards,
            posting_block_size=block_size,
            tiers=tiers,
            active_tier=active_tier,
            keep_positions=keep_positions,
            index_format_version=format_version,
            lexical_weight=lexical_weight,
            cooccurrence_weight=cooccurrence_weight,
            fusion_temperature=temperature,
            fixed_decimal_bits=bits,
            fixed_tolerance_units=tolerance,
            interaction_budget_ms=float(budget),
            concurrency=concurrency,
            snapshot_interval_batches=interval,
            top_k=top_k,
            compat=compat,
        )

    def patched(self, patch: dict) -> "Config":
        if not isinstance(patch, dict):
            raise PipelineError("E_EDIT_INVALID", "set_config patch must be an object")
        return Config.from_mapping(deep_merge(self.raw, patch), self.compat)


def _parse_tiers(value: Any) -> dict[int, TierSpec]:
    if value is None:
        value = [DEFAULT_TIER_SPECS[1], DEFAULT_TIER_SPECS[2]]
    items: list[dict]
    if isinstance(value, dict):
        items = []
        for key, item in value.items():
            if not isinstance(item, dict):
                raise ValidationError("E_INDEX_INVALID", "compression tier entries must be objects")
            candidate = dict(item)
            candidate.setdefault("id", key)
            items.append(candidate)
    elif isinstance(value, list):
        items = value
    else:
        raise ValidationError("E_INDEX_INVALID", "compression_tiers must be an array or object")

    tiers: dict[int, TierSpec] = {}
    for item in items:
        if not isinstance(item, dict):
            raise ValidationError("E_INDEX_INVALID", "compression tier entries must be objects")
        raw_id = item.get("id")
        if isinstance(raw_id, str) and raw_id.startswith("tier"):
            raw_id = raw_id[4:]
        try:
            tier_id = int(raw_id)
        except (TypeError, ValueError) as exc:
            raise ValidationError("E_INDEX_INVALID", "compression tier id must be an integer") from exc
        codec = item.get("codec", item.get("name"))
        if not isinstance(codec, str) or not codec:
            raise ValidationError("E_INDEX_INVALID", "compression codec must be non-empty")
        lowfreq = item.get("low_frequency_filter", tier_id == 2)
        if not isinstance(lowfreq, bool):
            raise ValidationError("E_INDEX_INVALID", "low_frequency_filter must be a boolean")
        threshold = item.get("low_frequency_threshold", 2)
        if not isinstance(threshold, int) or isinstance(threshold, bool) or threshold < 1:
            raise ValidationError("E_INDEX_INVALID", "low_frequency_threshold must be a positive integer")
        if tier_id in tiers:
            raise ValidationError("E_INDEX_INVALID", f"duplicate compression tier: {tier_id}")
        tiers[tier_id] = TierSpec(tier_id, codec, lowfreq, threshold)
    return tiers


def config_hash_of_mapping(raw: dict) -> str:
    return sha256_json(raw)

