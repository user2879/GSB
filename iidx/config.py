from __future__ import annotations

import copy
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

from .util import failure, is_int, sha256_json, sort_failures

NORMALIZATION_VERSION = "nfkc-casefold-collapse-v1"
BLOCK_SCHEMA_VERSION = 1
COMPRESSION_SCHEMA_VERSION = 1
QUERY_SCHEMA_VERSION = 1


DEFAULT_CONFIG: dict[str, Any] = {
    "chunk_size": 5,
    "batch_max_docs": 4,
    "thread_pool_size": 3,
    "concurrency_quota": 4,
    "global_token_budget": 1200,
    "per_shard_token_quota": 500,
    "snapshot_commit_interval_batches": 2,
    "interactive_latency_budget_ms": 150,
    "fixed_decimal_bits": 12,
    "quantization_tiers": [1, 2, 4, 8],
    "quantization_tier": 1,
    "compression_tier": "auto",
    "compression_switch_points": [3, 6],
    "low_frequency_threshold": 1,
    "query_top_k": 10,
}


ALIASES: dict[str, str] = {
    "inverted_block_size": "chunk_size",
    "inverted_index_chunk_size": "chunk_size",
    "inverted_index_block_size": "chunk_size",
    "block_size": "chunk_size",
    "max_docs_per_batch": "batch_max_docs",
    "batch_size": "batch_max_docs",
    "concurrency": "thread_pool_size",
    "max_workers": "thread_pool_size",
    "max_threads": "thread_pool_size",
    "thread_pool_concurrency": "thread_pool_size",
    "concurrency_quota": "concurrency_quota",
    "thread_pool_quota": "concurrency_quota",
    "max_concurrency_quota": "concurrency_quota",
    "shard_token_quota": "per_shard_token_quota",
    "per_shard_quota": "per_shard_token_quota",
    "token_quota_per_shard": "per_shard_token_quota",
    "token_budget": "global_token_budget",
    "global_budget": "global_token_budget",
    "global_term_token_budget": "global_token_budget",
    "snapshot_interval_batches": "snapshot_commit_interval_batches",
    "interactive_latency_budget": "interactive_latency_budget_ms",
    "latency_budget_ms": "interactive_latency_budget_ms",
    "interactive_latency_ms": "interactive_latency_budget_ms",
    "fixed_bits": "fixed_decimal_bits",
    "fixed_point_bits": "fixed_decimal_bits",
    "quantization_tier_index": "quantization_tier",
    "tier": "compression_tier",
    "compression_switchpoint": "compression_switch_points",
    "compression_switch_point": "compression_switch_points",
    "compression_tier_switch_points": "compression_switch_points",
    "compression_tier_switch_point": "compression_switch_points",
    "switch_points": "compression_switch_points",
    "low_freq_threshold": "low_frequency_threshold",
    "top_k": "query_top_k",
}


@dataclass(frozen=True)
class AppConfig:
    chunk_size: int = 5
    batch_max_docs: int = 4
    thread_pool_size: int = 3
    concurrency_quota: int = 4
    global_token_budget: int = 1200
    per_shard_token_quota: int = 500
    snapshot_commit_interval_batches: int = 2
    interactive_latency_budget_ms: int = 150
    fixed_decimal_bits: int = 12
    quantization_tiers: tuple[int, ...] = (1, 2, 4, 8)
    quantization_tier: int = 1
    compression_tier: str | int = "auto"
    compression_switch_points: tuple[int, ...] = (3, 6)
    low_frequency_threshold: int = 1
    query_top_k: int = 10

    def to_dict(self) -> dict[str, Any]:
        return {
            "chunk_size": self.chunk_size,
            "batch_max_docs": self.batch_max_docs,
            "thread_pool_size": self.thread_pool_size,
            "concurrency_quota": self.concurrency_quota,
            "global_token_budget": self.global_token_budget,
            "per_shard_token_quota": self.per_shard_token_quota,
            "snapshot_commit_interval_batches": self.snapshot_commit_interval_batches,
            "interactive_latency_budget_ms": self.interactive_latency_budget_ms,
            "fixed_decimal_bits": self.fixed_decimal_bits,
            "quantization_tiers": list(self.quantization_tiers),
            "quantization_tier": self.quantization_tier,
            "compression_tier": self.compression_tier,
            "compression_switch_points": list(self.compression_switch_points),
            "low_frequency_threshold": self.low_frequency_threshold,
            "query_top_k": self.query_top_k,
        }

    @property
    def config_hash(self) -> str:
        return sha256_json({"schema": 1, "config": self.to_dict()})

    @property
    def index_config_hash(self) -> str:
        return sha256_json(
            {
                "schema": BLOCK_SCHEMA_VERSION,
                "normalization": NORMALIZATION_VERSION,
                "chunk_size": self.chunk_size,
            }
        )

    @property
    def compression_config_hash(self) -> str:
        return sha256_json(
            {
                "schema": COMPRESSION_SCHEMA_VERSION,
                "tier_policy": self.compression_tier,
                "switch_points": list(self.compression_switch_points),
                "low_frequency_threshold": self.low_frequency_threshold,
            }
        )

    def encoding_parameters_hash(self, tier: int) -> str:
        if tier not in (1, 2):
            raise ValueError("tier must be 1 or 2")
        return sha256_json(
            {
                "schema": COMPRESSION_SCHEMA_VERSION,
                "encoding": "varint-delta",
                "tier": tier,
                "low_frequency_threshold": self.low_frequency_threshold if tier == 2 else None,
                "lossless_filtered_appendix": tier == 2,
            }
        )

    @property
    def query_config_hash(self) -> str:
        return sha256_json(
            {
                "schema": QUERY_SCHEMA_VERSION,
                "fixed_decimal_bits": self.fixed_decimal_bits,
                "quantization_tiers": list(self.quantization_tiers),
                "quantization_tier": self.quantization_tier,
                "compression_tier": self.compression_tier,
                "compression_switch_points": list(self.compression_switch_points),
                "low_frequency_threshold": self.low_frequency_threshold,
                "query_top_k": self.query_top_k,
            }
        )

    @property
    def quantization_quantum(self) -> int:
        return self.quantization_tiers[self.quantization_tier]

    def with_overrides(self, overrides: Mapping[str, Any] | None) -> tuple["AppConfig", list[dict[str, Any]]]:
        if not overrides:
            return self, []
        merged = self.to_dict()
        merged.update(overrides)
        return config_from_mapping(merged)

    def selected_tier(self, block_no: int, total_postings: int) -> tuple[int, str, int | None]:
        if self.compression_tier != "auto":
            return int(self.compression_tier), "override", None
        switches = 0
        matched: int | None = None
        for point in self.compression_switch_points:
            if point <= block_no:
                switches += 1
                matched = point
            else:
                break
        tier = 1 if switches % 2 == 0 else 2
        reason = "auto_toggle" if matched is not None else "auto_initial"
        return tier, reason, matched


@dataclass
class ConfigLoadResult:
    config: AppConfig
    failures: list[dict[str, Any]] = field(default_factory=list)
    source_path: Path | None = None
    raw: dict[str, Any] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return not self.failures


def _canonical_mapping(raw: Mapping[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in raw.items():
        canonical = ALIASES.get(key, key)
        result.setdefault(canonical, value)
    return result


def _parse_int(
    value: Any,
    *,
    field_name: str,
    default: int,
    minimum: int,
    failures: list[dict[str, Any]],
) -> int:
    if not is_int(value):
        failures.append(
            failure(
                "config",
                "E_CONFIG_INVALID_INTEGER",
                f"config:{field_name}",
                f"{field_name} must be an integer",
                value=value,
                fallback=default,
            )
        )
        return default
    if value < minimum:
        failures.append(
            failure(
                "config",
                "E_CONFIG_VALUE_RANGE",
                f"config:{field_name}",
                f"{field_name} must be >= {minimum}",
                value=value,
                fallback=default,
            )
        )
        return default
    return value


def _parse_points(value: Any, default: tuple[int, ...], failures: list[dict[str, Any]]) -> tuple[int, ...]:
    if is_int(value):
        items = [value]
    elif isinstance(value, (list, tuple)):
        items = list(value)
    else:
        failures.append(
            failure(
                "config",
                "E_CONFIG_SWITCH_POINTS",
                "config:compression_switch_points",
                "compression_switch_points must be an integer list",
                value=value,
                fallback=list(default),
            )
        )
        return default
    if not items or any(not is_int(item) or item < 0 for item in items):
        failures.append(
            failure(
                "config",
                "E_CONFIG_SWITCH_POINTS",
                "config:compression_switch_points",
                "compression_switch_points must contain non-negative integers",
                value=value,
                fallback=list(default),
            )
        )
        return default
    if any(left >= right for left, right in zip(items, items[1:])):
        failures.append(
            failure(
                "config",
                "E_CONFIG_SWITCH_POINTS_NOT_INCREASING",
                "config:compression_switch_points",
                "compression_switch_points must be strictly increasing",
                value=value,
                fallback=list(default),
            )
        )
        return default
    return tuple(items)


def _parse_tier(value: Any, failures: list[dict[str, Any]]) -> str | int:
    if value == "auto" or value is None:
        return "auto"
    if is_int(value) and value in (1, 2):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"auto", "adaptive"}:
            return "auto"
        if lowered in {"1", "tier1", "tier-1", "delta"}:
            return 1
        if lowered in {"2", "tier2", "tier-2", "delta-filter"}:
            return 2
    failures.append(
        failure(
            "config",
            "E_CONFIG_COMPRESSION_TIER",
            "config:compression_tier",
            "compression_tier must be auto, 1 or 2",
            value=value,
            fallback="auto",
        )
    )
    return "auto"


def _parse_quantization(
    raw_tiers: Any,
    raw_tier: Any,
    failures: list[dict[str, Any]],
) -> tuple[tuple[int, ...], int]:
    default_tiers = (1, 2, 4, 8)
    if isinstance(raw_tiers, (list, tuple)) and raw_tiers and all(is_int(v) and v > 0 for v in raw_tiers):
        tiers = tuple(int(v) for v in raw_tiers)
    else:
        tiers = default_tiers
        if raw_tiers != list(default_tiers):
            failures.append(
                failure(
                    "config",
                    "E_CONFIG_QUANTIZATION_TIERS",
                    "config:quantization_tiers",
                    "quantization_tiers must be a non-empty list of positive integers",
                    value=raw_tiers,
                    fallback=list(default_tiers),
                )
            )
    tier = raw_tier
    if isinstance(tier, str) and tier.strip().isdigit():
        tier = int(tier)
    if not is_int(tier) or not 0 <= tier < len(tiers):
        failures.append(
            failure(
                "config",
                "E_CONFIG_QUANTIZATION_TIER",
                "config:quantization_tier",
                "quantization_tier must index quantization_tiers",
                value=raw_tier,
                fallback=0,
            )
        )
        tier = 0
    return tiers, tier


def config_from_mapping(raw_mapping: Mapping[str, Any]) -> tuple[AppConfig, list[dict[str, Any]]]:
    raw = _canonical_mapping(raw_mapping)
    failures: list[dict[str, Any]] = []
    defaults = AppConfig()

    chunk_size = _parse_int(
        raw.get("chunk_size", defaults.chunk_size),
        field_name="chunk_size",
        default=defaults.chunk_size,
        minimum=1,
        failures=failures,
    )
    batch_max_docs = _parse_int(
        raw.get("batch_max_docs", defaults.batch_max_docs),
        field_name="batch_max_docs",
        default=defaults.batch_max_docs,
        minimum=1,
        failures=failures,
    )
    concurrency_quota = _parse_int(
        raw.get("concurrency_quota", defaults.concurrency_quota),
        field_name="concurrency_quota",
        default=defaults.concurrency_quota,
        minimum=1,
        failures=failures,
    )
    thread_pool_size = _parse_int(
        raw.get("thread_pool_size", defaults.thread_pool_size),
        field_name="thread_pool_size",
        default=min(defaults.thread_pool_size, concurrency_quota),
        minimum=1,
        failures=failures,
    )
    if thread_pool_size > concurrency_quota:
        failures.append(
            failure(
                "config",
                "E_CONFIG_CONCURRENCY_EXCEEDS_QUOTA",
                "config:thread_pool_size",
                "thread_pool_size must not exceed concurrency_quota",
                value=thread_pool_size,
                quota=concurrency_quota,
                fallback=concurrency_quota,
            )
        )
        thread_pool_size = concurrency_quota
    global_token_budget = _parse_int(
        raw.get("global_token_budget", defaults.global_token_budget),
        field_name="global_token_budget",
        default=defaults.global_token_budget,
        minimum=0,
        failures=failures,
    )
    per_shard_token_quota = _parse_int(
        raw.get("per_shard_token_quota", defaults.per_shard_token_quota),
        field_name="per_shard_token_quota",
        default=defaults.per_shard_token_quota,
        minimum=0,
        failures=failures,
    )
    snapshot_interval = _parse_int(
        raw.get("snapshot_commit_interval_batches", defaults.snapshot_commit_interval_batches),
        field_name="snapshot_commit_interval_batches",
        default=defaults.snapshot_commit_interval_batches,
        minimum=1,
        failures=failures,
    )
    latency_budget = _parse_int(
        raw.get("interactive_latency_budget_ms", defaults.interactive_latency_budget_ms),
        field_name="interactive_latency_budget_ms",
        default=defaults.interactive_latency_budget_ms,
        minimum=0,
        failures=failures,
    )
    fixed_bits = _parse_int(
        raw.get("fixed_decimal_bits", defaults.fixed_decimal_bits),
        field_name="fixed_decimal_bits",
        default=defaults.fixed_decimal_bits,
        minimum=1,
        failures=failures,
    )
    if fixed_bits > 30:
        failures.append(
            failure(
                "config",
                "E_CONFIG_FIXED_DECIMAL_BITS",
                "config:fixed_decimal_bits",
                "fixed_decimal_bits must be <= 30",
                value=fixed_bits,
                fallback=defaults.fixed_decimal_bits,
            )
        )
        fixed_bits = defaults.fixed_decimal_bits
    quantization_tiers, quantization_tier = _parse_quantization(
        raw.get("quantization_tiers", list(defaults.quantization_tiers)),
        raw.get("quantization_tier", defaults.quantization_tier),
        failures,
    )
    compression_tier = _parse_tier(raw.get("compression_tier", defaults.compression_tier), failures)
    switch_points = _parse_points(
        raw.get("compression_switch_points", list(defaults.compression_switch_points)),
        defaults.compression_switch_points,
        failures,
    )
    low_frequency_threshold = _parse_int(
        raw.get("low_frequency_threshold", defaults.low_frequency_threshold),
        field_name="low_frequency_threshold",
        default=defaults.low_frequency_threshold,
        minimum=0,
        failures=failures,
    )
    query_top_k = _parse_int(
        raw.get("query_top_k", defaults.query_top_k),
        field_name="query_top_k",
        default=defaults.query_top_k,
        minimum=1,
        failures=failures,
    )
    config = AppConfig(
        chunk_size=chunk_size,
        batch_max_docs=batch_max_docs,
        thread_pool_size=thread_pool_size,
        concurrency_quota=concurrency_quota,
        global_token_budget=global_token_budget,
        per_shard_token_quota=per_shard_token_quota,
        snapshot_commit_interval_batches=snapshot_interval,
        interactive_latency_budget_ms=latency_budget,
        fixed_decimal_bits=fixed_bits,
        quantization_tiers=quantization_tiers,
        quantization_tier=quantization_tier,
        compression_tier=compression_tier,
        compression_switch_points=switch_points,
        low_frequency_threshold=low_frequency_threshold,
        query_top_k=query_top_k,
    )
    return config, sort_failures(failures)


def load_config(path: Path, overrides: Mapping[str, Any] | None = None) -> ConfigLoadResult:
    failures: list[dict[str, Any]] = []
    raw: dict[str, Any] = {}
    if path.exists():
        try:
            loaded = json.loads(path.read_text(encoding="utf-8-sig"))
            if isinstance(loaded, dict):
                raw = loaded
            else:
                failures.append(
                    failure(
                        "config",
                        "E_CONFIG_JSON_OBJECT",
                        "config:root",
                        "configuration root must be a JSON object",
                        fallback=copy.deepcopy(DEFAULT_CONFIG),
                    )
                )
        except (OSError, json.JSONDecodeError) as exc:
            failures.append(
                failure(
                    "config",
                    "E_CONFIG_JSON_INVALID",
                    "config:root",
                    f"configuration could not be parsed: {exc}",
                    fallback=copy.deepcopy(DEFAULT_CONFIG),
                )
            )
    else:
        failures.append(
            failure(
                "config",
                "E_CONFIG_FILE_MISSING",
                "config:root",
                "configuration file is missing; built-in defaults are used",
                path=str(path),
                fallback=copy.deepcopy(DEFAULT_CONFIG),
            )
        )
    if overrides:
        raw.update(overrides)
    config, config_failures = config_from_mapping(raw)
    failures.extend(config_failures)
    return ConfigLoadResult(config=config, failures=sort_failures(failures), source_path=path, raw=raw)


def merge_overrides(config: AppConfig, overrides: Mapping[str, Any]) -> AppConfig:
    merged = config.to_dict()
    merged.update(overrides)
    updated, _ = config_from_mapping(merged)
    return updated
