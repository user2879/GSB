"""Deterministic category crossing and cardinality-budget degradation."""

from __future__ import annotations

from typing import Any

from .config import ENGINE_REVISION, PipelineConfig
from .sample import LoadedSample, MISSING_CATEGORY
from .util import fingerprint, sha256_bytes, stable_combinations


def combination_id(columns: tuple[str, ...]) -> str:
    return "+".join(columns)


def feature_key(columns: tuple[str, ...]) -> str:
    return "x__" + "__".join(columns)


def _scalar_token(value: Any) -> str:
    from .sample import _scalar_key

    return _scalar_key(value)


def _hash64(material: str) -> int:
    return int.from_bytes(bytes.fromhex(sha256_bytes(material.encode("utf-8")))[:8], "big")


def enumerate_combinations(config: PipelineConfig) -> list[tuple[str, ...]]:
    order = int(config.raw["crossing"]["order"])
    return stable_combinations(config.valid_whitelist, order)


def cross_data_fingerprint(columns: tuple[str, ...], sample: LoadedSample) -> str:
    return fingerprint({name: sample.column_fingerprints.get(name, "") for name in columns})


def cross_params_fingerprint(config: PipelineConfig, columns: tuple[str, ...]) -> str:
    crossing = config.raw["crossing"]
    return fingerprint(
        {
            "engine_revision": ENGINE_REVISION,
            "columns": list(columns),
            "order": len(columns),
            "cardinality_budget": int(crossing["cardinality_budget"]),
            "overflow_strategy": str(crossing["overflow_strategy"]),
            "hash_buckets": int(crossing["hash_buckets"]),
            "hash_seed": int(crossing["hash_seed"]),
        }
    )


def cross_cache_key(
    config: PipelineConfig,
    tier: str,
    columns: tuple[str, ...],
    data_fingerprint: str,
    params_fingerprint: str,
) -> str:
    return fingerprint(
        {
            "stage": "cross_encoding",
            "engine_revision": ENGINE_REVISION,
            "precision_fingerprint": config.precision_fingerprint(tier),
            "combination": combination_id(columns),
            "data_fingerprint": data_fingerprint,
            "params_fingerprint": params_fingerprint,
        }
    )


def encode_combination(
    columns: tuple[str, ...],
    values: dict[str, list[Any]],
    *,
    crossing: dict[str, Any],
) -> dict[str, Any]:
    """Return the canonical encoded feature and its degradation metadata."""

    identifier = combination_id(columns)
    rows = len(values[columns[0]]) if columns else 0
    composite_keys: list[str] = []
    for row_index in range(rows):
        tokens = []
        for column in columns:
            token = _scalar_token(values[column][row_index])
            tokens.append(f"{len(token)}:{token}")
        composite_keys.append("|".join(tokens))
    unique_keys = sorted(set(composite_keys))
    unique_count = len(unique_keys)
    budget = int(crossing["cardinality_budget"])
    strategy = str(crossing["overflow_strategy"])
    hash_seed = int(crossing["hash_seed"])
    degradation: dict[str, Any] | None = None

    if unique_count > budget:
        if strategy == "skip":
            encoded: list[int] = []
            mode = "skipped"
            encoded_cardinality = 0
            action = "skip"
            buckets = 0
        else:
            buckets = min(int(crossing["hash_buckets"]), budget)
            encoded = [
                _hash64(f"{hash_seed}|{identifier}|{key}") % buckets
                for key in composite_keys
            ]
            mode = "hash_bucket"
            encoded_cardinality = len(set(encoded))
            action = "hash_bucket"
        degradation = {
            "combination": identifier,
            "columns": list(columns),
            "unique_count": unique_count,
            "cardinality_budget": budget,
            "strategy": strategy,
            "action": action,
            "hash_buckets": buckets if strategy == "hash_bucket" else 0,
            "encoded_cardinality": encoded_cardinality,
            "sort_key": f"cross_encoding/{identifier}/degradation",
        }
    else:
        hashes = {
            key: _hash64(f"{hash_seed}|{identifier}|{key}") for key in unique_keys
        }
        key_to_code = {
            key: code
            for code, key in enumerate(sorted(unique_keys, key=lambda item: (hashes[item], item)))
        }
        encoded = [key_to_code[key] for key in composite_keys]
        mode = "dense_hash"
        encoded_cardinality = unique_count
        buckets = 0

    return {
        "content": {
            "feature_key": feature_key(columns),
            "combination": identifier,
            "columns": list(columns),
            "encoded": encoded,
            "unique_count": unique_count,
            "encoding_mode": mode,
            "encoded_cardinality": encoded_cardinality,
            "hash_buckets": buckets,
        },
        "degradation": degradation,
    }


def build_cross_tasks(config: PipelineConfig, sample: LoadedSample) -> list[dict[str, Any]]:
    crossing = config.raw["crossing"]
    tasks: list[dict[str, Any]] = []
    for columns in enumerate_combinations(config):
        data_fp = cross_data_fingerprint(columns, sample)
        params_fp = cross_params_fingerprint(config, columns)
        for tier in ("single", "double"):
            tasks.append(
                {
                    "columns": columns,
                    "combination": combination_id(columns),
                    "feature_key": feature_key(columns),
                    "tier": tier,
                    "sort_key": f"cross_encoding/{combination_id(columns)}/tier={tier}",
                    "cache_key": cross_cache_key(config, tier, columns, data_fp, params_fp),
                    "data_fingerprint": data_fp,
                    "params_fingerprint": params_fp,
                    "crossing": crossing,
                }
            )
    return sorted(tasks, key=lambda item: item["sort_key"])
