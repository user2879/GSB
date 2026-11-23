"""配置加载与校验（E_CONFIG_INVALID）。

配置 YAML 被解析为不可变的 AppConfig，并派生确定性 config_hash。
任何字段非法都收集为结构化错误，而非抛出中断。
"""
from __future__ import annotations

import math
from dataclasses import dataclass, asdict
from typing import Any

import yaml

from .errors import E_CONFIG_INVALID, FailureEntry, FailureLog
from .fixedpoint import (
    ALLOWED_DECIMAL_PLACES,
    ALLOWED_QUANT_MODES,
    to_fixed,
)

ALLOWED_TOKENIZER_VERSIONS = ("v1", "v2")


@dataclass(frozen=True)
class RewardWeights:
    prompt: float
    chosen: float
    rejected: float

    def to_fixed(self, decimal_places: int) -> dict[str, int]:
        return {
            "prompt": to_fixed(self.prompt, decimal_places),
            "chosen": to_fixed(self.chosen, decimal_places),
            "rejected": to_fixed(self.rejected, decimal_places),
        }


@dataclass(frozen=True)
class AppConfig:
    tokenizer_version: str
    normalize: bool
    feature_dim: int
    hash_buckets: int
    weights: RewardWeights
    boundary_threshold: float
    decimal_places: int
    quant_mode: str
    batch_size: int
    learning_rate: float
    global_pair_budget: int
    per_batch_quota: int
    max_workers: int
    snapshot_commit_pairs: int
    time_partition_ms: int

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        return data

    def canonical_yaml(self) -> str:
        # 规范化序列化用于 config_hash，sort_keys 保证跨机器一致
        return yaml.safe_dump(
            self.to_dict(), sort_keys=True, allow_unicode=True, default_flow_style=False
        )


def _is_pos_int(v: object) -> bool:
    return isinstance(v, int) and not isinstance(v, bool) and v > 0


def _is_finite_positive(v: object) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v) and v > 0


def load_config(path: str, failures: FailureLog) -> AppConfig | None:
    """加载并校验配置；非法时写入 E_CONFIG_INVALID 并返回 None。"""
    try:
        with open(path, "r", encoding="utf-8") as fh:
            raw = yaml.safe_load(fh)
    except (OSError, yaml.YAMLError) as exc:
        failures.add(FailureEntry(
            sort_key="config", code=E_CONFIG_INVALID, stage="load_config",
            message=f"cannot parse yaml: {exc}",
        ))
        return None

    if not isinstance(raw, dict):
        failures.add(FailureEntry(
            sort_key="config", code=E_CONFIG_INVALID, stage="load_config",
            message="top-level config must be a mapping",
        ))
        return None

    problems: list[str] = []

    tok = raw.get("tokenizer_version")
    if tok not in ALLOWED_TOKENIZER_VERSIONS:
        problems.append(f"tokenizer_version {tok!r} not in {ALLOWED_TOKENIZER_VERSIONS}")

    normalize = raw.get("normalize", True)
    if not isinstance(normalize, bool):
        problems.append("normalize must be bool")

    feature_dim = raw.get("feature_dim")
    if not _is_pos_int(feature_dim):
        problems.append("feature_dim must be positive int")

    hash_buckets = raw.get("hash_buckets")
    if not _is_pos_int(hash_buckets):
        problems.append("hash_buckets must be positive int")
    elif isinstance(feature_dim, int) and hash_buckets > feature_dim:
        problems.append("hash_buckets must not exceed feature_dim")

    weights_raw = raw.get("weights")
    weights: RewardWeights | None = None
    if not isinstance(weights_raw, dict) or not {"prompt", "chosen", "rejected"} <= set(weights_raw):
        problems.append("weights must contain prompt/chosen/rejected")
    else:
        wvals = []
        for name in ("prompt", "chosen", "rejected"):
            wv = weights_raw[name]
            if not isinstance(wv, (int, float)) or isinstance(wv, bool) or not math.isfinite(wv):
                problems.append(f"weights.{name} must be finite number")
                wvals.append(0.0)
            else:
                wvals.append(float(wv))
        if len(wvals) == 3:
            weights = RewardWeights(wvals[0], wvals[1], wvals[2])

    threshold = raw.get("boundary_threshold")
    if not isinstance(threshold, (int, float)) or isinstance(threshold, bool) or not math.isfinite(threshold):
        problems.append("boundary_threshold must be finite number")
    elif threshold < 0:
        problems.append("boundary_threshold must be >= 0")

    decimal_places = raw.get("decimal_places")
    if decimal_places not in ALLOWED_DECIMAL_PLACES:
        problems.append(f"decimal_places {decimal_places!r} not in {ALLOWED_DECIMAL_PLACES}")

    quant_mode = raw.get("quant_mode")
    if quant_mode not in ALLOWED_QUANT_MODES:
        problems.append(f"quant_mode {quant_mode!r} not in {ALLOWED_QUANT_MODES}")

    batch_size = raw.get("batch_size")
    if not _is_pos_int(batch_size):
        problems.append("batch_size must be positive int")

    learning_rate = raw.get("learning_rate")
    if not _is_finite_positive(learning_rate):
        problems.append("learning_rate must be positive finite number")

    global_pair_budget = raw.get("global_pair_budget")
    if not _is_pos_int(global_pair_budget):
        problems.append("global_pair_budget must be positive int")

    per_batch_quota = raw.get("per_batch_quota")
    if not _is_pos_int(per_batch_quota):
        problems.append("per_batch_quota must be positive int")

    max_workers = raw.get("max_workers")
    if not _is_pos_int(max_workers):
        problems.append("max_workers must be positive int")

    snapshot_commit_pairs = raw.get("snapshot_commit_pairs")
    if not _is_pos_int(snapshot_commit_pairs):
        problems.append("snapshot_commit_pairs must be positive int")

    time_partition_ms = raw.get("time_partition_ms", 1000)
    if not _is_pos_int(time_partition_ms):
        problems.append("time_partition_ms must be positive int")

    if problems:
        for msg in problems:
            failures.add(FailureEntry(
                sort_key="config", code=E_CONFIG_INVALID,
                stage="load_config", message=msg,
            ))
        return None

    assert weights is not None
    return AppConfig(
        tokenizer_version=str(tok),
        normalize=normalize,
        feature_dim=int(feature_dim),
        hash_buckets=int(hash_buckets),
        weights=weights,
        boundary_threshold=float(threshold),
        decimal_places=int(decimal_places),
        quant_mode=str(quant_mode),
        batch_size=int(batch_size),
        learning_rate=float(learning_rate),
        global_pair_budget=int(global_pair_budget),
        per_batch_quota=int(per_batch_quota),
        max_workers=int(max_workers),
        snapshot_commit_pairs=int(snapshot_commit_pairs),
        time_partition_ms=int(time_partition_ms),
    )
