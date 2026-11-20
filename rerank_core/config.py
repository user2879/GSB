"""精排配置与特征分组表的加载、校验与指纹。"""

from __future__ import annotations

import json
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Any

from .errors import ConfigInvalidError
from .fixedpoint import parse_exact, stable_hash

MAX_FRAC_BITS = 12
MIN_FRAC_BITS = 0


@dataclass(frozen=True)
class RerankConfig:
    recall_channels: tuple[str, ...]
    dense_dim: int
    sparse_dim: int
    weights: tuple[Fraction, ...]
    normalization: str                 # "minmax" | "zscore"
    fusion_temperature: Fraction
    score_clip: tuple[Fraction, Fraction]
    frac_bits: int
    quant_levels: int
    rerank_window: int
    diversity_penalty: Fraction
    similarity_threshold: Fraction
    latency_budget_ms: float
    concurrency: int
    snapshot_every_commands: int
    groups: tuple[dict[str, Any], ...]
    feature_names: tuple[str, ...]

    @property
    def feature_dim(self) -> int:
        return len(self.recall_channels) + self.dense_dim + self.sparse_dim

    def fingerprint(self) -> str:
        payload = {
            "recall_channels": list(self.recall_channels),
            "dense_dim": self.dense_dim,
            "sparse_dim": self.sparse_dim,
            "weights": [str(w) for w in self.weights],
            "normalization": self.normalization,
            "fusion_temperature": str(self.fusion_temperature),
            "score_clip": [str(self.score_clip[0]), str(self.score_clip[1])],
            "frac_bits": self.frac_bits,
            "quant_levels": self.quant_levels,
            "rerank_window": self.rerank_window,
            "diversity_penalty": str(self.diversity_penalty),
            "similarity_threshold": str(self.similarity_threshold),
            "concurrency": self.concurrency,
            "snapshot_every_commands": self.snapshot_every_commands,
            "groups": list(self.groups),
            "feature_names": list(self.feature_names),
        }
        return stable_hash(payload)

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "recall_channels": list(self.recall_channels),
            "dense_dim": self.dense_dim,
            "sparse_dim": self.sparse_dim,
            "weights": [str(w) for w in self.weights],
            "normalization": self.normalization,
            "fusion_temperature": str(self.fusion_temperature),
            "score_clip": [str(self.score_clip[0]), str(self.score_clip[1])],
            "frac_bits": self.frac_bits,
            "quant_levels": self.quant_levels,
            "rerank_window": self.rerank_window,
            "diversity_penalty": str(self.diversity_penalty),
            "similarity_threshold": str(self.similarity_threshold),
            "latency_budget_ms": self.latency_budget_ms,
            "concurrency": self.concurrency,
            "snapshot_every_commands": self.snapshot_every_commands,
            "groups": [dict(g) for g in self.groups],
            "feature_names": list(self.feature_names),
        }


def _load_json(path: Path) -> Any:
    try:
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except FileNotFoundError as exc:
        raise ConfigInvalidError(f"文件不存在: {path}", details={"path": str(path)}) from exc
    except json.JSONDecodeError as exc:
        raise ConfigInvalidError(
            f"JSON 解析失败: {path}: {exc.msg}",
            details={"path": str(path), "line": exc.lineno},
        ) from exc


def _require_number(raw: Any, name: str) -> Fraction:
    try:
        return parse_exact(raw, field=name)
    except (TypeError, ValueError) as exc:
        raise ConfigInvalidError(f"配置项 {name} 非法: {exc}") from exc


def load_config(config_path: str | Path, groups_path: str | Path) -> RerankConfig:
    raw = _load_json(Path(config_path))
    groups_raw = _load_json(Path(groups_path))
    if not isinstance(raw, dict):
        raise ConfigInvalidError("精排配置根节点必须是 JSON 对象")

    channels = raw.get("recall_channels")
    if not isinstance(channels, list) or not channels or not all(
        isinstance(c, str) and c for c in channels
    ):
        raise ConfigInvalidError("recall_channels 必须是非空字符串数组")
    if len(set(channels)) != len(channels):
        raise ConfigInvalidError("recall_channels 存在重复通道名")

    dense_dim = raw.get("dense_dim")
    sparse_dim = raw.get("sparse_dim")
    if not isinstance(dense_dim, int) or isinstance(dense_dim, bool) or dense_dim < 0:
        raise ConfigInvalidError("dense_dim 必须是非负整数")
    if not isinstance(sparse_dim, int) or isinstance(sparse_dim, bool) or sparse_dim < 0:
        raise ConfigInvalidError("sparse_dim 必须是非负整数")

    feature_dim = len(channels) + dense_dim + sparse_dim
    feature_names = (
        [f"recall:{c}" for c in channels]
        + [f"dense:{i}" for i in range(dense_dim)]
        + [f"sparse:{i}" for i in range(sparse_dim)]
    )

    weights_raw = raw.get("weights")
    if not isinstance(weights_raw, list) or not weights_raw:
        raise ConfigInvalidError("weights 必须是非空数组")
    if len(weights_raw) != feature_dim:
        raise ConfigInvalidError(
            f"权重长度 {len(weights_raw)} 与特征维度 {feature_dim} 不一致",
            details={"weights_len": len(weights_raw), "feature_dim": feature_dim},
        )
    weights = tuple(_require_number(w, f"weights[{i}]") for i, w in enumerate(weights_raw))

    normalization = raw.get("normalization", "minmax")
    if normalization not in ("minmax", "zscore"):
        raise ConfigInvalidError("normalization 仅支持 minmax 或 zscore")

    temperature = _require_number(raw.get("fusion_temperature", 1), "fusion_temperature")
    if temperature <= 0:
        raise ConfigInvalidError("fusion_temperature 必须为正数")

    clip_raw = raw.get("score_clip", [-1000, 1000])
    if (not isinstance(clip_raw, list) or len(clip_raw) != 2):
        raise ConfigInvalidError("score_clip 必须是长度为 2 的数组")
    clip_lo = _require_number(clip_raw[0], "score_clip[0]")
    clip_hi = _require_number(clip_raw[1], "score_clip[1]")
    if clip_lo > clip_hi:
        raise ConfigInvalidError("score_clip 下界不得大于上界")

    frac_bits = raw.get("frac_bits", 4)
    if not isinstance(frac_bits, int) or isinstance(frac_bits, bool) or not (
        MIN_FRAC_BITS <= frac_bits <= MAX_FRAC_BITS
    ):
        raise ConfigInvalidError(
            f"frac_bits 必须是 [{MIN_FRAC_BITS}, {MAX_FRAC_BITS}] 内的整数"
        )

    quant_levels = raw.get("quant_levels", 256)
    if not isinstance(quant_levels, int) or isinstance(quant_levels, bool) or quant_levels < 2:
        raise ConfigInvalidError("quant_levels 必须是 >= 2 的整数")

    rerank_window = raw.get("rerank_window")
    if not isinstance(rerank_window, int) or isinstance(rerank_window, bool) or rerank_window <= 0:
        raise ConfigInvalidError("rerank_window 必须是正整数")

    penalty = _require_number(raw.get("diversity_penalty", 0), "diversity_penalty")
    if not (Fraction(0) <= penalty <= 1):
        raise ConfigInvalidError("diversity_penalty 必须位于 [0,1]")

    threshold = _require_number(raw.get("similarity_threshold", 1), "similarity_threshold")
    if not (Fraction(0) <= threshold <= 1):
        raise ConfigInvalidError("similarity_threshold 必须位于 [0,1]")

    budget = raw.get("latency_budget_ms")
    if not isinstance(budget, (int, float)) or isinstance(budget, bool) or budget <= 0:
        raise ConfigInvalidError("latency_budget_ms 必须是正数")

    concurrency = raw.get("concurrency")
    if not isinstance(concurrency, int) or isinstance(concurrency, bool) or concurrency <= 0:
        raise ConfigInvalidError("concurrency 必须是正整数")

    snapshot_every = raw.get("snapshot_every_commands", 1)
    if not isinstance(snapshot_every, int) or isinstance(snapshot_every, bool) or snapshot_every <= 0:
        raise ConfigInvalidError("snapshot_every_commands 必须是正整数")

    groups = _validate_groups(groups_raw, feature_dim)

    return RerankConfig(
        recall_channels=tuple(channels),
        dense_dim=dense_dim,
        sparse_dim=sparse_dim,
        weights=weights,
        normalization=normalization,
        fusion_temperature=temperature,
        score_clip=(clip_lo, clip_hi),
        frac_bits=frac_bits,
        quant_levels=quant_levels,
        rerank_window=rerank_window,
        diversity_penalty=penalty,
        similarity_threshold=threshold,
        latency_budget_ms=float(budget),
        concurrency=concurrency,
        snapshot_every_commands=snapshot_every,
        groups=tuple(groups),
        feature_names=tuple(feature_names),
    )


def _validate_groups(groups_raw: Any, feature_dim: int) -> list[dict[str, Any]]:
    if not isinstance(groups_raw, dict) or not isinstance(groups_raw.get("groups"), list):
        raise ConfigInvalidError("特征分组表根节点须为含 groups 数组的对象")
    normalized: list[dict[str, Any]] = []
    seen: list[int] = []
    for pos, group in enumerate(groups_raw["groups"]):
        if not isinstance(group, dict):
            raise ConfigInvalidError(f"分组表第 {pos} 项必须是对象")
        name = group.get("name")
        indexes = group.get("feature_indexes")
        conflict = group.get("conflict", False)
        if not isinstance(name, str) or not name:
            raise ConfigInvalidError(f"分组表第 {pos} 项 name 非法")
        if not isinstance(indexes, list) or not indexes:
            raise ConfigInvalidError(f"分组 {name} 的 feature_indexes 必须是非空数组")
        for idx in indexes:
            if not isinstance(idx, int) or isinstance(idx, bool) or not (0 <= idx < feature_dim):
                raise ConfigInvalidError(
                    f"分组 {name} 存在越界特征下标: {idx}（特征维度 {feature_dim}）"
                )
            if idx in seen:
                raise ConfigInvalidError(f"特征下标 {idx} 在分组表中重复出现")
            seen.append(idx)
        if not isinstance(conflict, bool):
            raise ConfigInvalidError(f"分组 {name} 的 conflict 必须是布尔值")
        normalized.append(
            {"name": name, "feature_indexes": list(indexes), "conflict": conflict}
        )
    if sorted(seen) != list(range(feature_dim)):
        raise ConfigInvalidError(
            f"分组表特征覆盖不完整：覆盖 {sorted(seen)}，期望 0..{feature_dim - 1}"
        )
    return normalized


def group_of_feature(config: RerankConfig) -> dict[int, str]:
    mapping: dict[int, str] = {}
    for group in config.groups:
        for idx in group["feature_indexes"]:
            mapping[idx] = group["name"]
    return mapping


def groups_fingerprint(config: RerankConfig) -> str:
    return stable_hash([dict(g) for g in config.groups])
