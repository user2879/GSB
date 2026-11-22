"""通道融合配置：解析、校验、规范化哈希。"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

from .errors import E_CONFIG_INVALID, Failure
from .serialization import canonical_json, content_hash, read_json, to_native

NORMALIZATIONS = ("minmax", "zscore")


@dataclass
class ChannelSpec:
    name: str
    weight: float
    normalization: str


@dataclass
class FusionConfig:
    channels: Dict[str, ChannelSpec]
    fusion_temperature: float
    feature_dim: int
    feature_weights: List[float]
    score_clip: Tuple[float, float]
    frac_bits_choices: List[int]
    gears: List[int]
    tolerance: float
    recompute_budget: int
    concurrency: int
    snapshot_commit_interval_batches: int
    raw: Dict[str, Any]

    @property
    def channel_names_sorted(self) -> List[str]:
        return sorted(self.channels)

    def hash(self) -> str:
        return content_hash(self.raw)

    def to_dict(self) -> Dict[str, Any]:
        return to_native(self.raw)


def _is_int(v: Any) -> bool:
    return isinstance(v, int) and not isinstance(v, bool)


def _finite(v: Any) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool) \
        and math.isfinite(float(v))


def _strict_pos_int_sorted(values: Any) -> bool:
    if not isinstance(values, list) or not values:
        return False
    if not all(_is_int(v) and v > 0 for v in values):
        return False
    return all(values[i] < values[i + 1] for i in range(len(values) - 1))


def load_config(path: str) -> Tuple[FusionConfig | None, List[Failure]]:
    failures: List[Failure] = []
    try:
        raw = read_json(path)
    except (OSError, ValueError) as exc:
        return None, [Failure(E_CONFIG_INVALID, path, f"配置无法读取: {exc}")]
    if not isinstance(raw, dict):
        return None, [Failure(E_CONFIG_INVALID, path, "配置根节点必须是对象")]

    def bad(reason: str) -> None:
        failures.append(Failure(E_CONFIG_INVALID, path, reason))

    channels_raw = raw.get("channels")
    channels: Dict[str, ChannelSpec] = {}
    if not isinstance(channels_raw, dict) or not channels_raw:
        bad("channels 必须是非空对象")
    else:
        for name in sorted(channels_raw):
            spec = channels_raw[name]
            if not isinstance(name, str) or not name:
                bad("通道名必须为非空字符串")
                continue
            if not isinstance(spec, dict):
                bad(f"通道 {name} 配置必须是对象")
                continue
            if not _finite(spec.get("weight")):
                bad(f"通道 {name} 权重必须为有限数")
                continue
            norm = spec.get("normalization")
            if norm not in NORMALIZATIONS:
                bad(f"通道 {name} 归一化方式非法: {norm}")
                continue
            channels[name] = ChannelSpec(name, float(spec["weight"]), norm)

    temp = raw.get("fusion_temperature")
    if not _finite(temp) or float(temp) <= 0:
        bad("fusion_temperature 必须为正有限数")
        temp = 1.0

    feature_dim = raw.get("feature_dim")
    if not _is_int(feature_dim) or feature_dim <= 0:
        bad("feature_dim 必须为正整数")
        feature_dim = 1

    fw = raw.get("feature_weights")
    feature_weights: List[float] = []
    if not isinstance(fw, list) or len(fw) != feature_dim \
            or not all(_finite(v) for v in fw):
        bad("feature_weights 长度须等于 feature_dim 且全部有限")
        feature_weights = [0.0] * int(feature_dim)
    else:
        feature_weights = [float(v) for v in fw]

    clip = raw.get("score_clip")
    if (not isinstance(clip, list) or len(clip) != 2
            or not all(_finite(v) for v in clip) or clip[0] > clip[1]):
        bad("score_clip 须为 [low, high] 且 low <= high")
        clip = [-1.0e9, 1.0e9]

    fb = raw.get("frac_bits_choices")
    if not _strict_pos_int_sorted(fb):
        bad("frac_bits_choices 须为非空且严格递增的正整数集合")
        fb = [8]

    gears = raw.get("gears")
    if not _strict_pos_int_sorted(gears):
        bad("gears 须为非空且严格递增的正整数集合")
        gears = [1]

    tolerance = raw.get("tolerance")
    if not _finite(tolerance) or float(tolerance) < 0:
        bad("tolerance 必须为非负有限数")
        tolerance = 0.0

    budget = raw.get("recompute_budget")
    if not _is_int(budget) or budget <= 0:
        bad("recompute_budget 必须为正整数")
        budget = 1

    concurrency = raw.get("concurrency")
    if not _is_int(concurrency) or concurrency <= 0:
        bad("concurrency 必须为正整数")
        concurrency = 1

    interval = raw.get("snapshot_commit_interval_batches")
    if not _is_int(interval) or interval <= 0:
        bad("snapshot_commit_interval_batches 必须为正整数")
        interval = 1

    # 档位步长必须小于 1，保证整数域仍能区分 (0,1) 的归一化值。
    for f in fb:
        for g in gears:
            if g >= (1 << f):
                bad(f"非法量化组合 frac_bits={f} gear={g}: 步长 >= 1")

    if failures:
        return None, failures

    cfg = FusionConfig(
        channels=channels,
        fusion_temperature=float(temp),
        feature_dim=int(feature_dim),
        feature_weights=feature_weights,
        score_clip=(float(clip[0]), float(clip[1])),
        frac_bits_choices=[int(v) for v in fb],
        gears=[int(v) for v in gears],
        tolerance=float(tolerance),
        recompute_budget=int(budget),
        concurrency=int(concurrency),
        snapshot_commit_interval_batches=int(interval),
        raw=to_native(raw),
    )
    return cfg, []


def config_canonical_hash(cfg: FusionConfig) -> str:
    return content_hash(canonical_json(cfg.raw))
