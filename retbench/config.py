"""对齐/召回配置与交互命令流的数据模型及严格校验。"""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass, field, asdict
from typing import Any

from .errors import BenchError, E_CONFIG_INVALID
from .fixedpoint import ALLOWED_FRAC_BITS, ALLOWED_TIERS, validate as validate_quant
from .miniyaml import parse_yaml

KNOWN_CHANNELS = ("lexical", "visual", "hybrid")
KNOWN_OPS = (
    "set_scale_tier",
    "set_weight",
    "set_quant",
    "replace_image",
    "replace_text",
    "enable_channel",
    "disable_channel",
)
RAW_STAT_DIM = 5  # 颜色均值 3 + 梯度强度 1 + 边缘密度 1
REDUCED_STAT_DIM = 4  # 降精度策略丢弃边缘密度


def _require(cond: bool, message: str, context: dict[str, Any] | None = None) -> None:
    if not cond:
        raise BenchError(E_CONFIG_INVALID, message, context)


def _is_finite_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))


@dataclass
class StrategyEntry:
    id: str
    capacity_samples: int
    capacity_primitives: int
    capacity_assertion_len: float
    unit_cost_limit: int
    feature_precision: str          # full | reduced
    tile_range: str                 # full | center_half
    channels: list[str]
    cap_discount: float

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "StrategyEntry":
        for key in (
            "id", "capacity_samples", "capacity_primitives",
            "capacity_assertion_len", "unit_cost_limit",
            "feature_precision", "tile_range", "channels", "cap_discount",
        ):
            _require(key in data, f"策略表条目缺少字段 {key}")
        _require(data["feature_precision"] in ("full", "reduced"),
                 f"策略 {data['id']} feature_precision 非法")
        _require(data["tile_range"] in ("full", "center_half"),
                 f"策略 {data['id']} tile_range 非法")
        channels = list(data["channels"])
        _require(channels and all(c in KNOWN_CHANNELS for c in channels),
                 f"策略 {data['id']} 通道组合非法")
        _require(0 < float(data["cap_discount"]) <= 1.0,
                 f"策略 {data['id']} cap_discount 须在 (0,1]")
        return cls(
            id=str(data["id"]),
            capacity_samples=int(data["capacity_samples"]),
            capacity_primitives=int(data["capacity_primitives"]),
            capacity_assertion_len=float(data["capacity_assertion_len"]),
            unit_cost_limit=int(data["unit_cost_limit"]),
            feature_precision=str(data["feature_precision"]),
            tile_range=str(data["tile_range"]),
            channels=channels,
            cap_discount=float(data["cap_discount"]),
        )


@dataclass
class Config:
    raw: dict[str, Any]
    resolution: tuple[int, int]
    vocabulary: dict[str, list[str]]
    tile_size: int
    visual_dim: int
    text_dim: int
    noise_level: int
    default_assertion_count: int
    weights: dict[str, float]
    channels_enabled: dict[str, bool]
    channel_weights: dict[str, float]
    fusion_temperature: float
    candidate_cap: int
    frac_bits: int
    tier: str
    scale_tiers: list[int]
    active_tier: int
    latency_budget_ms: int
    thread_pool_size: int
    commit_interval_commands: int
    temp_dir: str
    snapshot_dir: str
    strategy_table: list[StrategyEntry]
    work_ms_per_task: int

    @property
    def grid(self) -> tuple[int, int]:
        return (self.resolution[1] // self.tile_size, self.resolution[0] // self.tile_size)

    def to_fingerprint_obj(self) -> dict[str, Any]:
        """参与配置指纹的稳定结构（排除运行期目录等易变字段）。"""

        return {
            "resolution": list(self.resolution),
            "vocabulary": self.vocabulary,
            "tile_size": self.tile_size,
            "visual_dim": self.visual_dim,
            "text_dim": self.text_dim,
            "noise_level": self.noise_level,
            "default_assertion_count": self.default_assertion_count,
            "weights": self.weights,
            "channel_weights": self.channel_weights,
            "fusion_temperature": self.fusion_temperature,
            "candidate_cap": self.candidate_cap,
            "frac_bits": self.frac_bits,
            "tier": self.tier,
            "scale_tiers": self.scale_tiers,
            "strategy_table": [asdict(s) for s in self.strategy_table],
            "work_ms_per_task": self.work_ms_per_task,
        }


def load_config(path: str) -> Config:
    with open(path, "r", encoding="utf-8") as handle:
        data = parse_yaml(handle.read())
    return build_config(data)


def build_config(data: dict[str, Any]) -> Config:
    _require(isinstance(data, dict), "配置顶层必须是映射")

    res = data.get("resolution")
    _require(isinstance(res, list) and len(res) == 2
             and all(isinstance(v, int) and v > 0 for v in res),
             "resolution 必须为两个正整数 [width, height]")
    width, height = int(res[0]), int(res[1])

    vocab = data.get("vocabulary", {})
    for key in ("entities", "attributes", "relations"):
        _require(isinstance(vocab.get(key), list) and vocab[key]
                 and all(isinstance(v, str) and v for v in vocab[key]),
                 f"vocabulary.{key} 必须为非空字符串列表")
    _require(len(set(vocab["entities"])) == len(vocab["entities"])
             and len(set(vocab["attributes"])) == len(vocab["attributes"])
             and len(set(vocab["relations"])) == len(vocab["relations"]),
             "受控词表不允许重复条目")

    image = data.get("image", {})
    tile_size = image.get("tile_size")
    _require(isinstance(tile_size, int) and tile_size > 0, "tile_size 必须为正整数")
    _require(width % tile_size == 0 and height % tile_size == 0,
             f"分块尺寸 {tile_size} 必须能整除图像边长 {width}x{height}")
    visual_dim = image.get("visual_dim")
    _require(isinstance(visual_dim, int) and visual_dim > 0, "visual_dim 必须为正整数")
    noise_level = image.get("noise_level", 0)
    _require(isinstance(noise_level, int) and noise_level >= 0, "noise_level 必须为非负整数")

    text = data.get("text", {})
    text_dim = text.get("text_dim")
    _require(isinstance(text_dim, int) and text_dim > 0, "text_dim 必须为正整数")
    default_assertion_count = text.get("default_assertion_count", 4)
    _require(isinstance(default_assertion_count, int) and default_assertion_count > 0,
             "default_assertion_count 必须为正整数")

    alignment = data.get("alignment", {})
    weights = alignment.get("weights", {})
    _require(set(weights) == {"color", "gradient", "edge"},
             "alignment.weights 必须且只能包含 color/gradient/edge")
    for name, value in weights.items():
        _require(_is_finite_number(value) and float(value) >= 0,
                 f"对齐权重 {name} 必须为有限非负数")

    recall = data.get("recall", {})
    channels_enabled = recall.get("channels", {})
    _require(set(channels_enabled) == set(KNOWN_CHANNELS),
             f"recall.channels 必须包含 {KNOWN_CHANNELS}")
    _require(all(isinstance(v, bool) for v in channels_enabled.values()),
             "通道开关必须为布尔值")
    _require(any(channels_enabled.values()), "至少启用一个召回通道")
    channel_weights = recall.get("channel_weights", {})
    _require(set(channel_weights) == set(KNOWN_CHANNELS),
             "channel_weights 必须包含三个通道")
    for name, value in channel_weights.items():
        _require(_is_finite_number(value) and float(value) > 0,
                 f"通道权重 {name} 必须为有限正数")
    temperature = recall.get("fusion_temperature", 1.0)
    _require(_is_finite_number(temperature) and float(temperature) > 0,
             "fusion_temperature 必须为有限正数")
    candidate_cap = recall.get("candidate_cap")
    _require(isinstance(candidate_cap, int) and candidate_cap > 0,
             "candidate_cap 必须为正整数")

    quant = data.get("quantization", {})
    frac_bits = quant.get("frac_bits")
    tier = quant.get("tier")
    _require(isinstance(frac_bits, int) and frac_bits in ALLOWED_FRAC_BITS,
             f"frac_bits 必须属于 {ALLOWED_FRAC_BITS}")
    _require(isinstance(tier, str) and tier in ALLOWED_TIERS,
             f"tier 必须属于 {ALLOWED_TIERS}")
    validate_quant(frac_bits, tier)

    scale = data.get("scale", {})
    scale_tiers = scale.get("tiers")
    _require(isinstance(scale_tiers, list) and len(scale_tiers) >= 2
             and all(isinstance(v, int) and v > 0 for v in scale_tiers),
             "scale.tiers 必须为至少两个正整数")
    _require(all(b > a for a, b in zip(scale_tiers, scale_tiers[1:])),
             "规模档位集合必须严格单调递增")
    active_tier = scale.get("active_tier", scale_tiers[0])
    _require(isinstance(active_tier, int) and active_tier in scale_tiers,
             "active_tier 必须属于 scale.tiers")

    interaction = data.get("interaction", {})
    budget = interaction.get("latency_budget_ms")
    _require(isinstance(budget, int) and budget > 0, "latency_budget_ms 必须为正整数")
    pool = interaction.get("thread_pool_size")
    _require(isinstance(pool, int) and pool > 0, "thread_pool_size 必须为正整数")
    work_ms = int(interaction.get("work_ms_per_task", 0))
    _require(work_ms >= 0, "work_ms_per_task 必须为非负整数")

    snapshot = data.get("snapshot", {})
    interval = snapshot.get("commit_interval_commands")
    _require(isinstance(interval, int) and interval > 0,
             "commit_interval_commands 必须为正整数")
    temp_dir = str(snapshot.get("temp_dir", "_tmp"))
    snapshot_dir = str(snapshot.get("snapshot_dir", "snapshots"))

    table_raw = data.get("strategy_table")
    _require(isinstance(table_raw, list) and table_raw, "strategy_table 必须为非空列表")
    table = [StrategyEntry.from_dict(row) for row in table_raw]
    _require(len({s.id for s in table}) == len(table), "策略 id 必须唯一")

    return Config(
        raw=data,
        resolution=(width, height),
        vocabulary={key: list(vocab[key]) for key in ("entities", "attributes", "relations")},
        tile_size=tile_size,
        visual_dim=int(visual_dim),
        text_dim=int(text_dim),
        noise_level=int(noise_level),
        default_assertion_count=int(default_assertion_count),
        weights={k: float(v) for k, v in weights.items()},
        channels_enabled={k: bool(v) for k, v in channels_enabled.items()},
        channel_weights={k: float(v) for k, v in channel_weights.items()},
        fusion_temperature=float(temperature),
        candidate_cap=int(candidate_cap),
        frac_bits=int(frac_bits),
        tier=str(tier),
        scale_tiers=list(scale_tiers),
        active_tier=int(active_tier),
        latency_budget_ms=int(budget),
        thread_pool_size=int(pool),
        commit_interval_commands=int(interval),
        temp_dir=temp_dir,
        snapshot_dir=snapshot_dir,
        strategy_table=table,
        work_ms_per_task=work_ms,
    )


@dataclass
class Command:
    seq: int
    op: str
    payload: dict[str, Any] = field(default_factory=dict)


def load_commands(path: str, config: Config) -> list[Command]:
    with open(path, "r", encoding="utf-8") as handle:
        data = json.load(handle)
    _require(isinstance(data, list), "命令流必须为列表")
    commands: list[Command] = []
    for index, row in enumerate(data):
        _require(isinstance(row, dict), f"命令 #{index} 必须为对象")
        op = row.get("op")
        _require(op in KNOWN_OPS, f"命令 #{index} 未知操作 {op!r}")
        seq = row.get("seq", index)
        _require(isinstance(seq, int) and seq >= 0, f"命令 #{index} seq 非法")
        payload = {k: v for k, v in row.items() if k not in ("op", "seq")}
        _validate_command_payload(op, payload, config)
        commands.append(Command(seq=int(seq), op=op, payload=payload))
    _require([c.seq for c in commands] == sorted(c.seq for c in commands),
             "命令 seq 必须单调不递减")
    return commands


def _validate_command_payload(op: str, payload: dict[str, Any], config: Config) -> None:
    if op == "set_scale_tier":
        _require(isinstance(payload.get("value"), int)
                 and payload["value"] in config.scale_tiers,
                 "set_scale_tier 的 value 必须属于规模档位集合")
    elif op == "set_weight":
        _require(payload.get("channel") in ("color", "gradient", "edge"),
                 "set_weight 通道必须为 color/gradient/edge")
        _require(_is_finite_number(payload.get("value")) and float(payload["value"]) >= 0,
                 "set_weight 权重必须为有限非负数")
    elif op == "set_quant":
        _require(payload.get("frac_bits") in ALLOWED_FRAC_BITS
                 and payload.get("tier") in ALLOWED_TIERS,
                 "set_quant 参数必须在允许枚举内")
    elif op in ("replace_image", "replace_text"):
        _require(isinstance(payload.get("sample"), str) and payload["sample"],
                 f"{op} 必须指定 sample")
    elif op in ("enable_channel", "disable_channel"):
        _require(payload.get("channel") in KNOWN_CHANNELS,
                 "通道增删目标必须为 lexical/visual/hybrid")
