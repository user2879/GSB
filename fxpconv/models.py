"""算子描述数据模型与加载校验。

校验失败统一输出业务码并进入失败清单：
非法枚举/非法组合 -> E_DESC_INVALID，形状不符 -> E_SHAPE_MISMATCH，
预算非正 -> E_BUDGET_INVALID。
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path

from .errors import E_BUDGET_INVALID, E_DESC_INVALID, E_SHAPE_MISMATCH, FailureLedger
from .fingerprint import content_fingerprint

ALLOWED_ACC_BITS = (8, 16, 32, 64)
ALLOWED_QBITS = (4, 8, 16)
ALLOWED_SATURATION = ("saturate", "wrap")
ALLOWED_ROUNDING = ("trunc_zero", "round_nearest_even", "floor")

DIMENSION_KEYS = (
    "batch_n",
    "in_channels",
    "in_h",
    "in_w",
    "out_channels",
    "k_h",
    "k_w",
    "stride_h",
    "stride_w",
    "pad_h",
    "pad_w",
)


def int_range(bits: int) -> tuple[int, int]:
    """有符号整数缓冲的允许区间。"""
    bound = 1 << (bits - 1)
    return -bound, bound - 1


@dataclass
class OpDesc:
    name: str
    raw: dict
    batch_n: int
    in_channels: int
    in_h: int
    in_w: int
    out_channels: int
    k_h: int
    k_w: int
    stride_h: int
    stride_w: int
    pad_h: int
    pad_w: int
    acc_bits: int
    qbits: int
    saturation: str
    rounding: str
    frac_bits: int
    out_low: float
    out_high: float
    concurrency: int
    mac_budget: int
    snapshot_every: int
    tolerance: dict
    act_scale: float
    weight_scales: list[float]
    scale_sources: dict = field(default_factory=dict)

    @property
    def out_h(self) -> int:
        return (self.in_h + 2 * self.pad_h - self.k_h) // self.stride_h + 1

    @property
    def out_w(self) -> int:
        return (self.in_w + 2 * self.pad_w - self.k_w) // self.stride_w + 1

    @property
    def positions_per_sample(self) -> int:
        return self.out_channels * self.out_h * self.out_w

    @property
    def total_positions(self) -> int:
        return self.batch_n * self.positions_per_sample

    @property
    def macs_per_position(self) -> int:
        return self.in_channels * self.k_h * self.k_w

    @property
    def config_hash(self) -> str:
        return content_fingerprint(self.raw)

    @property
    def acc_lo(self) -> int:
        return int_range(self.acc_bits)[0]

    @property
    def acc_hi(self) -> int:
        return int_range(self.acc_bits)[1]

    @property
    def q_lo(self) -> int:
        return int_range(self.qbits)[0]

    @property
    def q_hi(self) -> int:
        return int_range(self.qbits)[1]

    def effective_scale(self, out_channel: int) -> float:
        return float(self.act_scale) * float(self.weight_scales[out_channel])

    def to_summary(self) -> dict:
        return {
            "name": self.name,
            "input_shape": [self.batch_n, self.in_channels, self.in_h, self.in_w],
            "weight_shape": [self.out_channels, self.in_channels, self.k_h, self.k_w],
            "kernel": [self.k_h, self.k_w],
            "stride": [self.stride_h, self.stride_w],
            "padding": [self.pad_h, self.pad_w],
            "output_shape": [self.batch_n, self.out_channels, self.out_h, self.out_w],
            "acc_bits": self.acc_bits,
            "qbits": self.qbits,
            "saturation": self.saturation,
            "rounding": self.rounding,
            "frac_bits": self.frac_bits,
            "mac_budget": self.mac_budget,
            "concurrency": self.concurrency,
            "config_hash": self.config_hash,
        }


def _is_positive_int(value) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value > 0


def _is_nonneg_int(value) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def load_descriptor(path: str | Path) -> tuple[OpDesc | None, FailureLedger]:
    ledger = FailureLedger()
    try:
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        ledger.add(E_DESC_INVALID, f"无法解析算子描述: {exc}", "load", (0, "descriptor"))
        return None, ledger

    if not isinstance(raw, dict):
        ledger.add(E_DESC_INVALID, "算子描述根节点必须是对象", "load", (0, "descriptor"))
        return None, ledger

    name = str(raw.get("name", "op"))

    for key in DIMENSION_KEYS:
        if not _is_nonneg_int(raw.get(key)):
            ledger.add(
                E_DESC_INVALID,
                f"维度参数 {key} 必须是非负整数，实际={raw.get(key)!r}",
                "validate",
                (0, "dimension", key),
            )

    # 预算与并发必须为正整数，预算违规使用独立业务码。
    if not _is_positive_int(raw.get("mac_budget")):
        ledger.add(
            E_BUDGET_INVALID,
            f"单批乘加点预算必须为正整数，实际={raw.get('mac_budget')!r}",
            "validate",
            (0, "mac_budget"),
        )
    if not _is_positive_int(raw.get("concurrency")):
        ledger.add(
            E_DESC_INVALID,
            f"并发数必须为正整数，实际={raw.get('concurrency')!r}",
            "validate",
            (0, "concurrency"),
        )
    if not _is_positive_int(raw.get("snapshot_every")):
        ledger.add(
            E_DESC_INVALID,
            f"快照提交间隔必须为正整数，实际={raw.get('snapshot_every')!r}",
            "validate",
            (0, "snapshot_every"),
        )

    acc_bits = raw.get("acc_bits")
    qbits = raw.get("qbits")
    if acc_bits not in ALLOWED_ACC_BITS:
        ledger.add(
            E_DESC_INVALID,
            f"累加缓冲位宽 {acc_bits!r} 不在允许枚举 {ALLOWED_ACC_BITS} 内",
            "validate",
            (0, "enum", "acc_bits"),
        )
    if qbits not in ALLOWED_QBITS:
        ledger.add(
            E_DESC_INVALID,
            f"量化位宽 {qbits!r} 不在允许枚举 {ALLOWED_QBITS} 内",
            "validate",
            (0, "enum", "qbits"),
        )
    saturation = raw.get("saturation")
    if saturation not in ALLOWED_SATURATION:
        ledger.add(
            E_DESC_INVALID,
            f"饱和策略 {saturation!r} 不在允许枚举 {ALLOWED_SATURATION} 内",
            "validate",
            (0, "enum", "saturation"),
        )
    rounding = raw.get("rounding")
    if rounding not in ALLOWED_ROUNDING:
        ledger.add(
            E_DESC_INVALID,
            f"舍入模式 {rounding!r} 不在允许枚举 {ALLOWED_ROUNDING} 内",
            "validate",
            (0, "enum", "rounding"),
        )
    if not _is_positive_int(raw.get("frac_bits")):
        ledger.add(
            E_DESC_INVALID,
            f"定点小数位宽必须为正整数，实际={raw.get('frac_bits')!r}",
            "validate",
            (0, "frac_bits"),
        )

    out_low = raw.get("out_low")
    out_high = raw.get("out_high")
    if not isinstance(out_low, (int, float)) or not math.isfinite(out_low):
        ledger.add(E_DESC_INVALID, f"输出区间下界非法: {out_low!r}", "validate", (0, "out_low"))
    if not isinstance(out_high, (int, float)) or not math.isfinite(out_high):
        ledger.add(E_DESC_INVALID, f"输出区间上界非法: {out_high!r}", "validate", (0, "out_high"))
    if isinstance(out_low, (int, float)) and isinstance(out_high, (int, float)):
        if math.isfinite(out_low) and math.isfinite(out_high) and out_low > out_high:
            ledger.add(E_DESC_INVALID, "输出区间下界不得大于上界", "validate", (0, "out_range"))

    tolerance = raw.get("tolerance")
    if not isinstance(tolerance, dict):
        ledger.add(E_DESC_INVALID, "容差参数必须是对象", "validate", (0, "tolerance"))
        tolerance = {}
    for key in ("accum", "trunc", "deq"):
        section = tolerance.get(key, {})
        if not isinstance(section, dict):
            ledger.add(E_DESC_INVALID, f"容差 {key} 必须是对象", "validate", (0, "tolerance", key))
            continue
        for bound in ("abs", "rel"):
            value = section.get(bound, 0.0)
            if not isinstance(value, (int, float)) or isinstance(value, bool) or value < 0:
                ledger.add(
                    E_DESC_INVALID,
                    f"容差 {key}.{bound} 必须是非负数，实际={value!r}",
                    "validate",
                    (0, "tolerance", key, bound),
                )

    act_scale = raw.get("act_scale")
    weight_scales = raw.get("weight_scales")
    if not isinstance(act_scale, (int, float)) or not math.isfinite(act_scale) or act_scale <= 0:
        ledger.add(
            E_DESC_INVALID,
            f"激活缩放因子必须是正有限数，实际={act_scale!r}",
            "validate",
            (0, "act_scale"),
        )
        act_scale = 1.0
    if not isinstance(weight_scales, list) or not weight_scales:
        ledger.add(
            E_DESC_INVALID,
            "权重缩放因子必须是非空列表（每输出通道一个）",
            "validate",
            (0, "weight_scales"),
        )
        weight_scales = []
    else:
        for index, value in enumerate(weight_scales):
            if not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
                ledger.add(
                    E_DESC_INVALID,
                    f"权重缩放因子[{index}] 必须是正有限数，实际={value!r}",
                    "validate",
                    (0, "weight_scales", index),
                )

    if len(ledger) > 0:
        return None, ledger

    # 形状可计算性检查（输出空间必须为正）。
    desc = OpDesc(
        name=name,
        raw=raw,
        batch_n=raw["batch_n"],
        in_channels=raw["in_channels"],
        in_h=raw["in_h"],
        in_w=raw["in_w"],
        out_channels=raw["out_channels"],
        k_h=raw["k_h"],
        k_w=raw["k_w"],
        stride_h=raw["stride_h"],
        stride_w=raw["stride_w"],
        pad_h=raw["pad_h"],
        pad_w=raw["pad_w"],
        acc_bits=acc_bits,
        qbits=qbits,
        saturation=saturation,
        rounding=rounding,
        frac_bits=raw["frac_bits"],
        out_low=float(out_low),
        out_high=float(out_high),
        concurrency=raw["concurrency"],
        mac_budget=raw["mac_budget"],
        snapshot_every=raw["snapshot_every"],
        tolerance=tolerance,
        act_scale=float(act_scale),
        weight_scales=[float(value) for value in weight_scales],
        scale_sources=raw.get(
            "scale_sources",
            {
                "act_scale": "descriptor.act_scale",
                "weight_scales": "descriptor.weight_scales[oc]",
            },
        ),
    )

    if desc.out_h <= 0 or desc.out_w <= 0:
        ledger.add(
            E_SHAPE_MISMATCH,
            f"由核/步长/填充推导出的输出空间非正: out_h={desc.out_h}, out_w={desc.out_w}",
            "validate",
            (0, "output_shape"),
        )
        return None, ledger
    if len(desc.weight_scales) != desc.out_channels:
        ledger.add(
            E_SHAPE_MISMATCH,
            f"权重缩放因子数量 {len(desc.weight_scales)} 与输出通道数 {desc.out_channels} 不一致",
            "validate",
            (0, "weight_scales_len"),
        )
        return None, ledger
    if saturation == "wrap" and (acc_bits < 16 or qbits < 16):
        ledger.add(
            E_DESC_INVALID,
            "饱和策略 wrap 仅允许 acc_bits>=16 且 qbits>=16 的组合，"
            f"实际 acc_bits={acc_bits}, qbits={qbits}",
            "validate",
            (0, "illegal_strategy_combo"),
        )
        return None, ledger
    return desc, ledger
