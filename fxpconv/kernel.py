"""整数域定点卷积内核。

累加顺序固定为 输出通道 -> 输入通道 -> 核行 -> 核列 的嵌套升序。
使用 Python 大整数先完成精确累加，再对照描述声明的累加缓冲位宽
做溢出前移处置，从而能记录“溢出发生的第一处累加坐标、方向与
溢出前部分和”，并保证同一输入多次运行逐元素一致。
"""

from __future__ import annotations

from dataclasses import dataclass

from .fingerprint import combine_fingerprints, ints_fingerprint
from .models import OpDesc
from .tensorpack import Tensor, TensorPack


@dataclass
class OverflowEvent:
    coord: tuple[int, int, int, int]
    direction: str
    partial_before: int

    def to_dict(self) -> dict:
        return {
            "first_overflow_coord": list(self.coord),
            "direction": self.direction,
            "partial_before": self.partial_before,
        }


@dataclass
class PositionResult:
    position: int
    batch_index: int
    out_channel: int
    out_h: int
    out_w: int
    raw_sum: int
    buffered: int
    overflow: OverflowEvent | None
    window_fingerprint: str
    act_row_ids: tuple[str, ...]
    weight_row_ids: tuple[str, ...]
    mac_count: int

    def to_dict(self) -> dict:
        return {
            "position": self.position,
            "batch_index": self.batch_index,
            "out_channel": self.out_channel,
            "out_h": self.out_h,
            "out_w": self.out_w,
            "raw_sum": self.raw_sum,
            "buffered": self.buffered,
            "overflow": None if self.overflow is None else self.overflow.to_dict(),
            "window_fingerprint": self.window_fingerprint,
            "act_row_ids": list(self.act_row_ids),
            "weight_row_ids": list(self.weight_row_ids),
            "mac_count": self.mac_count,
        }


def split_position(desc: OpDesc, position: int) -> tuple[int, int, int, int]:
    """全局位置索引 -> (批次, 输出通道, 输出行, 输出列)。"""
    per_sample = desc.positions_per_sample
    batch_index = position // per_sample
    within = position % per_sample
    out_channel, remainder = divmod(within, desc.out_h * desc.out_w)
    out_row, out_col = divmod(remainder, desc.out_w)
    return batch_index, out_channel, out_row, out_col


def _padded_act(activation: Tensor, n: int, c: int, h: int, w: int) -> int:
    """填充区按定点卷积“清零”规则贡献 0。"""
    _, _, in_h, in_w = activation.shape
    if h < 0 or w < 0 or h >= in_h or w >= in_w:
        return 0
    return activation.get(n, c, h, w)


def compute_position(
    desc: OpDesc,
    pack: TensorPack,
    position: int,
    *,
    saturation: str | None = None,
    acc_bits: int | None = None,
    activation_override: Tensor | None = None,
) -> PositionResult:
    """计算单个输出位置的整数累加（含缓冲溢出处置）。"""
    saturation = saturation or desc.saturation
    acc_bits = acc_bits or desc.acc_bits
    lo = -(1 << (acc_bits - 1))
    hi = (1 << (acc_bits - 1)) - 1

    activation = activation_override if activation_override is not None else pack.get("activation")
    weight = pack.get("weight")
    batch_index, out_channel, out_row, out_col = split_position(desc, position)

    # 缓冲先显式清零，再按固定嵌套顺序累加。
    buffered = 0
    raw_sum = 0
    first_overflow: OverflowEvent | None = None
    window_values: list[int] = []
    act_row_ids: list[str] = []
    weight_row_ids: list[str] = []
    mac_count = 0

    for in_channel in range(desc.in_channels):
        for kernel_row in range(desc.k_h):
            for kernel_col in range(desc.k_w):
                in_h = out_row * desc.stride_h + kernel_row - desc.pad_h
                in_w = out_col * desc.stride_w + kernel_col - desc.pad_w
                act_value = _padded_act(activation, batch_index, in_channel, in_h, in_w)
                weight_value = weight.get(out_channel, in_channel, kernel_row, kernel_col)
                product = act_value * weight_value
                partial_before = buffered
                candidate = buffered + product
                raw_sum += product
                window_values.extend((act_value, weight_value))
                mac_count += 1

                # 每个 (ic,kh,kw) 坐标都检查一次缓冲区间，记录第一处溢出。
                if candidate < lo or candidate > hi:
                    direction = "positive" if candidate > hi else "negative"
                    if first_overflow is None:
                        first_overflow = OverflowEvent(
                            coord=(in_channel, kernel_row, kernel_col),
                            direction=direction,
                            partial_before=partial_before,
                        )
                    if saturation == "saturate":
                        candidate = hi if candidate > hi else lo
                    else:  # wrap：二补数回绕（仅在 >=16 位组合下放行）
                        candidate = _twos_complement_wrap(candidate, acc_bits)
                buffered = candidate

        # 行标识按该位置实际引用的激活/权重逻辑行登记（填充行标注 PAD）。
        weight_row_ids.append(
            f"{weight.name}:oc{out_channel}:ic{in_channel}:rows0..{desc.k_h - 1}"
        )
        act_rows = set()
        for kernel_row in range(desc.k_h):
            in_h = out_row * desc.stride_h + kernel_row - desc.pad_h
            if 0 <= in_h < activation.shape[2]:
                act_rows.add(in_h)
        if act_rows:
            act_row_ids.append(
                f"{activation.name}:n{batch_index}:ic{in_channel}:h{sorted(act_rows)}"
            )
        else:
            act_row_ids.append(f"{activation.name}:n{batch_index}:ic{in_channel}:PAD")

    window_fp = combine_fingerprints(
        ints_fingerprint(window_values),
        f"pos={position}",
        f"acc_bits={acc_bits}",
    )
    return PositionResult(
        position=position,
        batch_index=batch_index,
        out_channel=out_channel,
        out_h=out_row,
        out_w=out_col,
        raw_sum=raw_sum,
        buffered=buffered,
        overflow=first_overflow,
        window_fingerprint=window_fp,
        act_row_ids=tuple(act_row_ids),
        weight_row_ids=tuple(weight_row_ids),
        mac_count=mac_count,
    )


def _twos_complement_wrap(value: int, bits: int) -> int:
    span = 1 << bits
    wrapped = value % span
    if wrapped >= 1 << (bits - 1):
        wrapped -= span
    return wrapped


def gather_window_fingerprint(
    desc: OpDesc,
    pack: TensorPack,
    position: int,
    *,
    acc_bits: int | None = None,
    activation_override: Tensor | None = None,
) -> str:
    """只采集输入窗口内容并计算指纹，不执行任何乘加/截断计算。"""
    acc_bits = acc_bits or desc.acc_bits
    activation = activation_override if activation_override is not None else pack.get("activation")
    weight = pack.get("weight")
    batch_index, _out_channel, out_row, out_col = split_position(desc, position)
    window_values: list[int] = []
    for in_channel in range(desc.in_channels):
        for kernel_row in range(desc.k_h):
            for kernel_col in range(desc.k_w):
                in_h = out_row * desc.stride_h + kernel_row - desc.pad_h
                in_w = out_col * desc.stride_w + kernel_col - desc.pad_w
                act_value = _padded_act(activation, batch_index, in_channel, in_h, in_w)
                weight_value = weight.get(
                    _out_channel, in_channel, kernel_row, kernel_col
                )
                window_values.extend((act_value, weight_value))
    return combine_fingerprints(
        ints_fingerprint(window_values),
        f"pos={position}",
        f"acc_bits={acc_bits}",
    )


def positions_referencing_activation_cell(
    desc: OpDesc, batch_index: int, in_channel: int, in_h: int, in_w: int
) -> set[int]:
    """几何判定：哪些输出位置的输入窗口引用了给定激活单元。"""
    affected: set[int] = set()
    for out_channel in range(desc.out_channels):
        for out_row in range(desc.out_h):
            for out_col in range(desc.out_w):
                hit = False
                for kernel_row in range(desc.k_h):
                    for kernel_col in range(desc.k_w):
                        h = out_row * desc.stride_h + kernel_row - desc.pad_h
                        w = out_col * desc.stride_w + kernel_col - desc.pad_w
                        if h == in_h and w == in_w:
                            hit = True
                if hit:
                    within = (
                        out_channel * desc.out_h * desc.out_w + out_row * desc.out_w + out_col
                    )
                    affected.add(batch_index * desc.positions_per_sample + within)
    return affected
