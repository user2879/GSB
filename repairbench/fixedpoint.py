"""整数定点量化与误差界核算（需求 4）。

定点模型（显式规则）：
  - 会话缩放因子 scale = 10^e，e 为 0..6 的整数（含端点）。
  - 定点单位 = 1 KB（1024 字节）；1 个量子(quantum) = 1/scale KB = 1024/scale 字节。
  - 量化 quantize_bytes(B) = trunc_toward_zero(B * scale / 1024)，截断方向恒为向零截断。
  - 乘法 mul(a,b) = trunc_toward_zero(a * b / scale)；除法 div(a,b) = trunc_toward_zero(a * scale / b)。
  - 溢出饱和规则：任何结果超出 int64 范围时饱和到 INT64_MAX / INT64_MIN；
    除以零按被除数符号饱和（0 视为正）。
  - 所有中间量用 Python 任意精度整数计算后再饱和，聚合统计使用 numpy int64。

误差界三段拆分（均以量子为单位计量，1 量子 = 1024/scale 字节）：
  1) quantization_trunc  量化截断：每次 quantize/div 截断误差 < 1 量子。
  2) batch_split_rounding 批次切分取整：预算按整段预留，块末段向上取整
     引入 0..segment_bytes-1 字节的取整量。
  3) queue_delay_conv     排队延迟折算：等待毫秒按节点带宽折算字节当量时
     每次折算截断误差 < 1 量子。
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

INT64_MAX = 2**63 - 1
INT64_MIN = -(2**63)

SCALE_EXP_MIN = 0
SCALE_EXP_MAX = 6

BYTES_PER_KB = 1024


def _saturate(value: int) -> int:
    if value > INT64_MAX:
        return INT64_MAX
    if value < INT64_MIN:
        return INT64_MIN
    return value


def _trunc_div(numer: int, denom: int) -> int:
    """向零截断的整数除法（denom 不得为 0）。"""
    q = abs(numer) // abs(denom)
    if (numer < 0) != (denom < 0):
        q = -q
    return q


class FixedPoint:
    """会话级定点上下文。"""

    def __init__(self, scale_exponent: int):
        if not isinstance(scale_exponent, int) or isinstance(scale_exponent, bool):
            raise ValueError("scale_exponent 必须为整数")
        if not (SCALE_EXP_MIN <= scale_exponent <= SCALE_EXP_MAX):
            raise ValueError(
                f"scale_exponent 须在 [{SCALE_EXP_MIN},{SCALE_EXP_MAX}]，得到 {scale_exponent}"
            )
        self.exponent = scale_exponent
        self.scale = 10**scale_exponent

    # ---- 基本运算（全部整数、向零截断、int64 饱和） ----
    def quantize_units(self, units: int) -> int:
        """整数单位量（如 KB/s、权重）量化：units * scale，精确无误差。"""
        return _saturate(units * self.scale)

    def quantize_bytes(self, byte_count: int) -> int:
        """字节量量化为 KB 定点：trunc(B * scale / 1024)，截断误差 < 1 量子。"""
        return _saturate(_trunc_div(byte_count * self.scale, BYTES_PER_KB))

    def dequantize_bytes(self, quanta: int) -> int:
        """KB 定点还原为字节：trunc(quanta * 1024 / scale)，向零截断。"""
        return _saturate(_trunc_div(quanta * BYTES_PER_KB, self.scale))

    def mul(self, a: int, b: int) -> int:
        return _saturate(_trunc_div(a * b, self.scale))

    def div(self, a: int, b: int) -> int:
        if b == 0:
            return INT64_MIN if a < 0 else INT64_MAX
        return _saturate(_trunc_div(a * self.scale, b))

    def quantize_ratio(self, numer: int, denom: int) -> int:
        """比例量化 trunc(numer * scale / denom)，如失配千分比 -> 定点比例。"""
        if denom == 0:
            return INT64_MIN if numer < 0 else INT64_MAX
        return _saturate(_trunc_div(numer * self.scale, denom))

    # ---- 误差界换算 ----
    def quantum_bytes(self) -> float:
        """1 量子折合字节数 = 1024 / scale。"""
        return BYTES_PER_KB / self.scale

    def quanta_to_bytes_bound(self, quanta: int) -> int:
        """quanta 数量子误差上界折合字节（向上取整，保守界）。"""
        return -((-quanta * BYTES_PER_KB) // self.scale)


@dataclass
class ErrorContribution:
    """三段误差贡献，内部以量子(int)计量。"""

    quantization_trunc: int = 0
    batch_split_rounding: int = 0
    queue_delay_conv: int = 0

    def total(self) -> int:
        return self.quantization_trunc + self.batch_split_rounding + self.queue_delay_conv

    def dominant(self) -> str:
        parts = {
            "quantization_trunc": self.quantization_trunc,
            "batch_split_rounding": self.batch_split_rounding,
            "queue_delay_conv": self.queue_delay_conv,
        }
        # 稳定决胜：并列时按固定字段序取先者，保证与处理顺序无关。
        return max(sorted(parts), key=lambda k: parts[k])

    def to_dict(self, fp: FixedPoint) -> dict:
        return {
            "quantization_trunc_quanta": self.quantization_trunc,
            "batch_split_rounding_quanta": self.batch_split_rounding,
            "queue_delay_conv_quanta": self.queue_delay_conv,
            "total_quanta": self.total(),
            "total_bytes_bound": fp.quanta_to_bytes_bound(self.total()),
            "dominant": self.dominant(),
        }


@dataclass
class ErrorLedger:
    """误差界累计器：单块、逐节点、全局三级。"""

    fp: FixedPoint
    per_block: dict = field(default_factory=dict)
    per_node: dict = field(default_factory=dict)

    def _slot(self, table: dict, key: str) -> ErrorContribution:
        if key not in table:
            table[key] = ErrorContribution()
        return table[key]

    def add_block_quantization(self, block_id: str, node_id: str, quanta: int) -> None:
        self._slot(self.per_block, block_id).quantization_trunc += quanta
        self._slot(self.per_node, node_id).quantization_trunc += quanta

    def add_batch_split(self, block_id: str, node_id: str, quanta: int) -> None:
        self._slot(self.per_block, block_id).batch_split_rounding += quanta
        self._slot(self.per_node, node_id).batch_split_rounding += quanta

    def add_queue_delay(self, block_id: str, node_id: str, quanta: int) -> None:
        self._slot(self.per_block, block_id).queue_delay_conv += quanta
        self._slot(self.per_node, node_id).queue_delay_conv += quanta

    def block_byte_error_quanta(self, block_id: str) -> int:
        """单块字节量化误差（量化截断段，量子）。"""
        return self._slot(self.per_block, block_id).quantization_trunc

    def node_bound(self, node_id: str) -> ErrorContribution:
        return self._slot(self.per_node, node_id)

    def global_bound(self) -> ErrorContribution:
        """全局修复量误差上界 = 各节点三段贡献之和（numpy int64 聚合）。"""
        totals = np.zeros(3, dtype=np.int64)
        for contrib in self.per_node.values():
            totals += np.array(
                [
                    contrib.quantization_trunc,
                    contrib.batch_split_rounding,
                    contrib.queue_delay_conv,
                ],
                dtype=np.int64,
            )
        return ErrorContribution(
            quantization_trunc=int(totals[0]),
            batch_split_rounding=int(totals[1]),
            queue_delay_conv=int(totals[2]),
        )

    def report(self) -> dict:
        return {
            "scale_exponent": self.fp.exponent,
            "quantum_bytes": self.fp.quantum_bytes(),
            "per_block": {
                bid: c.to_dict(self.fp) for bid, c in sorted(self.per_block.items())
            },
            "per_node": {
                nid: c.to_dict(self.fp) for nid, c in sorted(self.per_node.items())
            },
            "global": self.global_bound().to_dict(self.fp),
        }
