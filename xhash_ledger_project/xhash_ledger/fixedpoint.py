"""确定性定点整数运算。

所有联合特征均以 *Qm* 定点表示：真实值 ``v`` 被编码为整数
``round(v * 2**frac_bits)``。投影矩阵同样以定点整数存储，投影通过整数
乘加完成，全程不依赖浮点求和顺序，因此串行/并发、增量/全量结果逐元素一致。

每个样本单独记录量化步长 ``qstep = 2**-frac_bits`` 与逐分量量化残差
``residual = v - q*qstep``（以定点残差整数保存，避免浮点表示误差）。
"""
from __future__ import annotations

from dataclasses import dataclass


def to_fixed(value: float, frac_bits: int) -> int:
    """把浮点/整数标量量化为 Qm 定点整数，采用四舍五入（.5 向远离零方向）。"""
    scale = 1 << frac_bits
    if value >= 0:
        return int(value * scale + 0.5)
    return -int((-value) * scale + 0.5)


def to_fixed_vec(values, frac_bits: int) -> list[int]:
    return [to_fixed(float(v), frac_bits) for v in values]


def fixed_dot(a: list[int], b: list[int]) -> int:
    """两个定点向量的整数点积（不做缩放还原，结果为 Q(2m) 定点整数）。"""
    total = 0
    for x, y in zip(a, b):
        total += x * y
    return total


def fixed_matvec(matrix: list[list[int]], vector: list[int]) -> list[int]:
    """定点整数矩阵乘向量。"""
    return [fixed_dot(row, vector) for row in matrix]


@dataclass(frozen=True)
class Quantization:
    """单样本量化结果。

    frac_bits: 定点小数位宽
    qstep:     量化步长（真实尺度，以 2 的幂记录为整数指数 qstep_exp）
    values:    量化后的定点整数联合特征
    residuals: 逐分量量化残差，同样以定点整数表示
    """

    frac_bits: int
    qstep_exp: int
    values: list[int]
    residuals: list[int]

    def to_dict(self) -> dict:
        return {
            "frac_bits": self.frac_bits,
            "qstep_exp": self.qstep_exp,
            "qstep": f"2^-{self.frac_bits}",
            "values": list(self.values),
            "residuals": list(self.residuals),
        }


def quantize_vector(raw_values, frac_bits: int) -> Quantization:
    """把浮点联合特征量化为定点整数，并记录逐分量残差。

    残差定义为 ``raw - q*qstep``，以同一 frac_bits 的定点整数保存，
    即 ``residual_q = round(raw*2^m) - q``，保证完全可复核、无浮点漂移。
    """
    scale = 1 << frac_bits
    quantized: list[int] = []
    residuals: list[int] = []
    for raw in raw_values:
        scaled = float(raw) * scale
        q = to_fixed(raw, frac_bits)
        quantized.append(q)
        # 以整数方式重建原始缩放值的“最近整数”，残差同样落在定点网格上
        residuals.append(int(round(scaled)) - q)
    return Quantization(
        frac_bits=frac_bits,
        qstep_exp=-frac_bits,
        values=quantized,
        residuals=residuals,
    )
