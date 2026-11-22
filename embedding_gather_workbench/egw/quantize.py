"""定点量化与整数域聚合原语（全程确定性，无随机源）。"""
import numpy as np


def quant_bounds(quant_level):
    """quant_level 为有符号定点整数总位宽，返回 (qmin, qmax)。"""
    if quant_level >= 63:
        qmax = np.iinfo(np.int64).max
        qmin = np.iinfo(np.int64).min
    else:
        qmax = (1 << (quant_level - 1)) - 1
        qmin = -(1 << (quant_level - 1))
    return np.int64(qmin), np.int64(qmax)


def accumulator_bounds(quant_level, headroom_bits):
    acc_bits = quant_level - 1 + headroom_bits
    if acc_bits >= 63:
        return np.iinfo(np.int64).min, np.iinfo(np.int64).max
    return np.int64(-(1 << acc_bits)), np.int64((1 << acc_bits) - 1)


def quantize_step(fractional_bits):
    """每元素量化步长 = 2^-frac_bits。"""
    return np.float64(2.0) ** np.int64(-fractional_bits)


def quantize_row(vector, quant_level, fractional_bits):
    """浮点向量 -> 定点整数（向零取整后按档位截断），返回 (int64数组, 步长, 残差数组)。

    残差 = 原始浮点值 - 反量化值，按行内维度序号升序天然保留。
    """
    values = np.asarray(vector, dtype=np.float64)
    step = quantize_step(fractional_bits)
    scaled = values / step
    truncated = np.trunc(scaled).astype(np.int64)
    qmin, qmax = quant_bounds(quant_level)
    clipped = np.clip(truncated, qmin, qmax)
    dequant = clipped.astype(np.float64) * step
    residual = values - dequant
    return clipped, step, residual


def dequantize(int_vector, fractional_bits):
    step = quantize_step(fractional_bits)
    return np.asarray(int_vector, dtype=np.float64) * step


def saturating_add(acc, row, acc_min, acc_max):
    total = acc.astype(np.int64) + np.asarray(row, dtype=np.int64)
    return np.clip(total, acc_min, acc_max)


def divide_mean(total, count, rounding):
    """定点域均值除法。返回 (商数组, 余数数组)，余数满足 total = quotient*count + remainder。"""
    total = np.asarray(total, dtype=np.int64)
    n = int(count)
    if rounding == "TRUNCATE":
        quotient = np.trunc(total / n).astype(np.int64)
    else:
        quotient, remainder = np.divmod(total, n)
        if rounding == "FLOOR":
            pass
        elif rounding == "HALF_UP":
            quotient = quotient + (np.where(2 * remainder >= n, 1, 0))
        elif rounding == "HALF_EVEN":
            bump = np.where((2 * remainder > n) | ((2 * remainder == n) & (quotient % 2 != 0)), 1, 0)
            quotient = quotient + bump
        else:
            raise ValueError(f"未知舍入规则 {rounding}")
    remainder = total - quotient * n
    return quotient.astype(np.int64), remainder.astype(np.int64)
