"""定点表示：浮点张量 -> 有符号定点整数 -> 反量化，并记录步长与残差。

- 位宽枚举：8/16/32（有符号，范围 -2^(w-1) .. 2^(w-1)-1）。
- 量化档位枚举：
  * uniform   : 逐张量对称均匀，step=max(abs(x))/qmax
  * pow2      : 逐张量 2 的幂步长，step=2^floor(log2(max(abs(x))/qmax))
  * per_element: 逐元素独立 step（按元素量级，pow2），逐元素记录
所有量都在配置的精度档位（fp32/fp64）下计算，切换位宽/档位会真实改变
整数表示与反量化值。
"""

from __future__ import annotations

import numpy as np

ALLOWED_BIT_WIDTHS = (8, 16, 32)
ALLOWED_QUANT_MODES = ("uniform", "pow2", "per_element")


def _float_dtype(precision: str) -> np.dtype:
    return np.float64 if precision == "fp64" else np.float32


def _int_dtype(bit_width: int):
    return {8: np.int8, 16: np.int16, 32: np.int32}[bit_width]


def quantize(x: np.ndarray, bit_width: int, mode: str):
    """返回 dict: q(int ndarray), step, dequant(float), residual(float),
    以及枚举信息。step/residual 形状：per_element 时与 x 同形，否则标量。
    """
    if bit_width not in ALLOWED_BIT_WIDTHS:
        raise ValueError(f"bit width {bit_width} not in {ALLOWED_BIT_WIDTHS}")
    if mode not in ALLOWED_QUANT_MODES:
        raise ValueError(f"quant mode {mode} not in {ALLOWED_QUANT_MODES}")

    dt = x.dtype
    idt = _int_dtype(bit_width)
    qmax = float(2 ** (bit_width - 1) - 1)
    eps = np.finfo(dt).tiny

    if mode == "per_element":
        amax = np.maximum(np.abs(x).astype(dt), eps)
        step = np.power(
            np.array(2.0, dtype=dt),
            np.floor(np.log2(amax / qmax)).astype(np.int64),
        ).astype(dt)
        step_b = step
    else:
        amax = float(np.max(np.abs(x)))
        if amax == 0.0:
            step_s = np.array(eps, dtype=dt)
        elif mode == "uniform":
            step_s = np.array(amax / qmax, dtype=dt)
        else:  # pow2
            step_s = np.power(
                np.array(2.0, dtype=dt),
                np.int64(np.floor(np.log2((amax / qmax)))),
            ).astype(dt)
        step = step_s
        step_b = np.broadcast_to(step_s, x.shape)

    q = np.clip(np.rint(x / step_b), -qmax - 1, qmax).astype(idt)
    dequant = (q.astype(dt) * step_b).astype(dt)
    residual = (x - dequant).astype(dt)
    return {
        "q": q,
        "step": step,
        "dequant": dequant,
        "residual": residual,
        "bit_width": bit_width,
        "mode": mode,
        "qmax": qmax,
    }
