"""确定性逐元素数学。

所有近似形式、多项式阶数与迭代次数都是固定常量，并在产物元数据中显式
记录。运算顺序固定（Horner 法 / 固定次数 Newton 迭代），同一精度档位
同一输入多次运行逐元素一致；切换 fp32/fp64 会真实改变结果。

为了同时得到"每个输出元素的中间舍入次数"，本模块在返回 ndarray 的同时
返回与之同形状的舍入次数数组（每个元素标量运算的浮点 round 次数）。
"""

from __future__ import annotations

import numpy as np

EXP_POLY_DEGREE = 12          # exp 余项 Taylor 多项式阶数
SQRT_NEWTON_ITERATIONS = 10   # sqrt Newton-Raphson 固定迭代次数
EXP_RANGE_BITS = 16           # exp 范围归约 k = round(x / ln2)


def _dt(dtype: np.dtype) -> np.dtype:
    if dtype == np.float64:
        return np.float64
    if dtype == np.float32:
        return np.float32
    raise ValueError(f"unsupported precision dtype: {dtype}")


def approx_exp(x: np.ndarray):
    """确定性 exp：范围归约 + 固定阶 Taylor 多项式（Horner 求值）。

    令 k=round(x/ln2)，r=x-k*ln2，则 exp(x)=2^k * P(r)，
    P 为 exp 的 0..EXP_POLY_DEGREE 次 Taylor 多项式。
    返回 (y, round_count)。
    """
    dt = _dt(x.dtype)
    ln2 = np.array(np.log(2.0), dtype=dt)
    k = np.rint(x / ln2).astype(np.int64)
    r = x - k.astype(dt) * ln2
    # 范围内的 exp 用 Horner：P = 1 + r(1 + r(1/2 + r(1/6 + ...)))
    coeffs = [np.array(1.0, dtype=dt)]
    fact = 1.0
    for n in range(1, EXP_POLY_DEGREE + 1):
        fact *= n
        coeffs.append(np.array(1.0 / fact, dtype=dt))
    poly = coeffs[-1].copy()
    poly_rounds = 0
    for c in reversed(coeffs[:-1]):
        poly = poly * r + c      # 1 mul + 1 add
        poly_rounds += 2
    y = np.ldexp(poly, k.astype(np.int64)).astype(dt)
    # 舍入计数（每元素标量 round 次数，与形状无关的常数）：
    # x/ln2=1, rint 归约=1, k*ln2=1, 减法=1, Horner=2*degree, ldexp=1
    rounds = 1 + 1 + 1 + 1 + poly_rounds + 1
    return y, np.full(x.shape, rounds, dtype=np.int64)


def approx_sqrt(x: np.ndarray):
    """确定性 sqrt：固定次数 Newton-Raphson，初值 y0 用定点位估计。

    迭代式 y_{n+1} = (y_n + x / y_n) / 2。
    返回 (y, round_count)。
    """
    dt = _dt(x.dtype)
    x = x.astype(dt)
    # 初值：用 frexp 取指数，y0 = 2**floor(m/2)，全程确定性。
    mantissa, exponent = np.frexp(x)
    y0 = np.ldexp(np.ones_like(x), (exponent // 2).astype(np.int64)).astype(dt)
    y = y0
    for _ in range(SQRT_NEWTON_ITERATIONS):
        y = (y + x / y) * np.array(0.5, dtype=dt)  # div, add, mul 各 1
    # 初值构造 round 次数保守计 2（frexp/ldexp 各 1），每次迭代 3
    rounds = 2 + SQRT_NEWTON_ITERATIONS * 3
    return y.astype(dt), np.full(x.shape, rounds, dtype=np.int64)


def relu(x: np.ndarray, slope: float):
    """ReLU（带非负斜率）：x>=0 -> x，否则 slope*x。固定分支顺序。"""
    dt = _dt(x.dtype)
    s = np.array(slope, dtype=dt)
    y = np.where(x >= 0, x, s * x).astype(dt)
    # 比较 1 + 选择后乘法 1（负分支），统一记 2
    return y, np.full(x.shape, 2, dtype=np.int64)


def gelu(x: np.ndarray):
    """GELU tanh 近似（固定运算顺序），exp 用 approx_exp。

    gelu(x) = 0.5 x (1 + tanh(z))
    z = sqrt(2/pi) (x + 0.044715 x^3)
    tanh(z) = 1 - 2 / (exp(2z)^2 + 1)
    """
    dt = _dt(x.dtype)
    c = np.array(np.sqrt(2.0 / np.pi), dtype=dt)
    cubic_coef = np.array(0.044715, dtype=dt)
    half = np.array(0.5, dtype=dt)
    one = np.array(1.0, dtype=dt)
    two = np.array(2.0, dtype=dt)

    x3 = x * x * x                       # 2 rounds
    inner = x + cubic_coef * x3          # mul 1 + add 1
    z = c * inner                        # 1
    two_z = two * z                      # 1
    e2z, exp_rounds = approx_exp(two_z)  # exp 内部若干
    denom = e2z * e2z + one              # mul 1 + add 1
    tanh_z = one - two / denom           # div 1 + sub 1
    y = half * x * (one + tanh_z)        # add 1 + 2 mul
    rounds = (2 + 2 + 1 + 1 + exp_rounds
              + 2 + 2 + 3)
    return y.astype(dt), rounds.astype(np.int64)


def sigmoid(x: np.ndarray, scale: float):
    """sigmoid(scale*x) = 1/(1+exp(-scale*x))，exp 用 approx_exp。"""
    dt = _dt(x.dtype)
    s = np.array(scale, dtype=dt)
    one = np.array(1.0, dtype=dt)
    sx = s * x                            # 1
    neg = -sx                             # 1
    ex, exp_rounds = approx_exp(neg)
    denom = one + ex                      # 1
    y = one / denom                       # 1
    rounds = 1 + 1 + exp_rounds + 1 + 1
    return y.astype(dt), rounds.astype(np.int64)


def tanh_act(x: np.ndarray, gain: float):
    """tanh(gain*x)，经由 exp 近似：tanh(a)=(exp(2a)^2-1)/(exp(2a)^2+1)。"""
    dt = _dt(x.dtype)
    g = np.array(gain, dtype=dt)
    one = np.array(1.0, dtype=dt)
    two = np.array(2.0, dtype=dt)
    a = g * x                             # 1
    e2a, exp_rounds = approx_exp(two * a) # mul 1 + exp
    e2 = e2a * e2a                        # 1
    num = e2 - one                        # 1
    den = e2 + one                        # 1
    y = num / den                         # 1
    rounds = 1 + 1 + exp_rounds + 1 + 1 + 1 + 1
    return y.astype(dt), rounds.astype(np.int64)


ACTIVATIONS = ("relu", "gelu", "sigmoid", "tanh")


def apply_activation(op: str, x: np.ndarray, params: dict):
    """统一入口，返回 (y, round_count)。"""
    if op == "relu":
        return relu(x, float(params.get("slope", 0.0)))
    if op == "gelu":
        return gelu(x)
    if op == "sigmoid":
        return sigmoid(x, float(params.get("scale", 1.0)))
    if op == "tanh":
        return tanh_act(x, float(params.get("gain", 1.0)))
    raise KeyError(op)


def approx_metadata() -> dict:
    """写入产物元数据的近似形式说明。"""
    return {
        "exp": {
            "form": "range_reduction_taylor_horner",
            "range_bits": EXP_RANGE_BITS,
            "taylor_degree": EXP_POLY_DEGREE,
            "ldexp_2k": True,
        },
        "sqrt": {
            "form": "newton_raphson_fixed_iterations",
            "iterations": SQRT_NEWTON_ITERATIONS,
            "initial_guess": "ldexp_frexp_exponent_half",
        },
        "gelu": "tanh_approximation_via_approx_exp",
        "operation_order": "fixed_horner_and_fixed_newton_iterations",
    }


def gelu_fused_scalar(xv: np.ndarray):
    """融合路径专用 GELU：标量求值 + 显式替代结合顺序。

    与 gelu() 的差异：tanh 表达式改写为
        0.5*x + 0.5*x*tanh(z)
    并在标量上下文按 (0.5*x)*tanh(z) 重新结合，运算顺序与向量化路径不同，
    因此在 fp32 下可产生真实（通常很小）的舍入差异，用于融合容差诊断。
    返回 (y(同形状), round_count)。
    """
    dt = _dt(xv.dtype)
    x = xv.astype(dt, copy=True).reshape(-1)
    out = np.empty_like(x)
    c = np.array(np.sqrt(2.0 / np.pi), dtype=dt)
    cubic_coef = np.array(0.044715, dtype=dt)
    half = np.array(0.5, dtype=dt)
    one = np.array(1.0, dtype=dt)
    two = np.array(2.0, dtype=dt)
    for i in range(x.size):
        xi = x[i:i + 1]
        x3 = xi * xi * xi
        inner = xi + cubic_coef * x3
        z = c * inner
        e2z, exp_rounds = approx_exp(two * z)
        denom = e2z * e2z + one
        tanh_z = one - two / denom
        hx = half * xi                 # 替代结合：先取半
        out[i] = (hx + hx * tanh_z)[0]  # 0.5x + (0.5x)*tanh
    # 舍入次数与 gelu 相同量级（exp 内部 + 外层算术）
    rounds = (2 + 2 + 1 + 1 + 12 + 2 + 2 + 3)
    return out.reshape(xv.shape), np.full(xv.shape, rounds, dtype=np.int64)
