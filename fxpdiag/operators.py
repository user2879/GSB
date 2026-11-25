"""行向数值稳定 Softmax 与 LayerNorm（定点整数路径 + float64 真值路径）。"""

from __future__ import annotations

import math
from dataclasses import dataclass

from .fixedpoint import QuantSpec, Quantizer, RoundingEvent, exp_fixed, rsqrt_int
from .models import Scheme

EXP_APPROX = "EXP_RANGE_REDUCTION_TAYLOR4"
ACCUM_ORDER = "COL_ASC"
DEGENERATE_RSQRT_FALLBACK = "ZERO_VAR_RSQRT_FALLBACK_ONE"


@dataclass
class RowOpResult:
    """单行单算子的完整产物（定点路径）。"""

    layer: str
    row_index: int
    version: str
    quant_key: str
    step: float
    softmax_int: list[int]
    softmax_float: list[float]
    softmax_argmax: int
    softmax_intermediates: list[dict]
    softmax_events: list[dict]
    ln_int: list[int]
    ln_float: list[float]
    ln_stats: dict
    ln_events: list[dict]
    degenerate: bool
    bias_enabled: bool

    def to_dict(self) -> dict:
        return {
            "layer": self.layer,
            "row_index": self.row_index,
            "version": self.version,
            "quant_key": self.quant_key,
            "step": self.step,
            "softmax": {
                "int": self.softmax_int,
                "float": self.softmax_float,
                "argmax": self.softmax_argmax,
                "intermediates": self.softmax_intermediates,
                "events": self.softmax_events,
                "exp_approx": EXP_APPROX,
                "accum_order": ACCUM_ORDER,
            },
            "layernorm": {
                "int": self.ln_int,
                "float": self.ln_float,
                "stats": self.ln_stats,
                "events": self.ln_events,
                "degenerate": self.degenerate,
                "bias_enabled": self.bias_enabled,
            },
        }


def _events_to_dicts(events: list[RoundingEvent]) -> list[dict]:
    return [
        {
            "op": event.op,
            "index": event.index,
            "direction": event.direction,
            "before": event.before,
            "after_q": event.after_q,
            "step": event.step,
        }
        for event in events
    ]


def fixed_softmax(row: list[float], quantizer: Quantizer) -> dict:
    """定点数值稳定 Softmax：减行最大值 -> 定点指数 -> 升序累加 -> 归一。"""

    width = len(row)
    q_inputs = [quantizer.quantize(value, "sm.input", col) for col, value in enumerate(row)]
    q_max = max(q_inputs)
    intermediates: list[dict] = []
    q_exp: list[int] = []
    for col in range(width):
        q_diff = quantizer.add(-q_max, q_inputs[col], "sm.shift", col)
        q_e = exp_fixed(q_diff, quantizer, col)
        q_exp.append(q_e)
        intermediates.append(
            {
                "col": col,
                "q_input": q_inputs[col],
                "q_shift": q_diff,
                "q_exp": q_e,
            }
        )
    # 列序号升序累加（显式固定顺序）
    q_sum = 0
    for col in range(width):
        q_sum = quantizer.add(q_sum, q_exp[col], "sm.sum.asc", col)
    outputs = [
        quantizer.div(q_exp[col], q_sum, "sm.normalize", col) for col in range(width)
    ]
    floats = [quantizer.dequant(value) for value in outputs]
    argmax = max(range(width), key=lambda col: (outputs[col], -col))
    return {
        "int": outputs,
        "float": floats,
        "argmax": argmax,
        "intermediates": intermediates,
    }


def fixed_layernorm(
    row: list[float],
    gamma: float,
    beta: float,
    bias_enabled: bool,
    quantizer: Quantizer,
) -> dict:
    """定点 LayerNorm：升序均值/方差 -> 归一 -> 缩放偏置；零方差走退化规则。"""

    width = len(row)
    f = quantizer.frac_bits
    one = 1 << f
    q_inputs = [quantizer.quantize(value, "ln.input", col) for col, value in enumerate(row)]
    q_sum = 0
    for col in range(width):
        q_sum = quantizer.add(q_sum, q_inputs[col], "ln.mean.asc", col)
    q_mean = quantizer.div(q_sum, quantizer.quantize(float(width), "ln.n", 0),
                           "ln.mean.div", 0)
    q_sq_sum = 0
    q_diffs: list[int] = []
    for col in range(width):
        q_diff = quantizer.add(q_inputs[col], -q_mean, "ln.diff", col)
        q_diffs.append(q_diff)
        q_sq = quantizer.mul(q_diff, q_diff, "ln.sq", col)
        q_sq_sum = quantizer.add(q_sq_sum, q_sq, "ln.var.asc", col)
    q_var = quantizer.div(q_sq_sum, quantizer.quantize(float(width), "ln.n", 1),
                          "ln.var.div", 0)
    degenerate = q_var <= 0
    if degenerate:
        # 退化行：倒数标准差回退为 1（即不做缩放），并显式标记
        q_rsqrt = one
        rsqrt_iters = 0
    else:
        q_rsqrt = rsqrt_int(q_var, f)
        rsqrt_iters = 32
    q_gamma = quantizer.quantize(gamma, "ln.gamma", 0)
    q_beta = quantizer.quantize(beta if bias_enabled else 0.0, "ln.beta", 0)
    outputs: list[int] = []
    for col in range(width):
        q_norm = quantizer.mul(q_diffs[col], q_rsqrt, "ln.norm", col)
        q_scaled = quantizer.mul(q_norm, q_gamma, "ln.scale", col)
        if bias_enabled:
            q_out = quantizer.add(q_scaled, q_beta, "ln.bias.add", col)
        else:
            q_out = q_scaled
        outputs.append(q_out)
    stats = {
        "mean_q": q_mean,
        "mean_float": quantizer.dequant(q_mean),
        "var_q": q_var,
        "var_float": quantizer.dequant(q_var),
        "rsqrt_q": q_rsqrt,
        "rsqrt_float": quantizer.dequant(q_rsqrt),
        "rsqrt_iters": rsqrt_iters,
        "degenerate_rule": DEGENERATE_RSQRT_FALLBACK if degenerate else None,
        "gamma_q": q_gamma,
        "beta_q": q_beta,
        "accum_order": ACCUM_ORDER,
    }
    return {
        "int": outputs,
        "float": [quantizer.dequant(value) for value in outputs],
        "stats": stats,
        "degenerate": degenerate,
    }


def compute_row(
    layer: str,
    row_index: int,
    row: list[float],
    gamma: float,
    beta: float,
    scheme: Scheme,
    spec: QuantSpec,
    version: str,
) -> RowOpResult:
    """执行单行双算子（v1/v2 各自独立的量化器与事件轨迹）。"""

    sm_trace: list[RoundingEvent] = []
    sm_quantizer = Quantizer(spec, sm_trace)
    softmax = fixed_softmax(row, sm_quantizer)

    ln_trace: list[RoundingEvent] = []
    ln_quantizer = Quantizer(spec, ln_trace)
    layernorm = fixed_layernorm(row, gamma, beta, scheme.use_bias, ln_quantizer)

    return RowOpResult(
        layer=layer,
        row_index=row_index,
        version=version,
        quant_key=spec.key(),
        step=sm_quantizer.step,
        softmax_int=softmax["int"],
        softmax_float=softmax["float"],
        softmax_argmax=softmax["argmax"],
        softmax_intermediates=softmax["intermediates"],
        softmax_events=_events_to_dicts(sm_trace),
        ln_int=layernorm["int"],
        ln_float=layernorm["float"],
        ln_stats=layernorm["stats"],
        ln_events=_events_to_dicts(ln_trace),
        degenerate=layernorm["degenerate"],
        bias_enabled=scheme.use_bias,
    )


# ---------------------------------------------------------------------- #
# float64 无量化真值路径
# ---------------------------------------------------------------------- #
def float64_softmax(row: list[float]) -> list[float]:
    row_max = max(row)
    exps = [math.exp(value - row_max) for value in row]
    total = 0.0
    for value in exps:  # 同样固定升序累加
        total += value
    return [value / total for value in exps]


def float64_layernorm(
    row: list[float], gamma: float, beta: float, bias_enabled: bool
) -> dict:
    width = len(row)
    total = 0.0
    for value in row:
        total += value
    mean = total / width
    var_total = 0.0
    for value in row:
        var_total += (value - mean) ** 2
    variance = var_total / width
    degenerate = variance == 0.0
    rsqrt = 1.0 if degenerate else 1.0 / math.sqrt(variance)
    outputs = [
        (value - mean) * rsqrt * gamma + (beta if bias_enabled else 0.0)
        for value in row
    ]
    return {
        "float": outputs,
        "stats": {
            "mean": mean,
            "variance": variance,
            "rsqrt": rsqrt,
            "degenerate": degenerate,
        },
    }
